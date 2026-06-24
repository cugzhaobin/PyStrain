"""Strain Time Series workflow for the PyStrain2 web application.

Three sequential steps:
  1. stations — import GPS station coordinates, display on map
  2. config   — choose method, configure data source & epoch range, run computation
  3. results  — map + polygon list with per-location time-series dialog

All state lives in ``st.session_state`` keys prefixed with ``ts_``.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np
import streamlit as st

from pystrain.web import session
from pystrain.web.components.sidebar import render_step_indicator, render_navigation, render_reset_button
from pystrain.web.components.upload import gps_info_file_uploader


# ---------------------------------------------------------------------------
# Top-level rendering
# ---------------------------------------------------------------------------


def render_workflow() -> None:
    """Entry point called from ``app.py``."""
    step = st.session_state.ts_step
    steps = session.steps_list("timeseries")
    labels = session.step_labels("timeseries")

    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 应变时间序列")
        render_step_indicator(steps, labels, step)
        st.divider()
        _render_step_sidebar(step)
        can_adv = session.can_advance("ts_step", step)
        render_navigation("ts_step", step, steps, can_advance=can_adv, key_suffix="ts")
        render_reset_button("timeseries", key_suffix="ts")

    # --- Main area ---
    _render_step_main(step)


# ---------------------------------------------------------------------------
# Step sidebar renderers
# ---------------------------------------------------------------------------


def _render_step_sidebar(step: str) -> None:
    if step == "stations":
        _sidebar_stations()
    elif step == "config":
        _sidebar_config()
    elif step == "results":
        _sidebar_ts_results()


def _render_step_main(step: str) -> None:
    if step == "stations":
        _main_stations()
    elif step == "config":
        _main_config()
    elif step == "results":
        _main_ts_results()


# ===================================================================
# Step 1: Station Import
# ===================================================================


def _sidebar_stations() -> None:
    """Sidebar: upload GPS station info file."""
    lon, lat, names, heights = gps_info_file_uploader(key="ts_gps_info")

    if lon is not None:
        st.session_state.ts_gps_info_data = (lon, lat, names, heights)
        st.markdown(f"**已加载 {len(lon)} 个测站**")
        st.caption(f"经度范围: {lon.min():.2f}° – {lon.max():.2f}°")
        st.caption(f"纬度范围: {lat.min():.2f}° – {lat.max():.2f}°")


def _main_stations() -> None:
    """Main: show station locations on map."""
    st.header("Step 1: 导入测站坐标")

    data = st.session_state.ts_gps_info_data
    if data is not None:
        lon, lat, names, _ = data
        all_polys = st.session_state.get("ts_all_polygons", None)
        from pystrain.web.plots import _make_geo_figure, _polygon_trace
        import plotly.graph_objects as go

        pad = 0.5
        lon_min = float(np.nanmin(lon) - pad)
        lon_max = float(np.nanmax(lon) + pad)
        lat_min = float(np.nanmin(lat) - pad)
        lat_max = float(np.nanmax(lat) + pad)
        if all_polys:
            for p in all_polys:
                lon_min = min(lon_min, float(np.nanmin(p[:, 0]) - pad))
                lon_max = max(lon_max, float(np.nanmax(p[:, 0]) + pad))
                lat_min = min(lat_min, float(np.nanmin(p[:, 1]) - pad))
                lat_max = max(lat_max, float(np.nanmax(p[:, 1]) + pad))

        fig = _make_geo_figure(
            title=f"GPS 测站分布 — {len(lon)} 个站点",
            lon_range=(lon_min, lon_max),
            lat_range=(lat_min, lat_max),
            height=600,
        )
        hover_texts = [
            f"<b>{names[i]}</b><br>Lon: {lon[i]:.4f}°<br>Lat: {lat[i]:.4f}°"
            for i in range(len(lon))
        ]
        fig.add_trace(go.Scattergeo(
            lon=list(lon), lat=list(lat), mode="markers",
            marker=dict(size=8, color="steelblue", symbol="circle"),
            text=hover_texts, hoverinfo="text",
            name=f"Stations ({len(lon)})",
        ))
        if all_polys:
            for i, p in enumerate(all_polys):
                fig.add_trace(_polygon_trace(
                    p, opacity=0.15,
                    name=f"Polygon {i}" if i > 0 else "Boundary",
                ))
        elif st.session_state.ts_poly_data is not None:
            fig.add_trace(_polygon_trace(st.session_state.ts_poly_data, opacity=0.2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 请在左侧上传测站坐标文件 (lon lat height site_name 格式)")
        st.markdown("""
        **文件格式要求：**
        - 每行一个站点: `lon  lat  height  site_name`
        - 经纬度为十进制度
        - 高度单位为米
        """)


# ===================================================================
# Step 2: Time-Series Configuration
# ===================================================================


def _sidebar_config() -> None:
    """Sidebar: method → polygon (user) → data → epochs → compute."""
    st.subheader("⏱️ 时序配置")

    # ---- 1. Method selection (default: user) ----
    method = st.selectbox(
        "计算方法",
        ["user", "grid", "tri"],
        index=0,
        format_func=lambda x: {
            "user": "用户自定义 (User)",
            "grid": "规则网格 (Grid)",
            "tri": "三角网 (Triangle)",
        }[x],
        key="ts_method_sel",
    )
    st.session_state.ts_method = method

    st.divider()

    # ---- 2. If user method: polygon definition ----
    if method == "user":
        _ts_user_polygon_section()
        st.divider()

    # ---- 3. Data source ----
    with st.expander("数据源", expanded=True):
        ts_type = st.selectbox(
            "时间序列格式",
            ["pos", "dat"],
            format_func=lambda x: ".pos (PBO 格式)" if x == "pos" else ".dat (PyTsfit 格式)",
            key="ts_type_sel",
        )
        st.session_state.ts_ts_type = ts_type

        st.markdown("**数据目录路径**")
        data_dir = st.text_input(
            "数据目录路径",
            value=st.session_state.ts_data_dir or "",
            key="ts_dir",
            placeholder="/path/to/ts/data/",
        )
        if data_dir:
            st.session_state.ts_data_dir = data_dir

        dd = st.session_state.ts_data_dir
        if dd and os.path.isdir(dd):
            ext = ".pos" if ts_type == "pos" else ".dat"
            try:
                n_files = len([f for f in os.listdir(dd) if f.endswith(ext)])
                st.caption(f"✅ 目录存在 — 找到 {n_files} 个 {ext} 文件")
            except Exception:
                st.caption("✅ 目录存在")
        elif dd:
            st.caption("⚠️ 目录不存在，请检查路径")

    # ---- 4. Epoch range ----
    with st.expander("历元范围", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            se = st.number_input("起始历元 (decyr)", value=st.session_state.ts_epoch_start or 2020.0, key="ts_se")
            st.session_state.ts_epoch_start = se
        with c2:
            ee = st.number_input("结束历元 (decyr)", value=st.session_state.ts_epoch_end or 2021.0, key="ts_ee")
            st.session_state.ts_epoch_end = ee

    # ---- 5. Method-specific additional parameters ----
    if method == "grid":
        with st.expander("网格参数", expanded=True):
            _ts_grid_params()
    elif method == "tri":
        with st.expander("三角网参数", expanded=True):
            _ts_tri_params()

    # ---- 6. Compute button (only when essentials are ready) ----
    st.divider()
    if method == "user" and st.session_state.ts_user_poly_data is None:
        st.info("请先定义多边形（上传文件或从地图选取）")
    else:
        disabled = False
        if not st.session_state.ts_data_dir:
            disabled = True
        elif st.session_state.ts_epoch_start is None or st.session_state.ts_epoch_end is None:
            disabled = True
        elif method == "user" and st.session_state.ts_user_poly_data is None:
            disabled = True
        if st.button("🚀 运行计算", type="primary", use_container_width=True,
                     disabled=disabled, key="ts_run_from_config"):
            _run_ts_computation()


# ---------------------------------------------------------------------------
# User method: polygon definition
# ---------------------------------------------------------------------------


def _ts_user_polygon_section() -> None:
    """Polygon definition for the user-defined method.

    Supports:
      - Upload any text file with lon lat pairs (blank/comment lines ignored)
      - Select stations from the loaded station list to form a convex-hull polygon
    """
    st.markdown("**📐 自定义多边形**")
    st.caption("每个多边形必须 ≥ 3 个点")

    data = st.session_state.ts_gps_info_data
    all_names: list = []
    name_to_coord: dict = {}
    if data is not None:
        lon_arr, lat_arr, names_list, _ = data
        all_names = list(names_list)
        for i, n in enumerate(names_list):
            name_to_coord[n] = (float(lon_arr[i]), float(lat_arr[i]))

    source = st.radio(
        "多边形来源",
        ["file", "map"],
        format_func=lambda x: "上传文件" if x == "file" else "从地图选取",
        index=0 if st.session_state.ts_user_poly_source == "file" else 1,
        key="ts_user_poly_source_radio",
    )
    st.session_state.ts_user_poly_source = source

    if source == "file":
        _ts_user_polygon_from_file(name_to_coord)
    else:
        _ts_user_polygon_from_map(all_names, name_to_coord)


def _ts_user_polygon_from_file(name_to_coord: dict) -> None:
    """Upload and parse a polygon points file (any text with lon lat pairs)."""
    poly_file = st.file_uploader(
        "上传多边形文件",
        type=None,
        key="ts_user_poly_file",
        help="任意文本文件，每行包含经度 纬度（空格或制表符分隔），空行被忽略",
    )
    if poly_file is not None:
        try:
            content = poly_file.getvalue().decode("utf-8")
            polygons = _parse_polygon_points(content, name_to_coord)
            if polygons:
                st.session_state.ts_user_poly_data = polygons[0]
                st.session_state.ts_poly_data = polygons[0]
                st.session_state.ts_all_polygons = polygons
                # Extract station names: try direct name parsing, then nearest-station fallback
                raw_lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.startswith("#")]
                poly_station_names = []
                for p in polygons:
                    vnames = []
                    for v in range(len(p) - 1):
                        vlon, vlat = float(p[v, 0]), float(p[v, 1])
                        # Check if any raw line matches this polygon's vertex names
                        matched = ""
                        if name_to_coord:
                            matched = _find_nearest_station(vlon, vlat, name_to_coord)
                        vnames.append(matched)
                    poly_station_names.append(vnames)
                # Try direct station-name matching from file content
                direct_names = []
                for ln in raw_lines:
                    parts = ln.split()
                    if name_to_coord and all(p in name_to_coord for p in parts):
                        direct_names.append(parts)
                if direct_names and len(direct_names) == len(polygons):
                    poly_station_names = direct_names
                st.session_state.ts_user_poly_station_names = poly_station_names
                st.success(f"已加载 {len(polygons)} 个多边形 "
                           f"(顶点数: {[len(p) for p in polygons]})")
            else:
                st.error("未找到有效多边形（至少需要 3 个不共线的点）")
        except Exception as e:
            st.error(f"解析失败: {e}")


def _ts_user_polygon_from_map(all_names: list, name_to_coord: dict) -> None:
    """Select stations from the map to form a polygon (convex hull)."""
    if not all_names:
        st.info("请先在 Step 1 导入测站数据")
        return

    selected_names = st.multiselect(
        "选择测站",
        options=all_names,
        default=st.session_state.ts_user_poly_data is not None and
                len(st.session_state.ts_user_poly_data) <= len(all_names) and
                [_find_nearest_station(lon, lat, name_to_coord)
                 for lon, lat in st.session_state.ts_user_poly_data[:0]] or [],
        key="ts_user_poly_stations",
        help="至少选择 3 个测站来生成多边形",
        placeholder="点击选择测站...",
    )
    if not selected_names:
        if st.session_state.ts_user_poly_data is not None:
            st.session_state.ts_user_poly_data = None
            st.session_state.ts_poly_data = None
        st.info("请选择至少 3 个测站")
        return

    found_coords = [name_to_coord[n] for n in selected_names if n in name_to_coord]
    if len(found_coords) >= 3:
        poly = _convex_hull_polygon(found_coords)
        if poly is not None:
            st.session_state.ts_user_poly_data = poly
            st.session_state.ts_poly_data = poly
            st.success(f"已选择 {len(selected_names)} 个测站 → 凸包 {len(poly)} 顶点")
        else:
            st.warning("所选测站共线，无法生成凸包")
    else:
        st.info(f"已选择 {len(found_coords)} 个测站（需要 ≥ 3 个）")
        if len(found_coords) < 3 and st.session_state.ts_user_poly_data is not None:
            st.session_state.ts_user_poly_data = None
            st.session_state.ts_poly_data = None


def _find_nearest_station(target_lon: float, target_lat: float,
                          name_to_coord: dict) -> str:
    """Find the station name nearest to (target_lon, target_lat)."""
    best_name = ""
    best_dist = float("inf")
    for name, (clon, clat) in name_to_coord.items():
        d = (clon - target_lon)**2 + (clat - target_lat)**2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def _convex_hull_polygon(coords: list) -> np.ndarray | None:
    """Build a convex-hull polygon from a list of (lon, lat) tuples.

    Returns (M, 2) array of hull vertices in CCW order, or None if the
    points are degenerate (collinear / < 3).
    """
    from scipy.spatial import ConvexHull
    pts = np.array(coords)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        return np.vstack([hull_pts, hull_pts[:1]])
    except Exception:
        return None


def _parse_polygon_points(content: str, name_to_coord: dict) -> list:
    """Parse an arbitrary text file for lon/lat pairs.

    Strategies tried in order:
      1. Each line is a ``lon lat`` pair → collect successive pairs into polygons
         separated by blank lines.
      2. One line contains ≥ 3 coordinate pairs (lon1 lat1 lon2 lat2 ...).
      3. Each line contains station names known in name_to_coord.

    Returns a list of (N, 2) numpy arrays, each a closed polygon ring.
    """
    raw_lines = content.splitlines()

    # ---- Heuristic: check if most non-blank lines are single lon lat pairs ----
    data_lines = []
    blank_indices = []
    for i, ln in enumerate(raw_lines):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            blank_indices.append(i)
            data_lines.append(None)
        else:
            data_lines.append(s)

    numeric_counts = []
    for ln in data_lines:
        if ln is None:
            numeric_counts.append(0)
            continue
        parts = ln.split()
        try:
            vals = [float(x) for x in parts]
            numeric_counts.append(len(vals))
        except ValueError:
            numeric_counts.append(-1)

    n_pair_lines = sum(1 for c in numeric_counts if c == 2)
    n_multi_lines = sum(1 for c in numeric_counts if c >= 6 and c % 2 == 0)
    n_name_lines = sum(1 for c in numeric_counts if c == -1)

    all_polygons = []

    # Strategy 1: GMT-style (one lon lat pair per line, blank-line separated)
    if n_pair_lines >= n_multi_lines and n_name_lines == 0:
        current_ring = []
        for ln in data_lines:
            if ln is None:
                if len(current_ring) >= 3:
                    pts = np.array(current_ring)
                    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
                        pts = np.vstack([pts, pts[0]])
                    all_polygons.append(pts)
                current_ring = []
                continue
            parts = ln.split()
            current_ring.append([float(parts[0]), float(parts[1])])
        if len(current_ring) >= 3:
            pts = np.array(current_ring)
            if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
                pts = np.vstack([pts, pts[0]])
            all_polygons.append(pts)
        return all_polygons

    # Strategy 2 & 3: one polygon per line (coordinate pairs or station names)
    for ln in data_lines:
        if ln is None:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue

        # Try numeric (coordinate pairs)
        try:
            coords = [float(x) for x in parts]
            if len(coords) >= 6 and len(coords) % 2 == 0:
                pts = np.array(coords).reshape(-1, 2)
                if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
                    pts = np.vstack([pts, pts[0]])
                all_polygons.append(pts)
                continue
        except ValueError:
            pass

        # Try station names
        if name_to_coord:
            found = [name_to_coord[n] for n in parts if n in name_to_coord]
            if len(found) >= 3:
                pts = np.array(found)
                if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
                    pts = np.vstack([pts, pts[0]])
                all_polygons.append(pts)

    return all_polygons


# ---------------------------------------------------------------------------
# Grid parameters (for grid method)
# ---------------------------------------------------------------------------


def _ts_grid_params() -> None:
    gk = st.session_state.ts_grid_kwargs or {}
    c1, c2 = st.columns(2)
    with c1:
        slon = st.number_input("起始经度", value=gk.get("slon", 70.0), key="tsg_slon")
        slat = st.number_input("起始纬度", value=gk.get("slat", 15.0), key="tsg_slat")
    with c2:
        elon = st.number_input("结束经度", value=gk.get("elon", 110.0), key="tsg_elon")
        elat = st.number_input("结束纬度", value=gk.get("elat", 55.0), key="tsg_elat")
    dn = st.number_input("经度间距 (°)", 0.05, 2.0, gk.get("dn", 0.5), key="tsg_dn")
    de = st.number_input("纬度间距 (°)", 0.05, 2.0, gk.get("de", 0.5), key="tsg_de")
    stagger = st.checkbox("交错排列", value=gk.get("stagger", True), key="tsg_stag")
    maxdist = st.number_input("最大距离 (km)", 50.0, 1000.0, gk.get("maxdist_km", 300.0), key="tsg_md")
    min_sites = st.number_input("最少站点数", 3, 50, gk.get("min_sites", 6), key="tsg_ms")
    check_az = st.checkbox("检查方位角覆盖", value=gk.get("check_azimuth", True), key="tsg_ca")
    st.session_state.ts_grid_kwargs = {
        "slon": slon, "elon": elon, "slat": slat, "elat": elat,
        "dn": dn, "de": de, "stagger": stagger,
        "maxdist_km": maxdist, "min_sites": int(min_sites), "check_azimuth": check_az,
    }


def _ts_tri_params() -> None:
    tk = st.session_state.ts_tri_kwargs or {}
    ma = st.slider("最小角度 (°)", 1.0, 30.0, tk.get("min_angle_deg", 10.0), 1.0, key="tst_ma")
    ep = st.slider("边长百分位数", 50.0, 99.0, tk.get("max_edge_pctl", 95.0), 1.0, key="tst_ep")
    ef = st.number_input("边长倍数因子", 1.0, 5.0, tk.get("max_edge_factor", 1.5), 0.1, key="tst_ef")
    ar = st.number_input("最小面积比", 0.01, 1.0, tk.get("min_area_ratio", 0.1), 0.01, key="tst_ar")
    st.session_state.ts_tri_kwargs = {
        "min_angle_deg": ma, "max_edge_pctl": ep,
        "max_edge_factor": ef, "min_area_ratio": ar,
    }


def _render_directory_browser(start_path: str) -> None:
    """(Removed) Directory browser replaced by ZIP upload + manual path input."""
    pass


def _main_config() -> None:
    """Main: show stations + polygon on map."""
    st.header("时序配置")

    data = st.session_state.ts_gps_info_data
    if data is None:
        st.info("请先导入测站坐标。")
        return

    lon, lat, names, _ = data

    # Determine which polygon to show
    method = st.session_state.ts_method
    poly = st.session_state.ts_poly_data
    if method == "user":
        poly = st.session_state.ts_user_poly_data

    from pystrain.web.plots import plot_stations_with_polygon

    if method == "user" and poly is not None:
        all_polys = st.session_state.get("ts_all_polygons", None)
        from pystrain.web.plots import _make_geo_figure, _polygon_trace
        import plotly.graph_objects as go

        pad = 0.5
        lon_min = float(np.nanmin(lon) - pad)
        lon_max = float(np.nanmax(lon) + pad)
        lat_min = float(np.nanmin(lat) - pad)
        lat_max = float(np.nanmax(lat) + pad)
        if all_polys:
            for p in all_polys:
                lon_min = min(lon_min, float(np.nanmin(p[:, 0]) - pad))
                lon_max = max(lon_max, float(np.nanmax(p[:, 0]) + pad))
                lat_min = min(lat_min, float(np.nanmin(p[:, 1]) - pad))
                lat_max = max(lat_max, float(np.nanmax(p[:, 1]) + pad))

        fig = _make_geo_figure(
            title=f"用户自定义多边形 — {len(poly)} 顶点 ({len(all_polys) if all_polys else 1} 个多边形)",
            lon_range=(lon_min, lon_max),
            lat_range=(lat_min, lat_max),
            height=600,
        )
        hover_texts = [
            f"<b>{names[i]}</b><br>Lon: {lon[i]:.4f}°<br>Lat: {lat[i]:.4f}°"
            for i in range(len(lon))
        ]
        fig.add_trace(go.Scattergeo(
            lon=list(lon), lat=list(lat), mode="markers",
            marker=dict(size=8, color="steelblue", symbol="circle"),
            text=hover_texts, hoverinfo="text",
            name=f"Stations ({len(lon)})",
        ))
        if all_polys:
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                      "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#d35400"]
            for i, p in enumerate(all_polys):
                c = colors[i % len(colors)]
                fig.add_trace(go.Scattergeo(
                    lon=list(p[:, 0]) + [p[0, 0]],
                    lat=list(p[:, 1]) + [p[0, 1]],
                    mode="lines+markers",
                    line=dict(color=c, width=2),
                    marker=dict(size=5, color=c, symbol="circle"),
                    name=f"多边形 {i+1}",
                    showlegend=False,
                ))
        else:
            fig.add_trace(_polygon_trace(poly, opacity=0.2))
        st.plotly_chart(fig, use_container_width=True)

        from matplotlib.path import Path
        mpath = Path(poly)
        pts = np.column_stack([lon, lat])
        inside = mpath.contains_points(pts)
        n_inside = inside.sum()
        st.success(f"多边形内测站: {n_inside}/{len(lon)}")

        # ---- Polygon info table (manual layout with color swatches) ----
        poly_station_names = st.session_state.get("ts_user_poly_station_names", None)
        if all_polys and len(all_polys) > 0:
            # Init selection state when poly count changes
            if "ts_poly_sel" not in st.session_state or len(st.session_state.ts_poly_sel) != len(all_polys):
                st.session_state.ts_poly_sel = [False] * len(all_polys)
            colors_tbl = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                          "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#d35400"]
            st.subheader("📋 多边形列表")
            st.caption("勾选要参与计算的多边形")
            # Header
            hc = st.columns([0.8, 0.8, 0.8, 1, 3.6])
            with hc[0]: st.markdown("**选择**")
            with hc[1]: st.markdown("**序号**")
            with hc[2]: st.markdown("**颜色**")
            with hc[3]: st.markdown("**顶点**")
            with hc[4]: st.markdown("**构成站点**")
            # Rows
            for i, p in enumerate(all_polys):
                rc = st.columns([0.8, 0.8, 0.8, 1, 3.6])
                with rc[0]:
                    sel = st.checkbox("", value=st.session_state.ts_poly_sel[i], key=f"ts_poly_sel_{i}")
                    st.session_state.ts_poly_sel[i] = sel
                with rc[1]: st.write(i + 1)
                with rc[2]:
                    c = colors_tbl[i % len(colors_tbl)]
                    st.markdown(f'<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:{c}"></span>', unsafe_allow_html=True)
                with rc[3]: st.write(len(p) - 1)
                with rc[4]:
                    if poly_station_names and i < len(poly_station_names) and poly_station_names[i] and len(poly_station_names[i]) > 0:
                        st.caption(", ".join(poly_station_names[i]))
                    else:
                        st.caption("-")

    elif method == "grid":
        gk = st.session_state.ts_grid_kwargs
        if gk:
            from pystrain.grid import Grid
            from pystrain.web.plots import plot_grid_points_overlay
            grid = Grid(
                gk["slon"], gk["elon"], gk["slat"], gk["elat"],
                gk["dn"], gk["de"], stagger=gk.get("stagger", True),
            )
            fig = plot_grid_points_overlay(
                grid_lon=grid.lon, grid_lat=grid.lat,
                site_lon=lon, site_lat=lat,
                polygon=poly,
                title=f"Grid 预览 — {len(grid.lon)} 个格网点",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plot_stations_with_polygon(lon, lat, station_names=names, polygon=poly,
                                             title="测站分布 — 请在左侧配置网格参数")
            st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_stations_with_polygon(lon, lat, station_names=names, polygon=poly,
                                         title="测站与多边形" if poly is not None else "测站分布")
        st.plotly_chart(fig, use_container_width=True)

    config_summary = _build_config_summary()
    if config_summary:
        st.info(config_summary)


def _build_config_summary() -> str:
    """Build a one-line summary of the current configuration."""
    parts = []
    parts.append(f"方法: {st.session_state.ts_method}")
    if st.session_state.ts_data_dir:
        parts.append(f"数据目录: {st.session_state.ts_data_dir}")
    if st.session_state.ts_epoch_start and st.session_state.ts_epoch_end:
        parts.append(f"历元: {st.session_state.ts_epoch_start:.1f} – {st.session_state.ts_epoch_end:.1f}")
    return " | ".join(parts) if parts else ""


# ===================================================================
# Step 3: Compute
# ===================================================================


def _run_ts_computation() -> None:
    """Execute the time-series pipeline using the CLI's run_timeseries_pipeline."""
    data = st.session_state.ts_gps_info_data
    method = st.session_state.ts_method

    if data is None:
        st.error("请先导入测站坐标。")
        return

    lon_arr, lat_arr, names_list, heights = data

    # Mark stations being displayed
    n_sta = len(lon_arr)
    sta_status = st.empty()
    sta_status.info(f"准备计算 {n_sta} 个测站...\n" + "\n".join(str(n) for n in names_list[:15])
                    + ("\n..." if n_sta > 15 else ""))

    # 1. Write GPS info file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".llh", delete=False) as tf:
        for i in range(len(lon_arr)):
            h = heights[i] if heights is not None else 0.0
            tf.write(f"{lon_arr[i]:.6f} {lat_arr[i]:.6f} {h:.3f} {names_list[i]}\n")
        gps_info_path = tf.name

    # 2. Write site_groups_file from selected polygon stations
    poly = st.session_state.ts_user_poly_data if method == "user" else st.session_state.ts_poly_data
    sel = st.session_state.get("ts_poly_sel", [])
    poly_station_names = st.session_state.get("ts_user_poly_station_names", None)
    groups_file_path = None

    if method == "user" and poly_station_names and len(sel) > 0:
        selected_groups = []
        for i in range(len(poly_station_names)):
            if i < len(sel) and sel[i] and poly_station_names[i]:
                valid_names = [n for n in poly_station_names[i] if n and n != ""]
                if valid_names:
                    selected_groups.append(valid_names)
        if selected_groups:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".groups", delete=False) as tf:
                for g in selected_groups:
                    tf.write(" ".join(g) + "\n")
                groups_file_path = tf.name

    # 3. Generate YAML config file
    import yaml
    config = {
        "timeseries": {
            "method": "user",
            "gps_info_file": gps_info_path,
            "ts_type": st.session_state.ts_ts_type,
            "ts_path": st.session_state.ts_data_dir,
            "sepoch": st.session_state.ts_epoch_start,
            "eepoch": st.session_state.ts_epoch_end,
            "output_dir": tempfile.mkdtemp(prefix="pystrain_ts_out_"),
            "user": {
                "site_groups_file": groups_file_path or "",
                "max_sigma_mm": 5.0,
            },
        }
    }
    config_path = tempfile.mktemp(suffix=".yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # 4. Run computation via CLI pipeline
    from pystrain.cli.main import run_timeseries_pipeline
    from pystrain.config import Config

    progress_bar = st.progress(0, text="计算中...")
    status_text = st.empty()
    log_buf = io.StringIO()
    sta_status.success(f"已加载 {n_sta} 个测站")

    try:
        old_stdout = sys.stdout
        sys.stdout = log_buf

        cfg = Config(config_path)
        result = run_timeseries_pipeline(cfg)

        sys.stdout = old_stdout

        progress_bar.progress(100, text="计算完成！")
        status_text.empty()

        with st.expander("运行日志", expanded=False):
            st.code(log_buf.getvalue())

        # 5. Build display result (match the format expected by results step)
        tsc = result["tsc"]
        results_list = result["results"]
        loc_lon = np.array([float(r.lon) for r in results_list])
        loc_lat = np.array([float(r.lat) for r in results_list])

        display_result = {
            "tsc": tsc,
            "results": results_list,
            "method": "user",
            "n_locations": len(results_list),
            "n_epochs": len(tsc.decyr),
            "n_sites": len(tsc.sites),
            "sites": tsc.sites,
            "sites_lon": np.array(lon_arr),
            "sites_lat": np.array(lat_arr),
            "loc_lon": loc_lon,
            "loc_lat": loc_lat,
            "decyr": tsc.decyr,
            "polygon": poly,
        }

        st.session_state.ts_results = display_result
        st.session_state.ts_selected_index = 0
        st.session_state.ts_dialog_open_idx = -1
        st.success(f"计算完成！{display_result['n_locations']} 个位置, {display_result['n_epochs']} 个历元")
        st.session_state.ts_step = "results"
        st.rerun()

    except Exception as e:
        sys.stdout = old_stdout if "old_stdout" in dir() else sys.__stdout__
        progress_bar.empty()
        status_text.empty()
        with st.expander("运行日志", expanded=True):
            st.code(log_buf.getvalue())
        st.error(f"计算失败: {e}")
    finally:
        try:
            os.unlink(gps_info_path)
        except Exception:
            pass
        try:
            if groups_file_path:
                os.unlink(groups_file_path)
            os.unlink(config_path)
        except Exception:
            pass

# ===================================================================
# Step 3: Results
# ===================================================================


def _sidebar_ts_results() -> None:
    """Sidebar: result summary, download, and navigation."""
    result = st.session_state.ts_results
    if result is None:
        return

    st.subheader("📊 计算摘要")
    st.metric("方法", result["method"])
    st.metric("计算位置数", result["n_locations"])
    st.metric("历元数", result["n_epochs"])
    st.metric("站点数", result["n_sites"])

    st.divider()
    st.subheader("📥 数据导出")

    idx = st.session_state.ts_selected_index
    res_list = result["results"]

    if st.button("📥 导出全部结果 (CSV)", use_container_width=True, key="ts_dl_all"):
        import pandas as pd
        rows = []
        for j, r in enumerate(res_list):
            rows.append({
                "index": j,
                "lon": float(r.lon),
                "lat": float(r.lat),
                "dilation_mean": float(np.nanmean(r.dilation)),
                "shear_mean": float(np.nanmean(r.shear)),
                "e1_mean": float(np.nanmean(r.e1)),
                "e2_mean": float(np.nanmean(r.e2)),
                "omega_mean": float(np.nanmean(r.omega)),
                "sec_inv_mean": float(np.nanmean(r.sec_inv)),
            })
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        st.download_button("下载 CSV", data=csv, file_name="strain_ts_all_locations.csv",
                           mime="text/csv", key="ts_dl_all_btn")

    if 0 <= idx < len(res_list):
        if st.button("📥 导出当前选中时序 (CSV)", use_container_width=True, key="ts_dl_one"):
            r = res_list[idx]
            import pandas as pd
            df = pd.DataFrame({
                "decyr": r.decyr,
                "dilation": r.dilation,
                "shear": r.shear,
                "e1": r.e1,
                "e2": r.e2,
                "omega": r.omega,
                "sec_inv": r.sec_inv,
                "exx": r.exx,
                "exy": r.exy,
                "eyy": r.eyy,
            })
            csv = df.to_csv(index=False)
            st.download_button("下载 CSV", data=csv,
                               file_name=f"strain_ts_loc{idx}.csv", mime="text/csv",
                               key="ts_dl_one_btn")

    st.divider()
    st.subheader("🗺️ 断层线加载")

    fault_files = st.file_uploader(
        "导入断层文件（KML/SHP/JSON）",
        type=["kml", "kmz", "shp", "zip", "json", "geojson"],
        key="ts_fault_upload",
        accept_multiple_files=True,
        help="支持 KML、Shapefile (.shp/.zip)、GeoJSON 等多文件",
    )
    if fault_files:
        from pystrain.web.faults import parse_fault_file
        all_traces = list(st.session_state.get("ts_faults") or [])
        new_count = 0
        for f in fault_files:
            traces, err = parse_fault_file(f)
            if err:
                st.warning(f"{f.name}: {err}")
            elif traces:
                all_traces.extend(traces)
                new_count += len(traces)
        if new_count:
            st.session_state.ts_faults = all_traces
            st.success(f"已加载 {new_count} 条断层迹线（共 {len(all_traces)} 条）")

    if st.session_state.get("ts_faults"):
        n = len(st.session_state.ts_faults)
        st.metric("断层迹线数", n)
        if st.button("🗑️ 清除断层", key="ts_clear_faults", use_container_width=True):
            st.session_state.ts_faults = None
            st.rerun()

    st.divider()
    if st.button("🔄 返回配置", use_container_width=True, key="ts_back_to_config"):
        st.session_state.ts_step = "config"
        st.rerun()


def _main_ts_results() -> None:
    """Main: map (top) + polygon list (bottom) with time-series dialog on demand."""
    result = st.session_state.ts_results
    if result is None:
        st.info("没有计算结果。请返回上一步运行计算。")
        return

    st.header("结果查看")

    loc_lon = result["loc_lon"]
    loc_lat = result["loc_lat"]
    poly = result.get("polygon")
    res_list = result["results"]
    poly_station_names = st.session_state.get("ts_user_poly_station_names", None)
    n = len(res_list)

    # ================================================================
    # TOP: Map
    # ================================================================
    from pystrain.web.plots import _make_geo_figure
    import plotly.graph_objects as go

    # Build map with strain locations + all polygon boundaries
    all_polys = st.session_state.get("ts_all_polygons", None)
    if all_polys is None and poly is not None:
        all_polys = [poly]

    # Compute map extent (minimum coverage: China region)
    pad = 1.0
    CHINA_LON_MIN, CHINA_LON_MAX = 73.0, 135.0
    CHINA_LAT_MIN, CHINA_LAT_MAX = 18.0, 54.0

    lon_min = float(np.nanmin(loc_lon) - pad)
    lon_max = float(np.nanmax(loc_lon) + pad)
    lat_min = float(np.nanmin(loc_lat) - pad)
    lat_max = float(np.nanmax(loc_lat) + pad)
    if all_polys:
        for p in all_polys:
            lon_min = min(lon_min, float(np.nanmin(p[:, 0]) - pad))
            lon_max = max(lon_max, float(np.nanmax(p[:, 0]) + pad))
            lat_min = min(lat_min, float(np.nanmin(p[:, 1]) - pad))
            lat_max = max(lat_max, float(np.nanmax(p[:, 1]) + pad))

    # Enforce minimum extent to at least cover China
    lon_min = min(lon_min, CHINA_LON_MIN)
    lon_max = max(lon_max, CHINA_LON_MAX)
    lat_min = min(lat_min, CHINA_LAT_MIN)
    lat_max = max(lat_max, CHINA_LAT_MAX)

    fig = _make_geo_figure(
        title=f"应变计算位置 ({result['n_locations']} 个)",
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=700,
    )

    # Strain location markers (scatter)
    hover_texts = [
        f"<b>Location {i}</b><br>Lon: {x:.4f}°<br>Lat: {y:.4f}°"
        for i, x, y in zip(range(len(loc_lon)), loc_lon, loc_lat)
    ]
    fig.add_trace(go.Scattergeo(
        lon=list(loc_lon),
        lat=list(loc_lat),
        mode="markers",
        marker=dict(size=10, color="crimson", symbol="circle",
                   line=dict(width=1, color="white")),
        text=hover_texts,
        hoverinfo="text",
        name=f"Locations ({len(loc_lon)})",
    ))

    # Polygon boundaries — lines+markers, no fill (same as config step)
    if all_polys:
        poly_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                       "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#d35400"]
        for i, p in enumerate(all_polys):
            c = poly_colors[i % len(poly_colors)]
            fig.add_trace(go.Scattergeo(
                lon=list(p[:, 0]) + [p[0, 0]],
                lat=list(p[:, 1]) + [p[0, 1]],
                mode="lines+markers",
                line=dict(color=c, width=2),
                marker=dict(size=5, color=c, symbol="circle"),
                name=f"多边形 {i+1}",
                showlegend=False,
            ))

    # Fault traces overlay
    faults = st.session_state.get("ts_faults")
    if faults:
        from pystrain.web.plots import _add_fault_traces
        _add_fault_traces(fig, faults)

    st.plotly_chart(fig, use_container_width=True, key="ts_map")

    # ================================================================
    # BOTTOM: Polygon result table + time-series dialog (fragment)
    # ================================================================
    if n == 0:
        st.warning("无计算结果。")
        return

    @st.fragment
    def _render_table_and_dialog():
        """Fragment: table + dialog — button clicks only rerun this, not the map."""
        st.divider()
        st.subheader("📋 多边形计算结果")

        # Pre-compute row data
        row_data = []
        for j in range(n):
            names_str = "—"
            if (poly_station_names and j < len(poly_station_names)
                    and poly_station_names[j] and len(poly_station_names[j]) > 0):
                names_str = ", ".join(poly_station_names[j])
            try:
                r = res_list[j]
                rlon = float(r.lon)
                rlat = float(r.lat)
            except (AttributeError, IndexError, TypeError):
                rlon = float(loc_lon[j]) if j < len(loc_lon) else 0.0
                rlat = float(loc_lat[j]) if j < len(loc_lat) else 0.0
            try:
                r = res_list[j]
                has_data = bool(np.any(np.isfinite(r.dilation)))
            except Exception:
                has_data = False
            row_data.append((j + 1, f"({rlon:.2f}°, {rlat:.2f}°)", names_str, has_data))

        colors_tbl = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                      "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#d35400"]

        # Table header
        header_cols = st.columns([0.6, 0.6, 1.3, 3.2, 0.8, 1.5])
        with header_cols[0]: st.markdown("**序号**")
        with header_cols[1]: st.markdown("**颜色**")
        with header_cols[2]: st.markdown("**位置**")
        with header_cols[3]: st.markdown("**构成站点**")
        with header_cols[4]: st.markdown("**状态**")
        with header_cols[5]: st.markdown("**操作**")

        # Track which row's dialog to open
        open_dialog_idx = st.session_state.get("ts_dialog_open_idx", -1)

        # Table rows
        for j in range(n):
            seq, pos, sname, has_data = row_data[j]
            row_cols = st.columns([0.6, 0.6, 1.3, 3.2, 0.8, 1.5])
            with row_cols[0]: st.write(seq)
            with row_cols[1]:
                c = colors_tbl[j % len(colors_tbl)]
                st.markdown(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'border-radius:3px;background:{c}"></span>',
                    unsafe_allow_html=True,
                )
            with row_cols[2]: st.caption(pos)
            with row_cols[3]: st.caption(sname)
            with row_cols[4]:
                if has_data:
                    st.markdown("✅")
                else:
                    st.markdown("⚠️")
            with row_cols[5]:
                clicked = st.button(
                    "📈 查看图", key=f"ts_view_{j}",
                    use_container_width=True,
                    disabled=not has_data,
                )
                if clicked:
                    st.session_state.ts_dialog_open_idx = j
                    st.session_state.ts_selected_index = j

        # Time-series dialog
        if open_dialog_idx >= 0 and open_dialog_idx < n:
            r = res_list[open_dialog_idx]
            rlon = float(r.lon)
            rlat = float(r.lat)
            st.divider()
            st.subheader(f"📈 应变时间序列 — 位置 {open_dialog_idx} ({rlon:.4f}°, {rlat:.4f}°)")

            c_close, _ = st.columns([1, 5])
            with c_close:
                if st.button("✖ 关闭", key="ts_close_dialog", use_container_width=True):
                    st.session_state.ts_dialog_open_idx = -1

            from pystrain.web.plots import plot_timeseries_all_components_2col
            fig_ts = plot_timeseries_all_components_2col(
                r.decyr,
                dilation=r.dilation,
                shear=r.shear,
                e1=r.e1,
                e2=r.e2,
                omega=r.omega,
                sec_inv=r.sec_inv,
                exx=r.exx,
                exy=r.exy,
                eyy=r.eyy,
                title=f"应变时间序列 — 位置 {open_dialog_idx} ({rlon:.4f}°, {rlat:.4f}°)",
                height=1100,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            st.subheader("📋 统计摘要")
            stat_cols = st.columns(3)
            components_stats = [
                ("Dilatation", r.dilation, "nstrain/yr"),
                ("Max Shear", r.shear, "nstrain/yr"),
                ("e1", r.e1, "nstrain/yr"),
                ("e2", r.e2, "nstrain/yr"),
                ("Omega", r.omega, "nrad/yr"),
                ("2nd Inv", r.sec_inv, "nstrain²/yr²"),
                ("exx", r.exx, "nstrain/yr"),
                ("exy", r.exy, "nstrain/yr"),
                ("eyy", r.eyy, "nstrain/yr"),
            ]
            for i, (label, arr, unit) in enumerate(components_stats):
                col_idx = i % 3
                if arr is not None and np.any(np.isfinite(arr)):
                    stat_cols[col_idx].metric(
                        f"{label} 均值",
                        f"{np.nanmean(arr):.3f}",
                    )

    _render_table_and_dialog()
