"""Strain Rate Field workflow for the PyStrain web application.

Six sequential steps:
  1. load      — upload velocity field file, show map
  2. outlier   — detect & remove outliers
  3. algorithm — select strain estimation algorithm, show preview
  4. params    — configure algorithm-specific parameters
  5. compute   — run the full pipeline
  6. results   — interactive result tabs

All state lives in ``st.session_state`` keys prefixed with ``sr_``.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np
import streamlit as st

from pystrain.data import VelocityField
from pystrain.grid import Grid
from pystrain.outlier import iterative_outlier_removal
from pystrain.triangulation import delaunay_triangulation

from pystrain.web import session
from pystrain.web.components.sidebar import render_step_indicator, render_navigation, render_reset_button
from pystrain.web.components.upload import velocity_file_uploader, polygon_file_uploader
from pystrain.web.pipeline import run_pystrain_pipeline


# ---------------------------------------------------------------------------
# Top-level rendering
# ---------------------------------------------------------------------------


def render_workflow() -> None:
    """Entry point called from ``app.py``."""
    step = st.session_state.sr_step
    steps = session.steps_list("strain_rate")
    labels = session.step_labels("strain_rate")

    # --- Sidebar ---
    with st.sidebar:
        st.header("📈 应变率场")
        render_step_indicator(steps, labels, step)
        st.divider()
        _render_step_sidebar(step)
        can_adv = session.can_advance("sr_step", step)
        render_navigation("sr_step", step, steps, can_advance=can_adv, key_suffix="sr")
        render_reset_button("strain_rate", key_suffix="sr")

    # --- Main area ---
    _render_step_main(step)


# ---------------------------------------------------------------------------
# Step sidebar renderers
# ---------------------------------------------------------------------------


def _render_step_sidebar(step: str) -> None:
    """Dispatch sidebar content to the appropriate step function."""
    if step == "load":
        _sidebar_load()
    elif step == "outlier":
        _sidebar_outlier()
    elif step == "algorithm":
        _sidebar_algorithm()
    elif step == "params":
        _sidebar_params()
    elif step == "compute":
        _sidebar_compute()
    elif step == "results":
        _sidebar_results()


def _render_step_main(step: str) -> None:
    """Dispatch main area content to the appropriate step function."""
    if step == "load":
        _main_load()
    elif step == "outlier":
        _main_outlier()
    elif step == "algorithm":
        _main_algorithm()
    elif step == "params":
        _main_params()
    elif step == "compute":
        _main_compute()
    elif step == "results":
        _main_results()


# ===================================================================
# Step 1: Data Loading
# ===================================================================


def _sidebar_load() -> None:
    """Sidebar: velocity file upload + format selection."""
    vf, fname, fmt = velocity_file_uploader(key="sr_vel")
    if vf is not None:
        st.session_state.sr_vel_data = vf
        st.session_state.sr_vel_file = fname

    # Show stats
    if st.session_state.sr_vel_data is not None:
        vf = st.session_state.sr_vel_data
        st.markdown(f"**已加载 {len(vf)} 个 GPS 站点**")
        st.caption(f"经度范围: {vf.lon.min():.2f}° – {vf.lon.max():.2f}°")
        st.caption(f"纬度范围: {vf.lat.min():.2f}° – {vf.lat.max():.2f}°")


def _main_load() -> None:
    """Main: show velocity field map once data is loaded."""
    st.header("Step 1: 数据加载")
    vf = st.session_state.sr_vel_data
    if vf is not None:
        arrow_scale = st.slider("箭头缩放", 0.01, 1.0, 0.15, 0.01,
                                key="sr_arrow_scale1",
                                help="控制速度矢量箭头的长度，值越大箭头越长")
        from pystrain.web.plots import plot_velocity_field
        fig = plot_velocity_field(
            vf.lon, vf.lat, vf.ve, vf.vn,
            site_names=vf.names if hasattr(vf, 'names') and vf.names is not None and len(vf.names) > 0 else None,
            polygon=st.session_state.sr_poly_data,
            title="GNSS Velocity Field — 已加载数据",
            arrow_scale=arrow_scale,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 请在左侧上传速度场文件开始。")
        st.markdown("""
        **支持的文件格式：**
        - **GMT8** (8列)：`lon lat Ve Vn Se Sn Rho SiteName`
        - **GMT7** (7列)：`lon lat Ve Vn Se Sn SiteName`
        - **GLOBK** (13列)：GLOBK 输出格式
        """)


# ===================================================================
# Step 2: Outlier Detection
# ===================================================================


def _sidebar_outlier() -> None:
    """Sidebar: outlier detection parameters."""
    st.subheader("🔍 异常值检测设置")

    enabled = st.checkbox("启用异常值检测", value=st.session_state.sr_outlier_enabled, key="sr_outlier_check")
    st.session_state.sr_outlier_enabled = enabled

    if enabled:
        method = st.selectbox(
            "检测方法",
            ["knn_iqr", "loo_strain"],
            format_func=lambda x: {"knn_iqr": "KNN + IQR (快速)", "loo_strain": "LOO Strain (严格)"}[x],
            key="sr_outlier_method_sel",
        )
        st.session_state.sr_outlier_method = method

        if method == "knn_iqr":
            k_neighbors = st.slider("KNN 邻居数", 3, 20, 8, key="sr_knn")
            mad_factor = st.slider("MAD 阈值倍数", 1.0, 10.0, 3.5, 0.1, key="sr_mad")
            iqr_factor = st.slider("IQR 阈值倍数", 0.5, 5.0, 1.5, 0.1, key="sr_iqr")
            max_iter = st.slider("最大迭代次数", 1, 20, 5, key="sr_oi")
            st.session_state.sr_outlier_kwargs = {
                "k_neighbors": k_neighbors,
                "mad_factor": mad_factor,
                "iqr_factor": iqr_factor,
                "max_iterations": max_iter,
            }
        else:
            maxdist = st.slider("搜索半径 (km)", 50.0, 500.0, 200.0, 10.0, key="sr_loo_d")
            min_sites = st.slider("最少邻居数", 3, 20, 8, key="sr_loo_n")
            min_res = st.slider("残差阈值 (mm/yr)", 0.5, 10.0, 3.0, 0.5, key="sr_loo_r")
            st.session_state.sr_outlier_kwargs = {
                "maxdist_km": maxdist,
                "min_sites": min_sites,
                "min_residual_mm": min_res,
                "max_iterations": 3,
            }

        if st.button("🔍 运行异常值检测", type="primary", use_container_width=True, key="sr_run_outlier"):
            _run_outlier_detection()


def _run_outlier_detection() -> None:
    """Execute outlier detection and store results in session state."""
    vf = st.session_state.sr_vel_data
    if vf is None:
        st.error("请先加载速度场数据")
        return

    kwargs = st.session_state.sr_outlier_kwargs
    method = st.session_state.sr_outlier_method

    with st.spinner("正在检测异常值..."):
        try:
            # Build triangulation function adapter
            def _tri_fn(lon, lat, site_indices):
                tri, good, xy, _ = delaunay_triangulation(lon[site_indices], lat[site_indices])
                return tri, good, xy

            if method == "knn_iqr":
                vf_clean, history = iterative_outlier_removal(
                    vf, triangulation_fn=_tri_fn, **kwargs
                )
            else:
                # loo_strain_outlier_detection returns (mask, residuals, pred_ve, pred_vn)
                from pystrain.outlier import loo_strain_outlier_detection
                outlier_mask, residuals, _, _ = loo_strain_outlier_detection(vf, **kwargs)
                # Build vf_clean and history from the mask
                keep_mask = ~outlier_mask
                vf_clean = vf.subset(keep_mask)
                history = []
                for i in np.where(outlier_mask)[0]:
                    name = vf.names[i] if hasattr(vf, 'names') and vf.names is not None else f"S{i:04d}"
                    history.append({
                        "name": name, "lon": float(vf.lon[i]), "lat": float(vf.lat[i]),
                        "residual": float(residuals[i]), "iteration": 1,
                    })

            st.session_state.sr_outlier_result = (vf_clean, history)
            st.success(f"检测完成：剔除 {len(history)} 个异常点，保留 {len(vf_clean)} 个站点")
        except Exception as e:
            st.error(f"异常值检测失败: {e}")


def _main_outlier() -> None:
    """Main: show outlier detection results or raw velocity map."""
    st.header("Step 2: 异常值检测")

    vf = st.session_state.sr_vel_data
    if vf is None:
        st.info("请先在左侧加载数据，然后返回此步骤。")
        return

    outlier_result = st.session_state.sr_outlier_result

    if outlier_result is not None:
        vf_clean, history = outlier_result
        from pystrain.web.plots import plot_outlier_map

        # Build outlier mask
        outlier_names = {h.get("name", "") for h in history}
        outlier_mask = np.zeros(len(vf), dtype=bool)
        if hasattr(vf, 'names') and vf.names is not None and len(vf.names) > 0:
            for i, name in enumerate(vf.names):
                if name in outlier_names:
                    outlier_mask[i] = True

        fig = plot_outlier_map(
            vf.lon, vf.lat, outlier_mask,
            site_names=vf.names if hasattr(vf, 'names') else None,
            polygon=st.session_state.sr_poly_data,
            title=f"异常值检测结果 — 剔除 {len(history)} 个，保留 {len(vf_clean)} 个",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show outlier list
        if history:
            st.subheader(f"异常点列表 ({len(history)} 个)")
            import pandas as pd
            df = pd.DataFrame(history)
            st.dataframe(df, use_container_width=True)
    else:
        arrow_scale2 = st.slider("箭头缩放", 0.01, 1.0, 0.15, 0.01, key="sr_arrow_scale2",
                                 help="控制速度矢量箭头的长度")
        from pystrain.web.plots import plot_velocity_field
        fig = plot_velocity_field(
            vf.lon, vf.lat, vf.ve, vf.vn,
            site_names=vf.names if hasattr(vf, 'names') else None,
            polygon=st.session_state.sr_poly_data,
            title="原始速度场 — 尚未进行异常值检测",
            arrow_scale=arrow_scale2,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("👈 在左侧配置参数并点击「运行异常值检测」")


# ===================================================================
# Step 3: Algorithm Selection
# ===================================================================


def _sidebar_algorithm() -> None:
    """Sidebar: algorithm radio selection."""
    st.subheader("🧮 选择算法")

    algorithms = {
        "shen2015": "Shen et al. (2015) — 网格插值",
        "wang2012": "Wang (2012) — 自适应网格",
        "delaunay": "Delaunay 三角网最小二乘",
    }
    current = st.session_state.sr_algorithm
    selected = st.radio(
        "应变率计算方法",
        list(algorithms.keys()),
        format_func=lambda x: algorithms[x],
        index=list(algorithms.keys()).index(current) if current in algorithms else 0,
        key="sr_algo_radio",
    )
    if selected != st.session_state.sr_algorithm:
        st.session_state.sr_algorithm = selected
        # Clear downstream state when algorithm changes
        st.session_state.sr_algo_params = {}
        st.session_state.sr_result = None
        st.session_state.sr_show_preview = True
        st.session_state.sr_poly_data = None


def _main_algorithm() -> None:
    """Main: show algorithm description + parameter inputs + preview button."""
    st.header("Step 3: 算法选择与参数")

    algo = st.session_state.sr_algorithm
    vf = st.session_state.sr_vel_data
    poly = st.session_state.sr_poly_data

    if vf is None:
        st.info("请先加载数据。")
        return

    # --- Algorithm description ---
    descs = {
        "shen2015": (
            "**Shen et al. (2015)** — 网格最优插值法\n\n"
            "在规则格网点上，用高斯距离加权 + 空间覆盖加权对 GPS 速度场插值，"
            "通过最小二乘解算每个格网点的应变率张量。"
            "适用于大范围、均匀分布的 GNSS 台网。"
        ),
        "wang2012": (
            "**Wang (2012)** — 自适应三角网格反演法\n\n"
            "根据 GPS 站点密度自适应生成三角网格，用线性形函数插值 GPS 速度到网格顶点，"
            "通过 Laplacian 光滑正则化 + L-curve 最优光滑搜索反演应变率。"
            "适用于站点分布不均匀的区域。"
        ),
        "delaunay": (
            "**Delaunay 三角网最小二乘法**\n\n"
            "直接对 GPS 站点做 Delaunay 三角剖分，在每个三角形质心用三个顶点速度"
            "做最小二乘解算应变率张量。适用于站点密集、分布较均匀的局部区域。"
        ),
    }
    st.markdown(descs.get(algo, ""))

    # --- Algorithm-specific parameter inputs ---
    st.divider()
    st.subheader("参数设置")
    _show_algo_params(algo, vf, poly)


def _show_algo_params(algo: str, vf, poly) -> None:
    """Show parameter inputs and preview for the selected algorithm."""
    # Initialize params from session state
    params = st.session_state.sr_algo_params
    generate = False

    if algo == "shen2015":
        # Auto-compute defaults
        lon_min, lon_max = float(vf.lon.min()), float(vf.lon.max())
        lat_min, lat_max = float(vf.lat.min()), float(vf.lat.max())
        dlon, dlat = lon_max - lon_min, lat_max - lat_min
        def_region = f"{lon_min - 0.25*dlon:.1f} {lon_max + 0.25*dlon:.1f} {lat_min - 0.25*dlat:.1f} {lat_max + 0.25*dlat:.1f}"
        def_sp = min(max(dlon, dlat) / 20.0, 5.0)

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            region_str = st.text_input(
                "网格范围 (slon elon slat elat)",
                value=params.get("region_str", def_region),
                key="shen_region",
                help="四个数字，空格分隔：最小经度 最大经度 最小纬度 最大纬度",
            )
        with c2:
            spacing = st.number_input("格网间距 (°)", 0.05, 10.0,
                                      params.get("spacing", round(def_sp, 2)),
                                      0.05, key="shen_spacing")
        with c3:
            stagger = st.checkbox("交错排列", value=params.get("stagger", True), key="shen_stagger")

        with st.expander("高级加权参数", expanded=False):
            cw1, cw2 = st.columns(2)
            with cw1:
                dist_w = st.selectbox("距离权重", ["gaussian", "quadratic"],
                                      index=0 if params.get("distance_weight", "gaussian") == "gaussian" else 1,
                                      key="shen_dw2")
                Wt = st.number_input("权重阈值 Wt", 1.0, 100.0, params.get("Wt", 24.0), key="shen_wt2")
                min_sites = st.number_input("最少站点数", 3, 50, params.get("min_sites", 6), key="shen_ms2")
            with cw2:
                spat_w = st.selectbox("空间权重", ["voronoi", "none"],
                                      index=0 if params.get("spatial_weight", "voronoi") == "voronoi" else 1,
                                      key="shen_sw2")
                L0 = st.number_input("截断权重 L0", 1e-6, 1.0, params.get("L0", 0.01), format="%.4f", key="shen_l02")

        # Parse region string
        try:
            parts = region_str.split()
            region = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
        except (ValueError, IndexError):
            region = [lon_min - 0.25*dlon, lon_max + 0.25*dlon, lat_min - 0.25*dlat, lat_max + 0.25*dlat]

        params.update({
            "region_str": region_str, "region": region,
            "spacing": spacing, "stagger": stagger,
            "distance_weight": dist_w,
            "spatial_weight": None if spat_w == "none" else spat_w,
            "Wt": Wt, "L0": L0, "min_sites": int(min_sites),
            "auto_region": True,
        })
        st.session_state.sr_algo_params = params

    elif algo == "wang2012":
        c1, c2, c3 = st.columns(3)
        with c1:
            mesh_method = st.selectbox("网格方法",
                                       ["adaptive", "poisson", "gmsh"],
                                       format_func=lambda x: {"adaptive": "自适应", "poisson": "Poisson", "gmsh": "Gmsh"}[x],
                                       index=["adaptive", "poisson", "gmsh"].index(params.get("mesh_method", "adaptive")),
                                       key="wang_mm2")
        with c2:
            spacing = st.number_input("网格间距 (°)", 0.05, 2.0, params.get("mesh_spacing", 0.25), 0.05, key="wang_sp2")
        with c3:
            max_sta = st.number_input("每格最大站点数", 2, 20, params.get("max_stations_per_cell", 6), key="wang_msc2")

        # Polygon boundary file (required for Wang2012 mesh clipping)
        st.divider()
        st.markdown("**📐 边界多边形** *(Wang2012 需要多边形来裁剪网格)*")
        poly = polygon_file_uploader(key="sr_poly")
        if poly is not None:
            st.session_state.sr_poly_data = poly
        if st.session_state.sr_poly_data is not None:
            st.caption(f"✅ 已加载 {len(st.session_state.sr_poly_data)} 个顶点")
        else:
            st.warning("⚠️ 请上传多边形文件，否则将使用 GPS 站点范围自动生成矩形边界")

        with st.expander("光滑参数", expanded=False):
            smooth_search = st.checkbox("L-curve 搜索", value=params.get("smooth_search", True), key="wang_ss2")
            if smooth_search:
                c1, c2 = st.columns(2)
                with c1:
                    s_min = st.number_input("log10(smooth_min)", -5.0, 0.0, params.get("smooth_min", -2.2), 0.1, key="wang_smin2")
                with c2:
                    s_max = st.number_input("log10(smooth_max)", -3.0, 2.0, params.get("smooth_max", -0.8), 0.1, key="wang_smax2")
                params.update({"smooth_search": True, "smooth_range": [s_min, s_max]})
            else:
                sf = st.number_input("光滑因子", 0.0, 1.0, params.get("smooth_factor", 0.01), 0.001, format="%.4f", key="wang_sf2")
                params.update({"smooth_search": False, "smooth_factor": sf})

        params.update({
            "mesh_method": mesh_method, "mesh_spacing": spacing,
            "max_stations_per_cell": int(max_sta),
        })
        st.session_state.sr_algo_params = params

    elif algo == "delaunay":
        c1, c2, c3 = st.columns(3)
        with c1:
            min_angle = st.slider("最小角度 (°)", 1.0, 30.0, params.get("min_angle_deg", 10.0), 1.0, key="del_ma2")
            edge_pctl = st.slider("边长百分位数", 50.0, 99.0, params.get("max_edge_pctl", 95.0), 1.0, key="del_ep2")
        with c2:
            edge_factor = st.number_input("边长倍数", 1.0, 5.0, params.get("max_edge_factor", 1.5), 0.1, key="del_ef2")
            area_ratio = st.number_input("最小面积比", 0.01, 1.0, params.get("min_area_ratio", 0.1), 0.01, key="del_ar2")
        with c3:
            use_me = st.checkbox("边长上限", value=params.get("max_edge_km") is not None, key="del_ume2")
            max_edge = None
            if use_me:
                max_edge = st.number_input("最大边长 (km)", 10.0, 2000.0, params.get("max_edge_km") or 300.0, 10.0, key="del_me2")
        params.update({
            "min_angle_deg": min_angle, "max_edge_pctl": edge_pctl,
            "max_edge_factor": edge_factor, "min_area_ratio": area_ratio,
            "max_edge_km": max_edge,
        })
        st.session_state.sr_algo_params = params

    # --- Generate Preview button ---
    generate = st.button("🔄 生成预览图", use_container_width=True, key="sr_gen_preview")

    if generate or st.session_state.get("sr_show_preview", False):
        st.session_state.sr_show_preview = True
        st.divider()
        st.subheader("预览")
        with st.spinner("生成预览..."):
            try:
                if algo == "shen2015":
                    _show_shen2015_preview(vf, poly, params)
                elif algo == "wang2012":
                    _show_wang2012_preview(vf, poly, params)
                elif algo == "delaunay":
                    _show_delaunay_preview(vf, poly, params)
            except Exception as e:
                st.error(f"预览生成失败: {e}")


def _show_shen2015_preview(vf, poly, params) -> None:
    from pystrain.web.plots import plot_grid_points_overlay
    from pystrain.grid import Grid
    region = params.get("region")
    if region is None:
        lon_min, lon_max = float(vf.lon.min()), float(vf.lon.max())
        lat_min, lat_max = float(vf.lat.min()), float(vf.lat.max())
        dlon, dlat = lon_max - lon_min, lat_max - lat_min
        region = [lon_min - 0.25*dlon, lon_max + 0.25*dlon, lat_min - 0.25*dlat, lat_max + 0.25*dlat]
    sp = params.get("spacing", 0.2)
    grid = Grid(region[0], region[1], region[2], region[3], sp, sp, stagger=params.get("stagger", True))
    fig = plot_grid_points_overlay(grid.lon, grid.lat, vf.lon, vf.lat, polygon=poly,
                                   title=f"Shen et al. (2015) — {len(grid.lon)} 格网点, 间距 {sp}°")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"范围: {region[0]:.2f}–{region[1]:.2f}°, {region[2]:.2f}–{region[3]:.2f}° | 间距: {sp}° | 格点数: {len(grid.lon)}")


def _show_wang2012_preview(vf, poly, params) -> None:
    from pystrain.web.plots import plot_wang2012_mesh_preview
    from pystrain.wang2012.mesh import generate_mesh
    if poly is not None:
        lon_min, lon_max = float(poly[:, 0].min()), float(poly[:, 0].max())
        lat_min, lat_max = float(poly[:, 1].min()), float(poly[:, 1].max())
    else:
        lon_min, lon_max = float(vf.lon.min()), float(vf.lon.max())
        lat_min, lat_max = float(vf.lat.min()), float(vf.lat.max())
    dlon, dlat = lon_max - lon_min, lat_max - lat_min
    region = [lon_min - 0.1*dlon, lon_max + 0.1*dlon, lat_min - 0.1*dlat, lat_max + 0.1*dlat]
    vertices, simplices = generate_mesh(
        method=params.get("mesh_method", "adaptive"),
        region=region,
        spacing=params.get("mesh_spacing", 0.25),
        max_stations_per_cell=params.get("max_stations_per_cell", 6),
        max_spacing=None, randomize=True, randomize_fraction=0.2,
        sites_lon=vf.lon, sites_lat=vf.lat, polygon=poly,
    )
    fig = plot_wang2012_mesh_preview(vertices[:, 0], vertices[:, 1], simplices,
                                     vf.lon, vf.lat, polygon=poly,
                                     title=f"Wang (2012) — {len(vertices)} 顶点, {len(simplices)} 三角形")
    st.plotly_chart(fig, use_container_width=True)


def _show_delaunay_preview(vf, poly, params) -> None:
    from pystrain.web.plots import plot_triangulation_mesh
    tri, good, xy, _ = delaunay_triangulation(
        vf.lon, vf.lat, polygon=poly,
        min_angle_deg=params.get("min_angle_deg", 10.0),
        max_edge_pctl=params.get("max_edge_pctl", 95.0),
        max_edge_factor=params.get("max_edge_factor", 1.5),
        min_area_ratio=params.get("min_area_ratio", 0.1),
        max_edge_km=params.get("max_edge_km"),
    )
    fig = plot_triangulation_mesh(tri, vf.lon, vf.lat, good_mask=good, polygon=poly,
                                  title=f"Delaunay — {good.sum()} 个有效三角形")
    st.plotly_chart(fig, use_container_width=True)


# (old _preview_wang2012 removed — now using _show_wang2012_preview)


# ===================================================================
# Step 4: Parameter Configuration
# ===================================================================


def _sidebar_params() -> None:
    """Sidebar: dynamic parameter panel based on selected algorithm."""
    algo = st.session_state.sr_algorithm
    st.subheader("⚙️ 算法参数配置")

    params = st.session_state.sr_algo_params

    if algo == "shen2015":
        _params_shen2015(params)
    elif algo == "wang2012":
        _params_wang2012(params)
    elif algo == "delaunay":
        _params_delaunay(params)

    if st.button("🔄 刷新预览", use_container_width=True, key="sr_refresh_preview"):
        st.session_state.sr_show_preview = True


def _params_shen2015(params: Dict) -> None:
    """Shen2015 parameters."""
    with st.expander("网格范围与间距", expanded=True):
        auto_region = st.checkbox("自动计算范围", value=params.get("auto_region", True), key="shen_auto")
        if auto_region:
            spacing = st.number_input("网格间距 (°)", 0.05, 10.0, params.get("spacing", 0.2), 0.05, key="shen_sp")
            params.update({"auto_region": True, "spacing": spacing})
        else:
            c1, c2 = st.columns(2)
            with c1:
                slon = st.number_input("起始经度", value=params.get("slon", 70.0), key="shen_slon")
                slat = st.number_input("起始纬度", value=params.get("slat", 15.0), key="shen_slat")
            with c2:
                elon = st.number_input("结束经度", value=params.get("elon", 110.0), key="shen_elon")
                elat = st.number_input("结束纬度", value=params.get("elat", 55.0), key="shen_elat")
            spacing = st.number_input("网格间距 (°)", 0.05, 10.0, params.get("spacing", 0.2), 0.05, key="shen_sp_m")
            params.update({"auto_region": False, "slon": slon, "elon": elon, "slat": slat, "elat": elat,
                           "spacing": spacing, "region": [slon, elon, slat, elat]})
        stagger = st.checkbox("交错排列", value=params.get("stagger", True), key="shen_stagger")
        params["stagger"] = stagger

    with st.expander("加权参数", expanded=True):
        dist_w = st.selectbox("距离权重", ["gaussian", "quadratic"],
                              index=0 if params.get("distance_weight", "gaussian") == "gaussian" else 1,
                              key="shen_dw")
        spat_w = st.selectbox("空间权重", ["voronoi", "none"],
                              index=0 if params.get("spatial_weight", "voronoi") == "voronoi" else 1,
                              key="shen_sw")
        Wt = st.number_input("权重阈值 Wt", 1.0, 100.0, params.get("Wt", 24.0), key="shen_wt")
        L0 = st.number_input("截断权重 L0", 1e-6, 1.0, params.get("L0", 0.01), format="%.4f", key="shen_l0")
        min_sites = st.number_input("最少站点数", 3, 50, params.get("min_sites", 6), key="shen_ms")
        use_maxdist = st.checkbox("启用初始距离截断", value=params.get("maxdist_km") is not None, key="shen_umd")
        maxdist = None
        if use_maxdist:
            maxdist = st.number_input("初始截断距离 (km)", 10.0, 1000.0,
                                      params.get("maxdist_km") or 200.0, 10.0, key="shen_md")
        params.update({
            "distance_weight": dist_w,
            "spatial_weight": None if spat_w == "none" else spat_w,
            "Wt": Wt, "L0": L0, "min_sites": int(min_sites),
            "maxdist_km": maxdist,
        })

    st.session_state.sr_algo_params = params


def _params_wang2012(params: Dict) -> None:
    """Wang2012 parameters."""
    with st.expander("网格生成", expanded=True):
        mesh_method = st.selectbox(
            "网格方法",
            ["adaptive", "poisson", "gmsh"],
            index=["adaptive", "poisson", "gmsh"].index(params.get("mesh_method", "adaptive")),
            format_func=lambda x: {"adaptive": "自适应网格", "poisson": "Poisson 盘采样", "gmsh": "Gmsh 有限元"}[x],
            key="wang_mm",
        )
        spacing = st.number_input("网格间距 (°)", 0.05, 2.0, params.get("mesh_spacing", 0.25), 0.05, key="wang_sp")
        max_sta = st.number_input("每格最大站点数", 2, 20, params.get("max_stations_per_cell", 6), key="wang_msc")
        params.update({"mesh_method": mesh_method, "mesh_spacing": spacing, "max_stations_per_cell": int(max_sta)})

    with st.expander("光滑参数", expanded=True):
        smooth_search = st.checkbox("L-curve 搜索最优光滑因子", value=params.get("smooth_search", True), key="wang_ss")
        if smooth_search:
            c1, c2 = st.columns(2)
            with c1:
                s_min = st.number_input("log10(smooth_min)", -5.0, 0.0, params.get("smooth_min", -2.2), 0.1, key="wang_smin")
            with c2:
                s_max = st.number_input("log10(smooth_max)", -3.0, 2.0, params.get("smooth_max", -0.8), 0.1, key="wang_smax")
            params.update({"smooth_search": True, "smooth_range": [s_min, s_max]})
        else:
            sf = st.number_input("固定光滑因子", 0.0, 1.0, params.get("smooth_factor", 0.01), 0.001, format="%.4f", key="wang_sf")
            params.update({"smooth_search": False, "smooth_factor": sf})

    st.session_state.sr_algo_params = params


def _params_delaunay(params: Dict) -> None:
    """Delaunay parameters."""
    min_angle = st.slider("最小角度 (°)", 1.0, 30.0, params.get("min_angle_deg", 10.0), 1.0, key="del_ma")
    edge_pctl = st.slider("边长百分位数", 50.0, 99.0, params.get("max_edge_pctl", 95.0), 1.0, key="del_ep")
    edge_factor = st.number_input("边长倍数因子", 1.0, 5.0, params.get("max_edge_factor", 1.5), 0.1, key="del_ef")
    area_ratio = st.number_input("最小面积比", 0.01, 1.0, params.get("min_area_ratio", 0.1), 0.01, key="del_ar")
    use_me = st.checkbox("启用边长绝对上限", value=params.get("max_edge_km") is not None, key="del_ume")
    max_edge = None
    if use_me:
        max_edge = st.number_input("最大边长 (km)", 10.0, 2000.0, params.get("max_edge_km") or 300.0, 10.0, key="del_me")
    proj = st.selectbox("投影方式", ["utm", "polyconic"],
                        index=0 if params.get("projection", "utm") == "utm" else 1, key="del_proj")
    params.update({
        "min_angle_deg": min_angle, "max_edge_pctl": edge_pctl,
        "max_edge_factor": edge_factor, "min_area_ratio": area_ratio,
        "max_edge_km": max_edge, "projection": proj,
    })
    st.session_state.sr_algo_params = params


def _main_params() -> None:
    """Main: show updated preview based on current parameters."""
    st.header("Step 4: 参数配置")
    algo = st.session_state.sr_algorithm
    vf = st.session_state.sr_vel_data
    poly = st.session_state.sr_poly_data
    params = st.session_state.sr_algo_params

    if vf is None:
        st.info("请先加载数据。")
        return

    show = st.session_state.get("sr_show_preview", False)
    if not show and not params:
        st.info("👈 在左侧调整参数，然后点击「刷新预览」查看效果。")
        return

    try:
        if algo == "shen2015":
            _show_shen2015_preview(vf, poly, params)
        elif algo == "wang2012":
            _show_wang2012_preview(vf, poly, params)
        elif algo == "delaunay":
            _show_delaunay_preview(vf, poly, params)
    except Exception as e:
        st.error(f"预览生成失败: {e}")
        st.info("请调整参数后重试。")


# (old param-based preview functions merged into _show_*_preview above)


# ===================================================================
# Step 5: Compute
# ===================================================================


def _sidebar_compute() -> None:
    """Sidebar: Monte Carlo settings + compute button."""
    st.subheader("🚀 运行计算")

    with st.expander("Monte Carlo 不确定度", expanded=True):
        mc_enable = st.checkbox("启用 Monte Carlo", value=st.session_state.mc_enabled, key="sr_mc_enable")
        st.session_state.mc_enabled = mc_enable
        if mc_enable:
            mc_iter = st.number_input("MC 迭代次数", 50, 2000, st.session_state.mc_iterations, 50, key="sr_mc_iter")
            mc_seed = st.number_input("随机种子", 0, 999999, st.session_state.mc_seed, key="sr_mc_seed")
            st.session_state.mc_iterations = int(mc_iter)
            st.session_state.mc_seed = int(mc_seed)

    if st.button("🚀 运行应变计算", type="primary", use_container_width=True, key="sr_run_compute"):
        _run_computation()


def _run_computation() -> None:
    """Execute the full pipeline."""
    vf = st.session_state.sr_vel_data
    if vf is None:
        st.error("请先加载数据。")
        return

    algo = st.session_state.sr_algorithm
    params = st.session_state.sr_algo_params
    poly = st.session_state.sr_poly_data

    # Prepare paths
    with tempfile.TemporaryDirectory() as tmpdir:
        vel_path = os.path.join(tmpdir, "vel.dat")
        _write_vel_file(vf, vel_path, fmt="gmt8")

        poly_path = None
        if poly is not None:
            poly_path = os.path.join(tmpdir, "poly.dat")
            np.savetxt(poly_path, poly)

        # Prepare kwargs based on algorithm
        tri_kwargs = {"min_angle_deg": 10.0, "max_edge_pctl": 95.0, "max_edge_factor": 1.5, "min_area_ratio": 0.1}
        grid_kwargs = None
        wang2012_kwargs = None

        # Use outlier-cleaned data if available
        outlier_result = st.session_state.sr_outlier_result
        if outlier_result is not None:
            vf_clean, _ = outlier_result
            vel_path_clean = os.path.join(tmpdir, "vel_clean.dat")
            _write_vel_file(vf_clean, vel_path_clean, fmt="gmt8")
            # Re-write vel_path to use cleaned data
            vel_path = vel_path_clean

        if algo == "shen2015":
            # Build region — use stored region from Step 3/4 if available
            region = params.get("region")
            if region is None:
                # Try Step 4 manual parameters
                if not params.get("auto_region", True) and "slon" in params:
                    region = [params["slon"], params["elon"], params["slat"], params["elat"]]
                else:
                    # Fallback: auto-compute from GPS site bounds
                    lon_min, lon_max = float(vf.lon.min()), float(vf.lon.max())
                    lat_min, lat_max = float(vf.lat.min()), float(vf.lat.max())
                    dlon, dlat = lon_max - lon_min, lat_max - lat_min
                    buf = 0.25
                    region = [lon_min - buf * dlon, lon_max + buf * dlon,
                              lat_min - buf * dlat, lat_max + buf * dlat]
            grid_kwargs = {
                "region": region,
                "spacing": [params.get("spacing", 0.2), params.get("spacing", 0.2)],
                "distance_weight": params.get("distance_weight", "gaussian"),
                "spatial_weight": params.get("spatial_weight"),
                "Wt": params.get("Wt", 24.0),
                "L0": params.get("L0", 0.01),
                "min_sites": params.get("min_sites", 6),
                "maxdist_km": params.get("maxdist_km"),
            }
            algorithm = "shen2015"

        elif algo == "wang2012":
            wang2012_kwargs = {
                "mesh_method": params.get("mesh_method", "adaptive"),
                "mesh_spacing": params.get("mesh_spacing", 0.25),
                "max_stations_per_cell": params.get("max_stations_per_cell", 6),
                "smooth_search": params.get("smooth_search", True),
                "smooth_range": params.get("smooth_range", (-2.2, -0.8)),
                "smooth_factor": params.get("smooth_factor", 0.01),
                "smooth_step": params.get("smooth_step", 0.2),
                "smooth_boundary": params.get("smooth_boundary", True),
            }
            algorithm = "wang2012"

        elif algo == "delaunay":
            tri_kwargs = {
                "min_angle_deg": params.get("min_angle_deg", 10.0),
                "max_edge_pctl": params.get("max_edge_pctl", 95.0),
                "max_edge_factor": params.get("max_edge_factor", 1.5),
                "min_area_ratio": params.get("min_area_ratio", 0.1),
                "max_edge_km": params.get("max_edge_km"),
            }
            algorithm = "delaunay"

        else:
            st.error(f"未知算法: {algo}")
            return

        # Run pipeline
        progress_bar = st.progress(0, text="准备中...")
        status_text = st.empty()
        log_buf = io.StringIO()

        def stage_cb(stage: int, total: int, msg: str):
            pct = int(stage / total * 100)
            progress_bar.progress(pct, text=f"[{stage}/{total}] {msg}")
            status_text.text(msg)

        try:
            old_stdout = sys.stdout
            sys.stdout = log_buf

            mc_iter = st.session_state.mc_iterations if st.session_state.mc_enabled else 0
            result = run_pystrain_pipeline(
                vel_path=vel_path,
                poly_path=poly_path,
                fmt="auto",
                algorithm=algorithm,
                outlier_enable=False,  # already handled in step 2
                outlier_kwargs={},
                tri_kwargs=tri_kwargs,
                grid_kwargs=grid_kwargs,
                wang2012_kwargs=wang2012_kwargs,
                mc_iterations=mc_iter,
                mc_seed=st.session_state.mc_seed,
                stage_callback=stage_cb,
            )
            sys.stdout = old_stdout

            progress_bar.progress(100, text="计算完成！")
            status_text.empty()

            with st.expander("运行日志", expanded=False):
                st.code(log_buf.getvalue())

            st.session_state.sr_result = result
            st.success(f"计算完成！{result['n_good_triangles']} 个有效三角形/网格点")
            # Auto-advance to results
            st.session_state.sr_step = "results"
            st.rerun()

        except Exception as e:
            sys.stdout = old_stdout
            progress_bar.empty()
            status_text.empty()
            with st.expander("运行日志", expanded=True):
                st.code(log_buf.getvalue())
            st.error(f"计算失败: {e}")


def _main_compute() -> None:
    """Main: show status before computation."""
    st.header("Step 5: 运行计算")

    vf = st.session_state.sr_vel_data
    if vf is None:
        st.info("请先加载数据。")
        return

    st.markdown(f"""
    **准备计算：**
    - 算法: **{st.session_state.sr_algorithm}**
    - 站点数: **{len(vf)}**
    - Monte Carlo: **{'启用' if st.session_state.mc_enabled else '禁用'}**
    """)

    # Show last preview
    try:
        algo = st.session_state.sr_algorithm
        params = st.session_state.sr_algo_params
        poly = st.session_state.sr_poly_data
        if algo == "shen2015":
            _show_shen2015_preview(vf, poly, params)
        elif algo == "wang2012":
            _show_wang2012_preview(vf, poly, params)
        elif algo == "delaunay":
            _show_delaunay_preview(vf, poly, params)
    except Exception:
        pass

    st.info("👈 在左侧配置 Monte Carlo 参数并点击「运行应变计算」。")


# ===================================================================
# Step 6: Results
# ===================================================================


def _sidebar_results() -> None:
    """Sidebar: show summary and download options after computation."""
    result = st.session_state.sr_result
    if result is None:
        return

    st.subheader("📊 计算摘要")
    st.metric("算法", result.get("algorithm", "unknown"))
    st.metric("输入站点数", result.get("n_sites_total", "-"))
    st.metric("使用站点数", result.get("n_sites_used", "-"))
    st.metric("剔除异常点", result.get("n_outliers", "-"))
    st.metric("有效三角形", result.get("n_good_triangles", "-"))

    st.divider()
    from pystrain.web.components.results import render_result_downloads
    render_result_downloads(result, prefix="strain_rate")


def _main_results() -> None:
    """Main: show interactive result tabs with fault overlay controls."""
    result = st.session_state.sr_result
    if result is None:
        st.info("没有计算结果。请返回 Step 5 运行计算。")
        return

    st.header("Step 6: 计算结果")

    from pystrain.web.components.results import render_summary_metrics, render_strain_field_tabs
    render_summary_metrics(result)

    # --- Fault file uploader (right side, multi-file) ---
    st.divider()
    fcol1, fcol2, fcol3 = st.columns([3, 1, 1])
    with fcol1:
        fault_files = st.file_uploader(
            "🗺️ 导入断层文件（可选，支持多选）",
            type=["kml", "kmz", "shp", "zip", "json", "geojson"],
            key="sr_fault_upload",
            accept_multiple_files=True,
            help="KML / Shapefile / GeoJSON 格式的断层迹线，支持多文件",
        )
        if fault_files:
            from pystrain.web.faults import parse_fault_file
            all_traces = list(st.session_state.get("sr_faults") or [])
            new_count = 0
            for f in fault_files:
                traces, err = parse_fault_file(f)
                if err:
                    st.warning(f"{f.name}: {err}")
                elif traces:
                    all_traces.extend(traces)
                    new_count += len(traces)
            if new_count:
                st.session_state.sr_faults = all_traces
                st.success(f"已加载 {new_count} 条断层迹线（共 {len(all_traces)} 条）")
    with fcol2:
        if st.session_state.get("sr_faults"):
            n = len(st.session_state.sr_faults)
            st.metric("断层迹线数", n)
    with fcol3:
        if st.session_state.get("sr_faults"):
            if st.button("🗑️ 清除断层", key="sr_clear_faults", use_container_width=True):
                st.session_state.sr_faults = None
                st.rerun()

    st.divider()
    render_strain_field_tabs(result, faults=st.session_state.get("sr_faults"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vel_file(vf: VelocityField, path: str, fmt: str = "gmt8") -> None:
    """Write a VelocityField to a GMT8-format file."""
    with open(path, "w") as f:
        for i in range(len(vf)):
            rho = vf.rho[i] if hasattr(vf, 'rho') and vf.rho is not None else 0.0
            name = vf.names[i] if hasattr(vf, 'names') and vf.names is not None else f"S{i:04d}"
            f.write(
                f"{vf.lon[i]:12.6f} {vf.lat[i]:11.6f} "
                f"{vf.ve[i]:8.3f} {vf.vn[i]:8.3f} "
                f"{vf.se[i]:6.3f} {vf.sn[i]:6.3f} "
                f"{rho:6.3f} {name}\n"
            )
