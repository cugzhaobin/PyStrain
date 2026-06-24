"""Result display components for the PyStrain2 web app."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st


def render_summary_metrics(metrics: Dict[str, Any]) -> None:
    """Render a row of metric cards for pipeline results."""
    cols = st.columns(4)
    items = [
        ("输入站点数", metrics.get("n_sites_total", "-")),
        ("使用站点数", metrics.get("n_sites_used", "-")),
        ("剔除异常点", metrics.get("n_outliers", "-")),
        ("有效三角形/网格点", metrics.get("n_locations", metrics.get("n_good_triangles", "-"))),
    ]
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)


def _get_display_opts(algorithm: str, clon: np.ndarray, clat: np.ndarray) -> dict:
    """Render display-mode controls and return a dict of options.

    For Delaunay the only sensible visualisation is filled triangles, so
    we skip the scatter-vs-grid toggle entirely.
    Controls are conditional on the selected mode:
    - scatter → marker size slider
    - grid    → interpolation density slider
    """
    # Auto-compute a dense default grid spacing from the data extent
    dlon = float(np.nanmax(clon) - np.nanmin(clon))
    dlat = float(np.nanmax(clat) - np.nanmin(clat))
    def_dense = max(0.02, min(0.5, max(dlon, dlat) / 50.0))

    if algorithm == "delaunay":
        return {"mode": "triangles", "grid_spacing": round(def_dense, 3), "marker_size": 50}

    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio(
            "展示模式", ["scatter", "grid"],
            format_func=lambda x: "散点图" if x == "scatter" else "GMT 风格网格",
            key="sr_display_mode", horizontal=True,
        )
    with c2:
        if mode == "scatter":
            marker_size = st.slider("散点大小", 5, 200, 50, 5, key="sr_marker_size")
            grid_spacing = def_dense
        else:
            marker_size = 50  # unused in grid mode
            grid_spacing = st.slider(
                "内插密度", 0.02, 2.0, def_dense, 0.01,
                key="sr_grid_spacing",
                help="值越小网格越密",
            )
    return {"mode": mode, "grid_spacing": grid_spacing, "marker_size": marker_size}


def _plot_scalar(result: Dict[str, Any], field_key: str,
                 field_name: str, cmap: str, symmetric: bool,
                 colorbar_label: str, title: str, opts: dict,
                 faults=None) -> None:
    """Plot a scalar field — dispatches to the correct renderer."""
    values = result[field_key]

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        st.warning(f"无有效{field_name}数据")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("最大值", f"{finite.max():.2f}")
    c2.metric("最小值", f"{finite.min():.2f}")
    c3.metric("均值", f"{finite.mean():.2f}")

    polygon = result.get("_polygon")
    clon = result["centroids_lon"]
    clat = result["centroids_lat"]
    algo = result.get("algorithm", "")

    mode = opts["mode"]

    if algo == "delaunay" or mode == "triangles":
        from pystrain.web.plots import plot_triangle_strain_field
        tri = result.get("_tri")
        good_mask = result.get("_good_mask")
        site_lon = result.get("_lon")
        site_lat = result.get("_lat")
        if tri is None or good_mask is None or site_lon is None:
            st.warning("缺少三角网数据，无法渲染三角形填充")
            return
        fig = plot_triangle_strain_field(
            tri, good_mask, site_lon, site_lat,
            values,
            field_name=field_name, cmap=cmap, symmetric=symmetric,
            colorbar_label=colorbar_label, polygon=polygon, faults=faults,
            title=title,
        )

    elif mode == "grid":
        from pystrain.web.plots import plot_gridded_strain_field
        fig = plot_gridded_strain_field(
            clon, clat, values,
            field_name=field_name, cmap=cmap, symmetric=symmetric,
            colorbar_label=colorbar_label, polygon=polygon, faults=faults,
            title=title, grid_spacing=opts["grid_spacing"],
        )
    else:
        from pystrain.web.plots import plot_scalar_strain_field
        fig = plot_scalar_strain_field(
            clon, clat, values,
            field_name=field_name, cmap=cmap, symmetric=symmetric,
            colorbar_label=colorbar_label, polygon=polygon, faults=faults,
            title=title, marker_size=opts.get("marker_size", 50),
        )
    st.plotly_chart(fig, use_container_width=True)


def render_strain_field_tabs(result: Dict[str, Any], faults=None) -> None:
    """Render tabs for strain-rate-field results with display controls.

    Parameters
    ----------
    faults : optional list of (N, 2) arrays of fault trace coordinates.
    """
    from pystrain.web.plots import plot_principal_strain

    clon = result["centroids_lon"]
    clat = result["centroids_lat"]
    polygon = result.get("_polygon")
    algorithm = result.get("algorithm", "shen2015")

    # --- Global display controls ---
    opts = _get_display_opts(algorithm, clon, clat)

    tabs = st.tabs([
        "📐 面膨胀率",
        "🔪 最大剪应变",
        "✚ 主应变十字",
        "📏 第二不变量",
        "🔄 旋转率",
    ])

    # --- Dilatation ---
    with tabs[0]:
        _plot_scalar(
            result, "dilatation", "Dilatation", "RdBu_r", True,
            "Dilatation (nstrain/yr)", "面膨胀率 (Dilatation Rate)", opts,
            faults=faults,
        )

    # --- Max Shear ---
    with tabs[1]:
        _plot_scalar(
            result, "max_shear", "Max Shear", "YlOrRd", False,
            "Max Shear (nstrain/yr)", "最大剪应变率 (Max Shear Strain Rate)", opts,
            faults=faults,
        )

    # --- Principal Strain ---
    with tabs[2]:
        e1 = result["e1"]
        e2 = result["e2"]
        azimuth = result["azimuth"]
        finite_e = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(azimuth)
        if finite_e.any():
            c1, c2, c3 = st.columns(3)
            c1.metric("e1 均值", f"{np.mean(e1[finite_e]):.1f}")
            c2.metric("e2 均值", f"{np.mean(e2[finite_e]):.1f}")
            all_vals = np.concatenate([np.abs(e1[finite_e]), np.abs(e2[finite_e])])
            max_val = float(np.max(all_vals))
            def_scale = 0.3 / max(max_val, 1.0) if max_val > 0 else 0.1
            scale = c3.slider(
                "十字缩放", 0.01, 2.0, min(2.0, max(0.01, def_scale * 5.0)),
                0.01, key="sr_pscale", help="控制主应变十字的大小",
            )
            fig = plot_principal_strain(
                clon, clat, e1, e2, azimuth,
                polygon=polygon, faults=faults,
                title="主应变率 (Principal Strain) — 蓝=压缩, 红=拉张",
                scale=scale,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("无有效主应变数据")

    # --- Second Invariant ---
    with tabs[3]:
        sec_inv = result.get("sec_inv")
        if sec_inv is not None:
            _plot_scalar(
                result, "sec_inv", "Second Invariant", "plasma", False,
                "2nd Invariant (nstrain²/yr²)", "第二不变量 (Second Invariant)", opts,
                faults=faults,
            )

    # --- Omega ---
    with tabs[4]:
        omega = result.get("omega")
        if omega is not None:
            _plot_scalar(
                result, "omega", "Omega", "RdBu_r", True,
                "Omega (nrad/yr)", "旋转率 (Rotation Rate)", opts,
                faults=faults,
            )


def render_result_downloads(result: Dict[str, Any], prefix: str = "result") -> None:
    """Render download buttons for strain results as CSV."""
    import pandas as pd

    clon = result.get("centroids_lon", [])
    clat = result.get("centroids_lat", [])
    n = len(clon)

    df_data: Dict[str, Any] = {"lon": clon, "lat": clat}
    for key in ["dilatation", "max_shear", "e1", "e2", "azimuth", "sec_inv",
                "omega", "exx", "exy", "eyy", "ve", "vn"]:
        if key in result and result[key] is not None:
            arr = result[key]
            if isinstance(arr, np.ndarray) and len(arr) == n:
                df_data[key] = arr

    df = pd.DataFrame(df_data)
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 下载结果 (CSV)",
        data=csv,
        file_name=f"{prefix}_strain_results.csv",
        mime="text/csv",
    )
