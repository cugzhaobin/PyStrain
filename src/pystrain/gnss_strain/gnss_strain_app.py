"""
gnss_strain_app.py — GNSS 应变率计算 Streamlit 网页界面

启动方式:
    streamlit run gnss_strain_app.py
"""

import io
import sys
import os
import tempfile

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, validate_config, cfg_to_kwargs
from gnss_strain import run_full_pipeline

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GNSS Strain Rate Calculator",
    page_icon="🌍",
    layout="wide",
)

st.title("GNSS Strain Rate Calculator")
st.caption("GNSS 速度场 → 应变率计算  |  基于 Delaunay 三角化 + Monte Carlo 不确定度")

# ---------------------------------------------------------------------------
# 侧边栏参数控件
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("参数设置")

    # ---- 数据文件 ----
    st.subheader("数据文件")
    vel_file_upload = st.file_uploader(
        "速度场文件", type=["vel", "txt", "dat"],
        help="GMT 8列 或 GLOBK 13列 格式"
    )
    poly_file_upload = st.file_uploader(
        "边界多边形文件（可选）", type=["txt", "dat", "poly"],
        help="GMT 多边形格式，留空则自动使用矩形边界"
    )
    vel_format = st.selectbox(
        "输入格式", ["auto", "gmt", "globk"],
        help="auto=自动识别，gmt=8列，globk=13列"
    )
    output_dir = st.text_input("输出目录", value="output_app",
                               help="结果文件和图片保存路径")

    # ---- 密度控制 ----
    st.subheader("三角网密度控制")
    use_thinning = st.checkbox("启用站点抽稀", value=False)
    min_spacing_km = st.number_input(
        "最小站点间距 (km)", min_value=1.0, max_value=500.0,
        value=50.0, step=5.0,
        disabled=not use_thinning,
        help="距离小于此值的相邻站点将被合并（保留精度较好的站点）"
    ) if use_thinning else None

    use_max_edge = st.checkbox("启用边长绝对上限", value=False)
    max_edge_km = st.number_input(
        "最大三角形边长 (km)", min_value=10.0, max_value=2000.0,
        value=300.0, step=10.0,
        disabled=not use_max_edge,
        help="超过此长度的三角形边将被剔除"
    ) if use_max_edge else None

    # ---- 三角网质量 ----
    st.subheader("三角网质量")
    min_angle_deg = st.slider(
        "最小角度 (°)", min_value=1.0, max_value=30.0, value=10.0, step=1.0
    )
    max_edge_pctl = st.slider(
        "边长百分位数", min_value=50.0, max_value=99.0, value=95.0, step=1.0
    )
    max_edge_factor = st.number_input(
        "边长倍数因子", min_value=1.0, max_value=5.0, value=1.5, step=0.1
    )

    # ---- 异常值检测 ----
    st.subheader("异常值检测")
    k_neighbors = st.slider("KNN 邻居数", min_value=3, max_value=20, value=8)
    mad_factor = st.number_input(
        "MAD 阈值倍数", min_value=1.0, max_value=10.0, value=3.5, step=0.1
    )
    iqr_factor = st.number_input(
        "IQR 阈值倍数", min_value=0.5, max_value=5.0, value=1.5, step=0.1
    )
    max_outlier_iter = st.slider("最大迭代次数", min_value=1, max_value=20, value=5)

    # ---- 光滑 & 不确定度 ----
    st.subheader("光滑 & 不确定度")
    smooth_weight = st.slider(
        "光滑权重", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )
    smooth_iter = st.slider("光滑迭代次数", min_value=0, max_value=10, value=2)
    mc_iterations = st.number_input(
        "Monte Carlo 迭代次数", min_value=50, max_value=2000,
        value=200, step=50
    )

    st.divider()

    # ---- YAML 导入/导出 ----
    st.subheader("配置文件")
    yaml_upload = st.file_uploader("导入 YAML 配置", type=["yaml", "yml"])
    if yaml_upload is not None:
        cfg_text = yaml_upload.read().decode("utf-8")
        st.session_state['imported_yaml'] = cfg_text
        st.success("配置已导入，点击【运行计算】应用")

    # 导出当前配置
    current_cfg_dict = {
        'data': {
            'vel_file': '', 'poly_file': None,
            'output_dir': output_dir, 'vel_format': vel_format,
        },
        'triangulation': {
            'min_angle_deg': min_angle_deg,
            'max_edge_pctl': max_edge_pctl,
            'max_edge_factor': max_edge_factor,
            'min_spacing_km': min_spacing_km,
            'max_edge_km': max_edge_km,
        },
        'outlier': {
            'k_neighbors': k_neighbors,
            'mad_factor': mad_factor,
            'iqr_factor': iqr_factor,
            'max_outlier_iter': max_outlier_iter,
        },
        'smoothing': {
            'smooth_weight': smooth_weight,
            'smooth_iter': smooth_iter,
        },
        'uncertainty': {'mc_iterations': int(mc_iterations)},
    }
    import yaml as _yaml
    cfg_yaml_str = _yaml.dump(current_cfg_dict, allow_unicode=True,
                              default_flow_style=False)
    st.download_button(
        "导出当前配置 (YAML)", data=cfg_yaml_str,
        file_name="gnss_strain_config.yaml", mime="text/yaml"
    )

    st.divider()
    run_btn = st.button("运行计算", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# 主区域：运行 & 结果展示
# ---------------------------------------------------------------------------

if run_btn:
    if vel_file_upload is None:
        st.error("请先上传速度场文件！")
        st.stop()

    # 如果有导入的 YAML，先加载作为基础配置
    base_cfg_overrides = {}
    if 'imported_yaml' in st.session_state:
        import yaml as _yaml_imp
        base_cfg_overrides = _yaml_imp.safe_load(
            st.session_state['imported_yaml']) or {}

    # 将上传的文件写入临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        vel_path = os.path.join(tmpdir, vel_file_upload.name)
        with open(vel_path, 'wb') as f:
            f.write(vel_file_upload.read())

        poly_path = None
        if poly_file_upload is not None:
            poly_path = os.path.join(tmpdir, poly_file_upload.name)
            with open(poly_path, 'wb') as f:
                f.write(poly_file_upload.read())

        # 构建参数覆盖（GUI 优先级最高）
        gui_overrides = {
            'vel_file': vel_path,
            'poly_file': poly_path,
            'output_dir': os.path.join(tmpdir, output_dir),
            'vel_format': vel_format,
            'smooth_weight': smooth_weight,
            'smooth_iter': int(smooth_iter),
            'min_angle_deg': min_angle_deg,
            'max_edge_pctl': max_edge_pctl,
            'max_edge_factor': max_edge_factor,
            'min_spacing_km': min_spacing_km,
            'max_edge_km': max_edge_km,
            'mc_iterations': int(mc_iterations),
            'k_neighbors': int(k_neighbors),
            'mad_factor': mad_factor,
            'iqr_factor': iqr_factor,
            'max_outlier_iter': int(max_outlier_iter),
        }

        # 进度条
        progress_bar = st.progress(0, text="准备中...")
        status_text = st.empty()
        stage_total = 8

        def stage_cb(stage, msg):
            pct = int(stage / stage_total * 100)
            progress_bar.progress(pct, text=f"[{stage}/{stage_total}] {msg}")
            status_text.text(msg)

        log_expander = st.expander("运行日志", expanded=False)
        log_buf = io.StringIO()

        try:
            import sys as _sys
            old_stdout = _sys.stdout
            _sys.stdout = log_buf

            cfg = load_config(None, gui_overrides)
            validate_config(cfg)
            kwargs = cfg_to_kwargs(cfg)
            kwargs['stage_callback'] = stage_cb

            result, unc, outlier_history = run_full_pipeline(**kwargs)

            _sys.stdout = old_stdout
        except Exception as e:
            _sys.stdout = old_stdout
            progress_bar.empty()
            status_text.empty()
            with log_expander:
                st.code(log_buf.getvalue())
            st.error(f"计算失败：{e}")
            st.stop()

        progress_bar.progress(100, text="计算完成！")
        status_text.empty()

        with log_expander:
            st.code(log_buf.getvalue())

        # 缓存结果到 session_state
        st.session_state['result'] = result
        st.session_state['unc'] = unc
        st.session_state['outlier_history'] = outlier_history

# ---------------------------------------------------------------------------
# 结果展示（Tabs）
# ---------------------------------------------------------------------------

if 'result' in st.session_state:
    result = st.session_state['result']
    unc = st.session_state['unc']
    outlier_history = st.session_state['outlier_history']

    lon = result['_lon']
    lat = result['_lat']
    ve = result['_ve']
    vn = result['_vn']
    tri = result['_tri']
    xy = result['_xy']
    good_mask = result['_good_mask']
    polygon = result['_polygon']
    outlier_lon = result.get('_outlier_lon', np.array([]))
    outlier_lat = result.get('_outlier_lat', np.array([]))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "原始速度场",
        "异常点分析",
        "三角网",
        "面膨胀率",
        "最大剪应变",
        "主应变十字",
    ])

    def _fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf.read()

    # ---- Tab 1: 原始速度场 ----
    with tab1:
        st.subheader("原始速度场（含异常点）")
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        ax1.quiver(lon, lat, ve, vn, color='steelblue',
                   scale=None, scale_units='inches',
                   width=0.003, alpha=0.8, label='Used sites')
        if len(outlier_lon) > 0:
            ax1.scatter(outlier_lon, outlier_lat,
                        color='red', s=30, zorder=5, label='Outliers')
        ax1.set_xlabel("Longitude (°)")
        ax1.set_ylabel("Latitude (°)")
        ax1.set_title("GNSS Velocity Field")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        st.download_button(
            "下载图片 (PNG)", data=_fig_to_bytes(fig1),
            file_name="velocity_field.png", mime="image/png"
        )
        plt.close(fig1)

    # ---- Tab 2: 异常点分析 ----
    with tab2:
        st.subheader(f"异常点列表（共 {len(outlier_history)} 个）")
        if outlier_history:
            import pandas as pd
            df_out = pd.DataFrame(outlier_history)
            col_order = [c for c in
                         ['name', 'lon', 'lat', 'residual', 'reason', 'iteration']
                         if c in df_out.columns]
            df_out = df_out[col_order]
            st.dataframe(df_out, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(10, 7))
            # 用于站点散点底图
            ax2.scatter(lon, lat, s=8, c='lightgray', zorder=1, label='Used')
            if len(outlier_lon) > 0:
                ax2.scatter(outlier_lon, outlier_lat,
                            s=40, c='red', zorder=5, label='Outlier')
            ax2.set_xlabel("Longitude (°)")
            ax2.set_ylabel("Latitude (°)")
            ax2.set_title("Outlier Distribution")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            st.download_button(
                "下载图片 (PNG)", data=_fig_to_bytes(fig2),
                file_name="outliers.png", mime="image/png"
            )
            plt.close(fig2)
        else:
            st.info("未发现异常点。")

    # ---- Tab 3: 三角网 ----
    with tab3:
        st.subheader("Delaunay 三角网 + 速度矢量")
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        good_idx = np.where(good_mask)[0]
        bad_idx = np.where(~good_mask)[0]

        for i in good_idx:
            xs = lon[tri.simplices[i]]
            ys = lat[tri.simplices[i]]
            ax3.fill(xs, ys, facecolor='none', edgecolor='gray',
                     linewidth=0.5, alpha=0.6)
        for i in bad_idx:
            xs = lon[tri.simplices[i]]
            ys = lat[tri.simplices[i]]
            ax3.fill(xs, ys, facecolor='none', edgecolor='red',
                     linewidth=0.4, alpha=0.3, linestyle='--')

        ax3.quiver(lon, lat, ve, vn, color='steelblue',
                   scale=None, scale_units='inches',
                   width=0.003, alpha=0.8)
        if polygon is not None:
            ax3.plot(polygon[:, 0], polygon[:, 1],
                     'k-', linewidth=1.5, label='Boundary')
        ax3.set_xlabel("Longitude (°)")
        ax3.set_ylabel("Latitude (°)")
        ax3.set_title(f"Triangulation  (good={good_mask.sum()}, "
                      f"bad={(~good_mask).sum()})")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        st.download_button(
            "下载图片 (PNG)", data=_fig_to_bytes(fig3),
            file_name="triangulation.png", mime="image/png"
        )
        plt.close(fig3)

    # ---- Tab 4: 面膨胀率 ----
    with tab4:
        st.subheader("面膨胀率 (Dilatation Rate)")
        clat = result['centroids_lat']
        clon = result['centroids_lon']
        dilatation = result['dilatation']

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("最大值 (nstrain/yr)", f"{dilatation.max():.1f}")
            st.metric("最小值 (nstrain/yr)", f"{dilatation.min():.1f}")
        with col_b:
            st.metric("均值 (nstrain/yr)", f"{dilatation.mean():.1f}")
            if 'dilatation_std' in result:
                st.metric("平均不确定度", f"{result['dilatation_std'].mean():.1f}")

        fig4, ax4 = plt.subplots(figsize=(10, 7))
        vmax = np.percentile(np.abs(dilatation), 95)
        sc4 = ax4.scatter(clon, clat, c=dilatation, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, s=60, zorder=3)
        plt.colorbar(sc4, ax=ax4, label='Dilatation (nstrain/yr)')
        ax4.set_xlabel("Longitude (°)")
        ax4.set_ylabel("Latitude (°)")
        ax4.set_title("Dilatation Rate")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        st.download_button(
            "下载图片 (PNG)", data=_fig_to_bytes(fig4),
            file_name="dilatation.png", mime="image/png"
        )
        plt.close(fig4)

    # ---- Tab 5: 最大剪应变 ----
    with tab5:
        st.subheader("最大剪应变率 (Max Shear Strain Rate)")
        max_shear = result['max_shear']

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("最大值 (nstrain/yr)", f"{max_shear.max():.1f}")
            st.metric("中位数 (nstrain/yr)", f"{np.median(max_shear):.1f}")
        with col_b:
            st.metric("均值 (nstrain/yr)", f"{max_shear.mean():.1f}")
            if 'max_shear_std' in result:
                st.metric("平均不确定度", f"{result['max_shear_std'].mean():.1f}")

        fig5, ax5 = plt.subplots(figsize=(10, 7))
        sc5 = ax5.scatter(clon, clat, c=max_shear, cmap='YlOrRd',
                          vmin=0, s=60, zorder=3)
        plt.colorbar(sc5, ax=ax5, label='Max Shear (nstrain/yr)')
        ax5.set_xlabel("Longitude (°)")
        ax5.set_ylabel("Latitude (°)")
        ax5.set_title("Maximum Shear Strain Rate")
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5)
        st.download_button(
            "下载图片 (PNG)", data=_fig_to_bytes(fig5),
            file_name="max_shear.png", mime="image/png"
        )
        plt.close(fig5)

    # ---- Tab 6: 主应变十字 ----
    with tab6:
        st.subheader("主应变率 (Principal Strain Rate)")
        e1 = result['e1']
        e2 = result['e2']
        azimuth = result['azimuth']

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("e1 均值 (nstrain/yr)", f"{e1.mean():.1f}")
        with col_b:
            st.metric("e2 均值 (nstrain/yr)", f"{e2.mean():.1f}")

        fig6, ax6 = plt.subplots(figsize=(10, 7))
        # 将主应变绘制成十字（按方位角旋转）
        scale = 0.3 / max(np.abs(np.concatenate([e1, e2])).max(), 1.0)
        for i in range(len(clon)):
            az_rad = np.radians(azimuth[i])
            cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)
            # e1 方向（主压缩/拉张轴）
            dx1 = e1[i] * scale * cos_a
            dy1 = e1[i] * scale * sin_a
            # e2 方向（垂直轴）
            dx2 = e2[i] * scale * (-sin_a)
            dy2 = e2[i] * scale * cos_a
            c1 = 'blue' if e1[i] < 0 else 'red'
            c2 = 'blue' if e2[i] < 0 else 'red'
            ax6.annotate("", xy=(clon[i] + dx1, clat[i] + dy1),
                         xytext=(clon[i] - dx1, clat[i] - dy1),
                         arrowprops=dict(arrowstyle='->', color=c1,
                                        lw=0.8))
            ax6.annotate("", xy=(clon[i] + dx2, clat[i] + dy2),
                         xytext=(clon[i] - dx2, clat[i] - dy2),
                         arrowprops=dict(arrowstyle='->', color=c2,
                                         lw=0.8))

        ax6.set_xlabel("Longitude (°)")
        ax6.set_ylabel("Latitude (°)")
        ax6.set_title("Principal Strain Rate  (blue=compression, red=extension)")
        ax6.grid(True, alpha=0.3)
        st.pyplot(fig6)
        st.download_button(
            "下载图片 (PNG)", data=_fig_to_bytes(fig6),
            file_name="principal_strain.png", mime="image/png"
        )
        plt.close(fig6)

    # ---- 数值统计摘要 ----
    st.divider()
    st.subheader("计算摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("输入站点数", len(lon) + len(outlier_lon))
    c2.metric("使用站点数", len(lon))
    c3.metric("剔除异常点", len(outlier_history))
    c4.metric("有效三角形", int(good_mask.sum()))

else:
    st.info("请在左侧上传速度场文件并点击【运行计算】开始。")
