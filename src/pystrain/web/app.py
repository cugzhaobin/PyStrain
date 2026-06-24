"""PyStrain Streamlit web application.

Launch with:
    streamlit run src/pystrain/web/app.py
or:
    pystrain web

Two modes are supported:
  - Strain Rate Field (应变率场): velocity field → algorithm → parameters → results
  - Strain Time Series (应变时间序列): stations → polygon → config → results with interactive selection
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Streamlit is required for the PyStrain web app. "
        "Install it with: pip install pystrain[gui]"
    ) from exc

from pystrain.web import session

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PyStrain — GNSS 应变计算平台",
    page_icon="🌍",
    layout="wide",
)

# Initialise session state (must be first)
session.init()


# ---------------------------------------------------------------------------
# Mode selection (shown when no mode is active)
# ---------------------------------------------------------------------------

if st.session_state.mode is None:
    st.title("🌍 PyStrain — GNSS 应变计算平台")
    st.caption("选择一个模式开始")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📈 应变率场
        **Strain Rate Field**

        基于 GNSS 速度场，计算长期平均应变率空间分布。
        - 支持 Shen et al. (2015) / Wang (2012) / Delaunay 三种算法
        - 异常值检测与剔除
        - Monte Carlo 不确定度估计
        - 交互式地图展示
        """)
        if st.button("进入应变率场 →", key="btn_sr", use_container_width=True):
            session.reset_mode("strain_rate")
            st.rerun()

    with col2:
        st.markdown("""
        ### 📊 应变时间序列
        **Strain Time Series**

        基于 GPS 位置时间序列，计算每历元应变，得到应变随时间变化。
        - 导入测站坐标与时间序列数据
        - 多边形定义分析区域
        - 逐历元应变计算
        - 点击位置查看时间序列图
        """)
        if st.button("进入应变时间序列 →", key="btn_ts", use_container_width=True):
            session.reset_mode("timeseries")
            st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------

mode = st.session_state.mode

if mode == "strain_rate":
    from pystrain.web.strain_rate import render_workflow
    render_workflow()
elif mode == "timeseries":
    from pystrain.web.strain_timeseries import render_workflow
    render_workflow()

# ---- Sidebar: mode switcher & navigation ----
with st.sidebar:
    st.divider()
    if st.button("🏠 返回主页"):
        st.session_state.mode = None
        st.rerun()
