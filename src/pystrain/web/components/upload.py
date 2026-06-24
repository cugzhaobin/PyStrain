"""Standardized file-uploader widgets for the PyStrain2 web app."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import streamlit as st

from pystrain.io.velocity import read_velocity_file
from pystrain.io.polygon import read_polygon


def velocity_file_uploader(
    key: str = "vel_upload",
) -> Tuple[Optional[object], Optional[str], str]:
    """Render a velocity-file uploader with format selector.

    Returns
    -------
    (VelocityField or None, file_name or None, format_str)
    """
    st.subheader("📁 速度场文件")
    vel_file = st.file_uploader(
        "上传速度场文件",
        type=["vel", "txt", "dat", "gmtvec"],
        key=key,
        help="支持 GMT7/GMT8/GLOBK 格式",
    )
    fmt = st.selectbox(
        "输入格式",
        ["auto", "gmt8", "gmt7", "globk"],
        key=f"{key}_fmt",
        help="auto=自动识别",
    )

    if vel_file is not None:
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="wb", suffix=vel_file.name, delete=False) as tf:
                tf.write(vel_file.getvalue())
                tmp_path = tf.name
            vf = read_velocity_file(tmp_path, fmt=fmt)
            os.unlink(tmp_path)
            st.success(f"已加载 {len(vf)} 个站点")
            return vf, vel_file.name, fmt
        except Exception as e:
            st.error(f"文件解析失败: {e}")
            return None, None, fmt

    return None, None, fmt


def polygon_file_uploader(
    key: str = "poly_upload",
    label: str = "上传边界多边形文件（可选）",
) -> Optional[np.ndarray]:
    """Render a polygon-file uploader.

    Returns (N, 2) array or None.
    """
    st.subheader("📐 边界多边形")
    poly_file = st.file_uploader(
        label,
        type=["txt", "dat", "poly"],
        key=key,
        help="GMT 多边形格式：每行一个 lon lat 顶点",
    )
    if poly_file is not None:
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="wb", suffix=poly_file.name, delete=False) as tf:
                tf.write(poly_file.getvalue())
                tmp_path = tf.name
            rings = read_polygon(tmp_path)
            os.unlink(tmp_path)
            if rings:
                st.success(f"已加载多边形 ({len(rings[0])} 个顶点)")
                return rings[0]
            else:
                st.warning("多边形文件为空")
                return None
        except Exception as e:
            st.error(f"多边形解析失败: {e}")
            return None
    return None


def gps_info_file_uploader(
    key: str = "gps_info_upload",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list], Optional[np.ndarray]]:
    """Render a GPS station info file uploader.

    Expected format: lon  lat  height  site_name (space-separated).

    Returns
    -------
    (lon, lat, names, heights) — each may be None.
    """
    st.subheader("📍 测站坐标文件")
    gps_file = st.file_uploader(
        "上传测站信息文件 (lon lat height site_name)",
        type=["txt", "dat", "llh"],
        key=key,
    )
    if gps_file is not None:
        try:
            content = gps_file.getvalue().decode("utf-8")
            lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.startswith("#")]
            data = []
            names_list = []
            for ln in lines:
                parts = ln.split()
                if len(parts) >= 4:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    names_list.append(parts[3])
                elif len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1]), 0.0])
                    names_list.append(f"sta_{len(names_list)}")
            if not data:
                st.error("未能解析任何站点")
                return None, None, None, None
            arr = np.array(data)
            st.success(f"已加载 {len(arr)} 个测站")
            return arr[:, 0], arr[:, 1], names_list, arr[:, 2]
        except Exception as e:
            st.error(f"测站文件解析失败: {e}")
            return None, None, None, None
    return None, None, None, None
