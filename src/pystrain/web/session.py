"""Session state management for the PyStrain2 web application.

Defines all session-state keys with defaults and provides helper
functions for workflow-step gating.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    # ---- Mode ----
    "mode": None,  # "strain_rate" | "timeseries" | None (unselected)

    # ---- Strain-rate workflow ----
    "sr_step": "load",  # load | outlier | algorithm | params | compute | results
    "sr_vel_file": None,  # UploadedFile or None
    "sr_vel_data": None,  # VelocityField or None
    "sr_poly_file": None,  # UploadedFile or None
    "sr_poly_data": None,  # np.ndarray (N,2) or None
    "sr_outlier_enabled": False,
    "sr_outlier_method": "knn_iqr",  # "knn_iqr" | "loo_strain"
    "sr_outlier_kwargs": {},  # {k_neighbors, mad_factor, iqr_factor, max_iterations}
    "sr_outlier_result": None,  # (VelocityField, outlier_list) or None
    "sr_algorithm": "shen2015",  # "shen2015" | "wang2012" | "delaunay"
    "sr_algo_params": {},  # algorithm-specific parameter dict
    "sr_result": None,  # dict from run_pystrain_pipeline()
    "sr_show_preview": False,  # Whether to show algorithm preview
    "sr_faults": None,  # list of (N,2) np.ndarray — fault traces for overlay

   # ---- Time-series workflow ----
    "ts_step": "stations",  # stations | config | results
    "ts_gps_info_data": None,  # (lon, lat, height, names) tuple or None
    "ts_poly_mode": "file",  # "file" | "manual"
    "ts_poly_data": None,  # np.ndarray (N,2) or None — primary boundary polygon
    "ts_all_polygons": None,  # list of np.ndarray (N,2) — all loaded polygons
    "ts_poly_selected_names": [],  # list of station names selected in manual mode
    "ts_ts_type": "pos",  # "pos" | "dat"
    "ts_data_dir": None,  # str or None
    "ts_epoch_start": None,  # float or None
    "ts_epoch_end": None,  # float or None
   "ts_method": "user",  # "user" | "grid" | "tri"
    "ts_grid_kwargs": {},  # grid method parameters
    "ts_tri_kwargs": {},  # tri method parameters
    "ts_user_kwargs": {},  # user method parameters
   "ts_user_groups": None,  # list of site name lists from uploaded groups file
    "ts_user_poly_data": None,  # np.ndarray (N,2) or None — polygon points for user method
    "ts_user_poly_source": "file",  # "file" | "map"
    "ts_results": None,  # dict from run_timeseries_pipeline_web()
    "ts_selected_index": 0,  # selected strain location index
    "ts_dialog_open_idx": -1,  # index of currently open time-series dialog, -1 = closed
    "ts_faults": None,  # list of (N,2) np.ndarray — fault traces for results map overlay

    # ---- Common ----
    "mc_enabled": True,
    "mc_iterations": 200,
    "mc_seed": 42,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init() -> None:
    """Ensure every key defined in *DEFAULTS* exists in ``st.session_state``.

    Called once at the top of ``app.py`` before any rendering.
    Existing keys are left untouched so that already-populated state
    survives Streamlit re-runs.
    """
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_mode(mode: str) -> None:
    """Switch to a new mode, clearing all previous state for that mode."""
    st.session_state.mode = mode
    if mode == "strain_rate":
        _reset_strain_rate()
    elif mode == "timeseries":
        _reset_timeseries()


def step_forward(step_key: str, next_step: str) -> None:
    """Advance the workflow step."""
    st.session_state[step_key] = next_step


def step_back(step_key: str, prev_step: str) -> None:
    """Go back to a previous workflow step."""
    st.session_state[step_key] = prev_step


def steps_list(mode: str) -> list:
    """Return the ordered list of step IDs for *mode*."""
    if mode == "strain_rate":
        return ["load", "outlier", "algorithm", "params", "compute", "results"]
    if mode == "timeseries":
        return ["stations", "config", "results"]
    return []


def step_labels(mode: str) -> list:
    """Return human-readable step labels for *mode*."""
    if mode == "strain_rate":
        return [
            "数据加载",
            "异常值检测",
            "算法选择",
            "参数配置",
            "运行计算",
            "计算结果",
        ]
    if mode == "timeseries":
        return [
            "测站导入",
            "时序配置",
            "结果查看",
        ]
    return []


def can_advance(step_key: str, current_step: str) -> bool:
    """Return True if the prerequisites for *current_step* are met."""
    if step_key == "sr_step":
        return _sr_can_advance(current_step)
    if step_key == "ts_step":
        return _ts_can_advance(current_step)
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reset_strain_rate() -> None:
    """Clear all strain-rate-related session state by reassigning defaults."""
    for key, default in DEFAULTS.items():
        if key.startswith("sr_"):
            st.session_state[key] = default
    # Also clear any extra sr_ keys not in DEFAULTS.
    # Widget-managed keys (e.g. sr_vel, sr_poly) are skipped — Streamlit
    # forbids direct assignment to those session_state entries.
    for key in list(st.session_state.keys()):
        if key.startswith("sr_") and key not in DEFAULTS:
            try:
                st.session_state[key] = None
            except Exception:
                pass
    st.session_state.sr_step = "load"


def _reset_timeseries() -> None:
    """Clear all time-series-related session state by reassigning defaults."""
    for key, default in DEFAULTS.items():
        if key.startswith("ts_"):
            st.session_state[key] = default
    # Widget-managed keys (e.g. ts_gps_info, ts_poly_file) are skipped —
    # Streamlit forbids direct assignment to those session_state entries.
    for key in list(st.session_state.keys()):
        if key.startswith("ts_") and key not in DEFAULTS:
            try:
                st.session_state[key] = None
            except Exception:
                pass
    st.session_state.ts_step = "stations"


def _sr_can_advance(step: str) -> bool:
    """Check prerequisites for each strain-rate step."""
    if step == "outlier":
        return st.session_state.sr_vel_data is not None
    if step == "algorithm":
        return True  # outlier step is optional
    if step == "params":
        return st.session_state.sr_algorithm is not None
    if step == "compute":
        return bool(st.session_state.sr_algo_params)
    if step == "results":
        return st.session_state.sr_result is not None
    return True


def _ts_can_advance(step: str) -> bool:
    """Check prerequisites for each time-series step."""
    if step == "config":
        return st.session_state.ts_gps_info_data is not None
    if step == "results":
        return (
            st.session_state.ts_data_dir is not None
            and st.session_state.ts_epoch_start is not None
            and st.session_state.ts_epoch_end is not None
            and st.session_state.ts_results is not None
        )
    return True
