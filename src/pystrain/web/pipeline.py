"""Streamlit-independent pipeline runner for the PyStrain2 web app."""

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pystrain.data import StrainResult, TimeSeriesCollection, VelocityField
from pystrain.grid import Grid, ShenWangStrainRate
from pystrain.io import read_polygon, read_velocity_file
from pystrain.outlier import iterative_outlier_removal
from pystrain.timeseries.loader import TimeSeriesLoader
from pystrain.timeseries.strain_ts import (
    GridStrainTimeSeries,
    TriStrainTimeSeries,
    UserStrainTimeSeries,
)
from pystrain.tri import DelaunayStrainRate
from pystrain.triangulation import delaunay_triangulation
from pystrain.uncertainty import monte_carlo_strain_uncertainty
from pystrain.wang2012 import Wang2012StrainRate


def _triangulation_fn_for_outlier(
    lon: np.ndarray,
    lat: np.ndarray,
    site_indices: np.ndarray,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Adapter for the iterative outlier detector."""
    tri, good_triangles, xy, _ = delaunay_triangulation(
        lon[site_indices], lat[site_indices]
    )
    return tri, good_triangles, xy


def _auto_region(
    vf: VelocityField, buffer: float = 0.25
) -> Tuple[float, float, float, float]:
    """Compute a default grid region from site bounds."""
    lon_min, lon_max = float(vf.lon.min()), float(vf.lon.max())
    lat_min, lat_max = float(vf.lat.min()), float(vf.lat.max())
    dlon = lon_max - lon_min
    dlat = lat_max - lat_min
    return (
        lon_min - buffer * dlon,
        lon_max + buffer * dlon,
        lat_min - buffer * dlat,
        lat_max + buffer * dlat,
    )


def run_pystrain_pipeline(
    vel_path: str,
    poly_path: Optional[str],
    fmt: str,
    algorithm: str,
    outlier_enable: bool,
    outlier_kwargs: Dict[str, Any],
    tri_kwargs: Dict[str, Any],
    wang2012_kwargs: Optional[Dict[str, Any]] = None,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    mc_iterations: int = 200,
    mc_seed: int = 42,
    stage_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run a PyStrain2 pipeline and return a display-compatible result dict."""

    def _stage(stage: int, total: int, msg: str):
        if stage_callback is not None:
            stage_callback(stage, total, msg)

    total_stages = 6
    _stage(1, total_stages, "读取速度场文件...")
    vf = read_velocity_file(vel_path, fmt=fmt)

    polygon = None
    if poly_path is not None:
        _stage(1, total_stages, "读取边界多边形...")
        rings = read_polygon(poly_path)
        polygon = rings[0]

    outlier_history: List[Dict] = []
    if outlier_enable:
        _stage(2, total_stages, "异常点检测...")
        vf, outlier_history = iterative_outlier_removal(
            vf,
            triangulation_fn=_triangulation_fn_for_outlier,
            **outlier_kwargs,
        )

    _stage(3, total_stages, "构建三角网...")
    tri, good_triangles, xy, proj_params = delaunay_triangulation(
        vf.lon,
        vf.lat,
        polygon=polygon,
        min_angle_deg=tri_kwargs["min_angle_deg"],
        max_edge_pctl=tri_kwargs["max_edge_pctl"],
        max_edge_factor=tri_kwargs["max_edge_factor"],
        min_area_ratio=tri_kwargs.get("min_area_ratio", 0.1),
        max_edge_km=tri_kwargs.get("max_edge_km"),
    )

    _stage(4, total_stages, f"计算应变率 ({algorithm})...")
    if algorithm == "delaunay":
        tri_kwargs_local = dict(tri_kwargs)
        estimator = DelaunayStrainRate(vf, polygon=polygon, **tri_kwargs_local)

        def estimator_fn(vf_in: VelocityField) -> StrainResult:
            return DelaunayStrainRate(
                vf_in,
                polygon=polygon,
                **tri_kwargs_local,
            ).compute()

    elif algorithm == "wang2012":
        wang2012_kwargs_local = dict(wang2012_kwargs or {})
        smooth_range = wang2012_kwargs_local.pop("smooth_range", (-2.2, -0.8))
        # Remove keys not accepted by Wang2012StrainRate.__init__
        wang2012_kwargs_local.pop("smooth_method", None)
        wang2012_kwargs_local.pop("poly_file", None)
        estimator = Wang2012StrainRate(
            vf, polygon=polygon,
            smooth_range=tuple(smooth_range),
            **wang2012_kwargs_local
        )

        def estimator_fn(vf_in: VelocityField) -> StrainResult:
            return Wang2012StrainRate(
                vf_in,
                polygon=polygon,
                smooth_range=tuple(smooth_range),
                **wang2012_kwargs_local,
            ).compute()

    elif algorithm == "shen2015":
        if grid_kwargs is None:
            raise ValueError("Grid parameters are required for Shen-Wang algorithm.")
        region = grid_kwargs["region"]
        spacing = grid_kwargs["spacing"]
        grid = Grid(region[0], region[1], region[2], region[3], spacing[0], spacing[1])
        shen_kwargs = {
            "distance_kind": grid_kwargs.get("distance_weight", "gaussian"),
            "spatial_kind": grid_kwargs.get("spatial_weight", "voronoi"),
            "Wt": grid_kwargs.get("Wt", 24.0),
            "L0": grid_kwargs.get("L0", 0.01),
            "min_sites": grid_kwargs.get("min_sites", 6),
            "maxdist_km": grid_kwargs.get("maxdist_km"),
        }
        estimator = ShenWangStrainRate(vf, grid, **shen_kwargs)

        def estimator_fn(vf_in: VelocityField) -> StrainResult:
            return ShenWangStrainRate(vf_in, grid, **shen_kwargs).compute()

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result = estimator.compute()

    unc: Optional[Dict[str, np.ndarray]] = None
    if mc_iterations > 0:
        _stage(5, total_stages, "Monte Carlo 不确定度估计...")
        unc = monte_carlo_strain_uncertainty(
            estimator_fn, vf, n_iterations=mc_iterations, seed=mc_seed
        )

    _stage(6, total_stages, "整理结果...")

    outlier_lon = np.array([o["lon"] for o in outlier_history], dtype=float)
    outlier_lat = np.array([o["lat"] for o in outlier_history], dtype=float)

    display_result = {
        "algorithm": algorithm,
        "_lon": vf.lon,
        "_lat": vf.lat,
        "_ve": vf.ve,
        "_vn": vf.vn,
        "_tri": tri,
        "_xy": xy,
        "_good_mask": good_triangles,
        "_polygon": polygon,
        "_outlier_lon": outlier_lon,
        "_outlier_lat": outlier_lat,
        "centroids_lon": result.lon,
        "centroids_lat": result.lat,
        "dilatation": result.dilation,
        "max_shear": result.shear,
        "e1": result.e1,
        "e2": result.e2,
        "azimuth": result.azimuth,
        "exx": result.exx,
        "exy": result.exy,
        "eyy": result.eyy,
        "omega": result.omega,
        "sec_inv": result.sec_inv,
        "ve": result.ve,
        "vn": result.vn,
        "outlier_history": outlier_history,
        "n_sites_total": len(vf) + len(outlier_history),
        "n_sites_used": len(vf),
        "n_outliers": len(outlier_history),
        "n_good_triangles": int(good_triangles.sum()),
        "n_bad_triangles": int((~good_triangles).sum()),
    }

    if unc is not None:
        display_result["dilatation_std"] = unc.get("dilation")
        display_result["max_shear_std"] = unc.get("shear")
        display_result["e1_std"] = unc.get("e1")
        display_result["e2_std"] = unc.get("e2")
        display_result["exx_std"] = unc.get("exx")
        display_result["exy_std"] = unc.get("exy")
        display_result["eyy_std"] = unc.get("eyy")

    return display_result


# ---------------------------------------------------------------------------
# Time-series pipeline
# ---------------------------------------------------------------------------


def run_timeseries_pipeline_web(
    gps_info_path: str,
    ts_type: str,
    ts_path: str,
    sepoch: float,
    eepoch: float,
    method: str = "grid",
    polygon: Optional[np.ndarray] = None,
    grid_kwargs: Optional[Dict[str, Any]] = None,
    tri_kwargs: Optional[Dict[str, Any]] = None,
    user_kwargs: Optional[Dict[str, Any]] = None,
    stage_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run strain time-series pipeline for the web app.

    Parameters
    ----------
    gps_info_path : path to the GPS station metadata file
        (lon lat height site_name).
    ts_type : "pos" (PBO .pos format) or "dat" (PyTsfit .dat format).
    ts_path : directory containing per-site time-series files.
    sepoch, eepoch : start / end epoch in decimal years.
    method : "grid", "tri", or "user".
    polygon : optional (N,2) boundary polygon.
    grid_kwargs : parameters for GridStrainTimeSeries.
    tri_kwargs : parameters for TriStrainTimeSeries.
    user_kwargs : parameters for UserStrainTimeSeries.
    stage_callback : optional progress callback (stage, total, msg).

    Returns
    -------
    dict with keys:
        - tsc: TimeSeriesCollection
        - results: list of StrainTimeSeriesResult
        - method: str
        - n_locations: int
        - n_epochs: int
        - sites: list of str
        - loc_lon, loc_lat: arrays of strain computation coordinates
    """
    def _stage(stage: int, total: int, msg: str):
        if stage_callback is not None:
            stage_callback(stage, total, msg)

    total_stages = 4
    _stage(1, total_stages, "读取测站信息...")
    sites_lon = []
    sites_lat = []
    sites_names = []
    with open(gps_info_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) >= 4:
                sites_lon.append(float(parts[0]))
                sites_lat.append(float(parts[1]))
                sites_names.append(parts[3])
            elif len(parts) >= 2:
                sites_lon.append(float(parts[0]))
                sites_lat.append(float(parts[1]))
                sites_names.append(f"sta_{len(sites_names)}")

    sites_lon_arr = np.array(sites_lon)
    sites_lat_arr = np.array(sites_lat)

    # --- Build site_list to avoid loading unnecessary POS files ---
    # This mirrors the filtering done by the CLI `run_timeseries_pipeline`.
    site_list = None
    if method == "user":
        uk = user_kwargs or {}
        sgf = uk.get("site_groups_file")
        if sgf and os.path.exists(sgf):
            all_sites = []
            with open(sgf, "r") as f:
                for line in f:
                    all_sites.extend(line.strip().split())
            site_list = list(set(all_sites))
    elif method == "tri" and polygon is not None:
        from matplotlib.path import Path
        mpath = Path(polygon)
        inside_mask = mpath.contains_points(np.column_stack([sites_lon_arr, sites_lat_arr]))
        site_list = [sites_names[i] for i in np.where(inside_mask)[0]]
    # grid method: site_list stays None (needs all stations for spatial coverage)

    _stage(2, total_stages, "加载时间序列数据...")
    loader = TimeSeriesLoader(
        gps_info_file=gps_info_path,
        ts_type=ts_type,
        ts_path=ts_path,
        sepoch=sepoch,
        eepoch=eepoch,
        site_list=site_list,
    )
    tsc = loader.load()

    n_epochs = len(tsc.decyr)
    n_sites = len(tsc.sites)
    _stage(2, total_stages, f"加载完成: {n_epochs} 历元, {n_sites} 站点" +
           (f" (已过滤，总站点 {len(sites_names)})" if site_list else f" (全量)"))

    _stage(3, total_stages, f"计算应变时间序列 (method={method})...")

    results = []
    loc_lon = []
    loc_lat = []

    if method == "grid":
        gk = grid_kwargs or {}
        grid = Grid(
            gk.get("slon", float(np.nanmin(sites_lon_arr))),
            gk.get("elon", float(np.nanmax(sites_lon_arr))),
            gk.get("slat", float(np.nanmin(sites_lat_arr))),
            gk.get("elat", float(np.nanmax(sites_lat_arr))),
            gk.get("dn", 0.5),
            gk.get("de", 0.5),
            stagger=gk.get("stagger", True),
        )
        estimator = GridStrainTimeSeries(
            tsc,
            grid,
            maxdist_km=gk.get("maxdist_km", 300.0),
            min_sites=gk.get("min_sites", 6),
            check_azimuth=gk.get("check_azimuth", True),
        )
        results = estimator.compute()
        loc_lon = grid.lon
        loc_lat = grid.lat

    elif method == "tri":
        tk = tri_kwargs or {}
        tri, good_triangles, xy, _ = delaunay_triangulation(
            sites_lon_arr, sites_lat_arr,
            polygon=polygon,
            min_angle_deg=tk.get("min_angle_deg", 10.0),
            max_edge_pctl=tk.get("max_edge_pctl", 95.0),
            max_edge_factor=tk.get("max_edge_factor", 1.5),
            min_area_ratio=tk.get("min_area_ratio", 0.1),
        )
        estimator = TriStrainTimeSeries(tsc, tri, good_triangles, xy)
        results = estimator.compute()
        for simplex in tri.simplices:
            loc_lon.append(float(np.mean(sites_lon_arr[simplex])))
            loc_lat.append(float(np.mean(sites_lat_arr[simplex])))
        loc_lon = np.array(loc_lon)
        loc_lat = np.array(loc_lat)

    elif method == "user":
        uk = user_kwargs or {}
        site_groups_file = uk.get("site_groups_file")
        max_sigma_mm = uk.get("max_sigma_mm", 5.0)
        if site_groups_file and os.path.exists(site_groups_file):
            with open(site_groups_file, "r") as f:
                groups = [ln.strip().split() for ln in f if ln.strip()]
            estimator = UserStrainTimeSeries(tsc, groups, max_sigma_mm=max_sigma_mm)
            results = estimator.compute()
            for group in groups:
                indices = [tsc.sites.index(s) for s in group if s in tsc.sites]
                if indices:
                    loc_lon.append(float(np.mean(sites_lon_arr[indices])))
                    loc_lat.append(float(np.mean(sites_lat_arr[indices])))
            loc_lon = np.array(loc_lon)
            loc_lat = np.array(loc_lat)
        else:
            raise ValueError("user method requires a valid site_groups_file")
    else:
        raise ValueError(f"Unknown time-series method: {method}")

    _stage(4, total_stages, "整理结果...")

    # Extract common arrays for display
    n_locations = len(results)
    display = {
        "tsc": tsc,
        "results": results,
        "method": method,
        "n_locations": n_locations,
        "n_epochs": n_epochs,
        "n_sites": n_sites,
        "sites": tsc.sites,
        "sites_lon": sites_lon_arr,
        "sites_lat": sites_lat_arr,
        "loc_lon": np.array(loc_lon),
        "loc_lat": np.array(loc_lat),
        "decyr": tsc.decyr,
        "polygon": polygon,
    }
    return display
