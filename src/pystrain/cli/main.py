"""Command-line interface for PyStrain2."""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("pystrain2")

from pystrain2.config import Config
from pystrain2.data import VelocityField
from pystrain2.grid import Grid, ShenWangStrainRate
from pystrain2.io import read_polygon, read_velocity_file, write_strain_result
from pystrain2.outlier import iterative_outlier_removal, loo_strain_outlier_detection
from pystrain2.timeseries import (
    GridStrainTimeSeries,
    TimeSeriesLoader,
    TriStrainTimeSeries,
    UserStrainTimeSeries,
)
from pystrain2.tri import DelaunayStrainRate
from pystrain2.triangulation import delaunay_triangulation
from pystrain2.wang2012 import Wang2012StrainRate
from pystrain2.visualization import (
    plot_raw_velocity_field,
    plot_outlier_result,
    plot_clean_velocity_field,
    plot_triangulation,
    plot_grid_points,
    plot_search_radius_map,
    plot_scalar_field,
    plot_wang2012_mesh,
    plot_principal_strain_crosses,
)


def _triangulation_fn_for_outlier(lon, lat, site_indices):
    """Adapter for outlier detector."""
    tri, good_triangles, xy, _ = delaunay_triangulation(
        lon[site_indices], lat[site_indices]
    )
    return tri, good_triangles, xy


def _parse_numeric_list(val):
    """Convert a string like '70 140 25 40' or list to list of floats."""
    if isinstance(val, list):
        return [float(v) for v in val]
    if isinstance(val, str):
        return [float(v) for v in val.split()]
    raise TypeError(f"Cannot parse {type(val).__name__} as numeric list")


def _get_plot_region(cfg: Config, polygon=None):
    """Return [slon, elon, slat, elat] for plot extent.

    Priority:
    1. Algorithm-specific region/mesh_region from config (user-specified)
    2. Polygon bounding box (when a polygon boundaries are available)
    3. None (auto-compute from data extent)
    """
    try:
        method = cfg["algorithms"]["method"]
        if method == "shen2015":
            scfg = cfg["algorithms"]["shen2015"]
            if "region" in scfg and scfg["region"] is not None:
                return _parse_numeric_list(scfg["region"])
        elif method == "wang2012":
            wcfg = cfg["algorithms"]["wang2012"]
            if wcfg.get("mesh_region") is not None:
                return _parse_numeric_list(wcfg["mesh_region"])
    except Exception:
        pass

    # Fall back to polygon bounding box
    if polygon is not None and len(polygon) > 0:
        buf = 0.2
        return [
            float(polygon[:, 0].min()) - buf,
            float(polygon[:, 0].max()) + buf,
            float(polygon[:, 1].min()) - buf,
            float(polygon[:, 1].max()) + buf,
        ]

    return None


def _write_loo_residuals(output_dir, vf, residuals, pred_ve, pred_vn, outlier_mask):
    """Write LOO strain residuals for all sites to a detail file."""
    import numpy as np
    path = output_dir / "loo_strain_residuals.txt"
    with open(path, "w", encoding="utf-8") as fid:
        fid.write(
            "# name               lon(°)    lat(°)   "
            "ve_obs(mm/yr) vn_obs(mm/yr) "
            "ve_pred(mm/yr) vn_pred(mm/yr) "
            "res(mm/yr)  outlier\n"
        )
        for i in range(len(vf)):
            fid.write(
                f"{vf.names[i]:<14s}  "
                f"{vf.lon[i]:9.4f}  {vf.lat[i]:9.4f}  "
                f"{vf.ve[i]:13.4f}  {vf.vn[i]:13.4f}  "
                f"{pred_ve[i]:13.4f}  {pred_vn[i]:13.4f}  "
                f"{residuals[i]:10.4f}  "
                f"{'YES' if outlier_mask[i] else 'no'}\n"
            )
    logger.info("  -> LOO residuals written to '%s'", path)


def _run_shen2015(cfg: Config, vf: VelocityField, polygon):
    scfg = cfg["algorithms"]["shen2015"]
    region = _parse_numeric_list(scfg["region"])
    spacing = _parse_numeric_list(scfg["spacing"])
    stagger = scfg.get("stagger", True)
    grid = Grid(region[0], region[1], region[2], region[3], spacing[0], spacing[1], stagger=stagger)
    estimator = ShenWangStrainRate(
        vf,
        grid,
        distance_kind=scfg["distance_weight"],
        spatial_kind=scfg["spatial_weight"],
        Wt=scfg["weight_threshold_Wt"],
        L0=scfg["distance_cutoff_L0"],
        min_sites=scfg["min_sites"],
        maxdist_km=scfg.get("maxdist_km"),
        auto_search_radius=scfg.get("auto_search_radius", True),
        D_min=scfg.get("D_min_km", 1.0),
        D_max=scfg.get("D_max_km", 1000.0),
    )
    return estimator.compute()


def _run_delaunay(cfg: Config, vf: VelocityField, polygon):
    dcfg = cfg["algorithms"]["delaunay"]
    estimator = DelaunayStrainRate(
        vf,
        polygon=polygon,
        min_angle_deg=dcfg["min_angle_deg"],
        max_edge_pctl=dcfg["max_edge_pctl"],
        max_edge_factor=dcfg["max_edge_factor"],
        min_area_ratio=dcfg.get("min_area_ratio", 0.1),
        max_edge_km=dcfg.get("max_edge_km"),
    )
    return estimator.compute()


def _run_wang2012(cfg: Config, vf: VelocityField, polygon):
    wcfg = cfg["algorithms"]["wang2012"]
    smooth_range = wcfg.get("smooth_range", (-2.2, -0.8))
    estimator = Wang2012StrainRate(
        vf,
        polygon=polygon,
        mesh_region=wcfg.get("mesh_region"),
        mesh_method=wcfg.get("mesh_method", "adaptive"),
        mesh_spacing=wcfg.get("mesh_spacing", 0.25),
        mesh_randomize=wcfg.get("mesh_randomize", True),
        mesh_randomize_fraction=wcfg.get("mesh_randomize_fraction", 0.2),
        max_stations_per_cell=wcfg.get("max_stations_per_cell", 6),
        max_spacing=wcfg.get("max_spacing"),
        smooth_factor=wcfg.get("smooth_factor", 0.01),
        smooth_search=wcfg.get("smooth_search", True),
        smooth_range=tuple(smooth_range),
        smooth_step=wcfg.get("smooth_step", 0.2),
        smooth_boundary=wcfg.get("smooth_boundary", True),
        min_area_ratio=wcfg.get("min_area_ratio", 0.1),
    )
    return estimator.compute()


def run_pipeline(cfg: Config) -> dict:
    """Run the full PyStrain2 pipeline according to config."""
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    t0_total = time.time()
    vel_file = cfg["data"]["vel_file"]
    poly_file = cfg["data"].get("poly_file")
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = cfg["visualization"]
    fig_dir = output_dir / vis.get("figure_dir", "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = vis.get("dpi", 150)
    method = cfg["algorithms"]["method"]

    should_save = vis.get("save_figures", True)

    # ------------------------------------------------------------------
    # 1. Load polygon early (before plotting) so plot region can use it
    # ------------------------------------------------------------------
    polygon = None
    if poly_file:
        logger.info("  -> loading polygon from data section ...")
        rings = read_polygon(poly_file)
        polygon = rings[0]
        logger.info("  -> polygon with %d vertices loaded", len(polygon))

    # Resolve algorithm-specific polygon (wang2012)
    if polygon is None and method == "wang2012":
        wcfg = cfg["algorithms"]["wang2012"]
        wang_poly = wcfg.get("poly_file")
        if wang_poly:
            logger.info("  -> loading wang2012 polygon ...")
            rings = read_polygon(wang_poly)
            polygon = rings[0]
            logger.info("  -> polygon with %d vertices loaded", len(polygon))

    plot_region = _get_plot_region(cfg, polygon=polygon)

    # ------------------------------------------------------------------
    # 2. Load data — plot immediately
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PyStrain2 — GNSS strain-rate estimation pipeline")
    logger.info("=" * 60)
    logger.info("Step 1/5: Loading velocity field ...")
    t0 = time.time()
    vf = read_velocity_file(vel_file, fmt=cfg["data"]["format"])
    logger.info("  -> loaded %d sites from '%s' (%.1f s)", len(vf), vel_file, time.time() - t0)

    if plot_region:
        logger.info("  -> plot region: [%.1f, %.1f, %.1f, %.1f]", *plot_region)

    # --- SHOW: raw velocity field ---
    if should_save and vis.get("show_raw_velocity", True):
        logger.info("  -> plotting raw velocity field ...")
        plot_raw_velocity_field(vf, str(fig_dir / "01_raw_velocity.png"),
                                region=plot_region, dpi=dpi)

    vf_original = vf
    outlier_mask = np.zeros(len(vf), dtype=bool)

    # ------------------------------------------------------------------
    # 2. Outlier detection — plot result immediately
    # ------------------------------------------------------------------
    outliers = []
    loo_residuals = None   # LOO residuals for optional output
    if cfg["outlier_detection"]["enable"]:
        outlier_method = cfg["outlier_detection"].get("method", "knn_iqr")
        t0 = time.time()

        if outlier_method == "loo_strain":
            # ---- Leave-one-out strain-based outlier detection ----
            logger.info("Step 2/5: Detecting outliers (LOO strain) ...")
            loo_cfg = cfg["outlier_detection"]["loo_strain"]
            loo_mask, loo_residuals, loo_pred_ve, loo_pred_vn = \
                loo_strain_outlier_detection(
                    vf,
                    maxdist_km=loo_cfg.get("maxdist_km", 200.0),
                    min_sites=loo_cfg.get("min_sites", 8),
                    min_residual_mm=loo_cfg.get("min_residual_mm", 3.0),
                    show_progress=True,
                )
            # Convert to outlier list format
            for i in np.where(loo_mask)[0]:
                outliers.append({
                    "name": vf.names[i],
                    "lon": vf.lon[i],
                    "lat": vf.lat[i],
                    "residual": float(loo_residuals[i]),
                    "reason": f"LOO(res={loo_residuals[i]:.2f} mm/yr)",
                })
            vf = vf.subset(~loo_mask)
            outlier_mask = loo_mask.copy()
            elapsed = time.time() - t0

            # Write LOO residuals for all sites
            _write_loo_residuals(
                output_dir, vf_original, loo_residuals, loo_pred_ve, loo_pred_vn, loo_mask
            )
        else:
            # ---- Original KNN + triangulation IQR ----
            logger.info("Step 2/5: Detecting outliers (KNN + residual IQR) ...")
            knn_cfg = cfg["outlier_detection"]["knn_iqr"]
            vf, outliers = iterative_outlier_removal(
                vf,
                triangulation_fn=_triangulation_fn_for_outlier,
                k_neighbors=knn_cfg["k_neighbors"],
                mad_factor=knn_cfg["mad_factor"],
                iqr_factor=knn_cfg["iqr_factor"],
                max_iterations=knn_cfg["max_iterations"],
                min_residual=knn_cfg.get("min_residual_mm", 0.5),
            )
            elapsed = time.time() - t0

            if outliers:
                outlier_names = {o["name"] for o in outliers}
                for i, name in enumerate(vf_original.names):
                    if name in outlier_names:
                        outlier_mask[i] = True

        logger.info("  -> removed %d outliers, %d sites remain (%.1f s)",
                     len(outliers), len(vf), elapsed)

        # Write outliers
        from pystrain2.io import write_outliers
        write_outliers(str(output_dir / "outliers.txt"), outliers)

        # --- SHOW: outlier map ---
        if should_save and vis.get("show_outliers", True) and outliers:
            logger.info("  -> plotting outlier result ...")
            plot_outlier_result(vf_original, outlier_mask,
                                str(fig_dir / "02_outliers.png"),
                                region=plot_region, dpi=dpi)

        # --- SHOW: clean velocity field ---
        if should_save and vis.get("show_clean_velocity", True) and outliers:
            logger.info("  -> plotting clean velocity field ...")
            plot_clean_velocity_field(vf, str(fig_dir / "03_clean_velocity.png"),
                                      region=plot_region, dpi=dpi)
    else:
        logger.info("Step 2/5: Outlier detection disabled, skipped.")

    # ------------------------------------------------------------------
    # 3. Strain-rate computation — plot intermediate results on-the-fly
    # ------------------------------------------------------------------
    results = {}
    logger.info("Step 3/5: Computing strain rate (method=%s) ...", method)
    t0 = time.time()

    if method == "shen2015":
        scfg = cfg["algorithms"]["shen2015"]
        region = _parse_numeric_list(scfg["region"])
        spacing = _parse_numeric_list(scfg["spacing"])
        stagger = scfg.get("stagger", True)
        grid = Grid(region[0], region[1], region[2], region[3], spacing[0], spacing[1], stagger=stagger)
        logger.info("  -> grid: %.1f–%.1f °E  %.1f–%.1f °N  spacing=%.2f°×%.2f°  %d grid points",
                     region[0], region[1], region[2], region[3], spacing[0], spacing[1], len(grid))

        # --- SHOW: grid points layout (before computation) ---
        if should_save and vis.get("show_grid_points", True):
            logger.info("  -> plotting grid points layout ...")
            plot_grid_points(grid, vf, str(fig_dir / "04_grid_points.png"),
                             region=plot_region, dpi=dpi)

        results["shen2015"] = _run_shen2015(cfg, vf, polygon)
        elapsed = time.time() - t0
        n_valid = int(np.sum(np.isfinite(results["shen2015"].exx)))
        logger.info("  -> %d / %d grid points solved (%.1f s)", n_valid, len(grid), elapsed)
        write_strain_result(str(output_dir / "shen2015_strain.txt"), results["shen2015"])

        # --- SHOW: search radius map (after computation) ---
        if should_save and vis.get("show_search_radius", True):
            sr = results["shen2015"]
            if hasattr(sr, "meta") and "D_values" in sr.meta:
                logger.info("  -> plotting search radius map ...")
                plot_search_radius_map(grid, sr.meta["D_values"],
                                       str(fig_dir / "05_search_radius.png"),
                                       region=plot_region, dpi=dpi)

    elif method == "delaunay":
        # --- SHOW: triangulation ---
        if should_save and vis.get("show_triangulation", True):
            logger.info("  -> plotting triangulation ...")
            _plot_triangulation_safe(vf, polygon, str(fig_dir / "04_triangulation.png"),
                                     region=plot_region, dpi=dpi)

        results["delaunay"] = _run_delaunay(cfg, vf, polygon)
        elapsed = time.time() - t0
        logger.info("  -> %d valid triangles computed (%.1f s)", len(results["delaunay"]), elapsed)
        write_strain_result(str(output_dir / "delaunay_strain.txt"), results["delaunay"])

    elif method == "wang2012":
        # Polygon: use poly_file from wang2012 section; fall back to data section
        wcfg = cfg["algorithms"]["wang2012"]
        poly_path = wcfg.get("poly_file") or cfg["data"].get("poly_file")
        if poly_path and polygon is None:
            logger.info("  -> loading wang2012 polygon ...")
            rings = read_polygon(poly_path)
            polygon = rings[0]
        # Filter velocity field to sites inside polygon for wang2012
        if polygon is not None:
            from matplotlib.path import Path as MplPath
            path = MplPath(polygon)
            inside = path.contains_points(np.column_stack([vf.lon, vf.lat]))
            n_inside = np.sum(inside)
            if n_inside < 3:
                logger.warning("  -> only %d sites inside polygon, skipping filter", n_inside)
            else:
                logger.info("  -> keeping %d / %d sites inside polygon", n_inside, len(vf))
                vf = vf.subset(inside)
        results["wang2012"] = _run_wang2012(cfg, vf, polygon)
        elapsed = time.time() - t0
        logger.info("  -> %d mesh triangles computed (%.1f s)", len(results["wang2012"]), elapsed)
        write_strain_result(str(output_dir / "wang2012_strain.txt"), results["wang2012"])

        # --- Write mesh vertex estimated velocities ---
        meta = results["wang2012"].meta
        if "mesh_ve" in meta and "mesh_vn" in meta:
            from pystrain2.io import write_strain_velocity_file
            write_strain_velocity_file(
                str(output_dir / "wang2012_vertex_velocities.txt"),
                meta["mesh_lon"],
                meta["mesh_lat"],
                meta["mesh_ve"],
                meta["mesh_vn"],
            )
            logger.info("  -> vertex velocities written to '%s'",
                        output_dir / "wang2012_vertex_velocities.txt")

        # --- SHOW: wang2012 triangular mesh ---
        if should_save and vis.get("show_wang2012_mesh", True):
            logger.info("  -> plotting wang2012 mesh ...")
            plot_wang2012_mesh(results["wang2012"], vf,
                               str(fig_dir / "05_wang2012_mesh.png"),
                               polygon=polygon, region=plot_region, dpi=dpi)

    # ------------------------------------------------------------------
    # 3b. Monte Carlo uncertainty propagation (optional)
    # ------------------------------------------------------------------
    if cfg["uncertainty"]["enable"]:
        logger.info("Step 3b/5: Monte Carlo uncertainty (method=%s) ...", method)
        t0 = time.time()
        try:
            from pystrain2.uncertainty import monte_carlo_strain_uncertainty

            ucfg = cfg["uncertainty"]
            if method == "shen2015":
                def est_fn(vf_in):
                    scfg2 = cfg["algorithms"]["shen2015"]
                    region = _parse_numeric_list(scfg2["region"])
                    spacing = _parse_numeric_list(scfg2["spacing"])
                    stagger = scfg2.get("stagger", True)
                    grid = Grid(region[0], region[1], region[2], region[3],
                                spacing[0], spacing[1], stagger=stagger)
                    return ShenWangStrainRate(
                        vf_in, grid,
                        distance_kind=scfg2["distance_weight"],
                        spatial_kind=scfg2["spatial_weight"],
                        Wt=scfg2["weight_threshold_Wt"],
                        L0=scfg2["distance_cutoff_L0"],
                        min_sites=scfg2["min_sites"],
                        maxdist_km=scfg2.get("maxdist_km"),
                        auto_search_radius=scfg2.get("auto_search_radius", True),
                        D_min=scfg2.get("D_min_km", 1.0),
                        D_max=scfg2.get("D_max_km", 1000.0),
                    ).compute()
            elif method == "delaunay":
                def est_fn(vf_in):
                    dcfg2 = cfg["algorithms"]["delaunay"]
                    return DelaunayStrainRate(
                        vf_in, polygon=polygon,
                        min_angle_deg=dcfg2["min_angle_deg"],
                        max_edge_pctl=dcfg2["max_edge_pctl"],
                        max_edge_factor=dcfg2["max_edge_factor"],
                        min_area_ratio=dcfg2.get("min_area_ratio", 0.1),
                        max_edge_km=dcfg2.get("max_edge_km"),
                    ).compute()
            elif method == "wang2012":
                def est_fn(vf_in):
                    wcfg2 = cfg["algorithms"]["wang2012"]
                    sr = wcfg2.get("smooth_range", (-2.2, -0.8))
                    return Wang2012StrainRate(
                        vf_in, polygon=polygon,
                        mesh_region=wcfg2.get("mesh_region"),
                        mesh_method=wcfg2.get("mesh_method", "adaptive"),
                        mesh_spacing=wcfg2.get("mesh_spacing", 0.25),
                        mesh_randomize=wcfg2.get("mesh_randomize", True),
                        mesh_randomize_fraction=wcfg2.get("mesh_randomize_fraction", 0.2),
                        max_stations_per_cell=wcfg2.get("max_stations_per_cell", 6),
                        max_spacing=wcfg2.get("max_spacing"),
                        smooth_factor=wcfg2.get("smooth_factor", 0.01),
                        smooth_search=wcfg2.get("smooth_search", True),
                        smooth_range=tuple(sr),
                        smooth_step=wcfg2.get("smooth_step", 0.2),
                        smooth_boundary=wcfg2.get("smooth_boundary", True),
                        min_area_ratio=wcfg2.get("min_area_ratio", 0.1),
                    ).compute()
            else:
                raise ValueError(f"Unknown method: {method}")

            stds = monte_carlo_strain_uncertainty(
                est_fn, vf,
                n_iterations=ucfg.get("mc_iterations", 200),
                seed=ucfg.get("seed", 42),
            )
            # Attach uncertainty stds to the result meta
            result_key = method
            if result_key in results and stds:
                results[result_key].meta["uncertainty_stds"] = stds
            elapsed = time.time() - t0
            logger.info("  -> MC uncertainty (%d iterations) done (%.1f s)",
                        ucfg.get("mc_iterations", 200), elapsed)
        except Exception as exc:
            logger.warning("  -> uncertainty computation failed: %s", exc)
    else:
        logger.info("Step 3b/5: Uncertainty analysis disabled, skipped.")

    # ------------------------------------------------------------------
    # 4. Final strain visualisation
    # ------------------------------------------------------------------
    if should_save and results:
        logger.info("Step 4/5: Generating final strain figures ...")
        t0 = time.time()

        result_key = method
        if result_key in results:
            strain_result = results[result_key]

            if vis.get("show_strain_scalars", True):
                logger.info("  -> plotting dilatation field ...")
                plot_scalar_field(strain_result, "dilation",
                                  str(fig_dir / "06_dilatation.png"),
                                  region=plot_region, cmap="RdBu_r", dpi=dpi,
                                  symmetric=True)

                logger.info("  -> plotting max shear field ...")
                plot_scalar_field(strain_result, "shear",
                                  str(fig_dir / "07_max_shear.png"),
                                  region=plot_region, cmap="YlOrRd", dpi=dpi,
                                  symmetric=False)

            if vis.get("show_principal_crosses", True):
                logger.info("  -> plotting principal strain crosses ...")
                plot_principal_strain_crosses(
                    results[result_key], str(fig_dir / "08_principal_strain.png"),
                    polygon=polygon, region=plot_region, dpi=dpi
                )

        n_png = len(list(fig_dir.glob("*.png")))
        logger.info("  -> %d figures saved to '%s' (%.1f s)", n_png, fig_dir, time.time() - t0)
    else:
        logger.info("Step 4/5: Figures disabled, skipped.")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t0_total
    logger.info("Step 5/5: Pipeline complete in %.1f seconds.", elapsed_total)
    logger.info("  Output directory: %s", output_dir)

    summary = {
        "method": method,
        "sites_input": len(vf_original),
        "outliers_removed": len(outliers),
        "sites_used": len(vf),
        "grid_points_or_triangles": len(list(results.values())[0]) if results else 0,
        "elapsed_seconds": elapsed_total,
    }
    logger.info("  Summary: %s", summary)

    return {
        "vf": vf,
        "vf_original": vf_original,
        "polygon": polygon,
        "outliers": outliers,
        "outlier_mask": outlier_mask,
        "results": results,
    }


def _plot_triangulation_safe(vf, polygon, path, region=None, dpi=150):
    """Plot triangulation with error handling. Returns True on success."""
    try:
        tri, good_triangles, xy, _ = delaunay_triangulation(
            vf.lon, vf.lat, polygon=polygon
        )
        plot_triangulation(vf, tri, good_triangles, path,
                           region=region, dpi=dpi)
        return True
    except Exception:
        return False


def _run_plot(args) -> int:
    """Run the strain gridding and plotting pipeline."""
    import time
    from pathlib import Path

    from pystrain2.plot import strain_file_to_grid, plot_strain_map, plot_strain_overview
    from pystrain2.io import read_faults, read_velocity_file

    strain_file = Path(args.strain_file)
    if not strain_file.exists():
        logger.error("Strain file not found: %s", strain_file)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = strain_file.stem

    # Convert grid_size to tuple if provided
    grid_size = tuple(args.grid_size) if args.grid_size else None

    logger.info("=" * 60)
    logger.info("PyStrain2 — Strain rate gridding & plotting")
    logger.info("=" * 60)
    logger.info("Input:  %s", strain_file)
    logger.info("Output: %s", output_dir)

    # ------------------------------------------------------------------
    # 1. Grid the strain data
    # ------------------------------------------------------------------
    t0 = time.time()
    gridded = strain_file_to_grid(
        str(strain_file),
        output_dir=str(output_dir),
        grid_size=grid_size,
        max_distance=args.max_distance,
        method=args.method,
        output_format=args.output_format,
        fields=args.fields,
        region=args.region,
    )
    elapsed = time.time() - t0
    logger.info("  gridding done (%.1f s)", elapsed)

    # If --no-map, we're done
    if args.no_map:
        logger.info("  --no-map: skipping figure generation")
        logger.info("Done.")
        return 0

    # ------------------------------------------------------------------
    # 2. Generate maps
    # ------------------------------------------------------------------
    logger.info("Step 2/2: Generating GMT-style maps ...")
    t0 = time.time()

    grid_lon = gridded.pop("grid_lon")
    grid_lat = gridded.pop("grid_lat")

    available_fields = list(gridded.keys())
    logger.info("  available fields: %s", ", ".join(available_fields))

    region = args.region

    # --- load fault traces (if any) ---------------------------------------
    faults = None
    if args.faults:
        logger.info("  loading fault traces ...")
        faults = []
        for fault_file in args.faults:
            traces = read_faults(fault_file)
            faults.extend(traces)
            logger.info("    %s: %d traces", fault_file, len(traces))
        logger.info("  -> %d total fault traces loaded", len(faults))

    # --- load velocity field (if any) -------------------------------------
    vf = None
    if args.vel_file:
        logger.info("  loading velocity field: %s", args.vel_file)
        vf = read_velocity_file(args.vel_file)
        # Clip to region if provided
        if region is not None:
            slon, elon, slat, elat = region
            inside = (vf.lon >= slon) & (vf.lon <= elon) & (vf.lat >= slat) & (vf.lat <= elat)
            vf = vf.subset(inside)
        logger.info("  -> %d GPS sites", len(vf))

    # --- individual field maps (skip descriptive aliases if short name exists) ---
    _alias_of = {"shear": "shr", "dilation": "dil", "sec_inv": "inv2", "azimuth": "theta"}
    plotted = set()
    for field_name, field_data in gridded.items():
        # Skip descriptive alias if canonical short name is also present
        canonical = _alias_of.get(field_name)
        if canonical and canonical in gridded:
            continue

        n_valid = int(np.sum(np.isfinite(field_data)))
        if n_valid < 3:
            logger.info("  skipping '%s' (%d valid cells)", field_name, n_valid)
            continue

        output_path = str(output_dir / f"{base_name}_{field_name}.png")
        logger.info("  plotting '%s' → %s", field_name,
                     Path(output_path).name)
        plot_strain_map(
            grid_lon, grid_lat, field_data,
            field=field_name,
            output_path=output_path,
            region=region,
            faults=faults,
            vf=vf,
            dpi=args.dpi,
        )
        plotted.add(field_name)

    # --- overview (if we have the core 4 fields) ---
    core_fields = ["dilation", "shear", "sec_inv", "omega"]
    if all(f in gridded for f in core_fields):
        overview_path = str(output_dir / f"{base_name}_overview.png")
        logger.info("  plotting overview → %s", Path(overview_path).name)
        try:
            plot_strain_overview(
                grid_lon, grid_lat, gridded,
                output_path=overview_path,
                region=region,
                faults=faults,
                dpi=args.dpi,
            )
        except Exception as exc:
            logger.warning("  overview plot failed: %s", exc)

    # --- standalone velocity map (if vel-file provided) --------------------
    if vf is not None:
        from pystrain2.plot import plot_velocity_map
        vel_path = str(output_dir / f"{base_name}_velocity.png")
        logger.info("  plotting velocity field → %s", Path(vel_path).name)
        try:
            plot_velocity_map(
                vf, vel_path,
                region=region, faults=faults,
                dpi=args.dpi,
            )
        except Exception as exc:
            logger.warning("  velocity map failed: %s", exc)

    n_png = len(list(output_dir.glob("*.png")))
    elapsed = time.time() - t0
    logger.info("  -> %d figures saved (%.1f s)", n_png, elapsed)
    logger.info("Done.")

    return 0


def _web_app_path() -> str:
    """Return the absolute path to the Streamlit app script."""
    return str(Path(__file__).parent.parent / "web" / "app.py")


def _run_web_server(args):
    """Launch the PyStrain2 Streamlit web app."""
    streamlit = shutil.which("streamlit")
    if streamlit is None:
        print(
            "Error: 'streamlit' command not found. "
            "Install the GUI extras: pip install pystrain2[gui]",
            file=sys.stderr,
        )
        return 1

    app_path = _web_app_path()
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    if args.port is not None:
        cmd.extend(["--server.port", str(args.port)])
    if args.address is not None:
        cmd.extend(["--server.address", args.address])

    return subprocess.call(cmd)


def run_timeseries_pipeline(cfg: Config) -> dict:
    """Run the strain time-series pipeline according to config.

    Uses ``timeseries.method`` (one of ``"grid"``, ``"tri"``, ``"user"``)
    to select which estimator to run, matching the pattern used by the
    strain-rate ``algorithms.method`` field.

    Returns
    -------
    dict
        Keys: ``"tsc"`` (TimeSeriesCollection), ``"results"`` (list of
        StrainTimeSeriesResult).
    """
    import time
    from pathlib import Path

    tscfg = cfg["timeseries"]
    method = tscfg.get("method", "grid")
    output_dir = Path(tscfg.get("output_dir", "./pystrain2_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PyStrain2 — GNSS strain time-series estimation")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & align multi-site time series
    # ------------------------------------------------------------------
    logger.info("Step 1/3: Loading GPS time series ...")
    t0 = time.time()

    site_list = None
    if method == "user":
        ucfg = tscfg.get("user", {})
        groups_file = ucfg.get("site_groups_file")
        if groups_file and Path(groups_file).exists():
            all_sites = []
            with open(groups_file, "r") as fid:
                for line in fid:
                    all_sites.extend(line.strip().split())
            site_list = list(set(all_sites))
            logger.info("  -> %d unique sites from groups file", len(site_list))

    loader = TimeSeriesLoader(
        gps_info_file=tscfg["gps_info_file"],
        ts_type=tscfg["ts_type"],
        ts_path=tscfg["ts_path"],
        sepoch=tscfg["sepoch"],
        eepoch=tscfg["eepoch"],
        site_list=site_list,
    )
    tsc = loader.load()
    elapsed = time.time() - t0
    logger.info(
        "  -> loaded %d sites × %d epochs (%.1f s)",
        len(tsc.sites), len(tsc.decyr), elapsed,
    )

    # ------------------------------------------------------------------
    # 2. Compute strain time series (method dispatch)
    # ------------------------------------------------------------------
    logger.info("Step 2/3: Computing strain time series (method=%s) ...", method)
    t0 = time.time()
    results: list = []

    if method == "grid":
        gcfg = tscfg["grid"]
        grid = Grid(
            gcfg["slon"], gcfg["elon"],
            gcfg["slat"], gcfg["elat"],
            gcfg["dn"], gcfg["de"],
            stagger=gcfg.get("stagger", True),
        )
        logger.info(
            "  -> grid: %d points, maxdist=%.0f km, min_sites=%d",
            len(grid), gcfg["maxdist_km"], gcfg["min_sites"],
        )
        estimator = GridStrainTimeSeries(
            tsc, grid,
            maxdist_km=gcfg["maxdist_km"],
            min_sites=gcfg["min_sites"],
            check_azimuth=gcfg.get("check_azimuth", True),
            output_dir=str(output_dir),
        )
        results = estimator.compute()

    elif method == "tri":
        tcfg = tscfg["tri"]
        logger.info("  -> triangle mesh mode")
        estimator = TriStrainTimeSeries(
            tsc,
            polygon=None,
            min_angle_deg=tcfg["min_angle_deg"],
            max_edge_pctl=tcfg["max_edge_pctl"],
            max_edge_factor=tcfg["max_edge_factor"],
            min_area_ratio=tcfg.get("min_area_ratio", 0.1),
            max_edge_km=tcfg.get("max_edge_km"),
            projection=tcfg.get("projection", "utm"),
            output_dir=str(output_dir),
        )
        results = estimator.compute()

    elif method == "user":
        ucfg = tscfg["user"]
        groups_file = ucfg["site_groups_file"]
        logger.info("  -> user groups mode: '%s'", groups_file)
        estimator = UserStrainTimeSeries(
            tsc,
            site_groups_file=groups_file,
            max_sigma_mm=ucfg.get("max_sigma_mm", 5.0),
            output_dir=str(output_dir),
        )
        results = estimator.compute()

    else:
        raise ValueError(f"Unknown timeseries method: {method}")

    elapsed = time.time() - t0
    n_total = len(results)
    n_epochs = len(tsc.decyr)
    logger.info(
        "  -> %d locations × %d epochs computed (%.1f s)",
        n_total, n_epochs, elapsed,
    )

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    logger.info("Step 3/3: Done.")
    logger.info("  Output directory: %s", output_dir)

    summary = {
        "method": method,
        "sites": len(tsc.sites),
        "epochs": len(tsc.decyr),
        "locations_computed": n_total,
    }
    logger.info("  Summary: %s", summary)

    return {"tsc": tsc, "results": results}


def main(argv=None):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        prog="pystrain2", description="PyStrain2 GNSS strain-rate estimation"
    )
    subparsers = parser.add_subparsers(dest="command")

    # compute
    compute_parser = subparsers.add_parser(
        "compute", help="Run strain-rate computation"
    )
    compute_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )

    # shen2015 quick command
    shen_parser = subparsers.add_parser(
        "shen2015", help="Run Shen et al. (2015) grid-based strain rate"
    )
    shen_parser.add_argument("--vel-file", required=True)
    shen_parser.add_argument("--poly-file", default=None)
    shen_parser.add_argument("--region", nargs=4, required=True, type=float,
                             metavar=("SLON", "ELON", "SLAT", "ELAT"))
    shen_parser.add_argument("--spacing", nargs=2, required=True, type=float,
                             metavar=("DLON", "DLAT"))
    shen_parser.add_argument("--output-dir", default="./pystrain2_output")
    shen_parser.add_argument("--distance-weight", default="gaussian")
    shen_parser.add_argument("--spatial-weight", default="voronoi")
    shen_parser.add_argument("--Wt", default=24.0, type=float)
    shen_parser.add_argument("--no-stagger", action="store_true",
                             help="Disable staggered grid rows")
    shen_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate result figures in output_dir/figures",
    )
    shen_parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Enable all visualization steps",
    )

    # delaunay quick command
    del_parser = subparsers.add_parser(
        "delaunay", help="Run Delaunay triangulation strain rate"
    )
    del_parser.add_argument("--vel-file", required=True)
    del_parser.add_argument("--poly-file", default=None)
    del_parser.add_argument("--output-dir", default="./pystrain2_output")
    del_parser.add_argument("--min-angle-deg", default=10.0, type=float)
    del_parser.add_argument("--max-edge-pctl", default=95.0, type=float)
    del_parser.add_argument("--max-edge-factor", default=1.5, type=float)
    del_parser.add_argument("--min-area-ratio", default=0.1, type=float)
    del_parser.add_argument("--plot", action="store_true",
                            help="Generate result figures in output_dir/figures")
    del_parser.add_argument("--plot-all", action="store_true",
                            help="Enable all visualization steps")

    # wang2012 quick command
    wang_parser = subparsers.add_parser(
        "wang2012", help="Run Wang (2012) mesh-based strain rate"
    )
    wang_parser.add_argument("--vel-file", required=True)
    wang_parser.add_argument("--poly-file", default=None)
    wang_parser.add_argument("--output-dir", default="./pystrain2_output")
    wang_parser.add_argument("--mesh-method", default="adaptive",
                            choices=["adaptive", "poisson", "gmsh"],
                            help="Mesh strategy: 'adaptive' (grid-based), 'poisson' (natural triangles), 'gmsh' (Gmsh mesher)")
    wang_parser.add_argument("--mesh-spacing", default=0.25, type=float)
    wang_parser.add_argument("--max-stations-per-cell", default=6, type=int,
                            help="Quadtree subdivides cells with > this many stations")
    wang_parser.add_argument("--max-spacing", default=None, type=float,
                            help="Coarsest mesh spacing (default: mesh_spacing × 4)")
    wang_parser.add_argument("--smooth-factor", default=0.01, type=float)
    wang_parser.add_argument("--no-smooth-search", action="store_true",
                             help="Disable L-curve smoothing search")
    wang_parser.add_argument("--plot", action="store_true",
                             help="Generate result figures in output_dir/figures")
    wang_parser.add_argument("--plot-all", action="store_true",
                             help="Enable all visualization steps")

    # timeseries command
    ts_parser = subparsers.add_parser(
        "timeseries", help="Run strain time-series estimation"
    )
    ts_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )

    # plot command
    plot_parser = subparsers.add_parser(
        "plot", help="Grid strain output and generate GMT-style maps"
    )
    plot_parser.add_argument(
        "strain_file", help="Path to strain output .txt file"
    )
    plot_parser.add_argument(
        "--output-dir", default="./pystrain2_plots",
        help="Output directory (default: ./pystrain2_plots)"
    )
    plot_parser.add_argument(
        "--grid-size", nargs=2, type=int, default=None, metavar=("NLON", "NLAT"),
        help="Grid dimensions in pixels (default: auto)"
    )
    plot_parser.add_argument(
        "--max-distance", type=float, default=None, metavar="DEG",
        help="Max interpolation distance in degrees (default: auto, 2.5× median NN)"
    )
    plot_parser.add_argument(
        "--method", default="linear", choices=["linear", "cubic"],
        help="Interpolation method (default: linear)"
    )
    plot_parser.add_argument(
        "--format", default="netcdf", choices=["netcdf", "geotiff", "both"],
        dest="output_format",
        help="Grid output format (default: netcdf)"
    )
    plot_parser.add_argument(
        "--fields", nargs="+", default=None, metavar="FIELD",
        help="Fields to grid (default: all).  E.g. dilation shear e1 e2"
    )
    plot_parser.add_argument(
        "--region", nargs=4, type=float, default=None,
        metavar=("SLON", "ELON", "SLAT", "ELAT"),
        help="Map extent (default: auto from data)"
    )
    plot_parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output figure DPI (default: 150)"
    )
    plot_parser.add_argument(
        "--vel-file", default=None, metavar="VEL_FILE",
        help="GPS velocity file to overlay on strain maps (GMT8/GMT7 format)"
    )
    plot_parser.add_argument(
        "--faults", nargs="+", default=None, metavar="FAULT_FILE",
        help="Fault trace files to overlay (KML or SHP format)"
    )
    plot_parser.add_argument(
        "--no-map", action="store_true",
        help="Skip map generation; only save NetCDF/GeoTIFF"
    )

    # web quick command
    web_parser = subparsers.add_parser("web", help="Launch the PyStrain2 web app")
    web_parser.add_argument(
        "--port", type=int, default=None, help="Streamlit server port"
    )
    web_parser.add_argument("--address", default=None, help="Streamlit server address")

    args = parser.parse_args(argv)

    if args.command == "compute":
        cfg = Config(args.config)
        run_pipeline(cfg)

    elif args.command == "shen2015":
        plot_all = getattr(args, "plot_all", False)
        cfg = Config(
            overrides={
                "data": {
                    "vel_file": args.vel_file,
                    "poly_file": args.poly_file,
                    "output_dir": args.output_dir,
                },
                "algorithms": {
                    "method": "shen2015",
                    "shen2015": {
                        "region": args.region,
                        "spacing": args.spacing,
                        "stagger": not args.no_stagger,
                        "distance_weight": args.distance_weight,
                        "spatial_weight": args.spatial_weight,
                        "weight_threshold_Wt": args.Wt,
                    },
                },
                "visualization": {
                    "save_figures": args.plot or plot_all,
                    "show_raw_velocity": plot_all,
                    "show_outliers": plot_all,
                    "show_clean_velocity": plot_all,
                    "show_triangulation": False,
                    "show_grid_points": args.plot or plot_all,
                    "show_search_radius": plot_all,
                    "show_strain_scalars": args.plot or plot_all,
                    "show_principal_crosses": args.plot or plot_all,
                },
            }
        )
        run_pipeline(cfg)

    elif args.command == "delaunay":
        plot_all = getattr(args, "plot_all", False)
        cfg = Config(
            overrides={
                "data": {
                    "vel_file": args.vel_file,
                    "poly_file": args.poly_file,
                    "output_dir": args.output_dir,
                },
                "algorithms": {
                    "method": "delaunay",
                    "delaunay": {
                        "min_angle_deg": args.min_angle_deg,
                        "max_edge_pctl": args.max_edge_pctl,
                        "max_edge_factor": args.max_edge_factor,
                        "min_area_ratio": args.min_area_ratio,
                    },
                },
                "visualization": {
                    "save_figures": args.plot or plot_all,
                    "show_raw_velocity": plot_all,
                    "show_outliers": plot_all,
                    "show_clean_velocity": plot_all,
                    "show_triangulation": args.plot or plot_all,
                    "show_grid_points": False,
                    "show_search_radius": False,
                    "show_strain_scalars": args.plot or plot_all,
                    "show_principal_crosses": args.plot or plot_all,
                },
            }
        )
        run_pipeline(cfg)

    elif args.command == "wang2012":
        plot_all = getattr(args, "plot_all", False)
        cfg = Config(
            overrides={
                "data": {
                    "vel_file": args.vel_file,
                    "poly_file": args.poly_file,
                    "output_dir": args.output_dir,
                },
                "algorithms": {
                    "method": "wang2012",
                    "wang2012": {
                        "poly_file": args.poly_file,
                        "mesh_method": args.mesh_method,
                        "mesh_spacing": args.mesh_spacing,
                        "max_stations_per_cell": args.max_stations_per_cell,
                        "max_spacing": args.max_spacing,
                        "smooth_factor": args.smooth_factor,
                        "smooth_search": not args.no_smooth_search,
                    },
                },
                "visualization": {
                    "save_figures": args.plot or plot_all,
                    "show_raw_velocity": plot_all,
                    "show_outliers": plot_all,
                    "show_clean_velocity": plot_all,
                    "show_triangulation": False,
                    "show_grid_points": False,
                    "show_search_radius": False,
                    "show_strain_scalars": args.plot or plot_all,
                    "show_wang2012_mesh": args.plot or plot_all,
                    "show_principal_crosses": args.plot or plot_all,
                },
            }
        )
        run_pipeline(cfg)

    elif args.command == "timeseries":
        cfg = Config(args.config)
        run_timeseries_pipeline(cfg)

    elif args.command == "plot":
        return _run_plot(args)

    elif args.command == "web":
        return _run_web_server(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
