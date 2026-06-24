"""Per-epoch strain time-series estimation.

Three estimation modes:

* :class:`GridStrainTimeSeries` — at each grid point.
* :class:`TriStrainTimeSeries` — at each Delaunay triangle centroid.
* :class:`UserStrainTimeSeries` — at user-defined site-group centroids.

All modes call :func:`pystrain.strain.lsq.estimate_strain_rate` per epoch.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from sklearn.linear_model import RANSACRegressor

from pystrain.data import StrainTimeSeriesResult, TimeSeriesCollection
from pystrain.geodesy import distance_azimuth
from pystrain.grid.grid import Grid
from pystrain.strain.lsq import estimate_strain_rate
from pystrain.triangulation import delaunay_triangulation

logger = logging.getLogger("pystrain.timeseries.strain_ts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distance_weight(d: np.ndarray, D: float) -> np.ndarray:
    """Gaussian distance weight: exp(-d²/D²).

    .. note::
       Uses **negative** exponential (corrected from the legacy PyStrain
       implementation which had an inverted sign ``exp(+d²/D²)``).
    """
    return np.exp(-(d ** 2) / (D ** 2))


def _check_azimuth_coverage(azimuths: np.ndarray) -> bool:
    """Return True if sites cover all four quadrants."""
    quadrants = [0, 0, 0, 0]
    for azi in azimuths:
        if 0 < azi < 90:
            quadrants[0] = 1
        elif 90 < azi < 180:
            quadrants[1] = 1
        elif -180 < azi < -90:
            quadrants[2] = 1
        elif -90 < azi < 0:
            quadrants[3] = 1
    return sum(quadrants) == 4


def _write_strain_ts_header(filepath: str, lon: float, lat: float) -> None:
    """Write a strain-time-series file header."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as fid:
        fid.write(f"# {lon:.6f}  {lat:.6f}\n")
        fid.write(
            "#      decyr        Ve        Vn         Exx         Exy         Eyy"
            "       omega          E1          E2       shear    dilation"
            "    sec_inv      theta\n"
        )


def _append_strain_epoch(filepath: str, decyr: float, result: dict) -> None:
    """Append one epoch's strain result to a file."""
    with open(filepath, "a") as fid:
        fid.write(
            f"{decyr:12.5f} {result['dx']:8.2f} {result['dy']:8.2f}"
            f" {result['exx']:11.4e} {result['exy']:11.4e}"
            f" {result['eyy']:11.4e} {result['omega']:11.4e}"
            f" {result['e1']:11.4e} {result['e2']:11.4e}"
            f" {result['shear']:11.4e} {result['dilation']:11.4e}"
            f" {result['sec_inv']:11.4e} {result['azimuth']:8.2f}\n"
        )


def _make_ts_result(lon: float, lat: float, decyrs, arrays: dict, meta=None):
    """Pack per-epoch arrays into a ``StrainTimeSeriesResult``."""
    n = len(decyrs)
    nan_arr = np.full(n, np.nan)
    return StrainTimeSeriesResult(
        lon=lon,
        lat=lat,
        decyr=np.asarray(decyrs),
        exx=arrays.get("exx", nan_arr.copy()),
        exy=arrays.get("exy", nan_arr.copy()),
        eyy=arrays.get("eyy", nan_arr.copy()),
        omega=arrays.get("omega", nan_arr.copy()),
        e1=arrays.get("e1", nan_arr.copy()),
        e2=arrays.get("e2", nan_arr.copy()),
        azimuth=arrays.get("azimuth", nan_arr.copy()),
        shear=arrays.get("shear", nan_arr.copy()),
        dilation=arrays.get("dilation", nan_arr.copy()),
        sec_inv=arrays.get("sec_inv", nan_arr.copy()),
        ve=arrays.get("ve", nan_arr.copy()),
        vn=arrays.get("vn", nan_arr.copy()),
        condition_number=arrays.get("condition_number", nan_arr.copy()),
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# Grid-mesh strain time series
# ---------------------------------------------------------------------------

class GridStrainTimeSeries:
    """Strain time series at each point of a regular grid.

    Parameters
    ----------
    tsc : TimeSeriesCollection
        Aligned multi-site position time series.
    grid : Grid
        Regular lon/lat grid definition.
    maxdist_km : float
        Maximum site-to-grid-point distance (km).
    min_sites : int
        Minimum number of sites required around each grid point.
    check_azimuth : bool
        If True, require sites in all four quadrants.
    output_dir : str
        Directory for per-grid-point output files.
    """

    def __init__(
        self,
        tsc: TimeSeriesCollection,
        grid: Grid,
        maxdist_km: float = 300.0,
        min_sites: int = 6,
        check_azimuth: bool = True,
        output_dir: str = "./pystrain_output",
    ):
        self.tsc = tsc
        self.grid = grid
        self.maxdist_km = maxdist_km
        self.min_sites = min_sites
        self.check_azimuth = check_azimuth
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    def compute(self) -> List[StrainTimeSeriesResult]:
        """Run per-epoch strain estimation at every grid point.

        Returns
        -------
        list of StrainTimeSeriesResult
        """
        n_grid = len(self.grid)
        n_epochs = len(self.tsc.decyr)
        results: List[StrainTimeSeriesResult] = []

        logger.info(
            "Grid strain TS: %d grid points × %d epochs, maxdist=%.0f km",
            n_grid, n_epochs, self.maxdist_km,
        )

        # Build site coordinate array for distance computation
        site_lon = np.asarray(self.tsc.lon)
        site_lat = np.asarray(self.tsc.lat)

        for ig in range(n_grid):
            glon = float(self.grid.lon[ig])
            glat = float(self.grid.lat[ig])

            # ---- Precompute site selection for this grid point ----
            d, az = distance_azimuth(
                np.full(len(site_lon), glon),
                np.full(len(site_lat), glat),
                site_lon, site_lat,
            )
            near = d <= self.maxdist_km
            n_near = np.sum(near)
            if n_near < self.min_sites:
                logger.debug(
                    "Grid pt %d (%.2f, %.2f): only %d sites within %.0f km",
                    ig, glon, glat, n_near, self.maxdist_km,
                )
                continue

            if self.check_azimuth:
                if not _check_azimuth_coverage(az[near]):
                    logger.debug(
                        "Grid pt %d (%.2f, %.2f): poor azimuthal coverage",
                        ig, glon, glat,
                    )
                    continue

            near_idx = np.where(near)[0]

            # Determine smoothing distance D
            d_near = d[near]
            D = np.sort(d_near)[min(self.min_sites, n_near) - 1] * 1.5

            # Local coordinates: E/km, N/km from distance+azimuth
            az_rad = np.deg2rad(az[near])
            x = d_near * np.sin(az_rad)
            y = d_near * np.cos(az_rad)

            # Distance weights (fixed: exp(-d²/D²))
            w_dist = _distance_weight(d_near, D)

            # ---- Output file ----
            ts_dir = os.path.join(self.output_dir, "timeseries")
            outfile = os.path.join(ts_dir, f"ts_grd_{ig:04d}.txt")
            _write_strain_ts_header(outfile, glon, glat)

            # ---- Per-epoch arrays ----
            exx = np.full(n_epochs, np.nan)
            exy = np.full(n_epochs, np.nan)
            eyy = np.full(n_epochs, np.nan)
            omega = np.full(n_epochs, np.nan)
            e1 = np.full(n_epochs, np.nan)
            e2 = np.full(n_epochs, np.nan)
            azimuth = np.full(n_epochs, np.nan)
            shear = np.full(n_epochs, np.nan)
            dilation = np.full(n_epochs, np.nan)
            sec_inv = np.full(n_epochs, np.nan)
            ve_arr = np.full(n_epochs, np.nan)
            vn_arr = np.full(n_epochs, np.nan)
            cond = np.full(n_epochs, np.nan)

            for j in range(n_epochs):
                # Sites with valid data this epoch
                E_epoch = self.tsc.E[j, near_idx]
                N_epoch = self.tsc.N[j, near_idx]
                SE_epoch = self.tsc.SE[j, near_idx]
                SN_epoch = self.tsc.SN[j, near_idx]

                valid = ~np.isnan(E_epoch) & ~np.isnan(N_epoch)
                if np.sum(valid) < self.min_sites:
                    continue

                try:
                    res = estimate_strain_rate(
                        x[valid], y[valid],
                        E_epoch[valid], N_epoch[valid],
                        SE_epoch[valid], SN_epoch[valid],
                        weights=w_dist[valid],
                        normalize=True,
                    )
                except Exception:
                    continue

                _append_strain_epoch(outfile, self.tsc.decyr[j], res)

                exx[j] = res["exx"]
                exy[j] = res["exy"]
                eyy[j] = res["eyy"]
                omega[j] = res["omega"]
                e1[j] = res["e1"]
                e2[j] = res["e2"]
                azimuth[j] = res["azimuth"]
                shear[j] = res["shear"]
                dilation[j] = res["dilation"]
                sec_inv[j] = res["sec_inv"]
                ve_arr[j] = res["dx"]
                vn_arr[j] = res["dy"]
                cond[j] = res["condition_number"]

            results.append(_make_ts_result(glon, glat, self.tsc.decyr, {
                "exx": exx, "exy": exy, "eyy": eyy, "omega": omega,
                "e1": e1, "e2": e2, "azimuth": azimuth,
                "shear": shear, "dilation": dilation, "sec_inv": sec_inv,
                "ve": ve_arr, "vn": vn_arr, "condition_number": cond,
            }, meta={"grid_index": ig, "output_file": outfile}))

        logger.info("Grid strain TS: %d/%d grid points solved.", len(results), n_grid)
        return results


# ---------------------------------------------------------------------------
# Triangle-mesh strain time series
# ---------------------------------------------------------------------------

class TriStrainTimeSeries:
    """Strain time series at Delaunay triangle centroids.

    Parameters
    ----------
    tsc : TimeSeriesCollection
        Aligned multi-site position time series.
    polygon : np.ndarray, optional
        (N,2) polygon for clipping triangles.
    min_angle_deg : float
        Minimum triangle interior angle.
    max_edge_pctl : float
        Percentile for edge-length threshold.
    max_edge_factor : float
        Multiplier on percentile edge length.
    min_area_ratio : float
        Minimum area ratio relative to 5th percentile.
    max_edge_km : float, optional
        Absolute edge-length cap.
    projection : str
        Projection method (``"utm"`` or ``"polyconic"``).
    output_dir : str
        Directory for per-triangle output files.
    """

    def __init__(
        self,
        tsc: TimeSeriesCollection,
        polygon: Optional[np.ndarray] = None,
        min_angle_deg: float = 10.0,
        max_edge_pctl: float = 95.0,
        max_edge_factor: float = 1.5,
        min_area_ratio: float = 0.1,
        max_edge_km: Optional[float] = None,
        projection: str = "utm",
        output_dir: str = "./pystrain_output",
    ):
        self.tsc = tsc
        self.polygon = polygon
        self.min_angle_deg = min_angle_deg
        self.max_edge_pctl = max_edge_pctl
        self.max_edge_factor = max_edge_factor
        self.min_area_ratio = min_area_ratio
        self.max_edge_km = max_edge_km
        self.projection = projection
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    def compute(self) -> List[StrainTimeSeriesResult]:
        """Run per-epoch strain estimation at triangle centroids.

        Returns
        -------
        list of StrainTimeSeriesResult
        """
        n_epochs = len(self.tsc.decyr)
        site_lon = np.asarray(self.tsc.lon)
        site_lat = np.asarray(self.tsc.lat)

        # Build triangulation from site coordinates
        tri, good_tri, _, _ = delaunay_triangulation(
            site_lon, site_lat,
            polygon=self.polygon,
            min_angle_deg=self.min_angle_deg,
            max_edge_pctl=self.max_edge_pctl,
            max_edge_factor=self.max_edge_factor,
            min_area_ratio=self.min_area_ratio,
            max_edge_km=self.max_edge_km,
        )

        good_indices = np.where(good_tri)[0]
        n_tri = len(good_indices)
        if n_tri == 0:
            raise ValueError("No valid triangles after quality control.")

        logger.info(
            "Tri strain TS: %d triangles × %d epochs", n_tri, n_epochs,
        )

        results: List[StrainTimeSeriesResult] = []

        for k, tri_idx in enumerate(good_indices):
            simplex = tri.simplices[tri_idx]
            lon_tri = site_lon[simplex]
            lat_tri = site_lat[simplex]
            centroid_lon = float(np.mean(lon_tri))
            centroid_lat = float(np.mean(lat_tri))

            # Build local coordinates (relative to centroid)
            # Use simple Cartesian approximation for small triangles
            from pystrain.geodesy import local_to_origin_projection
            x, y, _ = local_to_origin_projection(
                lon_tri, lat_tri,
                (centroid_lon, centroid_lat),
                method=self.projection,
            )

            # ---- Output file ----
            ts_dir = os.path.join(self.output_dir, "timeseries")
            outfile = os.path.join(ts_dir, f"ts_tri_{k:04d}.txt")
            _write_strain_ts_header(outfile, centroid_lon, centroid_lat)

            # ---- Per-epoch arrays ----
            exx = np.full(n_epochs, np.nan)
            exy = np.full(n_epochs, np.nan)
            eyy = np.full(n_epochs, np.nan)
            omega = np.full(n_epochs, np.nan)
            e1 = np.full(n_epochs, np.nan)
            e2 = np.full(n_epochs, np.nan)
            azimuth = np.full(n_epochs, np.nan)
            shear = np.full(n_epochs, np.nan)
            dilation = np.full(n_epochs, np.nan)
            sec_inv = np.full(n_epochs, np.nan)
            ve_arr = np.full(n_epochs, np.nan)
            vn_arr = np.full(n_epochs, np.nan)
            cond = np.full(n_epochs, np.nan)

            for j in range(n_epochs):
                E_epoch = self.tsc.E[j, simplex]
                N_epoch = self.tsc.N[j, simplex]
                SE_epoch = self.tsc.SE[j, simplex]
                SN_epoch = self.tsc.SN[j, simplex]

                # Skip if any vertex is NaN
                if np.any(np.isnan(E_epoch)) or np.any(np.isnan(N_epoch)):
                    continue

                try:
                    res = estimate_strain_rate(
                        x, y,
                        E_epoch, N_epoch,
                        SE_epoch, SN_epoch,
                        normalize=True,
                    )
                except Exception:
                    continue

                _append_strain_epoch(outfile, self.tsc.decyr[j], res)

                exx[j] = res["exx"]
                exy[j] = res["exy"]
                eyy[j] = res["eyy"]
                omega[j] = res["omega"]
                e1[j] = res["e1"]
                e2[j] = res["e2"]
                azimuth[j] = res["azimuth"]
                shear[j] = res["shear"]
                dilation[j] = res["dilation"]
                sec_inv[j] = res["sec_inv"]
                ve_arr[j] = res["dx"]
                vn_arr[j] = res["dy"]
                cond[j] = res["condition_number"]

            results.append(_make_ts_result(
                centroid_lon, centroid_lat, self.tsc.decyr, {
                    "exx": exx, "exy": exy, "eyy": eyy, "omega": omega,
                    "e1": e1, "e2": e2, "azimuth": azimuth,
                    "shear": shear, "dilation": dilation, "sec_inv": sec_inv,
                    "ve": ve_arr, "vn": vn_arr, "condition_number": cond,
                }, meta={"tri_index": int(tri_idx), "output_file": outfile},
            ))

        logger.info("Tri strain TS: %d/%d triangles solved.", len(results), n_tri)
        return results


# ---------------------------------------------------------------------------
# User-defined site-group strain time series
# ---------------------------------------------------------------------------

class UserStrainTimeSeries:
    """Strain time series at user-defined site-group centroids.

    Parameters
    ----------
    tsc : TimeSeriesCollection
        Aligned multi-site position time series.
    site_groups_file : str
        Path to a file with one group per line (space-separated site names).
    max_sigma_mm : float
        Maximum allowed per-site uncertainty (mm) per epoch.
    output_dir : str
        Directory for per-group output files.
    """

    def __init__(
        self,
        tsc: TimeSeriesCollection,
        site_groups_file: str,
        max_sigma_mm: float = 5.0,
        output_dir: str = "./pystrain_output",
    ):
        self.tsc = tsc
        self.site_groups_file = site_groups_file
        self.max_sigma_mm = max_sigma_mm
        self.output_dir = output_dir

        # Parse site groups
        self._groups: List[List[str]] = []
        with open(site_groups_file, "r") as fid:
            for line in fid:
                parts = line.strip().split()
                if parts:
                    self._groups.append(parts)

    # ------------------------------------------------------------------
    def compute(self) -> List[StrainTimeSeriesResult]:
        """Run per-epoch strain estimation for each site group.

        Returns
        -------
        list of StrainTimeSeriesResult
        """
        n_epochs = len(self.tsc.decyr)
        results: List[StrainTimeSeriesResult] = []

        for ig, group in enumerate(self._groups):
            if len(group) < 3:
                logger.warning(
                    "Group %d has <3 sites (%s) — skipped.", ig, " ".join(group)
                )
                continue

            # Map site names → indices in TimeSeriesCollection
            idx_site = []
            for site_name in group:
                try:
                    idx = self.tsc.sites.index(site_name)
                except ValueError:
                    logger.warning("Site %s not in collection — skipped.", site_name)
                    continue
                idx_site.append(idx)

            if len(idx_site) < 3:
                logger.warning(
                    "Group %d: <3 valid sites after lookup — skipped.", ig
                )
                continue

            idx_site = np.array(idx_site)
            group_lon = self.tsc.lon[idx_site]
            group_lat = self.tsc.lat[idx_site]
            centroid_lon = float(np.mean(group_lon))
            centroid_lat = float(np.mean(group_lat))

            # Local coordinates
            from pystrain.geodesy import local_to_origin_projection
            x, y, _ = local_to_origin_projection(
                group_lon, group_lat,
                (centroid_lon, centroid_lat),
                method="utm",
            )

            # ---- Output file ----
            ts_dir = os.path.join(self.output_dir, "timeseries")
            group_label = "_".join(group)
            outfile = os.path.join(ts_dir, f"ts_usr_{group_label}.txt")
            _write_strain_ts_header(outfile, centroid_lon, centroid_lat)

            # ---- Per-epoch arrays ----
            exx = np.full(n_epochs, np.nan)
            exy = np.full(n_epochs, np.nan)
            eyy = np.full(n_epochs, np.nan)
            omega = np.full(n_epochs, np.nan)
            e1 = np.full(n_epochs, np.nan)
            e2 = np.full(n_epochs, np.nan)
            azimuth = np.full(n_epochs, np.nan)
            shear = np.full(n_epochs, np.nan)
            dilation = np.full(n_epochs, np.nan)
            sec_inv = np.full(n_epochs, np.nan)
            ve_arr = np.full(n_epochs, np.nan)
            vn_arr = np.full(n_epochs, np.nan)
            cond = np.full(n_epochs, np.nan)

            for j in range(n_epochs):
                E_epoch = self.tsc.E[j, idx_site]
                N_epoch = self.tsc.N[j, idx_site]
                SE_epoch = self.tsc.SE[j, idx_site]
                SN_epoch = self.tsc.SN[j, idx_site]

                # Skip epochs with NaN data
                if np.any(np.isnan(E_epoch)) or np.any(np.isnan(N_epoch)):
                    continue

                # Skip if any uncertainty > max_sigma_mm
                if np.any(SE_epoch > self.max_sigma_mm) or np.any(SN_epoch > self.max_sigma_mm):
                    continue

                try:
                    res = estimate_strain_rate(
                        x, y,
                        E_epoch, N_epoch,
                        SE_epoch, SN_epoch,
                        normalize=True,
                    )
                except Exception:
                    continue

                _append_strain_epoch(outfile, self.tsc.decyr[j], res)

                exx[j] = res["exx"]
                exy[j] = res["exy"]
                eyy[j] = res["eyy"]
                omega[j] = res["omega"]
                e1[j] = res["e1"]
                e2[j] = res["e2"]
                azimuth[j] = res["azimuth"]
                shear[j] = res["shear"]
                dilation[j] = res["dilation"]
                sec_inv[j] = res["sec_inv"]
                ve_arr[j] = res["dx"]
                vn_arr[j] = res["dy"]
                cond[j] = res["condition_number"]

            results.append(_make_ts_result(
                centroid_lon, centroid_lat, self.tsc.decyr, {
                    "exx": exx, "exy": exy, "eyy": eyy, "omega": omega,
                    "e1": e1, "e2": e2, "azimuth": azimuth,
                    "shear": shear, "dilation": dilation, "sec_inv": sec_inv,
                    "ve": ve_arr, "vn": vn_arr, "condition_number": cond,
                }, meta={
                    "group_index": ig,
                    "group_sites": group,
                    "output_file": outfile,
                }),
            )

            logger.info(
                "User strain TS group %d: %d sites at (%.2f, %.2f)",
                ig, len(idx_site), centroid_lon, centroid_lat,
            )

        return results
