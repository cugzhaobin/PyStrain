"""Shen et al. (2015) BSSA grid-based strain-rate estimation."""

import logging
from typing import Optional
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import KDTree

from pystrain2.data import StrainResult, VelocityField
from pystrain2.geodesy import distance_azimuth, llh2utm, local_to_origin_projection
from pystrain2.grid.grid import Grid
from pystrain2.strain.lsq import estimate_strain_rate
from pystrain2.triangulation import voronoi_areas

logger = logging.getLogger("pystrain2")


def _distance_weight(d: np.ndarray, D: float, kind: str = "gaussian") -> np.ndarray:
    """Compute distance-dependent weight L_i."""
    if kind == "gaussian":
        return np.exp(-(d ** 2) / (D ** 2))
    elif kind == "quadratic":
        return 1.0 / (1.0 + (d ** 2) / (D ** 2))
    else:
        raise ValueError(f"Unknown distance weight kind: {kind}")


def _azimuth_weight(azimuths: np.ndarray) -> np.ndarray:
    """Compute azimuth-span spatial coverage weight Z_i (vectorized)."""
    n = len(azimuths)
    if n == 0:
        return np.array([])
    order = np.argsort(azimuths)
    sorted_az = azimuths[order]
    # compute span = (next - prev) in circular order
    shifted_next = np.roll(sorted_az, -1)
    shifted_prev = np.roll(sorted_az, 1)
    spans = np.empty(n)
    spans[order] = (shifted_next - shifted_prev) % 360.0
    return n * np.deg2rad(spans) / (4.0 * np.pi)


def _solve_D(
    d_selected: np.ndarray,
    Z: np.ndarray,
    Wt: float,
    L0: float,
    distance_kind: str,
    D_min: float = 1.0,
    D_max: float = 1000.0,
) -> Optional[float]:
    """Find smoothing distance D such that total weight Σ L(D)*Z = Wt.

    Z is precomputed spatial coverage weight (azimuth or voronoi).
    """
    def total_weight(D):
        L = _distance_weight(d_selected, D, distance_kind)
        L = np.where(L >= L0, L, 0.0)
        return np.sum(L * Z)

    try:
        W_min = total_weight(D_min)
        W_max = total_weight(D_max)
    except Exception:
        return None

    if W_min > Wt:
        return D_min
    if W_max < Wt:
        return D_max

    try:
        D_opt = brentq(lambda Dv: total_weight(Dv) - Wt, D_min, D_max)
        return D_opt
    except ValueError:
        return None


def estimate_search_radius(lon: np.ndarray, lat: np.ndarray, k: int = 6, factor: float = 3.0) -> float:
    """Estimate a reasonable maximum search distance from GPS site density.
    
    Uses mean k-nearest-neighbor distance in UTM coordinates.
    
    Parameters
    ----------
    lon, lat : array
        Site coordinates in degrees.
    k : int
        Number of neighbors for KNN distance estimation.
    factor : float
        Multiplier on mean KNN distance to get D_max.
    
    Returns
    -------
    D_max_km : float
        Estimated maximum search distance in km.
    """
    from scipy.spatial import KDTree
    from pystrain2.geodesy import llh2utm
    
    x, y, _ = llh2utm(lon, lat)
    coords = np.column_stack([x, y])
    
    k_use = min(k, len(coords) - 1)
    if k_use < 1:
        return 500.0  # fallback
    
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=k_use + 1)
    mean_knn = np.mean(dists[:, 1:])  # exclude self (distance 0)
    
    return factor * mean_knn


class ShenWangStrainRate:
    """Shen et al. (2015) BSSA optimal interpolation strain rate."""

    def __init__(
        self,
        vf: VelocityField,
        grid: Grid,
        distance_kind: str = "gaussian",
        spatial_kind: str = "azimuth",
        Wt: float = 24.0,
        L0: float = 0.01,
        min_sites: int = 6,
        maxdist_km: Optional[float] = None,
        projection: str = "utm",
        return_covariance: bool = False,
        auto_search_radius: bool = True,
        D_min: float = 1.0,
        D_max: float = 1000.0,
    ):
        self.vf = vf
        self.grid = grid
        self.distance_kind = distance_kind
        self.spatial_kind = spatial_kind
        self.Wt = Wt
        self.L0 = L0
        self.min_sites = min_sites
        self.maxdist_km = maxdist_km
        self.projection = projection
        self.return_covariance = return_covariance
        self.auto_search_radius = auto_search_radius
        self.D_min = D_min
        self.D_max = D_max

        # Auto-compute maxdist_km if not specified
        if self.maxdist_km is None and self.auto_search_radius:
            self.maxdist_km = estimate_search_radius(vf.lon, vf.lat)

        # --- Precompute geographic KDTree for robust spatial pre-filter ---
        # A single UTM zone is invalid for near-global datasets; use
        # longitude/latitude instead.  The approximate degree radius
        # (1° ≈ 111 km) is fine for a cheap pre-filter — the precise
        # distance_azimuth call refines the selection afterwards.
        self._geo_xy = np.column_stack([self.vf.lon, self.vf.lat])
        self._tree = KDTree(self._geo_xy)
        # Convert km search radius to approximate degrees (1 deg ≈ 111 km at equator)
        self._tree_radius_deg = (self.maxdist_km * 1.2) / 111.0

        # Precompute full-site Voronoi areas once (only if voronoi spatial weight)
        self._voronoi_areas = None
        self._voronoi_scale = 1.0
        if self.spatial_kind == "voronoi":
            raw_areas = voronoi_areas(self.vf.lon, self.vf.lat)
            # SphericalVoronoi can assign huge cells to boundary/convex-hull
            # sites.  Normalise by the *median* area (robust to outliers)
            # so that typical interior sites have Z ≈ 1, permitting
            # Σ L(D)·Z = Wt to be satisfied with a physically meaningful D
            # (Shen et al. 2015, eq. 4).
            median_area = float(np.median(raw_areas))
            if median_area <= 0:
                median_area = 1.0
            # Cap extremely large boundary cells at 100× median
            cap = median_area * 100.0
            self._voronoi_areas = np.clip(raw_areas, 0.0, cap)
            self._voronoi_scale = median_area

    def compute(self) -> StrainResult:
        """Compute strain rate at each grid point.

        Uses a geographic (lon/lat) KDTree as a fast spatial pre-filter —
        only sites within an approximate degree radius are passed to the
        precise ``distance_azimuth`` call, avoiding O(n_grid × n_sites).
        """
        try:
            from tqdm import tqdm
            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

        n_grid = len(self.grid)
        logger.info("  -> computing strain at %d grid points "
                     "(maxdist=%.0f km, spatial=%s) ...",
                     n_grid, self.maxdist_km, self.spatial_kind)

        # Precompute voronoi weights for spatial_kind="voronoi" once
        # _solve_D and final weighting both need Z — compute here once per loop.

        exx = np.full(n_grid, np.nan)
        exy = np.full(n_grid, np.nan)
        eyy = np.full(n_grid, np.nan)
        omega = np.full(n_grid, np.nan)
        e1 = np.full(n_grid, np.nan)
        e2 = np.full(n_grid, np.nan)
        azimuth = np.full(n_grid, np.nan)
        shear = np.full(n_grid, np.nan)
        dilation = np.full(n_grid, np.nan)
        sec_inv = np.full(n_grid, np.nan)
        ve_avg = np.full(n_grid, np.nan)
        vn_avg = np.full(n_grid, np.nan)
        D_values = np.full(n_grid, np.nan)
        cond_values = np.full(n_grid, np.nan)

        n_solved = 0
        idx_range = range(n_grid)
        if _has_tqdm:
            idx_range = tqdm(idx_range, desc="  grid points", unit="pt")

        for i in idx_range:
            glon = float(self.grid.lon[i])
            glat = float(self.grid.lat[i])

            # ---- FAST: geographic KDTree pre-filter ----
            # Use lon/lat coordinates for the cheap spatial filter because a
            # single UTM zone is invalid for near-global extent.  1° ≈ 111 km
            # at the equator; the buffer is intentionally generous.
            candi = self._tree.query_ball_point(
                [glon, glat], r=self._tree_radius_deg
            )
            if len(candi) < self.min_sites:
                continue

            # ---- PRECISE: pyproj exact distance only on candidates ----
            d, az = distance_azimuth(
                np.full(len(candi), glon),
                np.full(len(candi), glat),
                self.vf.lon[candi],
                self.vf.lat[candi],
            )

            # Distance cutoff
            selected = d <= self.maxdist_km
            n_selected = np.sum(selected)
            if n_selected < self.min_sites:
                continue

            d_sel = d[selected]
            az_sel = az[selected]
            sel_idx = np.array(candi)[selected]  # map back to global site indices

            # Spatial coverage weight Z (precomputed where possible)
            if self.spatial_kind == "azimuth":
                Z = _azimuth_weight(az_sel)
            elif self.spatial_kind == "voronoi":
                # Z = A_i / A_median, so typical sites have Z ≈ 1
                Z = self._voronoi_areas[sel_idx] / self._voronoi_scale
            else:
                Z = np.ones(n_selected)

            # Solve optimal D
            D_opt = _solve_D(d_sel, Z, self.Wt, self.L0, self.distance_kind,
                             D_min=self.D_min, D_max=self.D_max)
            if D_opt is None:
                continue
            D_values[i] = D_opt

            # Final weights
            L = _distance_weight(d, D_opt, self.distance_kind)
            L = np.where(L >= self.L0, L, 0.0)
            if self.spatial_kind == "azimuth":
                Z_all = _azimuth_weight(az)
            elif self.spatial_kind == "voronoi":
                Z_all = self._voronoi_areas[candi] / self._voronoi_scale
            else:
                Z_all = np.ones(len(candi))
            weights = L * Z_all

            mask = weights > 0
            if np.sum(mask) < self.min_sites:
                continue

            # Local projection around grid point
            idx_mask = np.array(candi)[mask]
            x, y, _ = local_to_origin_projection(
                self.vf.lon[idx_mask],
                self.vf.lat[idx_mask],
                (glon, glat),
                method=self.projection,
            )

            try:
                res = estimate_strain_rate(
                    x, y,
                    self.vf.ve[idx_mask],
                    self.vf.vn[idx_mask],
                    self.vf.se[idx_mask],
                    self.vf.sn[idx_mask],
                    weights=weights[mask],
                    normalize=True,
                    return_covariance=self.return_covariance,
                )
            except Exception:
                continue

            exx[i] = res["exx"]
            exy[i] = res["exy"]
            eyy[i] = res["eyy"]
            omega[i] = res["omega"]
            e1[i] = res["e1"]
            e2[i] = res["e2"]
            azimuth[i] = res["azimuth"]
            shear[i] = res["shear"]
            dilation[i] = res["dilation"]
            sec_inv[i] = res["sec_inv"]
            ve_avg[i] = res["dx"]
            vn_avg[i] = res["dy"]
            cond_values[i] = res["condition_number"]
            n_solved += 1

        logger.info("  -> solved %d / %d grid points", n_solved, n_grid)

        return StrainResult(
            lon=self.grid.lon,
            lat=self.grid.lat,
            exx=exx,
            exy=exy,
            eyy=eyy,
            omega=omega,
            e1=e1,
            e2=e2,
            azimuth=azimuth,
            shear=shear,
            dilation=dilation,
            sec_inv=sec_inv,
            ve=ve_avg,
            vn=vn_avg,
            meta={
                "D_values": D_values,
                "condition_numbers": cond_values,
                "auto_search_radius_km": self.maxdist_km,
            },
        )
