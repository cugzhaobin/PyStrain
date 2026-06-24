"""Outlier detection for GNSS velocity fields."""

import logging
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy.spatial import Delaunay, KDTree

from pystrain.data import VelocityField

logger = logging.getLogger("pystrain")


def knn_prescreening(
    vf: VelocityField,
    k_neighbors: int = 8,
    mad_factor: float = 3.5,
    metric: str = "utm",
) -> Tuple[np.ndarray, np.ndarray]:
    """KNN + MAD pre-screening on projected coordinates.

    Parameters
    ----------
    metric : {"utm", "lonlat"}
        Use UTM km (recommended) or raw degrees for neighbor search.

    Returns
    -------
    outlier_mask : np.ndarray
        True for sites flagged as outliers.
    suspended_mask : np.ndarray
        True for sites with too few neighbors.
    """
    from pystrain.geodesy import llh2utm

    n = len(vf)
    if metric == "utm":
        x, y, _ = llh2utm(vf.lon, vf.lat)
        coords = np.column_stack([x, y])
    else:
        coords = np.column_stack([vf.lon, vf.lat])

    k = min(k_neighbors, n - 1)
    if k < 3:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)
    # Exclude self
    neighbor_idx = idxs[:, 1:]

    ve_med = np.median(vf.ve[neighbor_idx], axis=1)
    vn_med = np.median(vf.vn[neighbor_idx], axis=1)
    mad_e = np.median(np.abs(vf.ve[neighbor_idx] - ve_med[:, None]), axis=1)
    mad_n = np.median(np.abs(vf.vn[neighbor_idx] - vn_med[:, None]), axis=1)

    mad_e[mad_e == 0] = 1e-9
    mad_n[mad_n == 0] = 1e-9

    dev = np.sqrt(
        ((vf.ve - ve_med) / mad_e) ** 2 +
        ((vf.vn - vn_med) / mad_n) ** 2
    )
    threshold = mad_factor * np.sqrt(2)

    outlier_mask = dev > threshold
    suspended_mask = np.sum(neighbor_idx >= 0, axis=1) < 3

    return outlier_mask, suspended_mask


def compute_residuals_triangulation(
    tri: "Delaunay",
    xy: np.ndarray,
    good_triangles: np.ndarray,
    vf: VelocityField,
    site_indices: np.ndarray,
) -> Dict[int, Dict]:
    """Compute leave-one-out velocity residuals for sites in valid triangles.

    Parameters
    ----------
    tri : Delaunay
        Triangulation built on the subset ``site_indices``.
    xy : np.ndarray
        Projected coordinates of the subset sites (local indices).
    good_triangles : np.ndarray
        Boolean mask of valid triangles (in local index space).
    vf : VelocityField
        Full velocity field.
    site_indices : np.ndarray
        Global indices of the sites in the triangulation subset.

    Returns
    -------
    residuals : dict
        Mapping from global site index to residual info.
    """
    global_simplices = site_indices[tri.simplices]
    good_global_simplices = global_simplices[good_triangles]

    residuals = {}
    for global_idx in site_indices:
        # Find all valid triangles containing this global site
        mask = np.any(good_global_simplices == global_idx, axis=1)
        if not np.any(mask):
            continue

        pred_ve_list = []
        pred_vn_list = []
        for simplex in good_global_simplices[mask]:
            # Other two vertices
            other = simplex[simplex != global_idx]
            if len(other) != 2:
                continue
            # Predict velocity as average of neighbors (simple leave-one-out)
            pred_ve_list.append(np.mean(vf.ve[other]))
            pred_vn_list.append(np.mean(vf.vn[other]))

        if not pred_ve_list:
            continue

        pred_ve = np.median(pred_ve_list)
        pred_vn = np.median(pred_vn_list)
        res_e = vf.ve[global_idx] - pred_ve
        res_n = vf.vn[global_idx] - pred_vn
        res_norm = np.sqrt(res_e**2 + res_n**2)

        residuals[global_idx] = {
            "res_e": res_e,
            "res_n": res_n,
            "res_norm": res_norm,
        }

    return residuals


def iqr_outlier_detection(
    residuals: Dict[int, Dict],
    iqr_factor: float = 1.5,
    min_residual: float = 0.5,
) -> List[int]:
    """Identify sites whose residual norm exceeds Q3 + factor*IQR."""
    if not residuals:
        return []

    norms = np.array([v["res_norm"] for v in residuals.values()])
    q1, q3 = np.percentile(norms, [25, 75])
    iqr = q3 - q1
    threshold = q3 + iqr_factor * iqr

    outlier_sites = []
    for idx, info in residuals.items():
        if info["res_norm"] > threshold and info["res_norm"] > min_residual:
            outlier_sites.append(idx)

    return outlier_sites


def loo_strain_outlier_detection(
    vf: "VelocityField",
    maxdist_km: float = 200.0,
    min_sites: int = 8,
    min_residual_mm: float = 3.0,
    projection: str = "utm",
    show_progress: bool = True,
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Leave-one-out strain-based outlier detection.

    For each GPS station, the strain rate is estimated at that station's
    location using all **other** stations within *maxdist_km*.  The
    predicted velocity from the strain solution is compared with the
    observed velocity.  Stations whose residual magnitude exceeds
    *min_residual_mm* are flagged as outliers.

    Each point is judged independently — no population statistics (IQR,
    MAD) are involved, because the LOO residual is already a direct
    measure of how well the station agrees with its neighbours' strain
    field.

    This is more rigorous than the KNN+triangulation approach because:

    - It uses the full strain-rate model (6-parameter least squares),
      not simple neighbour averaging.
    - The target station's own observation is excluded from the inversion
      (true leave-one-out).
    - Distance-based Gaussian weighting follows the same scheme as the
      Shen et al. (2015) strain-rate estimator.

    Parameters
    ----------
    vf : VelocityField
        Input velocity field.
    maxdist_km : float
        Maximum distance (km) to include a station as a neighbour.
    min_sites : int
        Minimum number of neighbours required for a valid estimate.
    min_residual_mm : float
        Direct residual threshold (mm/yr).  Stations with
        |v_obs - v_pred| > min_residual_mm are flagged.
    projection : str
        Local projection method (``"utm"`` or ``"polyconic"``).
    show_progress : bool
        Display a progress bar via tqdm (if installed).

    Returns
    -------
    outlier_mask : (N,) bool ndarray
        True for flagged outliers.
    residuals_mm : (N,) float ndarray
        Residual norm (mm/yr) for each station.  NaN where estimation failed.
    predicted_ve, predicted_vn : (N,) ndarray
        Predicted velocities (mm/yr) at each station.
    """
    from pystrain.geodesy import distance_azimuth, local_to_origin_projection
    from pystrain.grid.shen_wang import _distance_weight
    from pystrain.strain.lsq import estimate_strain_rate

    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

        def _tqdm(iterable, **kwargs):  # type: ignore[no-redef]
            return iterable

    n = len(vf)
    residuals = np.full(n, np.nan)
    pred_ve = np.full(n, np.nan)
    pred_vn = np.full(n, np.nan)

    for i in _tqdm(range(n), desc="  LOO strain outlier",
                    disable=not (show_progress and _has_tqdm), unit="site"):
        # ---- distances to all other stations ----
        d, _ = distance_azimuth(
            np.full(n, vf.lon[i]),
            np.full(n, vf.lat[i]),
            vf.lon, vf.lat,
        )
        # Exclude self
        mask = (d <= maxdist_km) & (np.arange(n) != i)
        if np.sum(mask) < min_sites:
            continue

        # ---- distance weight ----
        sorted_d = np.sort(d[mask])
        R = 1.5 * sorted_d[min(min_sites - 1, len(sorted_d) - 1)]
        weights = _distance_weight(d[mask], R, "gaussian")

        # ---- local projection ----
        x, y, _ = local_to_origin_projection(
            vf.lon[mask], vf.lat[mask],
            (vf.lon[i], vf.lat[i]),
            method=projection,
        )

        # ---- strain estimation at station i (leave-one-out) ----
        try:
            res = estimate_strain_rate(
                x, y,
                vf.ve[mask], vf.vn[mask],
                vf.se[mask], vf.sn[mask],
                weights=weights,
                normalize=True,
            )
        except Exception:
            continue

        # Predicted velocity at origin (= station i)
        pred_ve[i] = res["dx"]
        pred_vn[i] = res["dy"]
        residuals[i] = np.sqrt(
            (vf.ve[i] - res["dx"]) ** 2 + (vf.vn[i] - res["dy"]) ** 2
        )

    # ---- Direct threshold (each point independent) ----
    valid = np.isfinite(residuals)
    outlier_mask = np.zeros(n, dtype=bool)
    if np.any(valid):
        outlier_mask[valid] = residuals[valid] > min_residual_mm

    return outlier_mask, residuals, pred_ve, pred_vn


def iterative_outlier_removal(
    vf: VelocityField,
    triangulation_fn: Callable,
    k_neighbors: int = 8,
    mad_factor: float = 3.5,
    iqr_factor: float = 1.5,
    max_iterations: int = 5,
    min_residual: float = 0.5,
    min_sites: int = 6,
) -> Tuple[VelocityField, List[Dict]]:
    """Iteratively remove outliers using KNN + residual IQR.

    Parameters
    ----------
    vf : VelocityField
        Input velocity field.
    triangulation_fn : callable
        Function taking (lon, lat, site_indices) -> (tri, good_triangles, xy).

    Returns
    -------
    clean_vf : VelocityField
        Velocity field with outliers removed.
    outlier_history : list of dict
        Record of removed sites.
    """
    n = len(vf)
    keep_mask = np.ones(n, dtype=bool)
    outlier_history: List[Dict] = []

    for it in range(max_iterations):
        n_current = keep_mask.sum()
        if n_current < min_sites:
            break

        global_idx = np.where(keep_mask)[0]
        current_vf = vf.subset(keep_mask)

        # KNN pre-screening
        knn_mask, _ = knn_prescreening(current_vf, k_neighbors=k_neighbors)
        knn_global_outliers = set(global_idx[knn_mask])

        # Triangulate non-KNN-outlier sites
        sub_keep = ~knn_mask
        sub_global_idx = global_idx[sub_keep]

        if len(sub_global_idx) < 3:
            break

        try:
            tri, good_triangles, xy = triangulation_fn(
                vf.lon, vf.lat, sub_global_idx
            )
        except Exception:
            break

        # Residual IQR with explicit local-to-global mapping
        residuals = compute_residuals_triangulation(
            tri, xy, good_triangles, vf, sub_global_idx
        )
        iqr_outliers = iqr_outlier_detection(
            residuals, iqr_factor=iqr_factor, min_residual=min_residual
        )
        iqr_global_outliers = set(iqr_outliers)

        all_new = knn_global_outliers | iqr_global_outliers
        new_outliers = [s for s in all_new if keep_mask[s]]

        if not new_outliers:
            break

        logger.info("  -> iteration %d: removing %d outliers (KNN=%d, IQR=%d)",
                     it + 1, len(new_outliers),
                     len(knn_global_outliers & set(new_outliers)),
                     len(iqr_global_outliers & set(new_outliers)))

        for s in new_outliers:
            reason = []
            if s in knn_global_outliers:
                reason.append("KNN")
            if s in iqr_global_outliers:
                res = residuals.get(s, {})
                reason.append(f"IQR(res={res.get('res_norm', 0):.2f})")
            outlier_history.append({
                "name": vf.names[s],
                "lon": vf.lon[s],
                "lat": vf.lat[s],
                "residual": residuals.get(s, {}).get("res_norm", 0.0),
                "reason": "+".join(reason),
                "iteration": it,
            })
            keep_mask[s] = False

    return vf.subset(keep_mask), outlier_history
