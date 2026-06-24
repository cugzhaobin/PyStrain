"""L-curve search for optimal smoothing factor (Wang 2012)."""

from typing import Tuple
import numpy as np
from scipy import sparse

from pystrain.wang2012.solver import solve_wang2012


def lcurve_search(
    G: sparse.spmatrix,
    S: sparse.spmatrix,
    v_gps: np.ndarray,
    sigma_e: np.ndarray,
    sigma_n: np.ndarray,
    smooth_range: Tuple[float, float] = (-2.2, -0.8),
    smooth_step: float = 0.2,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Search for optimal smoothing factor via the L-curve criterion.

    Sweeps ``smooth_factor`` over a log-spaced range, solves the system at
    each value, then identifies the corner of the L-curve (maximum discrete
    curvature in log–log space).

    Parameters
    ----------
    G : sparse matrix
        Interpolation matrix (2*n_gps × 2*n_vtx).
    S : sparse matrix
        Smoothing matrix (2*n_vtx × 2*n_vtx).
    v_gps : np.ndarray, shape (2*n_gps,)
        Stacked observed GPS velocities [ve; vn].
    sigma_e, sigma_n : np.ndarray, shape (n_gps,)
        Velocity uncertainties.
    smooth_range : (log10_min, log10_max)
        Decade range for the sweep (log₁₀ of smooth_factor).
    smooth_step : float
        Step size in log₁₀ units.

    Returns
    -------
    best_factor : float
        Smooth factor at the L-curve corner.
    all_factors : np.ndarray
        All tested smooth factors.
    all_wrss : np.ndarray
        Weighted residuals at each factor.
    all_roughness : np.ndarray
        Roughness at each factor.
    """
    log_min, log_max = smooth_range
    # Inclusive range
    log_vals = np.arange(log_min, log_max + smooth_step * 0.5, smooth_step)
    all_factors = 10.0 ** log_vals

    all_wrss = np.empty(len(all_factors))
    all_roughness = np.empty(len(all_factors))

    for k, sf in enumerate(all_factors):
        _, wrss, roughness = solve_wang2012(G, S, v_gps, sigma_e, sigma_n, sf)
        all_wrss[k] = wrss
        all_roughness[k] = roughness

    # L-curve corner: maximum discrete curvature in log-log space
    best_idx = _find_lcurve_corner(all_wrss, all_roughness)
    best_factor = float(all_factors[best_idx])

    return best_factor, all_factors, all_wrss, all_roughness


def _find_lcurve_corner(wrss: np.ndarray, roughness: np.ndarray) -> int:
    """Return the index of maximum curvature on the L-curve (log-log space).

    Uses the discrete Menger curvature at each interior point of the log-log
    curve.  Falls back to the midpoint if fewer than 3 points are available.
    """
    n = len(wrss)
    if n < 3:
        return n // 2

    log_x = np.log10(np.where(wrss > 0, wrss, 1e-30))
    log_y = np.log10(np.where(roughness > 0, roughness, 1e-30))

    # Normalise to [0, 1] for numerical stability
    x = (log_x - log_x.min()) / (log_x.ptp() + 1e-30)
    y = (log_y - log_y.min()) / (log_y.ptp() + 1e-30)

    curvature = np.zeros(n)
    for i in range(1, n - 1):
        # Menger curvature for three consecutive points
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i],     y[i]
        x3, y3 = x[i + 1], y[i + 1]

        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        denom = a * b * c
        curvature[i] = area2 / denom if denom > 1e-30 else 0.0

    return int(np.argmax(curvature))
