"""Least-squares solver for Wang (2012) strain-rate estimation."""

from typing import Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_wang2012(
    G: sparse.spmatrix,
    S: sparse.spmatrix,
    v_gps: np.ndarray,
    sigma_e: np.ndarray,
    sigma_n: np.ndarray,
    smooth_factor: float,
) -> Tuple[np.ndarray, float, float]:
    """Solve the Wang (2012) regularised least-squares problem.

    Minimises:
        || W^{1/2} (G v - d) ||^2  +  smooth_factor^2 || S v ||^2

    via the normal equations:
        (G^T W G + smooth_factor^2 S^T S) v = G^T W d

    Parameters
    ----------
    G : sparse matrix, shape (2*n_gps, 2*n_vtx)
        Interpolation matrix (east + north blocks).
    S : sparse matrix, shape (2*n_vtx, 2*n_vtx)
        Smoothing matrix (east + north blocks).
    v_gps : np.ndarray, shape (2*n_gps,)
        Observed GPS velocities stacked as [ve_all; vn_all].
    sigma_e, sigma_n : np.ndarray, shape (n_gps,)
        Velocity uncertainties (mm/yr or consistent units).
    smooth_factor : float
        Smoothing regularisation factor.

    Returns
    -------
    v_mesh : np.ndarray, shape (2*n_vtx,)
        Solved mesh-node velocities [ve_vtx; vn_vtx].
    wrss : float
        Weighted root-sum-of-squares residual (data misfit measure).
    roughness : float
        Solution roughness (Sv measure).
    """
    n_gps = len(sigma_e)

    # Build weight vector: 1/sigma^2
    sigma_e = np.asarray(sigma_e, dtype=float)
    sigma_n = np.asarray(sigma_n, dtype=float)
    # Replace zero or tiny sigmas with a small floor
    sigma_e = np.where(sigma_e > 1e-30, sigma_e, 1e-3)
    sigma_n = np.where(sigma_n > 1e-30, sigma_n, 1e-3)
    w = np.concatenate([1.0 / sigma_e ** 2, 1.0 / sigma_n ** 2])

    W = sparse.diags(w, 0, format="csr")

    # Normal equations
    GtW = G.T @ W
    N_mat = GtW @ G + (smooth_factor ** 2) * (S.T @ S)
    rhs = GtW @ v_gps

    # Ensure matrix is in CSC / CSR for spsolve
    N_csc = N_mat.tocsc()

    v_mesh = spsolve(N_csc, rhs)

    # Residuals
    res = G @ v_mesh - v_gps
    n_obs = len(v_gps)
    wrss = float(np.sqrt(np.dot(w * res, res) / n_obs))

    # Roughness
    Sv = S @ v_mesh
    n_smooth = S.shape[0]
    roughness = float(np.sqrt(np.dot(Sv, Sv) / n_smooth))

    return v_mesh, wrss, roughness
