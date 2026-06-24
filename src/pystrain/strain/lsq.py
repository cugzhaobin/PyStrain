"""Weighted least-squares strain-rate estimation core."""

from typing import Dict, Optional
import numpy as np

from pystrain.strain.tensor import principal_strain, strain_invariants


def _build_design_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build 2N x 6 design matrix for the local velocity model.

    Model:
        ve = Ux + ω·y + τxx·x + τxy·y
        vn = Uy - ω·x + τxy·x + τyy·y

    Columns: [Ux, Uy, ω, τxx, τxy, τyy].
    """
    n = len(x)
    G = np.zeros((2 * n, 6))
    for i in range(n):
        # East velocity equation
        G[2 * i, 0] = 1.0
        G[2 * i, 2] = y[i]
        G[2 * i, 3] = x[i]
        G[2 * i, 4] = y[i]
        # North velocity equation
        G[2 * i + 1, 1] = 1.0
        G[2 * i + 1, 2] = -x[i]
        G[2 * i + 1, 4] = x[i]
        G[2 * i + 1, 5] = y[i]
    return G


def estimate_strain_rate(
    x: np.ndarray,
    y: np.ndarray,
    ve: np.ndarray,
    vn: np.ndarray,
    se: Optional[np.ndarray] = None,
    sn: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    normalize: bool = True,
    return_covariance: bool = False,
    debug: bool = False,
) -> Dict:
    """Estimate strain rate from local velocities via weighted least squares.

    Parameters
    ----------
    x, y : np.ndarray
        Local coordinates in km.
    ve, vn : np.ndarray
        East/north velocities in mm/yr.
    se, sn : np.ndarray, optional
        Velocity uncertainties in mm/yr. If None, unit weights are used.
    weights : np.ndarray, optional
        Additional per-station weights (e.g., Shen & Wang spatial weights).
    normalize : bool
        If True, column-normalize the design matrix to avoid large-number-
        eating-small-number problems.
    return_covariance : bool
        If True, include parameter covariance matrix in the returned dict.
    debug : bool
        If True, print condition number information.

    Returns
    -------
    dict with keys:
        dx, dy : mean velocities (mm/yr)
        exx, exy, eyy, omega : strain/rotation rates (nstrain/yr and nrad/yr)
        e1, e2, azimuth : principal strains and azimuth
        shear, dilation, sec_inv : strain invariants
        condition_number : condition number of normal matrix
        param_cov : (6,6) covariance matrix (if return_covariance=True)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ve = np.asarray(ve, dtype=float)
    vn = np.asarray(vn, dtype=float)
    n = len(x)

    if se is None:
        se = np.ones(n)
    if sn is None:
        sn = np.ones(n)
    se = np.asarray(se, dtype=float)
    sn = np.asarray(sn, dtype=float)

    G = _build_design_matrix(x, y)
    U = np.empty(2 * n)
    U[0::2] = ve
    U[1::2] = vn

    # Observation weight matrix: inverse variance
    W = np.empty(2 * n)
    W[0::2] = 1.0 / (se ** 2)
    W[1::2] = 1.0 / (sn ** 2)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        W[0::2] *= weights
        W[1::2] *= weights

    # Column normalization to prevent large-number-eating-small-number
    if normalize:
        col_norms = np.sqrt(np.sum(G ** 2, axis=0))
        col_norms[col_norms == 0] = 1.0
        S_inv = np.diag(1.0 / col_norms)
        G_norm = G @ S_inv
        N = G_norm.T @ (W[:, None] * G_norm)
        rhs = G_norm.T @ (W * U)
    else:
        N = G.T @ (W[:, None] * G)
        rhs = G.T @ (W * U)

    cond = np.linalg.cond(N)
    if debug:
        print(f"Normal matrix condition number: {cond:.3e}")

    try:
        L = np.linalg.solve(N, rhs)
    except np.linalg.LinAlgError:
        L = np.linalg.lstsq(N, rhs, rcond=None)[0]

    if normalize:
        L = S_inv @ L
        if return_covariance:
            param_cov = S_inv @ np.linalg.pinv(N) @ S_inv
        else:
            param_cov = None
    else:
        if return_covariance:
            param_cov = np.linalg.pinv(N)
        else:
            param_cov = None

    dx, dy, omega_raw, exx_raw, exy_raw, eyy_raw = L

    # Convert from mm/(km·yr) to nstrain/yr (1 nstrain = 1e-9)
    unit_factor = 1000.0
    exx = exx_raw * unit_factor
    exy = exy_raw * unit_factor
    eyy = eyy_raw * unit_factor
    omega = omega_raw * unit_factor

    e1, e2, azimuth = principal_strain(exx, exy, eyy)
    dilation, shear, sec_inv = strain_invariants(e1, e2)

    result = {
        "dx": float(dx),
        "dy": float(dy),
        "exx": float(exx),
        "exy": float(exy),
        "eyy": float(eyy),
        "omega": float(omega),
        "e1": float(e1),
        "e2": float(e2),
        "azimuth": float(azimuth),
        "shear": float(shear),
        "dilation": float(dilation),
        "sec_inv": float(sec_inv),
        "condition_number": float(cond),
    }

    if return_covariance:
        # Transform covariance to nstrain units
        cov_scaled = param_cov.copy()
        for idx in (3, 4, 5, 2):  # strain and rotation parameters
            cov_scaled[idx, :] *= unit_factor
            cov_scaled[:, idx] *= unit_factor
        result["param_cov"] = cov_scaled

    return result
