"""Uncertainty propagation for strain-rate estimates."""

from typing import Callable, Dict
import numpy as np

from pystrain.data import StrainResult, VelocityField


def monte_carlo_strain_uncertainty(
    estimator: Callable[[VelocityField], StrainResult],
    vf: VelocityField,
    n_iterations: int = 200,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Estimate strain-rate standard deviations via Monte Carlo.

    Parameters
    ----------
    estimator : callable
        Function taking a VelocityField and returning a StrainResult.
    vf : VelocityField
        Original velocity field with uncertainties.
    n_iterations : int
        Number of Monte Carlo iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Standard deviations for exx, exy, eyy, omega, e1, e2, shear, dilation, sec_inv.
    """
    rng = np.random.default_rng(seed)
    n = len(vf)

    # Cholesky of per-site covariance
    perturbations_e = np.empty((n_iterations, n))
    perturbations_n = np.empty((n_iterations, n))

    for i in range(n):
        cov = np.array([
            [vf.se[i] ** 2, vf.rho[i] * vf.se[i] * vf.sn[i]],
            [vf.rho[i] * vf.se[i] * vf.sn[i], vf.sn[i] ** 2],
        ])
        # Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-30, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        noise = rng.standard_normal((n_iterations, 2))
        pert = noise @ L.T
        perturbations_e[:, i] = pert[:, 0]
        perturbations_n[:, i] = pert[:, 1]

    results = {
        "exx": [], "exy": [], "eyy": [], "omega": [],
        "e1": [], "e2": [], "shear": [], "dilation": [], "sec_inv": [],
    }

    for it in range(n_iterations):
        vf_pert = VelocityField(
            lon=vf.lon,
            lat=vf.lat,
            ve=vf.ve + perturbations_e[it],
            vn=vf.vn + perturbations_n[it],
            se=vf.se,
            sn=vf.sn,
            rho=vf.rho,
            names=vf.names,
            meta=vf.meta,
        )
        try:
            res = estimator(vf_pert)
        except Exception:
            continue
        for key in results:
            results[key].append(getattr(res, key))

    stds = {}
    for key, arr_list in results.items():
        if not arr_list:
            stds[key] = None
            continue
        stacked = np.array(arr_list)
        stds[key] = np.std(stacked, axis=0, ddof=1)

    return stds
