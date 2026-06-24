"""Strain tensor decomposition utilities."""

from typing import Tuple
import numpy as np


def velocity_gradient_to_strain(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert 2x2 velocity gradient tensor to strain-rate components.

    Parameters
    ----------
    L : np.ndarray
        Velocity gradient tensor with shape (..., 2, 2) where
        L[..., 0, 0] = dve/dx, L[..., 0, 1] = dve/dy,
        L[..., 1, 0] = dvn/dx, L[..., 1, 1] = dvn/dy.

    Returns
    -------
    e_ee, e_en, e_nn, omega : np.ndarray
        Strain-rate components and rotation rate in same raw units as L.
    """
    e_ee = L[..., 0, 0]
    e_nn = L[..., 1, 1]
    e_en = 0.5 * (L[..., 0, 1] + L[..., 1, 0])
    omega = 0.5 * (L[..., 1, 0] - L[..., 0, 1])
    return e_ee, e_en, e_nn, omega


def principal_strain(e_ee: np.ndarray, e_en: np.ndarray, e_nn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal strains and azimuth of maximum principal strain.

    Azimuth is measured from East counter-clockwise and normalized to [0, 180).
    """
    e_ee = np.asarray(e_ee, dtype=float)
    e_en = np.asarray(e_en, dtype=float)
    e_nn = np.asarray(e_nn, dtype=float)

    # Build 2x2 strain tensor
    tensor = np.stack([
        np.stack([e_ee, e_en], axis=-1),
        np.stack([e_en, e_nn], axis=-1),
    ], axis=-2)

    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(tensor)
    # eigh returns ascending order; e1 is largest (extensional)
    e1 = eigvals[..., 1]
    e2 = eigvals[..., 0]

    # Azimuth from East (x-axis) counter-clockwise
    vx = eigvecs[..., 0, 1]
    vy = eigvecs[..., 1, 1]
    azimuth = np.rad2deg(np.arctan2(vy, vx))
    azimuth = np.mod(azimuth, 180.0)

    return e1, e2, azimuth


def strain_invariants(e1: np.ndarray, e2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dilatation, max shear, and second invariant from principal strains."""
    dilation = e1 + e2
    max_shear = 0.5 * (e1 - e2)
    sec_inv = np.sqrt(e1**2 + e2**2)
    return dilation, max_shear, sec_inv
