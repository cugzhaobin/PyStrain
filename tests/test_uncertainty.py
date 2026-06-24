"""Tests for uncertainty propagation."""

import numpy as np

from pystrain2.data import VelocityField
from pystrain2.uncertainty import monte_carlo_strain_uncertainty
from pystrain2.tri import DelaunayStrainRate


def test_monte_carlo_uncertainty_shape():
    lon, lat = np.meshgrid(
        np.linspace(112.0, 113.0, 5),
        np.linspace(40.0, 41.0, 5),
    )
    lon = lon.ravel()
    lat = lat.ravel()
    n = len(lon)
    from pystrain2.geodesy import llh2utm
    x, y, _ = llh2utm(lon, lat)
    x0 = np.mean(x)
    eps = 0.05
    ve = eps * (x - x0)
    vn = np.zeros(n)
    vf = VelocityField(
        lon=lon, lat=lat, ve=ve, vn=vn,
        se=np.full(n, 0.1), sn=np.full(n, 0.1), rho=np.zeros(n),
        names=np.array([f"U{i:02d}" for i in range(n)]),
    )
    estimator = DelaunayStrainRate(vf, min_angle_deg=5.0)
    result = estimator.compute()

    def estimate(vf_pert):
        return DelaunayStrainRate(vf_pert, min_angle_deg=5.0).compute()

    std = monte_carlo_strain_uncertainty(estimate, vf, n_iterations=20, seed=1)
    assert "exx" in std
    assert len(std["exx"]) == len(result)
    assert np.all(std["exx"] >= 0)
