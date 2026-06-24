"""Tests for Velmap-style shape-function strain-rate algorithm."""

import numpy as np
import pytest

from pystrain2.data import VelocityField
from pystrain2.velmap import VelmapStrainRate


def test_velmap_strain_pure_extension():
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
        names=np.array([f"V{i:02d}" for i in range(n)]),
    )
    estimator = VelmapStrainRate(vf, min_angle_deg=5.0, smooth_iter=0)
    result = estimator.compute()
    assert len(result) > 0
    assert np.mean(result.exx) == pytest.approx(50.0, rel=0.3)


def test_velmap_strain_rigid_rotation():
    lon, lat = np.meshgrid(
        np.linspace(112.0, 113.0, 5),
        np.linspace(40.0, 41.0, 5),
    )
    lon = lon.ravel()
    lat = lat.ravel()
    n = len(lon)
    from pystrain2.geodesy import llh2utm
    x, y, _ = llh2utm(lon, lat)
    x0, y0 = np.mean(x), np.mean(y)
    omega = 50.0  # nrad/yr
    ve = -omega * (y - y0) / 1000.0
    vn = omega * (x - x0) / 1000.0
    vf = VelocityField(
        lon=lon, lat=lat, ve=ve, vn=vn,
        se=np.full(n, 0.1), sn=np.full(n, 0.1), rho=np.zeros(n),
        names=np.array([f"V{i:02d}" for i in range(n)]),
    )
    estimator = VelmapStrainRate(vf, min_angle_deg=5.0, smooth_iter=0)
    result = estimator.compute()
    assert len(result) > 0
    assert np.mean(result.exx) == pytest.approx(0.0, abs=10.0)
    assert np.mean(result.eyy) == pytest.approx(0.0, abs=10.0)
    assert np.abs(np.mean(result.omega)) > 10.0
