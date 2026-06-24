"""Tests for Delaunay strain-rate algorithm."""

import numpy as np
import pytest

from pystrain2.data import VelocityField
from pystrain2.tri import DelaunayStrainRate


def test_delaunay_strain_pure_extension():
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
        names=np.array([f"D{i:02d}" for i in range(n)]),
    )
    estimator = DelaunayStrainRate(vf, min_angle_deg=5.0)
    result = estimator.compute()
    assert len(result) > 0
    # Mean exx should approximate 50 nstrain/yr
    assert np.mean(result.exx) == pytest.approx(50.0, rel=0.3)
    assert np.mean(result.eyy) == pytest.approx(0.0, abs=10.0)


def test_delaunay_strain_no_valid_triangles():
    vf = VelocityField(
        lon=np.array([112.0, 113.0]),
        lat=np.array([40.0, 41.0]),
        ve=np.array([1.0, 1.0]),
        vn=np.array([0.0, 0.0]),
        se=np.ones(2),
        sn=np.ones(2),
        rho=np.zeros(2),
        names=np.array(["a", "b"]),
    )
    with pytest.raises(ValueError):
        DelaunayStrainRate(vf).compute()
