"""Tests for user-defined point strain rate."""

import numpy as np
import pytest

from pystrain2.data import VelocityField
from pystrain2.user import UserStrainRate


def test_user_strain_point(tmp_path):
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
    pts_file = tmp_path / "user_points.txt"
    pts_file.write_text("112.5 40.5 CENTER\n", encoding="utf-8")
    estimator = UserStrainRate(str(pts_file), vf, maxdist_km=200.0, min_sites=6)
    result = estimator.compute()
    assert len(result) == 1
    assert np.isfinite(result.exx[0])
    assert result.exx[0] == pytest.approx(50.0, rel=0.5)
