"""Shared fixtures for PyStrain2 tests."""

import numpy as np
import pytest

from pystrain2.data import VelocityField


@pytest.fixture
def synthetic_velocity_field() -> VelocityField:
    """A 4x4 synthetic velocity field with a simple E-W extensional strain."""
    lon, lat = np.meshgrid(
        np.linspace(112.0, 113.0, 4),
        np.linspace(40.0, 41.0, 4),
    )
    lon = lon.ravel()
    lat = lat.ravel()
    n = len(lon)

    # Reference at center, velocities in mm/yr
    # Pure E-W extension: ve = eps * (x - x0), vn = 0
    # eps = 50 nstrain/yr = 5e-8 -> in mm/(km.yr) raw = 5e-11
    # Choose a larger value so residuals dominate noise.
    from pystrain2.geodesy import llh2utm
    x, y, _ = llh2utm(lon, lat)
    x0 = np.mean(x)
    eps = 0.05  # mm/(km.yr) -> 50 nstrain/yr
    ve = eps * (x - x0)
    vn = np.zeros(n)
    se = np.full(n, 0.1)
    sn = np.full(n, 0.1)
    rho = np.zeros(n)
    names = np.array([f"S{i:02d}" for i in range(n)])
    return VelocityField(lon=lon, lat=lat, ve=ve, vn=vn, se=se, sn=sn, rho=rho, names=names)


@pytest.fixture
def rigid_rotation_field() -> VelocityField:
    """A small rigid-body rotation velocity field (zero strain)."""
    lon, lat = np.meshgrid(
        np.linspace(112.0, 113.0, 4),
        np.linspace(40.0, 41.0, 4),
    )
    lon = lon.ravel()
    lat = lat.ravel()
    n = len(lon)

    from pystrain2.geodesy import llh2utm
    x, y, _ = llh2utm(lon, lat)
    x0, y0 = np.mean(x), np.mean(y)
    omega = 50.0  # nrad/yr -> raw = 5e-8 in mm/(km.yr)
    ve = -omega * (y - y0) / 1000.0
    vn = omega * (x - x0) / 1000.0
    se = np.full(n, 0.1)
    sn = np.full(n, 0.1)
    rho = np.zeros(n)
    names = np.array([f"R{i:02d}" for i in range(n)])
    return VelocityField(lon=lon, lat=lat, ve=ve, vn=vn, se=se, sn=sn, rho=rho, names=names)


@pytest.fixture
def gmt8_text(tmp_path):
    """Write a small GMT8 velocity file and return its path."""
    content = """# lon lat Ve Vn Se Sn Rho Site
112.61 41.27 3.39 -0.84 0.35 0.35 0.00 A001
112.56 40.89 3.26 -1.20 0.28 0.21 0.00 A002
112.48 40.53 3.12 -0.79 0.49 0.36 0.00 A003
112.35 40.17 3.17 -1.36 0.29 0.29 0.00 A004
113.13 41.02 3.01 -1.38 0.25 0.25 0.00 A005
113.21 40.79 2.93 -0.93 0.22 0.16 0.00 A006
113.20 40.45 3.44 -0.96 0.19 0.19 0.00 A007
112.73 40.02 3.85 -1.17 0.24 0.26 0.00 A008
113.24 40.14 3.05 -1.08 0.27 0.20 0.00 A009
"""
    path = tmp_path / "vel.gmtvec"
    path.write_text(content, encoding="utf-8")
    return str(path)
