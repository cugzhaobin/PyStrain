"""Tests for grid generation and Shen & Wang algorithm."""

import numpy as np
import pytest

from pystrain2.data import VelocityField
from pystrain2.grid import Grid, ShenWangStrainRate


def test_grid_generation():
    grid = Grid(112.0, 113.0, 40.0, 41.0, 0.5, 0.5, stagger=False)
    assert len(grid) == 9  # 3x3
    assert grid.lon[0] == 112.0


def test_grid_stagger():
    grid = Grid(112.0, 113.0, 40.0, 41.0, 0.5, 0.5, stagger=True)
    # Second row should be offset by 0.25
    assert grid.lon[3] == pytest.approx(112.25)


def test_shen_wang_grid_basic():
    # Create a dense synthetic field
    lon, lat = np.meshgrid(
        np.linspace(112.0, 113.0, 7),
        np.linspace(40.0, 41.0, 7),
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
        lon=lon,
        lat=lat,
        ve=ve,
        vn=vn,
        se=np.full(n, 0.1),
        sn=np.full(n, 0.1),
        rho=np.zeros(n),
        names=np.array([f"G{i:02d}" for i in range(n)]),
    )
    grid = Grid(112.2, 112.8, 40.2, 40.8, 0.3, 0.3, stagger=False)
    estimator = ShenWangStrainRate(vf, grid, Wt=6.0, min_sites=6, maxdist_km=200.0)
    result = estimator.compute()
    assert len(result) == len(grid)
    # At least some grid points should have valid results
    valid = np.isfinite(result.exx)
    assert np.sum(valid) > 0
    # Average exx should be close to 50 nstrain/yr
    assert np.nanmean(result.exx) == pytest.approx(50.0, rel=0.5)
