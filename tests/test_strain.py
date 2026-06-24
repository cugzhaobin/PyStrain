"""Tests for strain tensor and least-squares estimation."""

import numpy as np
import pytest

from pystrain2.geodesy import llh2utm
from pystrain2.strain.lsq import estimate_strain_rate
from pystrain2.strain.tensor import (
    principal_strain,
    strain_invariants,
    velocity_gradient_to_strain,
)


def test_velocity_gradient_to_strain_pure_shear():
    # dve/dx = 10, dvn/dy = -10 => e_ee=10, e_nn=-10, e_en=0
    L = np.array([
        [10.0, 0.0],
        [0.0, -10.0],
    ])
    ee, en, nn, omega = velocity_gradient_to_strain(L)
    assert ee == pytest.approx(10.0)
    assert nn == pytest.approx(-10.0)
    assert en == pytest.approx(0.0)
    assert omega == pytest.approx(0.0)


def test_velocity_gradient_to_strain_simple_shear():
    # dve/dy = 10, dvn/dx = 10 => e_en=10, omega=0
    L = np.array([
        [0.0, 10.0],
        [10.0, 0.0],
    ])
    ee, en, nn, omega = velocity_gradient_to_strain(L)
    assert ee == pytest.approx(0.0)
    assert nn == pytest.approx(0.0)
    assert en == pytest.approx(10.0)
    assert omega == pytest.approx(0.0)


def test_velocity_gradient_to_strain_rotation():
    # dve/dy = -10, dvn/dx = 10 => e_en=0, omega=10
    L = np.array([
        [0.0, -10.0],
        [10.0, 0.0],
    ])
    ee, en, nn, omega = velocity_gradient_to_strain(L)
    assert en == pytest.approx(0.0)
    assert omega == pytest.approx(10.0)


def test_principal_strain_extension():
    e1, e2, az = principal_strain(50.0, 0.0, -20.0)
    assert e1 == pytest.approx(50.0)
    assert e2 == pytest.approx(-20.0)
    assert az == pytest.approx(0.0)


def test_principal_strain_45_degrees():
    # Pure shear with principal axes at 45 deg
    e1, e2, az = principal_strain(0.0, 30.0, 0.0)
    assert e1 == pytest.approx(30.0)
    assert e2 == pytest.approx(-30.0)
    assert az == pytest.approx(45.0)


def test_strain_invariants():
    dilation, shear, sec_inv = strain_invariants(np.array([30.0]), np.array([-30.0]))
    assert dilation[0] == pytest.approx(0.0)
    assert shear[0] == pytest.approx(30.0)
    assert sec_inv[0] == pytest.approx(np.sqrt(1800.0))


def test_estimate_strain_rate_pure_extension():
    # Create a simple 2D grid centered at origin
    x = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    eps = 0.05  # raw mm/(km.yr)
    ve = eps * x
    vn = np.zeros_like(x)
    res = estimate_strain_rate(x, y, ve, vn, normalize=True)
    # exx should be ~50 nstrain/yr
    assert res["exx"] == pytest.approx(50.0, rel=1e-3)
    assert res["eyy"] == pytest.approx(0.0, abs=1.0)
    assert res["exy"] == pytest.approx(0.0, abs=1.0)


def test_estimate_strain_rate_rigid_rotation():
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    omega = 0.05  # raw rotation
    ve = -omega * y
    vn = omega * x
    res = estimate_strain_rate(x, y, ve, vn, normalize=True)
    # Strain should be near zero. Rotation sign follows the design-matrix
    # convention: ve = Ux + omega*y, vn = Uy - omega*x, so a physically
    # positive rotation (vn = +omega*x, ve = -omega*y) yields omega_est = -omega.
    assert res["exx"] == pytest.approx(0.0, abs=1.0)
    assert res["eyy"] == pytest.approx(0.0, abs=1.0)
    assert res["exy"] == pytest.approx(0.0, abs=1.0)
    assert res["omega"] == pytest.approx(-50.0, rel=1e-3)


def test_estimate_strain_rate_with_uncertainties():
    x = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    eps = 0.05
    ve = eps * x
    vn = np.zeros_like(x)
    se = np.full_like(x, 0.01)
    sn = np.full_like(x, 0.01)
    res = estimate_strain_rate(x, y, ve, vn, se, sn, return_covariance=True)
    assert "param_cov" in res
    assert res["param_cov"].shape == (6, 6)


def test_estimate_strain_rate_real_coordinates():
    """Regression: ensure column normalization handles km-scale coordinates."""
    lon = np.array([112.0, 113.0, 112.0, 113.0, 112.5])
    lat = np.array([40.0, 40.0, 41.0, 41.0, 40.5])
    x, y, _ = llh2utm(lon, lat)
    x0, y0 = np.mean(x), np.mean(y)
    eps = 0.05
    ve = eps * (x - x0)
    vn = np.zeros_like(ve)
    res = estimate_strain_rate(x - x0, y - y0, ve, vn, normalize=True)
    assert res["exx"] == pytest.approx(50.0, rel=1e-2)
