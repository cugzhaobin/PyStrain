"""Tests for PyStrain2 I/O modules."""

import numpy as np

from pystrain2.data import VelocityField
from pystrain2.io import read_velocity_file, write_strain_result, write_velocity_file
from pystrain2.io.polygon import read_polygon
from pystrain2.io.velocity import VelocityParseError


def test_read_gmt8_auto(gmt8_text):
    vf = read_velocity_file(gmt8_text, fmt="auto")
    assert len(vf) == 9
    assert vf.names[0] == "A001"
    assert np.isclose(vf.ve[0], 3.39)
    assert vf.meta["detected_format"] == "gmt8"


def test_read_gmt7(tmp_path):
    content = "112.61 41.27 3.39 -0.84 0.35 0.35 A001\n"
    path = tmp_path / "vel.gmtvec"
    path.write_text(content, encoding="utf-8")
    vf = read_velocity_file(str(path), fmt="auto")
    assert len(vf) == 1
    assert vf.rho[0] == 0.0


def test_read_globk(tmp_path):
    content = "112.61 41.27 3.0 4.0 3.39 -0.84 0.35 0.21 0.00 0.0 0.0 0.0 A001\n"
    path = tmp_path / "vel.glb"
    path.write_text(content, encoding="utf-8")
    vf = read_velocity_file(str(path), fmt="globk")
    assert len(vf) == 1
    assert vf.ve[0] == pytest.approx(3.39)
    assert vf.vn[0] == pytest.approx(-0.84)


def test_read_strict_mode(tmp_path):
    content = "112.61 41.27 3.39 -0.84 0.35 0.35 A001\nbad line\n"
    path = tmp_path / "vel.gmtvec"
    path.write_text(content, encoding="utf-8")
    vf = read_velocity_file(str(path), fmt="auto", strict=False)
    assert len(vf) == 1
    with pytest.raises(VelocityParseError):
        read_velocity_file(str(path), fmt="auto", strict=True)


def test_write_velocity_file(tmp_path):
    vf = VelocityField(
        lon=np.array([1.0, 2.0]),
        lat=np.array([1.0, 2.0]),
        ve=np.array([3.0, 4.0]),
        vn=np.array([0.0, 1.0]),
        se=np.array([0.1, 0.2]),
        sn=np.array([0.1, 0.2]),
        rho=np.array([0.0, 0.0]),
        names=np.array(["a", "b"]),
    )
    path = tmp_path / "out.gmtvec"
    write_velocity_file(str(path), vf, fmt="gmt8")
    text = path.read_text(encoding="utf-8")
    assert "a" in text and "b" in text


def test_write_strain_result(tmp_path):
    from pystrain2.data import StrainResult
    sr = StrainResult(
        lon=np.array([112.0]),
        lat=np.array([40.0]),
        exx=np.array([10.0]),
        exy=np.array([5.0]),
        eyy=np.array([-10.0]),
        omega=np.array([1.0]),
        e1=np.array([12.0]),
        e2=np.array([-12.0]),
        azimuth=np.array([45.0]),
        shear=np.array([12.0]),
        dilation=np.array([0.0]),
        sec_inv=np.array([16.97]),
        ve=np.array([0.0]),
        vn=np.array([0.0]),
    )
    path = tmp_path / "strain.txt"
    write_strain_result(str(path), sr)
    text = path.read_text(encoding="utf-8")
    assert "exx" in text
    assert "A001" not in text


import pytest


def test_read_polygon_auto_close(tmp_path):
    content = "# polygon\n112.0 40.0\n113.0 40.0\n113.0 41.0\n112.0 41.0\n"
    path = tmp_path / "poly.txt"
    path.write_text(content, encoding="utf-8")
    rings = read_polygon(str(path))
    assert len(rings) == 1
    # Auto-close adds a repeated first point
    assert len(rings[0]) == 5
    # First and last should now be equal
    np.testing.assert_allclose(rings[0][0], rings[0][-1])
