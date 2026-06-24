"""Tests for visualization utilities."""

import numpy as np

from pystrain2.data import StrainResult, VelocityField
from pystrain2.visualization import plot_scalar_field, plot_velocity_field


def test_plot_velocity_field(tmp_path):
    vf = VelocityField(
        lon=np.array([112.0, 113.0]),
        lat=np.array([40.0, 41.0]),
        ve=np.array([1.0, 2.0]),
        vn=np.array([0.5, -0.5]),
        se=np.ones(2),
        sn=np.ones(2),
        rho=np.zeros(2),
        names=np.array(["a", "b"]),
    )
    out = tmp_path / "vel.png"
    plot_velocity_field(vf, output_path=str(out))
    assert out.exists()


def test_plot_scalar_field(tmp_path):
    sr = StrainResult(
        lon=np.array([112.0, 113.0]),
        lat=np.array([40.0, 41.0]),
        exx=np.array([10.0, 20.0]),
        exy=np.array([0.0, 0.0]),
        eyy=np.array([-10.0, -20.0]),
        omega=np.array([0.0, 0.0]),
        e1=np.array([10.0, 20.0]),
        e2=np.array([-10.0, -20.0]),
        azimuth=np.array([0.0, 0.0]),
        shear=np.array([10.0, 20.0]),
        dilation=np.array([0.0, 0.0]),
        sec_inv=np.array([14.14, 28.28]),
        ve=np.zeros(2),
        vn=np.zeros(2),
    )
    out = tmp_path / "scalar.png"
    plot_scalar_field(sr, "exx", str(out))
    assert out.exists()
