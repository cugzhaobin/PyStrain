"""Tests for PyStrain2 data containers."""

import numpy as np
import pytest

from pystrain2.data import StrainResult, VelocityField


def test_velocity_field_length_mismatch():
    with pytest.raises(ValueError):
        VelocityField(
            lon=np.array([1.0, 2.0]),
            lat=np.array([1.0, 2.0]),
            ve=np.array([1.0, 2.0]),
            vn=np.array([1.0]),
            se=np.array([1.0, 2.0]),
            sn=np.array([1.0, 2.0]),
            rho=np.array([0.0, 0.0]),
            names=np.array(["a", "b"]),
        )


def test_velocity_field_subset():
    vf = VelocityField(
        lon=np.array([1.0, 2.0, 3.0]),
        lat=np.array([1.0, 2.0, 3.0]),
        ve=np.array([0.0, 1.0, 2.0]),
        vn=np.array([0.0, -1.0, -2.0]),
        se=np.ones(3),
        sn=np.ones(3),
        rho=np.zeros(3),
        names=np.array(["a", "b", "c"]),
    )
    sub = vf.subset(np.array([True, False, True]))
    assert len(sub) == 2
    assert list(sub.names) == ["a", "c"]


def test_strain_result_length_mismatch():
    with pytest.raises(ValueError):
        StrainResult(
            lon=np.array([1.0]),
            lat=np.array([1.0]),
            exx=np.array([1.0]),
            exy=np.array([1.0]),
            eyy=np.array([1.0]),
            omega=np.array([1.0]),
            e1=np.array([1.0]),
            e2=np.array([1.0]),
            azimuth=np.array([1.0]),
            shear=np.array([1.0]),
            dilation=np.array([1.0]),
            sec_inv=np.array([1.0, 2.0]),
            ve=np.array([1.0]),
            vn=np.array([1.0]),
        )
