"""Tests for geodetic utilities."""

import numpy as np
import pytest

from pystrain2.geodesy import (
    distance_azimuth,
    llh2localxy,
    llh2utm,
    local_to_origin_projection,
    utm2ll,
)


def test_utm_zone():
    from pystrain2.geodesy import utm_zone
    assert utm_zone(112.5, 40.0) == 49
    assert utm_zone(-3.0, 52.0) == 30


def test_utm_round_trip():
    lon = np.array([112.61, 113.24])
    lat = np.array([41.27, 40.14])
    x, y, params = llh2utm(lon, lat)
    lon2, lat2 = utm2ll(x, y, params)
    np.testing.assert_allclose(lon, lon2, atol=1e-6)
    np.testing.assert_allclose(lat, lat2, atol=1e-6)


def test_localxy_round_trip():
    lon = np.array([112.0, 113.0, 112.5])
    lat = np.array([40.0, 41.0, 40.5])
    x, y = llh2localxy(lon, lat, origin=(112.5, 40.5))
    # Origin maps to (0, 0)
    assert np.isclose(x[2], 0.0, atol=1e-3)
    assert np.isclose(y[2], 0.0, atol=1e-3)


def test_local_to_origin_utm():
    lon = np.array([112.0, 113.0, 112.5])
    lat = np.array([40.0, 41.0, 40.5])
    x, y, params = local_to_origin_projection(lon, lat, (112.5, 40.5), method="utm")
    assert x.shape == lon.shape
    assert params["zone"] >= 49


def test_local_to_origin_polyconic():
    lon = np.array([112.0, 113.0, 112.5])
    lat = np.array([40.0, 41.0, 40.5])
    x, y, params = local_to_origin_projection(lon, lat, (112.5, 40.5), method="polyconic")
    assert x.shape == lon.shape
    assert params["method"] == "polyconic"


def test_distance_azimuth():
    lon1 = np.array([112.0])
    lat1 = np.array([40.0])
    lon2 = np.array([113.0])
    lat2 = np.array([40.0])
    d, az = distance_azimuth(lon1, lat1, lon2, lat2)
    assert d[0] > 80 and d[0] < 90  # ~86 km per degree at this latitude
    assert np.isclose(az[0], 90.0, atol=1.0)


def test_unknown_projection_method():
    with pytest.raises(ValueError):
        local_to_origin_projection(
            np.array([112.0]), np.array([40.0]), (112.0, 40.0), method="mars"
        )
