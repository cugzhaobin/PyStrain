"""Tests for outlier detection."""

import numpy as np

from pystrain2.data import VelocityField
from pystrain2.outlier import (
    compute_residuals_triangulation,
    iqr_outlier_detection,
    iterative_outlier_removal,
    knn_prescreening,
)
from pystrain2.triangulation import delaunay_triangulation


def _triangulation_fn(lon, lat, site_indices):
    tri, good_triangles, xy, _ = delaunay_triangulation(
        lon[site_indices], lat[site_indices]
    )
    return tri, good_triangles, xy


def test_knn_prescreening_detects_outlier():
    # Cluster of 3 similar sites and one spatially isolated outlier
    lon = np.array([112.0, 112.1, 112.2, 115.0])
    lat = np.array([40.0, 40.1, 40.0, 43.0])
    ve = np.array([1.0, 1.0, 1.0, 1.0])
    vn = np.array([0.0, 0.0, 0.0, 0.0])
    # The isolated site has a drastically different velocity
    ve[3] = 50.0
    vf = VelocityField(
        lon=lon, lat=lat, ve=ve, vn=vn,
        se=np.ones(4), sn=np.ones(4), rho=np.zeros(4),
        names=np.array(["a", "b", "c", "d"]),
    )
    mask, _ = knn_prescreening(vf, k_neighbors=3, mad_factor=1.5)
    assert mask[3]


def test_compute_residuals_triangulation_mapping():
    lon = np.array([112.0, 113.0, 112.5, 113.5])
    lat = np.array([40.0, 40.0, 41.0, 41.0])
    ve = np.array([1.0, 1.1, 1.0, 1.1])
    vn = np.zeros(4)
    vf = VelocityField(
        lon=lon, lat=lat, ve=ve, vn=vn,
        se=np.ones(4), sn=np.ones(4), rho=np.zeros(4),
        names=np.array(["a", "b", "c", "d"]),
    )
    site_indices = np.array([0, 1, 2, 3])
    tri, good_triangles, xy, _ = delaunay_triangulation(lon[site_indices], lat[site_indices])
    residuals = compute_residuals_triangulation(tri, xy, good_triangles, vf, site_indices)
    assert len(residuals) > 0
    # All velocities similar, residuals should be small
    for info in residuals.values():
        assert info["res_norm"] < 1.0


def test_iqr_outlier_detection():
    residuals = {
        0: {"res_norm": 0.1},
        1: {"res_norm": 0.2},
        2: {"res_norm": 0.15},
        3: {"res_norm": 5.0},
    }
    outliers = iqr_outlier_detection(residuals, iqr_factor=1.5)
    assert 3 in outliers


def test_iterative_outlier_removal():
    # Cluster of 4 similar sites plus one isolated outlier
    lon = np.array([112.0, 112.1, 112.2, 112.15, 115.0])
    lat = np.array([40.0, 40.1, 40.0, 40.15, 43.0])
    ve = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    vn = np.zeros(5)
    ve[4] = 50.0
    vf = VelocityField(
        lon=lon, lat=lat, ve=ve, vn=vn,
        se=np.ones(5), sn=np.ones(5), rho=np.zeros(5),
        names=np.array(["a", "b", "c", "d", "e"]),
    )
    clean_vf, history = iterative_outlier_removal(
        vf, triangulation_fn=_triangulation_fn,
        k_neighbors=3, mad_factor=2.0, iqr_factor=1.5, max_iterations=2,
        min_sites=3,
    )
    assert len(clean_vf) < len(vf)
    assert any(o["name"] == "e" for o in history)
