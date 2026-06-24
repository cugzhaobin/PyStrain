"""Tests for triangulation utilities."""

import numpy as np

from pystrain2.triangulation import (
    build_adjacency,
    compute_shape_function_derivatives,
    delaunay_triangulation,
    detect_hanging_sites,
    local_to_global,
    voronoi_areas,
)


def test_delaunay_triangulation_basic():
    lon = np.array([112.0, 113.0, 112.5, 112.7])
    lat = np.array([40.0, 40.0, 41.0, 40.5])
    tri, good_triangles, xy, params = delaunay_triangulation(lon, lat)
    assert len(tri.simplices) > 0
    assert good_triangles.dtype == bool
    assert xy.shape == (len(lon), 2)
    assert "zone" in params


def test_delaunay_quality_filter_rejects_sliver():
    lon = np.array([112.0, 113.0, 112.001, 112.5])
    lat = np.array([40.0, 40.0, 40.0, 41.0])
    tri, good_triangles, xy, _ = delaunay_triangulation(lon, lat, min_angle_deg=5.0)
    # The extremely thin triangle should be rejected
    assert np.sum(good_triangles) < len(tri.simplices)


def test_shape_function_derivatives():
    lon = np.array([112.0, 113.0, 112.5])
    lat = np.array([40.0, 40.0, 41.0])
    tri, good_triangles, xy, _ = delaunay_triangulation(lon, lat)
    B_list, good_indices, areas = compute_shape_function_derivatives(
        tri, xy, good_triangles
    )
    assert len(B_list) == len(good_indices)
    assert all(B.shape == (2, 3) for B in B_list)
    assert np.all(areas > 0)


def test_build_adjacency():
    lon = np.array([112.0, 113.0, 112.5, 113.5])
    lat = np.array([40.0, 40.0, 41.0, 41.0])
    tri, good_triangles, _, _ = delaunay_triangulation(lon, lat)
    adj = build_adjacency(tri, good_triangles)
    assert len(adj) > 0


def test_local_to_global_mapping():
    lon = np.array([112.0, 113.0, 112.5, 113.5])
    lat = np.array([40.0, 40.0, 41.0, 41.0])
    site_indices = np.array([0, 2, 3])
    tri, good_triangles, _, _ = delaunay_triangulation(
        lon[site_indices], lat[site_indices]
    )
    global_simplices = local_to_global(tri, site_indices)
    assert global_simplices.shape == tri.simplices.shape
    assert np.all(global_simplices >= 0)


def test_detect_hanging_sites():
    lon = np.array([112.0, 113.0, 112.5, 115.0])
    lat = np.array([40.0, 40.0, 41.0, 50.0])
    tri, good_triangles, _, _ = delaunay_triangulation(lon, lat, max_edge_km=50.0)
    hanging = detect_hanging_sites(tri, good_triangles, len(lon))
    # The far away point should not belong to any valid triangle
    assert hanging[3]


def test_voronoi_areas_positive():
    lon = np.array([112.0, 113.0, 112.5, 113.5])
    lat = np.array([40.0, 40.0, 41.0, 41.0])
    areas = voronoi_areas(lon, lat)
    assert np.all(areas > 0)
    assert len(areas) == len(lon)


def test_voronoi_areas_regular_grid():
    """Voronoi cells of a regular grid should have nearly equal areas."""
    lon, lat = np.meshgrid(
        np.linspace(112.0, 113.0, 5),
        np.linspace(40.0, 41.0, 5),
    )
    areas = voronoi_areas(lon.ravel(), lat.ravel())
    # Exclude boundary cells, which use the circular fallback
    interior = areas.reshape(5, 5)[1:4, 1:4].ravel()
    assert len(interior) == 9
    assert np.all(interior > 0)
    # Interior cells should be roughly equal (within 10%)
    assert np.std(interior) / np.mean(interior) < 0.1
