"""Delaunay triangulation, shape-function derivatives, and quality filters."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import Delaunay, KDTree
from matplotlib.path import Path

from pystrain.geodesy import llh2utm


def _ll_to_xy(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Project lon/lat to UTM km, returning (N,2) coordinate array."""
    x, y, params = llh2utm(lon, lat)
    return np.column_stack([x, y]), params


def _triangle_area(xy: np.ndarray) -> np.ndarray:
    """Compute signed area of triangles given (N,3,2) vertex array."""
    return 0.5 * (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )


def _edge_lengths(xy: np.ndarray) -> np.ndarray:
    """Return (N,3) edge lengths for triangles."""
    e1 = np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1)
    e2 = np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1)
    e3 = np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1)
    return np.column_stack([e1, e2, e3])


def _min_interior_angle(xy: np.ndarray) -> np.ndarray:
    """Return minimum interior angle (degrees) for each triangle."""
    edges = _edge_lengths(xy)
    a, b, c = edges[:, 0], edges[:, 1], edges[:, 2]
    # Law of cosines for all three angles
    cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1, 1)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1, 1)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1, 1)
    angles = np.degrees(np.arccos(np.column_stack([cos_A, cos_B, cos_C])))
    return np.min(angles, axis=1)


def delaunay_triangulation(
    lon: np.ndarray,
    lat: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    min_angle_deg: float = 10.0,
    max_edge_pctl: float = 95.0,
    max_edge_factor: float = 1.5,
    min_area_ratio: float = 0.1,
    max_edge_km: Optional[float] = None,
) -> Tuple[Delaunay, np.ndarray, np.ndarray, dict]:
    """Build a quality-controlled Delaunay triangulation of GNSS sites.

    Parameters
    ----------
    lon, lat : np.ndarray
        Site coordinates in degrees.
    polygon : np.ndarray, optional
        Boundary polygon (N,2) in degrees. Triangles with centroids outside
        are excluded.
    min_angle_deg : float
        Minimum allowed interior angle.
    max_edge_pctl : float
        Percentile of site spacing used for adaptive edge threshold.
    max_edge_factor : float
        Multiplier on percentile threshold.
    min_area_ratio : float
        Minimum area relative to median triangle area.
    max_edge_km : float, optional
        Absolute maximum edge length in km.

    Returns
    -------
    tri : Delaunay
        Delaunay triangulation object (local indices into the input arrays).
    good_triangles : np.ndarray
        Boolean mask of valid triangles.
    xy : np.ndarray
        Projected site coordinates in km.
    proj_params : dict
        Projection parameters.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    xy, proj_params = _ll_to_xy(lon, lat)

    if len(xy) < 3:
        raise ValueError("At least 3 sites are required for triangulation.")

    tri = Delaunay(xy)
    n_tri = len(tri.simplices)
    good_triangles = np.ones(n_tri, dtype=bool)

    vertices = xy[tri.simplices]  # (n_tri, 3, 2)
    areas = _triangle_area(vertices)
    edges = _edge_lengths(vertices)
    min_angles = _min_interior_angle(vertices)

    # Area filter (exclude negative/degenerate)
    good_triangles &= areas > 0

    # Minimum angle filter
    good_triangles &= min_angles >= min_angle_deg

    # Edge length filters (percentile computed per edge position)
    max_edge_threshold = np.percentile(edges, max_edge_pctl, axis=0) * max_edge_factor
    good_triangles &= np.all(edges <= max_edge_threshold[None, :], axis=1)

    if max_edge_km is not None:
        good_triangles &= np.all(edges <= max_edge_km, axis=1)

    # Area ratio filter
    median_area = np.median(areas[areas > 0])
    if median_area > 0:
        good_triangles &= areas >= min_area_ratio * median_area

    # Polygon filter
    if polygon is not None:
        centroids = np.mean(vertices, axis=1)
        # Project polygon to same UTM frame
        poly_xy, _ = _ll_to_xy(polygon[:, 0], polygon[:, 1])
        path = Path(poly_xy)
        good_triangles &= path.contains_points(centroids)

    return tri, good_triangles, xy, proj_params


def compute_shape_function_derivatives(
    tri: Delaunay,
    xy: np.ndarray,
    good_triangles: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Compute B = [dN/dx; dN/dy] (2x3) for each triangle.

    Returns
    -------
    B_list : list of np.ndarray
        One 2x3 matrix per valid triangle.
    good_indices : np.ndarray
        Indices of triangles for which B was computed.
    areas : np.ndarray
        Triangle areas (km^2).
    """
    if good_triangles is None:
        good_triangles = np.ones(len(tri.simplices), dtype=bool)

    good_indices = np.where(good_triangles)[0]
    B_list = []
    areas = []

    for idx in good_indices:
        verts = xy[tri.simplices[idx]]  # (3, 2)
        x1, y1 = verts[0]
        x2, y2 = verts[1]
        x3, y3 = verts[2]
        A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if abs(A) < 1e-12:
            continue
        # Shape function derivatives
        dN_dx = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * A)
        dN_dy = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * A)
        B = np.vstack([dN_dx, dN_dy])
        B_list.append(B)
        areas.append(A)

    return B_list, good_indices, np.array(areas)


def build_adjacency(
    tri: Delaunay,
    good_triangles: np.ndarray,
) -> Dict[int, List[int]]:
    """Build triangle adjacency map based on shared edges."""
    adjacency: Dict[int, List[int]] = {
        i: [] for i in range(len(tri.simplices)) if good_triangles[i]
    }
    good_set = set(np.where(good_triangles)[0])

    for i in good_set:
        verts_i = set(tri.simplices[i])
        for j in good_set:
            if j <= i:
                continue
            verts_j = set(tri.simplices[j])
            if len(verts_i & verts_j) == 2:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def local_to_global(tri: Delaunay, site_indices: np.ndarray) -> np.ndarray:
    """Map local triangle vertex indices to global site indices.

    Parameters
    ----------
    tri : Delaunay
        Triangulation built on ``site_indices`` subset.
    site_indices : np.ndarray
        Global indices of the sites fed to Delaunay.

    Returns
    -------
    np.ndarray
        Array of shape (n_tri, 3) with global site indices.
    """
    return site_indices[tri.simplices]


def global_to_local(
    tri: Delaunay, global_idx: int, site_indices: np.ndarray
) -> Optional[int]:
    """Map a global site index to its local index in the triangulation."""
    local = np.where(site_indices == global_idx)[0]
    return int(local[0]) if len(local) > 0 else None


def _shoelace_area(vertices: np.ndarray) -> float:
    """Signed area of a closed polygon using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


def voronoi_areas(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Compute Voronoi cell area for each site using spherical Voronoi.

    Uses ``scipy.spatial.SphericalVoronoi`` on the unit sphere, which works
    correctly for global (or any-scale) datasets without projection distortion.
    Areas are returned in km².
    """
    from scipy.spatial import SphericalVoronoi

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)

    # Convert to 3D unit sphere
    lam = np.radians(lon)
    phi = np.radians(lat)
    points_3d = np.column_stack([
        np.cos(phi) * np.cos(lam),
        np.cos(phi) * np.sin(lam),
        np.sin(phi),
    ])

    # WGS84 mean radius
    R_km = 6371.0

    try:
        sv = SphericalVoronoi(points_3d, radius=1.0, center=[0.0, 0.0, 0.0])
        sv.sort_vertices_of_regions()
    except Exception:
        total_area = 4 * np.pi * R_km ** 2
        return np.full(n, total_area / n)

    areas = np.zeros(n)
    for i, region in enumerate(sv.regions):
        if len(region) < 3:
            continue
        verts = sv.vertices[region]   # (m, 3) points on unit sphere
        # Spherical polygon area via sum of spherical triangle areas
        # Triangulate the polygon around its first vertex
        v0 = verts[0]
        for j in range(1, len(region) - 1):
            v1 = verts[j]
            v2 = verts[j + 1]
            # Compute solid angle of spherical triangle (v0, v1, v2)
            # A = 4 * arctan(sqrt(tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2)))
            # where s is half the sum of spherical sides
            a = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
            b = np.arccos(np.clip(np.dot(v0, v2), -1, 1))
            c = np.arccos(np.clip(np.dot(v0, v1), -1, 1))
            s_half = (a + b + c) / 2
            # Spherical excess via l'Huilier's formula
            t = np.sqrt(max(0,
                np.tan(s_half / 2)
                * np.tan((s_half - a) / 2)
                * np.tan((s_half - b) / 2)
                * np.tan((s_half - c) / 2)
            ))
            excess = 4 * np.arctan(t)
            areas[i] += excess

    # Convert solid angle (steradians) to km² on sphere
    areas *= R_km ** 2

    # Fallback for sites with near-zero or very large Voronoi cell area
    # (boundary / degenerate cells in SphericalVoronoi)
    tree = KDTree(np.column_stack([lon, lat]))
    for i in range(n):
        if areas[i] <= 0 or not np.isfinite(areas[i]):
            k = min(7, n)
            dists, _ = tree.query([lon[i], lat[i]], k=k)
            rd_deg = np.mean(dists[1:]) if len(dists) > 1 else 1.0
            rd_km = rd_deg * 111.0
            areas[i] = np.pi * rd_km ** 2

    return areas


def detect_hanging_sites(
    tri: Delaunay,
    good_triangles: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Return boolean mask of sites not belonging to any valid triangle."""
    used = np.zeros(n_sites, dtype=bool)
    good_simplices = tri.simplices[good_triangles]
    if len(good_simplices) > 0:
        used[good_simplices.ravel()] = True
    return ~used
