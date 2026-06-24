"""GPS interpolation matrix for Wang (2012) strain-rate estimation.

Implements England & Molnar (2005) linear shape functions for triangular
finite elements in geographic (lon/lat) coordinates.
"""

from typing import Tuple
import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse


def interpk(
    point: Tuple[float, float],
    tri_vertices: np.ndarray,
) -> np.ndarray:
    """Compute linear shape-function values at *point* for a triangle.

    Parameters
    ----------
    point : (lon, lat)
        Query point in geographic coordinates.
    tri_vertices : array_like, shape (3, 2)
        Triangle vertices [(x1,y1), (x2,y2), (x3,y3)] in geographic
        coordinates (lon/lat or any Cartesian 2-D system).

    Returns
    -------
    N : np.ndarray, shape (3,)
        Shape-function values [N1, N2, N3].  They sum to 1 inside the
        triangle (England & Molnar 2005, eqs 5–8).
    """
    (x1, y1), (x2, y2), (x3, y3) = tri_vertices
    lon, lat = point

    delta = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if abs(delta) < 1e-30:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    a1 = (x2 * y3 - x3 * y2) / delta
    b1 = (y2 - y3) / delta
    c1 = (x3 - x2) / delta

    a2 = (x3 * y1 - x1 * y3) / delta
    b2 = (y3 - y1) / delta
    c2 = (x1 - x3) / delta

    a3 = (x1 * y2 - x2 * y1) / delta
    b3 = (y1 - y2) / delta
    c3 = (x2 - x1) / delta

    N1 = a1 + b1 * lon + c1 * lat
    N2 = a2 + b2 * lon + c2 * lat
    N3 = a3 + b3 * lon + c3 * lat

    return np.array([N1, N2, N3])


def intri(
    point: Tuple[float, float],
    tri_vertices: np.ndarray,
) -> bool:
    """Check whether *point* lies inside a triangle using barycentric coords.

    Parameters
    ----------
    point : (lon, lat)
    tri_vertices : array_like, shape (3, 2)

    Returns
    -------
    bool
    """
    N = interpk(point, tri_vertices)
    return bool(np.all(N >= -1e-10) and np.all(N <= 1.0 + 1e-10))


def build_gps_interpolation_matrix(
    tri: Delaunay,
    mesh_lon: np.ndarray,
    mesh_lat: np.ndarray,
    gps_lon: np.ndarray,
    gps_lat: np.ndarray,
) -> sparse.csr_matrix:
    """Build the GPS-to-mesh interpolation matrix G.

    For each GPS station the containing triangle is found and shape-function
    weights are stored.  The returned matrix has the block-diagonal structure
    ``[G_e; G_n]`` (east block stacked above north block) so that the GPS
    velocity vector ``[ve_all; vn_all]`` (all east first, all north second)
    maps directly to predicted velocities.

    Parameters
    ----------
    tri : scipy.spatial.Delaunay
        Delaunay triangulation of the mesh.
    mesh_lon, mesh_lat : np.ndarray, shape (n_vtx,)
        Mesh node positions.
    gps_lon, gps_lat : np.ndarray, shape (n_gps,)
        GPS station positions.

    Returns
    -------
    G : scipy.sparse.csr_matrix, shape (2*n_gps, 2*n_vtx)
        Block-diagonal interpolation matrix.
    """
    n_gps = len(gps_lon)
    n_vtx = len(mesh_lon)

    rows: list = []
    cols: list = []
    data: list = []

    mesh_pts = np.column_stack([mesh_lon, mesh_lat])

    for i in range(n_gps):
        pt = (gps_lon[i], gps_lat[i])

        # Fast look-up using Delaunay.find_simplex
        simplex_idx = tri.find_simplex(np.array([[gps_lon[i], gps_lat[i]]]))[0]

        if simplex_idx >= 0:
            vtx_indices = tri.simplices[simplex_idx]
            vertices = mesh_pts[vtx_indices]
            N = interpk(pt, vertices)
        else:
            # Station outside convex hull: find nearest triangle centroid
            centroids = mesh_pts[tri.simplices].mean(axis=1)
            dists = np.sum((centroids - np.array([gps_lon[i], gps_lat[i]])) ** 2, axis=1)
            nearest_tri = int(np.argmin(dists))
            vtx_indices = tri.simplices[nearest_tri]
            vertices = mesh_pts[vtx_indices]
            N = interpk(pt, vertices)
            # Clamp negative shape functions to zero and renormalise
            N = np.clip(N, 0.0, None)
            s = N.sum()
            if s > 0:
                N /= s
            else:
                N[:] = 1.0 / 3.0

        for k, j in enumerate(vtx_indices):
            rows.append(i)
            cols.append(j)
            data.append(float(N[k]))

    # Build sparse east-component block G_e (rows 0..n_gps-1, cols 0..n_vtx-1)
    G_e = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_gps, n_vtx), dtype=float
    )

    # North block occupies rows n_gps..2*n_gps-1, cols n_vtx..2*n_vtx-1
    G_n = G_e  # identical weights

    G = sparse.block_diag([G_e, G_n], format="csr")
    return G
