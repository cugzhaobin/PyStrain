"""Laplacian smoothing matrix for Wang (2012) strain-rate estimation.

Implements the cotangent / edge-length Laplacian described in Desbrun et al.
(1999).  The smoothing is applied in geographic (lon/lat) space so that the
smoothing scale is consistent with the mesh spacing in degrees.
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse


def build_laplacian_smoothing_matrix(
    tri: Delaunay,
    mesh_lon: np.ndarray,
    mesh_lat: np.ndarray,
    boundary_smooth: bool = True,
) -> sparse.csr_matrix:
    """Build the Laplacian smoothing matrix S.

    Uses an edge-length weighted Laplacian:
    - Off-diagonal entry S(i,j) = 2 / (edge_ij * E_i)
    - Diagonal entry S(i,i) = -sum_j S(i,j)
    where edge_ij is the Euclidean distance between vertices i and j in
    degrees, and E_i = sum of all edge lengths connected to vertex i.

    The final matrix is block-diagonal ``[S; S]`` for the east/north
    components, shape ``(2*n_vtx, 2*n_vtx)``.

    Parameters
    ----------
    tri : scipy.spatial.Delaunay
        Delaunay triangulation of the mesh.
    mesh_lon, mesh_lat : np.ndarray, shape (n_vtx,)
        Mesh node positions in degrees.
    boundary_smooth : bool
        If True, all vertices (including boundary) are smoothed.
        If False, rows corresponding to convex-hull boundary vertices are
        zeroed (no smoothing constraint on boundary).

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape (2*n_vtx, 2*n_vtx)
        Block-diagonal smoothing matrix for [east; north] vertex velocities.
    """
    n_vtx = len(mesh_lon)
    simplices = tri.simplices  # shape (n_tri, 3)

    # Collect all unique edges and their lengths
    edge_dict: dict = {}  # (i, j) -> edge_length  (i < j)
    for tri_verts in simplices:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            i, j = int(tri_verts[a]), int(tri_verts[b])
            if i > j:
                i, j = j, i
            if (i, j) not in edge_dict:
                dlon = mesh_lon[i] - mesh_lon[j]
                dlat = mesh_lat[i] - mesh_lat[j]
                edge_dict[(i, j)] = np.sqrt(dlon ** 2 + dlat ** 2)

    # Sum of edge lengths per vertex (E_i)
    E = np.zeros(n_vtx)
    for (i, j), ell in edge_dict.items():
        E[i] += ell
        E[j] += ell

    # Build off-diagonal entries
    rows: list = []
    cols: list = []
    data: list = []

    for (i, j), ell in edge_dict.items():
        if ell < 1e-30:
            continue
        w_ij = 2.0 / (ell * E[i]) if E[i] > 1e-30 else 0.0
        w_ji = 2.0 / (ell * E[j]) if E[j] > 1e-30 else 0.0
        rows.append(i)
        cols.append(j)
        data.append(w_ij)
        rows.append(j)
        cols.append(i)
        data.append(w_ji)

    S_mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_vtx, n_vtx), dtype=float)

    # Diagonal: -sum of off-diagonal row entries
    diag_vals = -np.array(S_mat.sum(axis=1)).ravel()
    S_mat = S_mat + sparse.diags(diag_vals, 0, shape=(n_vtx, n_vtx), format="csr")

    # Optionally zero out boundary rows
    if not boundary_smooth:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([mesh_lon, mesh_lat])
        hull = ConvexHull(pts)
        boundary_verts = hull.vertices
        # Zero out boundary rows
        S_lil = S_mat.tolil()
        for bv in boundary_verts:
            S_lil[bv, :] = 0.0
        S_mat = S_lil.tocsr()

    # Block-diagonal for east+north
    S = sparse.block_diag([S_mat, S_mat], format="csr")
    return S
