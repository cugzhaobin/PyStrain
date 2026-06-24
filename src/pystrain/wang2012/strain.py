"""Per-triangle strain-rate computation for Wang (2012) estimation."""

import numpy as np
from scipy.spatial import Delaunay

from pystrain.data import StrainResult
from pystrain.geodesy import llh2utm
from pystrain.strain.tensor import principal_strain, strain_invariants


def compute_triangle_strain(
    tri: Delaunay,
    mesh_lon: np.ndarray,
    mesh_lat: np.ndarray,
    v_mesh: np.ndarray,
) -> StrainResult:
    """Compute strain rate for each triangle of the mesh.

    Parameters
    ----------
    tri : scipy.spatial.Delaunay
        Delaunay triangulation of the mesh nodes.
    mesh_lon, mesh_lat : np.ndarray, shape (n_vtx,)
        Mesh node positions in degrees.
    v_mesh : np.ndarray, shape (2*n_vtx,)
        Solved mesh-node velocities stacked as [ve_all; vn_all].

    Returns
    -------
    StrainResult
        Strain-rate tensors at triangle centroids (or incenters).
        Units of exx, exy, eyy are nanostrain/yr (nstrain/yr) and omega is
        the same.
    """
    n_vtx = len(mesh_lon)
    n_tri = len(tri.simplices)

    ve_vtx = v_mesh[:n_vtx]
    vn_vtx = v_mesh[n_vtx:]

    # Project all mesh nodes to UTM km once
    x_utm, y_utm, _ = llh2utm(mesh_lon, mesh_lat)

    lons_out = np.empty(n_tri)
    lats_out = np.empty(n_tri)
    exx_out = np.empty(n_tri)
    exy_out = np.empty(n_tri)
    eyy_out = np.empty(n_tri)
    omega_out = np.empty(n_tri)

    for k, simplex in enumerate(tri.simplices):
        i0, i1, i2 = int(simplex[0]), int(simplex[1]), int(simplex[2])

        # Vertex coordinates in km
        xy = np.array([
            [x_utm[i0], y_utm[i0]],
            [x_utm[i1], y_utm[i1]],
            [x_utm[i2], y_utm[i2]],
        ])

        # Vertex velocities in mm/yr
        vel = np.array([
            [ve_vtx[i0], vn_vtx[i0]],
            [ve_vtx[i1], vn_vtx[i1]],
            [ve_vtx[i2], vn_vtx[i2]],
        ])

        # Velocity differences from vertex 0
        dvel = vel[1:] - vel[0]   # (2, 2): rows = vtx 1,2; cols = e,n
        dxy  = xy[1:]  - xy[0]    # (2, 2): rows = vtx 1,2; cols = x,y

        # Solve for velocity gradient tensor [dvx/dx, dvx/dy, dvy/dx, dvy/dy]
        # A * [dvx/dx, dvx/dy, dvy/dx, dvy/dy]^T = [dve1, dvn1, dve2, dvn2]^T
        A = np.array([
            [dxy[0, 0], dxy[0, 1], 0.0,       0.0      ],
            [0.0,       0.0,       dxy[0, 0], dxy[0, 1]],
            [dxy[1, 0], dxy[1, 1], 0.0,       0.0      ],
            [0.0,       0.0,       dxy[1, 0], dxy[1, 1]],
        ])
        b = np.array([dvel[0, 0], dvel[0, 1], dvel[1, 0], dvel[1, 1]])

        try:
            vg, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            vg = np.zeros(4)

        # velgrad = [dvx/dx, dvx/dy, dvy/dx, dvy/dy]
        dvx_dx, dvx_dy, dvy_dx, dvy_dy = vg

        exx = dvx_dx                         # mm/yr / km = 1e-6 /yr
        eyy = dvy_dy
        exy = 0.5 * (dvx_dy + dvy_dx)
        om  = 0.5 * (dvy_dx - dvx_dy)

        # Convert mm/yr/km → nanostrain/yr  (1 mm/yr / km = 1e-6 /yr = 1000 nstrain/yr)
        unit_factor = 1000.0
        exx_out[k]   = exx   * unit_factor
        exy_out[k]   = exy   * unit_factor
        eyy_out[k]   = eyy   * unit_factor
        omega_out[k] = om    * unit_factor

        # Centroid as result location
        lons_out[k] = (mesh_lon[i0] + mesh_lon[i1] + mesh_lon[i2]) / 3.0
        lats_out[k] = (mesh_lat[i0] + mesh_lat[i1] + mesh_lat[i2]) / 3.0

    e1, e2, azimuth = principal_strain(exx_out, exy_out, eyy_out)
    dilation, shear, sec_inv = strain_invariants(e1, e2)

    # Representative centroid velocity from interpolated mesh values
    ve_cen = np.empty(n_tri)
    vn_cen = np.empty(n_tri)
    for k, simplex in enumerate(tri.simplices):
        ve_cen[k] = np.mean(ve_vtx[simplex])
        vn_cen[k] = np.mean(vn_vtx[simplex])

    return StrainResult(
        lon=lons_out,
        lat=lats_out,
        exx=exx_out,
        exy=exy_out,
        eyy=eyy_out,
        omega=omega_out,
        e1=e1,
        e2=e2,
        azimuth=azimuth,
        shear=shear,
        dilation=dilation,
        sec_inv=sec_inv,
        ve=ve_cen,
        vn=vn_cen,
    )
