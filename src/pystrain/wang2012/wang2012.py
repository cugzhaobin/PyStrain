"""Wang (2012) / England & Molnar (2005) strain-rate estimation class."""

import logging
from typing import Optional, Tuple
import numpy as np

from pystrain.data import StrainResult, VelocityField
from pystrain.wang2012.mesh import (
    generate_adaptive_mesh,
    generate_gmsh_mesh,
    generate_poisson_mesh,
)
from pystrain.wang2012.interpolation import build_gps_interpolation_matrix
from pystrain.wang2012.smoothing import build_laplacian_smoothing_matrix
from pystrain.wang2012.solver import solve_wang2012
from pystrain.wang2012.lcurve import lcurve_search
from pystrain.wang2012.strain import compute_triangle_strain

logger = logging.getLogger("pystrain")


class Wang2012StrainRate:
    """Compute strain rate using the Wang (2012) / England & Molnar (2005) method.

    The algorithm:
    1. Generates a triangular mesh (adaptive or Poisson-disk).
    2. Builds a GPS-to-mesh interpolation matrix using linear shape functions.
    3. Builds a Laplacian smoothing matrix on the mesh.
    4. Optionally searches for the optimal smoothing factor via the L-curve.
    5. Solves for mesh-vertex velocities via regularised least squares.
    6. Computes strain rate per triangle from the solved vertex velocities.

    Parameters
    ----------
    vf : VelocityField
        Input GPS velocity field.
    polygon : np.ndarray, optional
        Nx2 array of (lon, lat) polygon vertices used to clip the mesh.
    mesh_region : list, optional
        [slon, elon, slat, elat] bounding box for the mesh.  Defaults to the
        bounding box of the GPS data expanded by *mesh_spacing*.
    mesh_method : str
        Meshing strategy: ``"adaptive"`` (grid-based with perturbation, fast),
        ``"poisson"`` (Poisson disk sampling, better-shaped triangles), or
        ``"gmsh"`` (Gmsh boundary-conforming mesh, requires ``pip install gmsh``).
        Default ``"adaptive"``.
    mesh_spacing : float
        Finest mesh node spacing in degrees (default 0.25).  The adaptive
        mesh refines down to this spacing where stations are dense.
    mesh_randomize : bool
        If True, add random perturbation to interior mesh nodes (default True).
        Only used when *mesh_method* == ``"adaptive"``.
    mesh_randomize_fraction : float
        Magnitude of perturbation as a fraction of *mesh_spacing* (default 0.2).
        Only used when *mesh_method* == ``"adaptive"``.
    max_stations_per_cell : int
        Quadtree subdivides cells with more than this many stations (default 6).
    max_spacing : float, optional
        Coarsest mesh spacing in degrees.  Defaults to ``mesh_spacing * 4``.
    smooth_factor : float
        Fixed smoothing regularisation factor used when *smooth_search* is
        False (default 0.01).
    smooth_search : bool
        If True, use the L-curve to find the optimal smoothing factor
        (default True).
    smooth_range : tuple(float, float)
        Log₁₀ range for the L-curve search, e.g. (-2.2, -0.8).
    smooth_step : float
        Step size in log₁₀ units for the L-curve sweep (default 0.2).
    smooth_boundary : bool
        If True, apply smoothing on boundary mesh nodes (default True).
    min_area_ratio : float
        Unused in this method; kept for API compatibility (default 0.1).
    """

    def __init__(
        self,
        vf: VelocityField,
        polygon: Optional[np.ndarray] = None,
        mesh_region: Optional[list] = None,
        mesh_method: str = "adaptive",
        mesh_spacing: float = 0.25,
        mesh_randomize: bool = True,
        mesh_randomize_fraction: float = 0.2,
        max_stations_per_cell: int = 6,
        max_spacing: Optional[float] = None,
        smooth_factor: float = 0.01,
        smooth_search: bool = True,
        smooth_range: Tuple[float, float] = (-2.2, -0.8),
        smooth_step: float = 0.2,
        smooth_boundary: bool = True,
        min_area_ratio: float = 0.1,
    ):
        self.vf = vf
        self.polygon = polygon
        self.mesh_region = mesh_region
        self.mesh_method = mesh_method
        self.mesh_spacing = mesh_spacing
        self.mesh_randomize = mesh_randomize
        self.mesh_randomize_fraction = mesh_randomize_fraction
        self.max_stations_per_cell = max_stations_per_cell
        self.max_spacing = max_spacing
        self.smooth_factor = smooth_factor
        self.smooth_search = smooth_search
        self.smooth_range = smooth_range
        self.smooth_step = smooth_step
        self.smooth_boundary = smooth_boundary
        self.min_area_ratio = min_area_ratio

    def _default_region(self) -> list:
        """Return a bounding box for the mesh.

        When *polygon* is provided, the bounding box is derived from the
        polygon vertices (padded by *mesh_spacing*).  Otherwise it falls
        back to the GPS data bounding box.
        """
        pad = self.mesh_spacing
        if self.polygon is not None:
            return [
                float(self.polygon[:, 0].min()) - pad,
                float(self.polygon[:, 0].max()) + pad,
                float(self.polygon[:, 1].min()) - pad,
                float(self.polygon[:, 1].max()) + pad,
            ]
        return [
            float(self.vf.lon.min() - pad),
            float(self.vf.lon.max() + pad),
            float(self.vf.lat.min() - pad),
            float(self.vf.lat.max() + pad),
        ]

    def compute(self) -> StrainResult:
        """Run the Wang (2012) strain-rate computation.

        Returns
        -------
        StrainResult
            Strain-rate tensors at triangle centroids.  The ``meta`` dict
            contains:
            - ``mesh_lon``, ``mesh_lat`` : mesh node positions
            - ``smooth_factor``          : the smoothing factor used
            - ``smooth_search``          : whether L-curve search was used
            - ``lcurve_factors``         : all factors tested (or None)
            - ``lcurve_wrss``            : WRSS at each factor (or None)
            - ``lcurve_roughness``       : roughness at each factor (or None)
        """
        vf = self.vf

        # ------------------------------------------------------------------
        # 1. Generate mesh (refined where stations are dense)
        # ------------------------------------------------------------------
        region = self.mesh_region if self.mesh_region is not None else self._default_region()
        logger.info(
            "  -> generating %s mesh (region=%s, min_spacing=%.2f°, "
            "max_stations=%d) ...",
            self.mesh_method, region, self.mesh_spacing, self.max_stations_per_cell,
        )
        if self.mesh_method == "gmsh":
            tri, mesh_lon, mesh_lat = generate_gmsh_mesh(
                region,
                self.mesh_spacing,
                vf.lon, vf.lat,
                polygon=self.polygon,
                max_stations=self.max_stations_per_cell,
                max_spacing=self.max_spacing,
            )
        elif self.mesh_method == "poisson":
            tri, mesh_lon, mesh_lat = generate_poisson_mesh(
                region,
                self.mesh_spacing,
                vf.lon, vf.lat,
                polygon=self.polygon,
                max_stations=self.max_stations_per_cell,
                max_spacing=self.max_spacing,
            )
        else:
            tri, mesh_lon, mesh_lat = generate_adaptive_mesh(
                region,
                self.mesh_spacing,
                vf.lon, vf.lat,
                polygon=self.polygon,
                max_stations=self.max_stations_per_cell,
                max_spacing=self.max_spacing,
                randomize=self.mesh_randomize,
                randomize_fraction=self.mesh_randomize_fraction,
            )
        logger.info("  -> mesh: %d vertices, %d triangles", len(mesh_lon), len(tri.simplices))

        # ------------------------------------------------------------------
        # 2. Build interpolation matrix G  (2*n_gps × 2*n_vtx)
        # ------------------------------------------------------------------
        logger.info("  -> building GPS interpolation matrix ...")
        G = build_gps_interpolation_matrix(tri, mesh_lon, mesh_lat, vf.lon, vf.lat)

        # ------------------------------------------------------------------
        # 3. Build smoothing matrix S  (2*n_vtx × 2*n_vtx)
        # ------------------------------------------------------------------
        logger.info("  -> building Laplacian smoothing matrix ...")
        S = build_laplacian_smoothing_matrix(
            tri, mesh_lon, mesh_lat, boundary_smooth=self.smooth_boundary
        )

        # ------------------------------------------------------------------
        # 4. Stack GPS velocities as [ve_all; vn_all]
        # ------------------------------------------------------------------
        v_gps = np.concatenate([vf.ve, vf.vn])

        # ------------------------------------------------------------------
        # 5. Find smoothing factor (L-curve or fixed)
        # ------------------------------------------------------------------
        lcurve_factors = None
        lcurve_wrss = None
        lcurve_roughness = None
        used_factor = self.smooth_factor

        if self.smooth_search:
            logger.info("  -> searching optimal smoothing (L-curve, range=%s) ...", self.smooth_range)
            used_factor, lcurve_factors, lcurve_wrss, lcurve_roughness = lcurve_search(
                G, S, v_gps, vf.se, vf.sn,
                smooth_range=self.smooth_range,
                smooth_step=self.smooth_step,
            )
            logger.info("  -> optimal log10(smooth) = %.3f", used_factor)

        # ------------------------------------------------------------------
        # 6. Solve for vertex velocities
        # ------------------------------------------------------------------
        logger.info("  -> solving for mesh-vertex velocities ...")
        v_mesh, wrss, roughness = solve_wang2012(
            G, S, v_gps, vf.se, vf.sn, used_factor
        )
        logger.info("  -> WRSS=%.2f  roughness=%.2f", wrss, roughness)

        # ------------------------------------------------------------------
        # 7. Compute strain per triangle
        # ------------------------------------------------------------------
        logger.info("  -> computing triangle strain ...")
        result = compute_triangle_strain(tri, mesh_lon, mesh_lat, v_mesh)

        # Attach metadata
        n_vtx = len(mesh_lon)
        result.meta.update(
            {
                "mesh_lon": mesh_lon,
                "mesh_lat": mesh_lat,
                "mesh_ve": v_mesh[:n_vtx],          # solved east velocity at each vertex
                "mesh_vn": v_mesh[n_vtx:],           # solved north velocity at each vertex
                "mesh_simplices": tri.simplices,     # triangle connectivity for plotting
                "smooth_factor": used_factor,
                "smooth_search": self.smooth_search,
                "wrss": wrss,
                "roughness": roughness,
                "lcurve_factors": lcurve_factors,
                "lcurve_wrss": lcurve_wrss,
                "lcurve_roughness": lcurve_roughness,
                "region": region,
                "mesh_spacing": self.mesh_spacing,
            }
        )

        return result
