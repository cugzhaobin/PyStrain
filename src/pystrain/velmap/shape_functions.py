"""Velmap-style strain-rate estimation using linear triangular shape functions."""

from typing import Optional
import numpy as np

from pystrain.data import StrainResult, VelocityField
from pystrain.strain.tensor import (
    principal_strain,
    strain_invariants,
    velocity_gradient_to_strain,
)
from pystrain.triangulation import (
    build_adjacency,
    compute_shape_function_derivatives,
    delaunay_triangulation,
)


def _smooth_triangles(
    values: np.ndarray,
    adjacency: dict,
    weight: float,
    iterations: int,
) -> np.ndarray:
    """Laplacian-style smoothing over triangle neighbors."""
    smoothed = values.copy()
    for _ in range(iterations):
        new_vals = smoothed.copy()
        for idx, neighbors in adjacency.items():
            if len(neighbors) == 0:
                continue
            n_neigh = len(neighbors)
            boundary_factor = 0.5 if n_neigh < 3 else 1.0
            w = weight * boundary_factor
            new_vals[idx] = (1 - w) * smoothed[idx] + w * np.mean(smoothed[neighbors])
        smoothed = new_vals
    return smoothed


class VelmapStrainRate:
    """Compute strain rate using linear triangular finite-element shape functions."""

    def __init__(
        self,
        vf: VelocityField,
        polygon: Optional[np.ndarray] = None,
        min_angle_deg: float = 10.0,
        max_edge_pctl: float = 95.0,
        max_edge_factor: float = 1.5,
        min_area_ratio: float = 0.1,
        max_edge_km: Optional[float] = None,
        smooth_weight: float = 0.3,
        smooth_iter: int = 3,
        projection: str = "utm",
    ):
        self.vf = vf
        self.polygon = polygon
        self.min_angle_deg = min_angle_deg
        self.max_edge_pctl = max_edge_pctl
        self.max_edge_factor = max_edge_factor
        self.min_area_ratio = min_area_ratio
        self.max_edge_km = max_edge_km
        self.smooth_weight = smooth_weight
        self.smooth_iter = smooth_iter
        self.projection = projection

    def compute(self) -> StrainResult:
        tri, good_triangles, xy, proj_params = delaunay_triangulation(
            self.vf.lon,
            self.vf.lat,
            polygon=self.polygon,
            min_angle_deg=self.min_angle_deg,
            max_edge_pctl=self.max_edge_pctl,
            max_edge_factor=self.max_edge_factor,
            min_area_ratio=self.min_area_ratio,
            max_edge_km=self.max_edge_km,
        )

        B_list, good_indices, areas = compute_shape_function_derivatives(
            tri, xy, good_triangles
        )

        n_good = len(good_indices)
        if n_good == 0:
            raise ValueError("No valid triangles after quality control.")

        e_ee = np.empty(n_good)
        e_en = np.empty(n_good)
        e_nn = np.empty(n_good)
        omega = np.empty(n_good)

        centroids_lon = np.empty(n_good)
        centroids_lat = np.empty(n_good)
        tri_ids = np.empty(n_good, dtype=int)

        for k, tri_idx in enumerate(good_indices):
            simplex = tri.simplices[tri_idx]
            lon_tri = self.vf.lon[simplex]
            lat_tri = self.vf.lat[simplex]
            centroids_lon[k] = np.mean(lon_tri)
            centroids_lat[k] = np.mean(lat_tri)
            tri_ids[k] = tri_idx

            B = B_list[k]
            ve_tri = self.vf.ve[simplex]
            vn_tri = self.vf.vn[simplex]

            L = np.vstack(
                [
                    B @ ve_tri,
                    B @ vn_tri,
                ]
            )  # (2, 2)

            ee, en, nn, om = velocity_gradient_to_strain(L)
            e_ee[k] = ee
            e_en[k] = en
            e_nn[k] = nn
            omega[k] = om

        # Convert mm/(km·yr) -> nstrain/yr
        unit_factor = 1000.0
        e_ee *= unit_factor
        e_en *= unit_factor
        e_nn *= unit_factor
        omega *= unit_factor

        # Optional smoothing on tensor components
        if self.smooth_iter > 0 and self.smooth_weight > 0:
            adjacency = build_adjacency(tri, good_triangles)
            # build_adjacency keys are original triangle indices; remap to
            # compressed 0..n_good-1 indices used by the component arrays.
            idx_to_compressed = {idx: k for k, idx in enumerate(good_indices)}
            compressed_adjacency = {
                idx_to_compressed[idx]: [
                    idx_to_compressed[n] for n in neighbors if n in idx_to_compressed
                ]
                for idx, neighbors in adjacency.items()
            }
            e_ee = _smooth_triangles(
                e_ee, compressed_adjacency, self.smooth_weight, self.smooth_iter
            )
            e_en = _smooth_triangles(
                e_en, compressed_adjacency, self.smooth_weight, self.smooth_iter
            )
            e_nn = _smooth_triangles(
                e_nn, compressed_adjacency, self.smooth_weight, self.smooth_iter
            )
            omega = _smooth_triangles(
                omega, compressed_adjacency, self.smooth_weight, self.smooth_iter
            )

        e1, e2, azimuth = principal_strain(e_ee, e_en, e_nn)
        dilation, shear, sec_inv = strain_invariants(e1, e2)

        return StrainResult(
            lon=centroids_lon,
            lat=centroids_lat,
            exx=e_ee,
            exy=e_en,
            eyy=e_nn,
            omega=omega,
            e1=e1,
            e2=e2,
            azimuth=azimuth,
            shear=shear,
            dilation=dilation,
            sec_inv=sec_inv,
            ve=np.zeros(n_good),
            vn=np.zeros(n_good),
            meta={
                "tri_ids": tri_ids,
                "areas": areas,
                "proj_params": proj_params,
            },
        )
