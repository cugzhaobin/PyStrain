"""Delaunay triangulation-based strain-rate estimation."""

import logging
from typing import Optional
import numpy as np

from pystrain.data import StrainResult, VelocityField
from pystrain.geodesy import local_to_origin_projection
from pystrain.strain.lsq import estimate_strain_rate
from pystrain.triangulation import delaunay_triangulation

logger = logging.getLogger("pystrain")


class DelaunayStrainRate:
    """Compute strain rate at Delaunay triangle centroids via least squares."""

    def __init__(
        self,
        vf: VelocityField,
        polygon: Optional[np.ndarray] = None,
        min_angle_deg: float = 10.0,
        max_edge_pctl: float = 95.0,
        max_edge_factor: float = 1.5,
        min_area_ratio: float = 0.1,
        max_edge_km: Optional[float] = None,
        projection: str = "utm",
    ):
        self.vf = vf
        self.polygon = polygon
        self.min_angle_deg = min_angle_deg
        self.max_edge_pctl = max_edge_pctl
        self.max_edge_factor = max_edge_factor
        self.min_area_ratio = min_area_ratio
        self.max_edge_km = max_edge_km
        self.projection = projection

    def compute(self) -> StrainResult:
        try:
            from tqdm import tqdm
            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

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

        good_indices = np.where(good_triangles)[0]
        n_good = len(good_indices)
        if n_good == 0:
            raise ValueError("No valid triangles after quality control.")
        logger.info("  -> computing strain at %d triangle centroids ...", n_good)

        centroids_lon = np.empty(n_good)
        centroids_lat = np.empty(n_good)
        exx = np.empty(n_good)
        exy = np.empty(n_good)
        eyy = np.empty(n_good)
        omega = np.empty(n_good)
        e1 = np.empty(n_good)
        e2 = np.empty(n_good)
        azimuth = np.empty(n_good)
        shear = np.empty(n_good)
        dilation = np.empty(n_good)
        sec_inv = np.empty(n_good)
        ve_avg = np.empty(n_good)
        vn_avg = np.empty(n_good)
        tri_ids = np.empty(n_good, dtype=int)

        idx_iter = range(n_good)
        if _has_tqdm:
            idx_iter = tqdm(idx_iter, desc="  triangles", unit="tri")

        for k in idx_iter:
            tri_idx = good_indices[k]
            simplex = tri.simplices[tri_idx]
            lon_tri = self.vf.lon[simplex]
            lat_tri = self.vf.lat[simplex]
            ve_tri = self.vf.ve[simplex]
            vn_tri = self.vf.vn[simplex]
            se_tri = self.vf.se[simplex]
            sn_tri = self.vf.sn[simplex]

            centroids_lon[k] = np.mean(lon_tri)
            centroids_lat[k] = np.mean(lat_tri)

            # Local coordinates relative to centroid
            x, y, _ = local_to_origin_projection(
                lon_tri, lat_tri,
                (centroids_lon[k], centroids_lat[k]),
                method=self.projection,
            )

            res = estimate_strain_rate(
                x, y, ve_tri, vn_tri, se_tri, sn_tri,
                normalize=True,
            )

            exx[k] = res["exx"]
            exy[k] = res["exy"]
            eyy[k] = res["eyy"]
            omega[k] = res["omega"]
            e1[k] = res["e1"]
            e2[k] = res["e2"]
            azimuth[k] = res["azimuth"]
            shear[k] = res["shear"]
            dilation[k] = res["dilation"]
            sec_inv[k] = res["sec_inv"]
            ve_avg[k] = res["dx"]
            vn_avg[k] = res["dy"]
            tri_ids[k] = tri_idx

        return StrainResult(
            lon=centroids_lon,
            lat=centroids_lat,
            exx=exx,
            exy=exy,
            eyy=eyy,
            omega=omega,
            e1=e1,
            e2=e2,
            azimuth=azimuth,
            shear=shear,
            dilation=dilation,
            sec_inv=sec_inv,
            ve=ve_avg,
            vn=vn_avg,
            meta={
                "tri_ids": tri_ids,
                "proj_params": proj_params,
            },
        )
