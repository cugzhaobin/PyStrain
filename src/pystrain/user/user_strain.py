"""User-specified point strain-rate estimation."""

from pathlib import Path
from typing import Optional
import numpy as np

from pystrain.data import StrainResult, VelocityField
from pystrain.geodesy import distance_azimuth, local_to_origin_projection
from pystrain.grid.shen_wang import _distance_weight
from pystrain.strain.lsq import estimate_strain_rate


class UserStrainRate:
    """Compute strain rate at user-specified lon/lat points."""

    def __init__(
        self,
        user_points_file: str,
        vf: VelocityField,
        maxdist_km: float = 200.0,
        min_sites: int = 8,
        weight_distance: Optional[float] = None,
        projection: str = "utm",
    ):
        self.vf = vf
        self.maxdist_km = maxdist_km
        self.min_sites = min_sites
        self.weight_distance = weight_distance
        self.projection = projection
        self.user_lon, self.user_lat, self.user_names = self._read_points(user_points_file)

    def _read_points(self, filepath: str):
        """Read user points file: lon lat name."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"User points file not found: {filepath}")

        lons, lats, names = [], [], []
        with open(path, "r", encoding="utf-8") as fid:
            for raw in fid:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                lons.append(float(parts[0]))
                lats.append(float(parts[1]))
                names.append(parts[2] if len(parts) > 2 else "")

        return np.array(lons), np.array(lats), np.array(names)

    def compute(self) -> StrainResult:
        """Compute strain rate at each user point."""
        n = len(self.user_lon)
        exx = np.full(n, np.nan)
        exy = np.full(n, np.nan)
        eyy = np.full(n, np.nan)
        omega = np.full(n, np.nan)
        e1 = np.full(n, np.nan)
        e2 = np.full(n, np.nan)
        azimuth = np.full(n, np.nan)
        shear = np.full(n, np.nan)
        dilation = np.full(n, np.nan)
        sec_inv = np.full(n, np.nan)
        ve_avg = np.full(n, np.nan)
        vn_avg = np.full(n, np.nan)

        for i in range(n):
            d, _ = distance_azimuth(
                np.full(len(self.vf), self.user_lon[i]),
                np.full(len(self.vf), self.user_lat[i]),
                self.vf.lon,
                self.vf.lat,
            )
            mask = d <= self.maxdist_km
            if np.sum(mask) < self.min_sites:
                continue

            # If weight_distance not given, use 1.5 * distance to min_site-th nearest
            if self.weight_distance is None:
                sorted_d = np.sort(d[mask])
                R = 1.5 * sorted_d[min(self.min_sites - 1, len(sorted_d) - 1)]
            else:
                R = self.weight_distance

            weights = _distance_weight(d[mask], R, "gaussian")

            x, y, _ = local_to_origin_projection(
                self.vf.lon[mask],
                self.vf.lat[mask],
                (self.user_lon[i], self.user_lat[i]),
                method=self.projection,
            )

            try:
                res = estimate_strain_rate(
                    x, y,
                    self.vf.ve[mask],
                    self.vf.vn[mask],
                    self.vf.se[mask],
                    self.vf.sn[mask],
                    weights=weights,
                    normalize=True,
                )
            except Exception:
                continue

            exx[i] = res["exx"]
            exy[i] = res["exy"]
            eyy[i] = res["eyy"]
            omega[i] = res["omega"]
            e1[i] = res["e1"]
            e2[i] = res["e2"]
            azimuth[i] = res["azimuth"]
            shear[i] = res["shear"]
            dilation[i] = res["dilation"]
            sec_inv[i] = res["sec_inv"]
            ve_avg[i] = res["dx"]
            vn_avg[i] = res["dy"]

        return StrainResult(
            lon=self.user_lon,
            lat=self.user_lat,
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
            meta={"names": self.user_names},
        )
