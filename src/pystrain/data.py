"""Shared data containers for PyStrain2."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class VelocityField:
    """Container for GNSS velocity field data."""

    lon: np.ndarray
    lat: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    se: np.ndarray
    sn: np.ndarray
    rho: np.ndarray
    names: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.lon)
        for arr in (self.lat, self.ve, self.vn, self.se, self.sn, self.rho, self.names):
            if len(arr) != n:
                raise ValueError("All velocity-field arrays must have the same length.")

    def __len__(self):
        return len(self.lon)

    def subset(self, mask: np.ndarray) -> "VelocityField":
        """Return a subset of sites according to a boolean mask."""
        return VelocityField(
            lon=self.lon[mask],
            lat=self.lat[mask],
            ve=self.ve[mask],
            vn=self.vn[mask],
            se=self.se[mask],
            sn=self.sn[mask],
            rho=self.rho[mask],
            names=self.names[mask],
            meta=dict(self.meta),
        )

    def keep_indices(self) -> np.ndarray:
        """Return indices of currently kept sites."""
        return np.arange(len(self))


@dataclass
class StrainResult:
    """Container for strain-rate computation results."""

    lon: np.ndarray
    lat: np.ndarray
    exx: np.ndarray
    exy: np.ndarray
    eyy: np.ndarray
    omega: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    azimuth: np.ndarray
    shear: np.ndarray
    dilation: np.ndarray
    sec_inv: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.lon)
        for arr in (
            self.lat, self.exx, self.exy, self.eyy, self.omega,
            self.e1, self.e2, self.azimuth, self.shear, self.dilation,
            self.sec_inv, self.ve, self.vn,
        ):
            if len(arr) != n:
                raise ValueError("All StrainResult arrays must have the same length.")

    def __len__(self):
        return len(self.lon)


@dataclass
class TimeSeries:
    """Container for a single site coordinate time series."""

    site: str
    lon: float
    lat: float
    height: float
    decyr: np.ndarray
    E: np.ndarray
    N: np.ndarray
    U: np.ndarray
    SE: Optional[np.ndarray] = None
    SN: Optional[np.ndarray] = None
    SU: Optional[np.ndarray] = None


@dataclass
class TimeSeriesCollection:
    """Container for aligned multi-site time series."""

    sites: List[str]
    decyr: np.ndarray
    E: np.ndarray
    N: np.ndarray
    U: np.ndarray
    SE: np.ndarray
    SN: np.ndarray
    SU: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    def __post_init__(self):
        n_sites = len(self.sites)
        n_epochs = len(self.decyr)
        for arr in (self.E, self.N, self.U, self.SE, self.SN, self.SU):
            if arr.shape != (n_epochs, n_sites):
                raise ValueError("TimeSeriesCollection arrays must have shape (n_epochs, n_sites).")
        for arr in (self.lon, self.lat):
            if len(arr) != n_sites:
                raise ValueError("lon/lat must match number of sites.")


@dataclass
class StrainTimeSeriesResult:
    """Container for per-epoch strain-rate estimates at one location.

    Each array has shape ``(n_epochs,)`` — one value per epoch in the
    aligned time-series grid.  NaN indicates epochs where no valid
    solution could be obtained.
    """

    lon: float
    lat: float
    decyr: np.ndarray
    exx: np.ndarray
    exy: np.ndarray
    eyy: np.ndarray
    omega: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    azimuth: np.ndarray
    shear: np.ndarray
    dilation: np.ndarray
    sec_inv: np.ndarray
    ve: np.ndarray
    vn: np.ndarray
    condition_number: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.decyr)
        for arr in (
            self.exx, self.exy, self.eyy, self.omega,
            self.e1, self.e2, self.azimuth, self.shear,
            self.dilation, self.sec_inv, self.ve, self.vn,
            self.condition_number,
        ):
            if len(arr) != n:
                raise ValueError(
                    "All StrainTimeSeriesResult arrays must have "
                    f"length n_epochs ({n}), got {len(arr)}."
                )
