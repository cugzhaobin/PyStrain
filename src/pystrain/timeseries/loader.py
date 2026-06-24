"""Multi-site GPS time-series loading and alignment.

The :class:`TimeSeriesLoader` loads position time series for multiple GPS
stations and aligns them to a common daily epoch grid.  Raw displacements
are kept as-is — **no per-site velocity estimation or detrending is
performed**, because strain depends only on spatial displacement gradients,
not absolute offsets.  Subtracting a per-site constant (which is what
detrending does) is entirely absorbed by the translation parameters
*Ux, Uy* in the least-squares strain solver and has zero effect on the
estimated strain components *τxx, τxy, τyy, ω*.

This avoids the bias introduced by RANSAC velocity estimation over short
time windows, transient deformation, or incomplete seasonal cycles.
"""

import glob
import logging
import os
from typing import List, Optional

import numpy as np

from pystrain.data import TimeSeriesCollection
from pystrain.gps_time import daily_epoch_grid, decyrs_to_mjd
from pystrain.io.timeseries import DatData, PosData

logger = logging.getLogger("pystrain.timeseries.loader")


class TimeSeriesLoader:
    """Load and align multi-site GPS position time series to a common
    daily epoch grid.

    Raw displacements are returned **without** velocity estimation or
    detrending.  The strain solver's translation parameters absorb any
    per-site constant offsets, making detrending unnecessary for strain
    computation.

    Parameters
    ----------
    gps_info_file : str
        Path to a file with columns ``lon lat height site``.
    ts_type : str
        Time-series type: ``"pos"`` (PBO format) or ``"dat"`` (PyTsfit).
    ts_path : str
        Directory containing the per-site time-series files.
    sepoch : float
        Start epoch (decimal year).
    eepoch : float
        End epoch (decimal year).
    site_list : list of str, optional
        If provided, only these sites are loaded.
    """

    def __init__(
        self,
        gps_info_file: str,
        ts_type: str,
        ts_path: str,
        sepoch: float,
        eepoch: float,
        site_list: Optional[List[str]] = None,
    ):
        self.gps_info_file = gps_info_file
        self.ts_type = ts_type
        self.ts_path = ts_path
        self.sepoch = sepoch
        self.eepoch = eepoch
        self.site_list = site_list or []

        self.gps_lon: np.ndarray = np.array([])
        self.gps_lat: np.ndarray = np.array([])
        self.sitenames: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> TimeSeriesCollection:
        """Run the full load → align pipeline.

        Returns
        -------
        TimeSeriesCollection
            Aligned multi-site daily position time series.  Displacements
            are in mm.  Epochs where a site has no data are NaN.
        """
        self._load_gps_info()
        tslist = self._load_timeseries()
        return self._align_timeseries(tslist)

    # ------------------------------------------------------------------
    # Step 1: read GPS site metadata
    # ------------------------------------------------------------------

    def _load_gps_info(self) -> None:
        """Parse the GPS info file (lon lat height site)."""
        if not os.path.isfile(self.gps_info_file):
            raise FileNotFoundError(f"GPS info file not found: {self.gps_info_file}")

        info = np.genfromtxt(self.gps_info_file, comments="#", dtype=str)
        if info.ndim == 1:
            info = info.reshape((1, -1))

        lon = info[:, 0].astype(float)
        lat = info[:, 1].astype(float)
        site = info[:, 3]

        # Filter to user-requested sites
        if len(self.site_list) > 0:
            mask = np.isin(site, self.site_list)
            if not mask.any():
                raise ValueError(
                    "None of the requested sites found in GPS info file."
                )
            lon = lon[mask]
            lat = lat[mask]
            site = site[mask]

        self.gps_lon = lon
        self.gps_lat = lat
        self.sitenames = site
        logger.info("%d sites loaded from GPS info file.", len(self.sitenames))

    # ------------------------------------------------------------------
    # Step 2: load per-site time series
    # ------------------------------------------------------------------

    def _load_timeseries(self) -> list:
        """Load time series for every site in ``sitenames``.

        Returns a list of :class:`PosData` / :class:`DatData` instances
        (only sites that have data within [sepoch, eepoch]).
        """
        tslist = []
        keep_idx = []

        for i, stname in enumerate(self.sitenames):
            if self.ts_type == "pos":
                pattern = os.path.join(self.ts_path, stname + "*.pos")
                matches = glob.glob(pattern)
                if not matches:
                    logger.debug("No .pos file for site %s", stname)
                    continue
                ts_obj = PosData(matches[0])
            elif self.ts_type == "dat":
                pattern = os.path.join(self.ts_path, stname + "*_obs.dat")
                matches = glob.glob(pattern)
                if not matches:
                    logger.debug("No .dat file for site %s", stname)
                    continue
                ts_obj = DatData(matches[0])
            else:
                raise ValueError(f"Unknown ts_type: {self.ts_type}")

            if len(ts_obj.decyr) == 0:
                logger.debug("Empty time series for site %s", stname)
                continue

            if ts_obj.decyr.min() > self.eepoch or ts_obj.decyr.max() < self.sepoch:
                logger.debug(
                    "Site %s: no data in [%.2f, %.2f]", stname, self.sepoch, self.eepoch
                )
                continue

            # Transfer lon/lat from info file if not in the time-series file
            if np.isnan(ts_obj.lon) or ts_obj.lat == 0.0:
                ts_obj.lon = float(self.gps_lon[i])
                ts_obj.lat = float(self.gps_lat[i])

            tslist.append(ts_obj)
            keep_idx.append(i)
            logger.info("Loaded time series for site %s", stname)

        # Filter metadata to match loaded sites
        self.gps_lon = self.gps_lon[keep_idx]
        self.gps_lat = self.gps_lat[keep_idx]
        self.sitenames = self.sitenames[keep_idx]

        logger.info("%d time series loaded.", len(tslist))
        return tslist

    # ------------------------------------------------------------------
    # Step 3: align to common epoch grid (no detrending)
    # ------------------------------------------------------------------

    def _align_timeseries(self, tslist: list) -> TimeSeriesCollection:
        """Align all loaded time series to a common daily epoch grid.

        For each site and daily epoch, the nearest original record
        (within tolerance) is used.  No velocity estimation or
        detrending is applied — raw displacements are returned.
        """
        if self.ts_type == "pos":
            return self._align_pos(tslist)
        else:
            return self._align_dat(tslist)

    # ------------------------------------------------------------------
    def _align_pos(self, tslist: list) -> TimeSeriesCollection:
        """Align .pos (MJD-based) time series to daily grid."""
        smjd = decyrs_to_mjd(self.sepoch)
        emjd = decyrs_to_mjd(self.eepoch)
        mjd_grid = np.arange(smjd, emjd)
        decyr_grid = daily_epoch_grid(self.sepoch, self.eepoch)

        ndays = len(mjd_grid)
        nsite = len(tslist)

        N = np.full((ndays, nsite), np.nan)
        E = np.full((ndays, nsite), np.nan)
        SN = np.full((ndays, nsite), np.nan)
        SE = np.full((ndays, nsite), np.nan)
        U_arr = np.full((ndays, nsite), np.nan)
        SU_arr = np.full((ndays, nsite), np.nan)

        for i in range(nsite):
            ts = tslist[i]
            for j in range(ndays):
                idx = np.where(np.abs(ts.MJD - mjd_grid[j]) < 0.015)[0]
                if len(idx) == 1:
                    N[j, i] = ts.N[idx[0]]
                    E[j, i] = ts.E[idx[0]]
                    SN[j, i] = ts.SN[idx[0]]
                    SE[j, i] = ts.SE[idx[0]]
                    if hasattr(ts, "U") and len(ts.U) > 0:
                        U_arr[j, i] = ts.U[idx[0]]
                        SU_arr[j, i] = ts.SU[idx[0]]

        return TimeSeriesCollection(
            sites=list(self.sitenames),
            decyr=decyr_grid,
            E=E, N=N, U=U_arr,
            SE=SE, SN=SN, SU=SU_arr,
            lon=self.gps_lon.copy(),
            lat=self.gps_lat.copy(),
        )

    # ------------------------------------------------------------------
    def _align_dat(self, tslist: list) -> TimeSeriesCollection:
        """Align .dat (decyr-based) time series to daily grid."""
        decyr_grid = daily_epoch_grid(self.sepoch, self.eepoch)
        ndays = len(decyr_grid)
        nsite = len(tslist)

        N = np.full((ndays, nsite), np.nan)
        E = np.full((ndays, nsite), np.nan)
        SN = np.full((ndays, nsite), np.nan)
        SE = np.full((ndays, nsite), np.nan)
        U_arr = np.full((ndays, nsite), np.nan)
        SU_arr = np.full((ndays, nsite), np.nan)

        for i in range(nsite):
            ts = tslist[i]
            for j in range(ndays):
                idx = np.where(np.abs(ts.decyr - decyr_grid[j]) < 0.0013)[0]
                if len(idx) == 1:
                    N[j, i] = ts.N[idx[0]]
                    E[j, i] = ts.E[idx[0]]
                    SN[j, i] = ts.SN[idx[0]]
                    SE[j, i] = ts.SE[idx[0]]
                    if hasattr(ts, "U") and len(ts.U) > 0:
                        U_arr[j, i] = ts.U[idx[0]]
                        SU_arr[j, i] = ts.SU[idx[0]]

        return TimeSeriesCollection(
            sites=list(self.sitenames),
            decyr=decyr_grid,
            E=E, N=N, U=U_arr,
            SE=SE, SN=SN, SU=SU_arr,
            lon=self.gps_lon.copy(),
            lat=self.gps_lat.copy(),
        )
