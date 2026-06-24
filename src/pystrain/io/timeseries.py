"""GPS position time-series file readers.

Supports two formats:

- **PBO .pos** — SOPAC/UNR position time series (24 fixed-width columns).
- **PyTsfit .dat** — 7-column ASCII output from PyTsfit with offsets/seasonals
  already removed.

Each reader returns a :class:`~pystrain2.data.TimeSeries` dataclass instance.
"""

import logging
import os
from typing import Optional

import numpy as np

from pystrain2.data import TimeSeries
from pystrain2.gps_time import jd_to_decyrs

logger = logging.getLogger("pystrain2.io.timeseries")


# ---------------------------------------------------------------------------
# PBO .pos format
# ---------------------------------------------------------------------------

class PosData:
    """Read a PBO-format ``.pos`` position time-series file.

    The PBO format has a 37-line header followed by fixed-width data columns.
    Displacements are stored in meters and converted to millimeters on load.

    Parameters
    ----------
    posfile : str
        Path to the ``.pos`` file.
    """

    def __init__(self, posfile: str):
        self.site = ""
        self.lon: float = np.nan
        self.lat: float = np.nan
        self.height: float = np.nan
        self.MJD: np.ndarray = np.array([])
        self.decyr: np.ndarray = np.array([])
        self.N: np.ndarray = np.array([])
        self.E: np.ndarray = np.array([])
        self.U: np.ndarray = np.array([])
        self.SN: np.ndarray = np.array([])
        self.SE: np.ndarray = np.array([])
        self.SU: np.ndarray = np.array([])

        if os.path.isfile(posfile):
            self._load(posfile)
        else:
            logger.warning("Time-series file %s does not exist.", posfile)

    def _load(self, posfile: str) -> None:
        """Parse PBO .pos header and space-delimited data columns.

        Handles both legacy (v1.0) and newer (v1.1.1) PBO formats.
        """
        # --- header ---
        data_start_line = 0
        with open(posfile, "r") as fid:
            for lineno, line in enumerate(fid, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                # Site ID: "4-character ID: XXXX" (v1.1) or legacy ID line
                if "4-character ID:" in line:
                    self.site = stripped.split(":")[-1].strip()
                elif "ID:" in line and not self.site:
                    if len(line) >= 20:
                        self.site = line[16:20].strip()
                # NEU reference position
                if "NEU" in line and "Reference position" in line:
                    # "NEU Reference position :  lat  lon  height  (frame)"
                    nums = []
                    for p in stripped.split():
                        p_clean = p.strip("(),*")
                        try:
                            nums.append(float(p_clean))
                        except ValueError:
                            pass
                    if len(nums) >= 3:
                        self.lat = nums[-3]
                        self.lon = nums[-2]
                        self.height = nums[-1]
                elif stripped.startswith("NEU") and np.isnan(self.lat):
                    parts = stripped.split()
                    if len(parts) >= 7:
                        self.lat = float(parts[4].replace("*", "0"))
                        self.lon = float(parts[5].replace("*", "0"))
                        self.height = float(parts[6].replace("*", "0"))
                # Detect start of data: line after the "*YYYYMMDD ..." column desc
                if stripped.startswith("*YYYYMMDD"):
                    data_start_line = lineno + 1

        # --- data: space-delimited, skip all header lines ---
        if data_start_line == 0:
            # Fallback: try legacy skip_header=37
            data_start_line = 37

        try:
            # Use space-delimited parsing; the last column ("final"/"rapid")
            # is a string, so we must read as strings and convert manually
            raw = np.genfromtxt(
                posfile, skip_header=data_start_line,
                dtype=str, comments=None,
            )
        except Exception:
            logger.warning("Could not parse %s — returning empty.", posfile)
            return

        if raw.ndim == 1:
            raw = raw.reshape((1, -1))
        if raw.size == 0:
            return

        # Convert numeric columns; last column ("final"/"rapid") is ignored
        ncols = raw.shape[1]
        data = np.full((raw.shape[0], ncols - 1), np.nan)
        for c in range(ncols - 1):
            try:
                data[:, c] = raw[:, c].astype(float)
            except (ValueError, IndexError):
                pass

        # Remove rows where dN/dE/dU (cols 15,16,17) are NaN
        if data.shape[1] <= 17:
            logger.warning("Unexpected column count %d in %s", data.shape[1], posfile)
            return
        nan_mask = np.isnan(data[:, [15, 16, 17]]).any(axis=1)
        valid = ~nan_mask

        self.MJD = data[valid, 2]
        self.N = data[valid, 15] * 1e3   # m → mm
        self.E = data[valid, 16] * 1e3
        self.U = data[valid, 17] * 1e3
        self.SN = data[valid, 18] * 1e3
        self.SE = data[valid, 19] * 1e3
        self.SU = data[valid, 20] * 1e3

        # MJD → decimal year
        self.decyr = np.array([jd_to_decyrs(float(m)) for m in self.MJD])

    def to_timeseries(self) -> TimeSeries:
        """Return a :class:`TimeSeries` dataclass."""
        return TimeSeries(
            site=self.site,
            lon=float(self.lon),
            lat=float(self.lat),
            height=float(self.height),
            decyr=self.decyr,
            E=self.E,
            N=self.N,
            U=self.U,
            SE=self.SE,
            SN=self.SN,
            SU=self.SU,
        )


# ---------------------------------------------------------------------------
# PyTsfit .dat format
# ---------------------------------------------------------------------------

class DatData:
    """Read a PyTsfit-format ``.dat`` position time-series file.

    Expected columns: ``decyr  N  E  U  SN  SE  SU``
    (7 columns, space-separated, with displacements in mm).

    Parameters
    ----------
    neufile : str
        Path to the ``.dat`` file.
    scale : float
        Multiplier applied to all displacement columns (default 1.0 = mm).
    """

    def __init__(self, neufile: str, scale: float = 1.0):
        self.site = os.path.basename(neufile)[:4]
        self.lon: float = np.nan
        self.lat: float = np.nan
        self.height: float = np.nan
        self.decyr: np.ndarray = np.array([])
        self.N: np.ndarray = np.array([])
        self.E: np.ndarray = np.array([])
        self.U: np.ndarray = np.array([])
        self.SN: np.ndarray = np.array([])
        self.SE: np.ndarray = np.array([])
        self.SU: np.ndarray = np.array([])

        if os.path.isfile(neufile):
            self._load(neufile, scale)
        else:
            logger.warning("Time-series file %s does not exist.", neufile)

    def _load(self, neufile: str, scale: float) -> None:
        """Parse 7-column space-separated data."""
        try:
            data = np.genfromtxt(neufile)
        except Exception:
            logger.warning("Could not parse %s — returning empty.", neufile)
            return

        if data.ndim == 1:
            data = data.reshape((1, -1))

        # Remove rows where N/E/U are NaN
        nan_mask = np.isnan(data[:, [1, 2, 3]]).any(axis=1)
        valid = ~nan_mask

        self.decyr = data[valid, 0]
        self.N = data[valid, 1] * scale
        self.E = data[valid, 2] * scale
        self.U = data[valid, 3] * scale
        self.SN = data[valid, 4] * scale
        self.SE = data[valid, 5] * scale
        self.SU = data[valid, 6] * scale

    def to_timeseries(self) -> TimeSeries:
        """Return a :class:`TimeSeries` dataclass."""
        return TimeSeries(
            site=self.site,
            lon=float(self.lon),
            lat=float(self.lat),
            height=float(self.height),
            decyr=self.decyr,
            E=self.E,
            N=self.N,
            U=self.U,
            SE=self.SE,
            SN=self.SN,
            SU=self.SU,
        )
