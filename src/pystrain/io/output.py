"""Output writers for strain results and reports."""

from pathlib import Path
from typing import Dict, List

import numpy as np

from pystrain2.data import StrainResult


# Strain-rate quantities are in nanostrain/year (1e-9 yr⁻¹).
# Velocities are in mm/yr.  Coordinates are decimal degrees.
_STRAIN_HEADER = (
    "# lon(°) lat(°) ve(mm/yr) vn(mm/yr) "
    "exx(ns/yr) exy(ns/yr) eyy(ns/yr) omega(nrad/yr) "
    "e1(ns/yr) e2(ns/yr) shr(ns/yr) dil(ns/yr) "
    "inv2(ns/yr) theta(°)"
)

# Column widths for formatted output (fixed-width, right-aligned).
# lon, lat, azimuth use fixed-point; all others use scientific notation.
_FMT = {
    "lon": "8.2f",
    "lat": "8.2f",
    "ve": "10.3e",
    "vn": "10.3e",
    "strain": "10.3e",   #  x.xxxe±nn format for nanostrain rates
    "omega": "10.3e",
    "azimuth": "7.2f",
}

# Columns written in order, with their keys and format groups
_COLUMNS = [
    ("lon",     "lon"),
    ("lat",     "lat"),
    ("ve",      "ve"),
    ("vn",      "vn"),
    ("exx",     "strain"),
    ("exy",     "strain"),
    ("eyy",     "strain"),
    ("omega",   "omega"),
    ("e1",      "strain"),
    ("e2",      "strain"),
    ("shear",   "strain"),
    ("dilation","strain"),
    ("sec_inv", "strain"),
    ("azimuth", "azimuth"),
]


def _fmt_val(val: float, key: str) -> str:
    """Format a single value with the prescribed width/precision.

    ``key`` selects the column format group (e.g. "lon", "strain").
    NaN values are replaced by the string ``"       nan"``.
    """
    if not np.isfinite(val):
        return f"{'nan':>{_FMT[key].split('.')[0]}}"
    template = f"{{:{_FMT[key]}}}"
    return template.format(val)


def write_strain_result(
    filepath: str,
    result: StrainResult,
    include_uncertainty: bool = False,
) -> None:
    """Write strain result to a fixed-width, aligned text file.

    Columns
    -------
    lon(°)  lat(°)  ve(mm/yr)  vn(mm/yr)
    exx(ns/yr)  exy(ns/yr)  eyy(ns/yr)  omega(nrad/yr)
    e1(ns/yr)  e2(ns/yr)  shr(ns/yr)  dil(ns/yr)  inv2(ns/yr)  theta(°)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fid:
        fid.write(_STRAIN_HEADER + "\n")
        for i in range(len(result)):
            row_parts = []
            for attr, fmt_key in _COLUMNS:
                val = getattr(result, attr)[i]
                row_parts.append(_fmt_val(val, fmt_key))
            fid.write(" ".join(row_parts) + "\n")


def write_outliers(
    filepath: str,
    outliers: List[Dict],
) -> None:
    """Write outlier report."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# name           lon(°)    lat(°)  resid(mm/yr)  reason\n")
        for o in outliers:
            fid.write(
                f"{o['name']:<14s}  "
                f"{o['lon']:9.4f}  {o['lat']:9.4f}  "
                f"{o.get('residual', 0):13.4f}  {o['reason']}\n"
            )


def write_report(
    filepath: str,
    summary: Dict,
) -> None:
    """Write a human-readable summary report."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# PyStrain2 computation report\n")
        for key, value in summary.items():
            fid.write(f"{key}: {value}\n")
