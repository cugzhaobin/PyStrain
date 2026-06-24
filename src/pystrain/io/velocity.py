"""Velocity field file readers and writers."""

from pathlib import Path
from typing import List, Tuple
import numpy as np

from pystrain2.data import VelocityField


class VelocityParseError(Exception):
    """Raised when velocity file parsing fails in strict mode."""
    pass


def _parse_line_gmt8(parts: List[str], line_num: int) -> Tuple[float, ...]:
    if len(parts) < 8:
        raise ValueError(f"GMT8 line {line_num}: expected 8 columns, got {len(parts)}")
    lon, lat, ve, vn, se, sn, rho = map(float, parts[:7])
    name = parts[7]
    return lon, lat, ve, vn, se, sn, rho, name


def _parse_line_gmt7(parts: List[str], line_num: int) -> Tuple[float, ...]:
    if len(parts) < 7:
        raise ValueError(f"GMT7 line {line_num}: expected 7 columns, got {len(parts)}")
    lon, lat, ve, vn, se, sn = map(float, parts[:6])
    name = parts[6]
    return lon, lat, ve, vn, se, sn, 0.0, name


def _parse_line_globk(parts: List[str], line_num: int) -> Tuple[float, ...]:
    if len(parts) < 13:
        raise ValueError(f"GLOBK line {line_num}: expected 13 columns, got {len(parts)}")
    # GLOBK: lon lat ve vn Ve_adj Vn_adj Se Sn Rho Vu Vu_adj Su site
    # Use original ve, vn (columns 3,4) and Se, Sn (columns 8,9)?
    # Legacy PyStrain uses Ve_adj/Vn_adj (5,6) and Se/Sn (7,8)
    lon = float(parts[0])
    lat = float(parts[1])
    ve = float(parts[4])
    vn = float(parts[5])
    se = float(parts[6])
    sn = float(parts[7])
    rho = float(parts[8])
    name = parts[12]
    return lon, lat, ve, vn, se, sn, rho, name


def read_velocity_file(
    filepath: str,
    fmt: str = "auto",
    strict: bool = False,
) -> VelocityField:
    """Read a GNSS velocity file.

    Supported formats:
      - gmt8: lon lat Ve Vn Se Sn Rho SiteName
      - gmt7: lon lat Ve Vn Se Sn SiteName
      - globk: lon lat ve vn Ve_adj Vn_adj Se Sn Rho Vu Vu_adj Su site
      - auto: choose by column count

    Parameters
    ----------
    filepath : str
        Path to velocity file.
    fmt : str
        Format name or 'auto'.
    strict : bool
        If True, raise VelocityParseError on malformed rows.

    Returns
    -------
    VelocityField
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    rows = []
    parse_log = []
    with open(path, "r", encoding="utf-8") as fid:
        for line_num, raw in enumerate(fid, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            detected = fmt
            if detected == "auto":
                if len(parts) == 8:
                    detected = "gmt8"
                elif len(parts) == 7:
                    detected = "gmt7"
                elif len(parts) == 13:
                    detected = "globk"
                else:
                    msg = f"Line {line_num}: unsupported column count {len(parts)}"
                    parse_log.append(msg)
                    if strict:
                        raise VelocityParseError(msg)
                    continue

            try:
                if detected == "gmt8":
                    row = _parse_line_gmt8(parts, line_num)
                elif detected == "gmt7":
                    row = _parse_line_gmt7(parts, line_num)
                elif detected == "globk":
                    row = _parse_line_globk(parts, line_num)
                else:
                    raise ValueError(f"Unknown format: {detected}")
                rows.append(row)
            except ValueError as exc:
                msg = f"Line {line_num}: {exc}"
                parse_log.append(msg)
                if strict:
                    raise VelocityParseError(msg) from exc

    if not rows:
        raise ValueError(f"No valid velocity rows found in {filepath}")

    data = np.array(rows, dtype=object)
    lon = data[:, 0].astype(float)
    lat = data[:, 1].astype(float)
    ve = data[:, 2].astype(float)
    vn = data[:, 3].astype(float)
    se = data[:, 4].astype(float)
    sn = data[:, 5].astype(float)
    rho = data[:, 6].astype(float)
    names = data[:, 7].astype(str)

    return VelocityField(
        lon=lon,
        lat=lat,
        ve=ve,
        vn=vn,
        se=se,
        sn=sn,
        rho=rho,
        names=names,
        meta={
            "filepath": str(path),
            "format": detected if fmt != "auto" else "auto",
            "detected_format": detected,
            "n_total": line_num,
            "n_parsed": len(rows),
            "n_skipped": len(parse_log),
            "parse_log": parse_log,
        },
    )


def write_velocity_file(
    filepath: str,
    vf: VelocityField,
    fmt: str = "gmt8",
) -> None:
    """Write velocity field to a GMT-style file."""
    with open(filepath, "w", encoding="utf-8") as fid:
        for i in range(len(vf)):
            if fmt == "gmt8":
                fid.write(
                    f"{vf.lon[i]:.6f} {vf.lat[i]:.6f} "
                    f"{vf.ve[i]:.4f} {vf.vn[i]:.4f} "
                    f"{vf.se[i]:.4f} {vf.sn[i]:.4f} "
                    f"{vf.rho[i]:.4f} {vf.names[i]}\n"
                )
            elif fmt == "gmt7":
                fid.write(
                    f"{vf.lon[i]:.6f} {vf.lat[i]:.6f} "
                    f"{vf.ve[i]:.4f} {vf.vn[i]:.4f} "
                    f"{vf.se[i]:.4f} {vf.sn[i]:.4f} {vf.names[i]}\n"
                )
            else:
                raise ValueError(f"Unsupported output format: {fmt}")


def write_strain_velocity_file(
    filepath: str,
    lon: "np.ndarray",
    lat: "np.ndarray",
    ve: "np.ndarray",
    vn: "np.ndarray",
    names: "np.ndarray" = None,
) -> None:
    """Write estimated velocities (grid points / mesh vertices) to a GMT8 file.

    Since estimated velocities at grid/vertex points have no formal
    uncertainties, ``se`` and ``sn`` are written as 0.0 and ``rho`` as 0.0.

    Parameters
    ----------
    filepath : str
        Output file path.
    lon, lat : ndarray
        Point coordinates (degrees).
    ve, vn : ndarray
        Estimated east / north velocities (mm/yr).
    names : ndarray, optional
        Point labels.  Defaults to sequential integers.
    """
    import numpy as np
    n = len(lon)
    if names is None:
        names = np.array([f"P{i:04d}" for i in range(n)])
    with open(filepath, "w", encoding="utf-8") as fid:
        for i in range(n):
            fid.write(
                f"{lon[i]:.6f} {lat[i]:.6f} "
                f"{ve[i]:.4f} {vn[i]:.4f} "
                f"0.0000 0.0000 0.0000 {names[i]}\n"
            )
