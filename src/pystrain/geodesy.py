"""Geodetic coordinate transformations and distance calculations."""

from typing import Tuple
import numpy as np
from pyproj import Geod, Transformer

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


def utm_zone(lon: float, lat: float) -> int:
    """Return UTM zone number for a given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    # Special zones for Norway and Svalbard omitted for simplicity
    return zone


def llh2utm(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Project lon/lat (degrees) to UTM coordinates in kilometers.

    Returns
    -------
    x, y : np.ndarray
        Easting and northing in km.
    proj_params : dict
        Parameters for potential inverse transform.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    zone = utm_zone(float(np.mean(lon)), float(np.mean(lat)))
    hemisphere = "north" if np.mean(lat) >= 0 else "south"
    crs_src = "EPSG:4326"
    crs_dst = f"EPSG:326{zone:02d}" if hemisphere == "north" else f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x / 1000.0, y / 1000.0, {"zone": zone, "hemisphere": hemisphere, "crs": crs_dst}


def utm2ll(x: np.ndarray, y: np.ndarray, proj_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse UTM projection from km to lon/lat (degrees)."""
    crs_src = proj_params["crs"]
    transformer = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(np.asarray(x) * 1000.0, np.asarray(y) * 1000.0)
    return lon, lat


def llh2localxy(lon: np.ndarray, lat: np.ndarray, origin: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Polyconic local projection to km, preserving legacy behavior approximately.

    For simplicity this uses an azimuthal equidistant projection centered at
    ``origin`` (lon0, lat0) in degrees.  Results are in km.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon0, lat0 = origin
    geod = Geod(ellps="WGS84")
    az, _, dist = geod.inv(
        np.full_like(lon, lon0), np.full_like(lat, lat0), lon, lat
    )
    # az is clockwise from north; convert to EN
    az_rad = np.deg2rad(az)
    x = dist * np.sin(az_rad) / 1000.0
    y = dist * np.cos(az_rad) / 1000.0
    return x, y


def distance_azimuth(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Geodetic distance (km) and azimuth (deg) from (lon1,lat1) to (lon2,lat2).

    Azimuth is clockwise from north.
    """
    geod = Geod(ellps="WGS84")
    az, _, dist = geod.inv(
        np.asarray(lon1, dtype=float),
        np.asarray(lat1, dtype=float),
        np.asarray(lon2, dtype=float),
        np.asarray(lat2, dtype=float),
    )
    return dist / 1000.0, az


def local_to_origin_projection(
    lon: np.ndarray,
    lat: np.ndarray,
    origin: Tuple[float, float],
    method: str = "utm",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Project lon/lat to local km coordinates with origin at (lon0, lat0).

    Parameters
    ----------
    method : {"utm", "polyconic"}
        UTM uses a single zone; polyconic uses azimuthal-equidistant approximation.
    """
    if method == "utm":
        x, y, params = llh2utm(lon, lat)
        x0, y0, _ = llh2utm(np.array([origin[0]]), np.array([origin[1]]))
        return x - x0[0], y - y0[0], params
    elif method == "polyconic":
        x, y = llh2localxy(lon, lat, origin)
        return x, y, {"method": "polyconic", "origin": origin}
    else:
        raise ValueError(f"Unknown projection method: {method}")
