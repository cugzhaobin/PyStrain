"""Interpolate scattered strain-rate data to a regular grid.

The key function is :func:`scattered_to_grid`, which interpolates from
arbitrary (lon, lat) points (e.g. triangle centroids, grid points) to a
regular lon×lat raster.  A distance-based mask is applied to prevent
interpolation across data voids — grid cells whose nearest valid data
point is farther than *max_distance* are set to NaN.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

logger = logging.getLogger("pystrain2.plot")

# ---------------------------------------------------------------------------
# Strain output file I/O
# ---------------------------------------------------------------------------

# Strain field columns as written by write_strain_result().
# These match the _COLUMNS list in pystrain2.io.output:
#   shr = max shear, dil = dilatation, inv2 = second invariant,
#   theta = azimuth of e1.
_STRAIN_FIELDS = [
    "ve", "vn", "exx", "exy", "eyy", "omega",
    "e1", "e2", "shr", "dil", "inv2", "theta",
]

# Map descriptive / legacy header names to short canonical names.
# Handles both the current output format (shr/dil/inv2/theta) and
# older/alternative formats (shear/dilation/sec_inv/azimuth).
_HEADER_ALIAS = {
    "shear":    "shr",
    "dilation": "dil",
    "inv2":     "sec_inv",   # legacy name → canonical
    "sec_inv":  "inv2",
    "azimuth":  "theta",
    # Also map short→short for identity
    "shr":    "shr",
    "dil":    "dil",
    "theta":  "theta",
}

# Map short canonical names to descriptive names for NetCDF / plot labels.
_FIELD_ALIAS = {
    "shr":   "shear",
    "dil":   "dilation",
    "inv2":  "sec_inv",
    "theta": "azimuth",
}


def read_strain_file(path: str) -> Dict[str, np.ndarray]:
    """Read a PyStrain2 strain output file.

    Parameters
    ----------
    path : str
        Path to a strain ``.txt`` file produced by
        :func:`pystrain2.io.write_strain_result`.

    Returns
    -------
    dict
        Keys: ``lon``, ``lat``, ``ve``, ``vn``, ``exx``, ``exy``, ``eyy``,
        ``omega``, ``e1``, ``e2``, ``shear``, ``dilation``, ``sec_inv``,
        ``azimuth``.  Each is a 1D float64 ndarray.
    """
    # Read header to count columns
    with open(path, "r") as fh:
        header = fh.readline().strip().lstrip("#").strip()
    col_names = header.split()
    n_cols = len(col_names)

    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Map header column names to canonical short names,
    # stripping units in parentheses — e.g. "lon(°)" → "lon", "ve(mm/yr)" → "ve"
    name_to_idx = {}
    for i, name in enumerate(col_names):
        # Strip trailing "(...)" containing units
        clean = name.split("(")[0].strip()
        canonical = _HEADER_ALIAS.get(clean, clean)
        if canonical not in name_to_idx:
            name_to_idx[canonical] = i

    result = {}
    for field in _STRAIN_FIELDS + ["lon", "lat"]:
        if field in name_to_idx:
            result[field] = data[:, name_to_idx[field]]
    return result


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _haversine_deg(lon1: np.ndarray, lat1: np.ndarray,
                   lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Haversine distance in **degrees** (approximate for small/medium areas).

    For true km distances over large areas the full Haversine formula is
    used, then converted back to equivalent degrees at the mean latitude.
    """
    R = 6371.0  # km
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_km = R * c
    # Convert back to approximate degrees at the mean latitude
    mean_lat = np.radians((lat1 + lat2) / 2)
    deg_per_km = 1.0 / (111.32 * np.cos(mean_lat))
    return dist_km * deg_per_km


def _median_neighbor_distance(lon: np.ndarray, lat: np.ndarray) -> float:
    """Median nearest-neighbour distance (in degrees) among valid points."""
    if len(lon) < 2:
        return 0.0
    # Use Cartesian approximation for speed; good enough for distance stats
    mean_lat = np.mean(lat)
    scale_lon = np.cos(np.radians(mean_lat))
    xy = np.column_stack([lon * scale_lon, lat])
    tree = KDTree(xy)
    dists, _ = tree.query(xy, k=2)
    nn_dists = dists[:, 1]  # closest (first is self with dist=0)
    # Convert back to degrees
    return float(np.median(nn_dists))


# ---------------------------------------------------------------------------
# Core interpolation
# ---------------------------------------------------------------------------

def scattered_to_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    max_distance: Optional[float] = None,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate scattered (lon, lat, value) to a regular grid.

    Parameters
    ----------
    lon, lat : 1D ndarray
        Coordinates of the input data points.
    values : 1D ndarray
        Scalar values at each input point.  May contain NaN; those points
        are excluded from the interpolation.
    grid_lon, grid_lat : 1D ndarray
        1D coordinate arrays defining the output grid (``len(grid_lon)`` ×
        ``len(grid_lat)``).
    max_distance : float or None
        Maximum interpolation distance in **degrees**.  Grid cells farther
        than this from the nearest valid data point are set to NaN.  When
        *None* (default), the threshold is auto-computed as
        ``2.5 × median_nearest_neighbour_distance``.
    method : str
        ``"linear"`` (default) uses :class:`~scipy.interpolate.LinearNDInterpolator`.
        ``"cubic"`` uses :class:`~scipy.interpolate.CloughTocher2DInterpolator`.

    Returns
    -------
    gridded : 2D ndarray (nlat × nlon)
        Interpolated field with NaN in data voids.
    """
    # --- select valid source points ----------------------------------------
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full((len(grid_lat), len(grid_lon)), np.nan)

    src_lon = lon[valid].astype(np.float64)
    src_lat = lat[valid].astype(np.float64)
    src_val = values[valid].astype(np.float64)

    # --- build interpolator ------------------------------------------------
    points = np.column_stack([src_lon, src_lat])
    if method == "cubic":
        from scipy.interpolate import CloughTocher2DInterpolator
        interpolator = CloughTocher2DInterpolator(points, src_val)
    else:
        interpolator = LinearNDInterpolator(points, src_val)

    # --- query grid --------------------------------------------------------
    glon2d, glat2d = np.meshgrid(grid_lon, grid_lat)
    result = interpolator(glon2d, glat2d)  # shape (nlat, nlon)

    # --- distance mask -----------------------------------------------------
    # Build a KDTree of source points for distance queries
    mean_lat = np.mean(src_lat)
    scale_lon = np.cos(np.radians(mean_lat))
    src_xy = np.column_stack([src_lon * scale_lon, src_lat])

    grid_pts = np.column_stack([glon2d.ravel() * scale_lon, glat2d.ravel()])
    tree = KDTree(src_xy)
    dists_deg, _ = tree.query(grid_pts, k=1)
    dists_deg = dists_deg.reshape(glon2d.shape)

    if max_distance is None:
        max_distance = 2.5 * _median_neighbor_distance(src_lon, src_lat)
        logger.info("  auto max_distance = %.4f° (2.5 × median NN dist)", max_distance)

    # Mask cells too far from any data point
    result[dists_deg > max_distance] = np.nan

    return result


# ---------------------------------------------------------------------------
# Grid coordinate helpers
# ---------------------------------------------------------------------------

def _make_grid_coords(
    lon: np.ndarray,
    lat: np.ndarray,
    grid_size: Optional[Tuple[int, int]] = None,
    padding: float = 0.1,
    region: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 1D grid coordinate arrays from data extent or explicit region.

    Parameters
    ----------
    lon, lat : 1D ndarray
        Coordinates of data points (used when *region* is None).
    grid_size : (nlon, nlat) or None
        Number of grid cells in each dimension.  When *None*,
        auto-computed for ~0.02° cell size.
    padding : float
        Extra padding in degrees added around the data extent
        (ignored when *region* is provided).
    region : [slon, elon, slat, elat] or None
        Explicit grid extent.  When provided, overrides auto-computation
        and *padding*.

    Returns
    -------
    grid_lon, grid_lat : 1D ndarrays
    """
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
    else:
        valid = np.isfinite(lon) & np.isfinite(lat)
        lon_min = lon[valid].min() - padding
        lon_max = lon[valid].max() + padding
        lat_min = lat[valid].min() - padding
        lat_max = lat[valid].max() + padding

    if grid_size is None:
        cell_size = 0.02  # ~2 km at mid-lat
        nlon = max(int((lon_max - lon_min) / cell_size) + 1, 50)
        nlat = max(int((lat_max - lat_min) / cell_size) + 1, 50)
    else:
        nlon, nlat = grid_size

    grid_lon = np.linspace(lon_min, lon_max, nlon)
    grid_lat = np.linspace(lat_min, lat_max, nlat)
    return grid_lon, grid_lat


# ---------------------------------------------------------------------------
# NetCDF / GeoTIFF output
# ---------------------------------------------------------------------------

def save_netcdf(
    path: str,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    fields: Dict[str, np.ndarray],
    title: str = "PyStrain2 strain rate",
) -> None:
    """Save gridded strain fields to a CF-1.6 NetCDF file.

    Parameters
    ----------
    path : str
        Output ``.nc`` file path.
    grid_lon : 1D ndarray
        Longitude coordinate (dimension ``lon``).
    grid_lat : 1D ndarray
        Latitude coordinate (dimension ``lat``).
    fields : dict
        Mapping of field name → 2D ndarray (nlat × nlon).
    title : str
        Global attribute for the dataset.
    """
    from scipy.io import netcdf_file

    nlat, nlon = len(grid_lat), len(grid_lon)

    with netcdf_file(path, "w") as nc:
        nc.title = title
        nc.source = "PyStrain2"
        nc.Conventions = "CF-1.6"

        nc.createDimension("lon", nlon)
        nc.createDimension("lat", nlat)

        v_lon = nc.createVariable("lon", "f8", ("lon",))
        v_lon.units = "degrees_east"
        v_lon.long_name = "longitude"
        v_lon[:] = grid_lon.astype(np.float64)

        v_lat = nc.createVariable("lat", "f8", ("lat",))
        v_lat.units = "degrees_north"
        v_lat.long_name = "latitude"
        v_lat[:] = grid_lat.astype(np.float64)

        unit_map = {
            "ve": "mm/yr", "vn": "mm/yr",
            "exx": "ns/yr", "exy": "ns/yr", "eyy": "ns/yr",
            "omega": "nrad/yr",
            "e1": "ns/yr", "e2": "ns/yr",
            "shr": "ns/yr", "dil": "ns/yr", "inv2": "ns/yr",
            "theta": "degrees",
            # descriptive aliases
            "shear": "ns/yr", "dilation": "ns/yr",
            "sec_inv": "ns/yr", "azimuth": "degrees",
        }

        long_name_map = {
            "shr": "max_shear", "dil": "dilatation",
            "inv2": "second_invariant", "theta": "azimuth_e1",
        }

        for name, data in fields.items():
            if data.shape != (nlat, nlon):
                logger.warning("  skipping '%s': shape mismatch %s vs (%d,%d)",
                               name, data.shape, nlat, nlon)
                continue
            var = nc.createVariable(name, "f8", ("lat", "lon"))
            var.units = unit_map.get(name, "")
            var.long_name = long_name_map.get(name, name)
            var[:] = data.astype(np.float64)

    logger.info("  NetCDF saved to '%s' (%d fields)", path, len(fields))


def save_geotiff(
    path: str,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    field: np.ndarray,
    field_name: str = "data",
) -> None:
    """Save a single gridded field as a GeoTIFF.

    Requires ``rasterio``.  Installs with ``pip install rasterio``.

    Parameters
    ----------
    path : str
        Output ``.tif`` file path.
    grid_lon : 1D ndarray
        Longitude coordinates.
    grid_lat : 1D ndarray
        Latitude coordinates (must be ascending for rasterio).
    field : 2D ndarray (nlat × nlon)
        Data values.
    field_name : str
        Description written to the file metadata.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        logger.error(
            "  rasterio is required for GeoTIFF output. "
            "Install with: pip install rasterio"
        )
        return

    nlat, nlon = field.shape
    if len(grid_lon) != nlon or len(grid_lat) != nlat:
        raise ValueError(
            f"Shape mismatch: field {field.shape} vs "
            f"grid ({len(grid_lon)}×{len(grid_lat)})"
        )

    # rasterio expects row-major (nlat × nlon) with lat decreasing
    # Our grids have lat increasing; flip
    if grid_lat[0] < grid_lat[-1]:
        data = np.asarray(field)[::-1, :].astype(np.float32)
        lat_min, lat_max = float(grid_lat[0]), float(grid_lat[-1])
    else:
        data = np.asarray(field).astype(np.float32)
        lat_max, lat_min = float(grid_lat[0]), float(grid_lat[-1])

    lon_min, lon_max = float(grid_lon[0]), float(grid_lon[-1])
    dx = (lon_max - lon_min) / (nlon - 1) if nlon > 1 else 1.0
    dy = (lat_max - lat_min) / (nlat - 1) if nlat > 1 else 1.0

    transform = from_bounds(
        lon_min - dx / 2, lat_min - dy / 2,
        lon_max + dx / 2, lat_max + dy / 2,
        nlon, nlat,
    )

    # Use WKT for CRS to avoid PROJ version conflicts
    crs_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

    try:
        dst = rasterio.open(
            path, "w",
            driver="GTiff",
            height=nlat, width=nlon,
            count=1, dtype="float32",
            crs=crs_wkt,
            transform=transform,
            nodata=np.nan,
        )
    except Exception:
        # Fallback: try without explicit CRS
        dst = rasterio.open(
            path, "w",
            driver="GTiff",
            height=nlat, width=nlon,
            count=1, dtype="float32",
            transform=transform,
            nodata=np.nan,
        )

    with dst:
        dst.write(data, 1)
        dst.set_band_description(1, field_name)

    logger.info("  GeoTIFF saved to '%s'", path)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def strain_file_to_grid(
    strain_path: str,
    output_dir: str = ".",
    grid_size: Optional[Tuple[int, int]] = None,
    max_distance: Optional[float] = None,
    method: str = "linear",
    output_format: str = "netcdf",
    fields: Optional[List[str]] = None,
    region: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Read a strain file, interpolate to grid, save NetCDF / GeoTIFF.

    Parameters
    ----------
    strain_path : str
        Path to the strain output ``.txt`` file.
    output_dir : str
        Directory for gridded output files.
    grid_size : (nlon, nlat) or None
        Grid dimensions.  Auto-computed when *None*.
    max_distance : float or None
        Max interpolation distance in degrees.  Auto-computed when *None*.
    method : str
        ``"linear"`` (default) or ``"cubic"``.
    output_format : str
        ``"netcdf"``, ``"geotiff"``, or ``"both"``.
    fields : list of str or None
        Which fields to grid.  Default: all strain fields.
    region : [slon, elon, slat, elat] or None
        Clip grid and data to this extent.  When *None*, full data extent.

    Returns
    -------
    grid_data : dict
        ``grid_lon``, ``grid_lat``, and gridded 2D arrays for each field.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = Path(strain_path).stem  # e.g. "shen2015_strain"

    # --- read --------------------------------------------------------------
    logger.info("Reading strain file: %s", strain_path)
    data = read_strain_file(strain_path)
    lon, lat = data["lon"], data["lat"]
    n_orig = len(lon)
    logger.info("  %d data points, %d fields available", n_orig,
                sum(1 for f in _STRAIN_FIELDS if f in data))

    # --- clip to region (if given) -----------------------------------------
    if region is not None:
        slon, elon, slat, elat = region
        inside = (lon >= slon) & (lon <= elon) & (lat >= slat) & (lat <= elat)
        n_clipped = int(np.sum(inside))
        if n_clipped < 3:
            logger.warning("  region clip leaves only %d points; continuing anyway", n_clipped)
        else:
            logger.info("  clipping to region [%.1f, %.1f, %.1f, %.1f]: %d / %d points kept",
                        slon, elon, slat, elat, n_clipped, n_orig)
            lon = lon[inside]
            lat = lat[inside]
            for key in list(data.keys()):
                if isinstance(data[key], np.ndarray) and len(data[key]) == n_orig:
                    data[key] = data[key][inside]

    # --- build grid --------------------------------------------------------
    grid_lon, grid_lat = _make_grid_coords(lon, lat, grid_size=grid_size, region=region)
    logger.info("  grid: %d lon × %d lat = %d cells",
                len(grid_lon), len(grid_lat), len(grid_lon) * len(grid_lat))

    # --- which fields to process -------------------------------------------
    if fields is None:
        fields = [f for f in _STRAIN_FIELDS if f in data]

    # --- grid each field ---------------------------------------------------
    gridded = {}
    for field in fields:
        if field not in data:
            logger.warning("  field '%s' not in strain file, skipping", field)
            continue
        vals = data[field]
        n_valid = int(np.sum(np.isfinite(vals)))
        if n_valid < 3:
            logger.warning("  field '%s' has only %d valid points, skipping", field, n_valid)
            gridded[field] = np.full((len(grid_lat), len(grid_lon)), np.nan)
            continue

        logger.info("  gridding '%s' (%d valid points) ...", field, n_valid)
        gridded[field] = scattered_to_grid(
            lon, lat, vals, grid_lon, grid_lat,
            max_distance=max_distance, method=method,
        )

    # --- save --------------------------------------------------------------
    if output_format in ("netcdf", "both"):
        nc_path = out / f"{base}.nc"
        save_netcdf(str(nc_path), grid_lon, grid_lat, gridded)

    if output_format in ("geotiff", "both"):
        for field_name, field_data in gridded.items():
            tif_path = out / f"{base}_{field_name}.tif"
            save_geotiff(str(tif_path), grid_lon, grid_lat, field_data, field_name)

    # Attach grid coords to returned dict for downstream plotting
    gridded["grid_lon"] = grid_lon
    gridded["grid_lat"] = grid_lat

    # Add descriptive aliases (e.g. "shear" → alias for "shr")
    for short, long in _FIELD_ALIAS.items():
        if short in gridded and long not in gridded:
            gridded[long] = gridded[short]

    return gridded
