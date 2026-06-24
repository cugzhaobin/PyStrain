"""Read fault (active-fault) geometry from KML or ESRI Shapefile format.

Returns a list of (N, 2) numpy arrays, one per fault trace, in (lon, lat)
order compatible with Cartopy ``PlateCarree`` and all PyStrain2 plotting
functions.

Supported formats (auto-detected by extension, or explicit *fmt*):

* ``"kml"`` — Keyhole Markup Language (Google Earth).  Reads ``<LineString>``
  and ``<LinearRing>`` geometries from any nesting level (``<Placemark>``,
  ``<Folder>``, ``<Document>``).
* ``"shp"`` — ESRI Shapefile (requires ``pyshp``).  Reads polyline features.
* ``"auto"`` — detect from file extension.

Usage::

    from pystrain2.io import read_faults

    faults = read_faults("active_faults.kml")
    for trace in faults:
        print(f"{len(trace)} vertices")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("pystrain2.io.faults")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_faults(filepath: str, fmt: str = "auto") -> List[np.ndarray]:
    """Read fault traces from a KML or SHP file.

    Parameters
    ----------
    filepath : str
        Path to ``.kml`` or ``.shp`` file.
    fmt : str
        ``"auto"`` (default), ``"kml"``, or ``"shp"``.

    Returns
    -------
    list of np.ndarray
        Each element is an (N, 2) float64 array of ``[lon, lat]`` vertices
        for one fault trace.  An empty list is returned for empty files.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fault file not found: {filepath}")

    if fmt == "auto":
        ext = path.suffix.lower()
        if ext in (".kml", ".kmz"):
            fmt = "kml"
        elif ext == ".shp":
            fmt = "shp"
        else:
            raise ValueError(
                f"Cannot auto-detect format for '{filepath}'. "
                f"Use fmt='kml' or fmt='shp'."
            )

    if fmt == "kml":
        return _read_faults_kml(filepath)
    elif fmt == "shp":
        return _read_faults_shp(filepath)
    else:
        raise ValueError(f"Unknown fault format: '{fmt}'. Use 'kml' or 'shp'.")


# ---------------------------------------------------------------------------
# KML reader
# ---------------------------------------------------------------------------

# KML namespace — handle both 2.2 and older versions
_KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "kml2": "http://earth.google.com/kml/2.2",
    "kml1": "http://earth.google.com/kml/2.1",
    "kml0": "http://earth.google.com/kml/2.0",
}


def _find_kml_elements(element, tag_names):
    """Recursively find all elements matching *tag_names* regardless of depth.

    Handles both namespaced (``{uri}tag``) and plain tag names.
    """
    results = []
    tag_set = set(tag_names)
    # Queue-based BFS avoids deep recursion
    queue = list(element)
    while queue:
        child = queue.pop(0)
        # Check local name (strip namespace)
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local in tag_set:
            results.append(child)
        queue.extend(list(child))
    return results


def _parse_kml_coordinates(text: str) -> np.ndarray:
    """Parse KML ``<coordinates>`` text into an (N, 2) or (N, 3) array.

    KML format: ``lon,lat[,alt] [whitespace] lon,lat[,alt] ...``
    """
    points = []
    for token in text.strip().split():
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) < 2:
            continue
        points.append([float(parts[0]), float(parts[1])])
    if not points:
        return np.empty((0, 2))
    return np.array(points, dtype=np.float64)


def _read_faults_kml(filepath: str) -> List[np.ndarray]:
    """Read LineString / LinearRing geometries from a KML file.

    Parameters
    ----------
    filepath : str
        Path to a ``.kml`` file.

    Returns
    -------
    list of np.ndarray
        Each is an (N, 2) array of [lon, lat] vertices.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()

    traces: List[np.ndarray] = []

    # Collect <LineString> and <LinearRing> elements
    geom_elems = _find_kml_elements(root, {"LineString", "LinearRing"})

    for elem in geom_elems:
        coord_elem = None
        for child in elem:
            local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local == "coordinates":
                coord_elem = child
                break

        if coord_elem is None or coord_elem.text is None:
            continue

        pts = _parse_kml_coordinates(coord_elem.text)
        if len(pts) >= 2:
            traces.append(pts)

    logger.info("  read %d fault traces from KML: %s", len(traces), filepath)
    return traces


# ---------------------------------------------------------------------------
# Shapefile reader
# ---------------------------------------------------------------------------


def _read_faults_shp(filepath: str) -> List[np.ndarray]:
    """Read polyline features from an ESRI Shapefile.

    Requires ``pyshp`` (``pip install pyshp``).

    Parameters
    ----------
    filepath : str
        Path to a ``.shp`` file.  The companion ``.dbf`` and ``.shx`` files
        must be present in the same directory.

    Returns
    -------
    list of np.ndarray
        Each is an (N, 2) array of [lon, lat] vertices.
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "Reading shapefiles requires 'pyshp'. "
            "Install with: pip install pyshp"
        )

    sf = shapefile.Reader(filepath)
    traces: List[np.ndarray] = []

    for shape_record in sf.iterShapeRecords():
        geom = shape_record.shape
        pts = np.array(geom.points, dtype=np.float64)
        # shapefile points can be (0,2) for null geometry
        if pts.ndim == 2 and pts.shape[0] >= 2:
            # Handle multi-part polylines: if the shape has "parts",
            # split at part boundaries
            parts = getattr(geom, "parts", None)
            if parts and len(parts) > 1:
                for start_idx, end_idx in zip(parts, list(parts[1:]) + [len(pts)]):
                    segment = pts[start_idx:end_idx]
                    if len(segment) >= 2:
                        traces.append(segment)
            else:
                traces.append(pts)

    logger.info("  read %d fault traces from SHP: %s", len(traces), filepath)
    return traces
