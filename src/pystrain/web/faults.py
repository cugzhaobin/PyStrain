"""Fault file parsing for the PyStrain2 web app.

Supports KML (Keyhole Markup Language) and ESRI Shapefile (.shp)
formats.  Each parser returns a list of ``(N, 2)`` numpy arrays in
*(lon, lat)* order suitable for overlay on Plotly Scattergeo maps.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from typing import List, Optional, Tuple

import numpy as np


def parse_kml(content: str) -> List[np.ndarray]:
    """Parse a KML string and extract all LineString coordinates.

    Returns a list of (N, 2) arrays of *(lon, lat)* points.
    """
    import xml.etree.ElementTree as ET

    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(content)

    # KML namespace may or may not be present
    def _find_all(element, tag):
        # Try with and without namespace
        results = element.findall(f"kml:{tag}", ns)
        if not results:
            results = element.findall(tag)
        return results

    def _find(element, tag):
        result = element.find(f"kml:{tag}", ns)
        if result is None:
            result = element.find(tag)
        return result

    traces: List[np.ndarray] = []

    # Collect all LineString elements (may be inside Placemark, MultiGeometry, etc.)
    for linestring in root.iter():
        tag = linestring.tag.split("}")[-1] if "}" in linestring.tag else linestring.tag
        if tag != "LineString":
            continue

        coord_elem = _find(linestring, "coordinates")
        if coord_elem is None:
            coord_elem = linestring.find("coordinates")
        if coord_elem is None or not coord_elem.text:
            continue

        text = coord_elem.text.strip()
        points = []
        for block in text.split():
            block = block.strip()
            if not block:
                continue
            parts = block.split(",")
            if len(parts) >= 2:
                try:
                    points.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        if len(points) >= 2:
            traces.append(np.array(points))

    return traces


def parse_shapefile(file_bytes: bytes, shx_bytes: Optional[bytes] = None,
                    dbf_bytes: Optional[bytes] = None) -> List[np.ndarray]:
    """Parse an ESRI Shapefile and extract polyline geometries.

    Parameters
    ----------
    file_bytes : contents of the ``.shp`` file.
    shx_bytes : contents of the ``.shx`` index file (optional, but needed
        for a valid shapefile).
    dbf_bytes : contents of the ``.dbf`` attribute file (optional).

    Returns a list of (N, 2) arrays of *(lon, lat)* points.
    """
    import shapefile

    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, "faults.shp")
        shx_path = os.path.join(tmpdir, "faults.shx")
        dbf_path = os.path.join(tmpdir, "faults.dbf")

        with open(shp_path, "wb") as f:
            f.write(file_bytes)
        with open(shx_path, "wb") as f:
            f.write(shx_bytes if shx_bytes else b"")
        with open(dbf_path, "wb") as f:
            f.write(dbf_bytes if dbf_bytes else b"")

        try:
            reader = shapefile.Reader(shp=shp_path, shx=shx_path, dbf=dbf_path)
        except Exception:
            # Try with just the .shp file
            reader = shapefile.Reader(shp=shp_path)

        traces: List[np.ndarray] = []
        for shape_record in reader.iterShapeRecords():
            shape = shape_record.shape
            if shape.shapeType != 3:  # POLYLINE
                # Also accept POLYGON (5) — extract rings
                if shape.shapeType == 5:
                    pts = np.array(shape.points)
                    if len(pts) >= 2:
                        traces.append(pts)
                    # Also extract parts for multi-part polygons
                    if len(shape.parts) > 1:
                        for i, start in enumerate(shape.parts):
                            end = shape.parts[i + 1] if i + 1 < len(shape.parts) else len(shape.points)
                            part_pts = np.array(shape.points[start:end])
                            if len(part_pts) >= 2:
                                traces.append(part_pts)
                continue

            if len(shape.points) < 2:
                continue

            # Handle multi-part polylines
            if len(shape.parts) <= 1:
                traces.append(np.array(shape.points))
            else:
                for i, start in enumerate(shape.parts):
                    end = shape.parts[i + 1] if i + 1 < len(shape.parts) else len(shape.points)
                    part_pts = np.array(shape.points[start:end])
                    if len(part_pts) >= 2:
                        traces.append(part_pts)

    return traces


def parse_geojson(content: str) -> List[np.ndarray]:
    """Parse a GeoJSON string and extract all LineString/MultiLineString coordinates.

    Returns a list of (N, 2) arrays of *(lon, lat)* points.
    """
    import json

    data = json.loads(content)
    traces: List[np.ndarray] = []

    def _extract_coords(geom: dict) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "LineString":
            pts = np.array(coords)
            if len(pts) >= 2:
                result.append(pts)
        elif gtype == "MultiLineString":
            for line in coords:
                pts = np.array(line)
                if len(pts) >= 2:
                    result.append(pts)
        elif gtype == "Polygon":
            for ring in coords:
                pts = np.array(ring)
                if len(pts) >= 2:
                    result.append(pts)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    pts = np.array(ring)
                    if len(pts) >= 2:
                        result.append(pts)
        return result

    if data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            geom = feature.get("geometry")
            if geom:
                traces.extend(_extract_coords(geom))
    elif data.get("type") == "Feature":
        geom = data.get("geometry")
        if geom:
            traces.extend(_extract_coords(geom))
    else:
        traces.extend(_extract_coords(data))

    return traces


def parse_fault_file(uploaded_file) -> Tuple[List[np.ndarray], Optional[str]]:
    """Auto-detect format (KML / Shapefile / ZIP / GeoJSON) and parse fault traces.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile object.

    Returns
    -------
    (traces, error_message)
        traces : list of (N, 2) numpy arrays of (lon, lat) coordinates.
        error_message : None on success, or a string describing the parse error.
    """
    name = (uploaded_file.name or "").lower()
    content = uploaded_file.getvalue()

    # --- KML ---
    if name.endswith(".kml") or name.endswith(".kmz"):
        try:
            if name.endswith(".kmz"):
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    kml_content = None
                    for fname in zf.namelist():
                        if fname.lower().endswith(".kml"):
                            kml_content = zf.read(fname).decode("utf-8")
                            break
                    if kml_content is None:
                        return [], "KMZ 文件中未找到 .kml 文件"
                    content = kml_content.encode("utf-8")
            text = content.decode("utf-8")
            traces = parse_kml(text)
            if not traces:
                return [], "KML 文件中未找到 LineString 几何"
            return traces, None
        except Exception as e:
            return [], f"KML 解析失败: {e}"

    # --- ZIP containing shapefile components ---
    if name.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                shp_data = shx_data = dbf_data = None
                for fname in zf.namelist():
                    lower = fname.lower()
                    data = zf.read(fname)
                    if lower.endswith(".shp"):
                        shp_data = data
                    elif lower.endswith(".shx"):
                        shx_data = data
                    elif lower.endswith(".dbf"):
                        dbf_data = data
                if shp_data is None:
                    return [], "ZIP 文件中未找到 .shp 文件"
                traces = parse_shapefile(shp_data, shx_data, dbf_data)
                if not traces:
                    return [], "Shapefile 中未找到折线几何"
                return traces, None
        except Exception as e:
            return [], f"ZIP/Shapefile 解析失败: {e}"

    # --- Single .shp file ---
    if name.endswith(".shp"):
        try:
            traces = parse_shapefile(content)
            if not traces:
                return [], "Shapefile 中未找到折线几何"
            return traces, None
        except Exception as e:
            return [], f"Shapefile 解析失败: {e}"

    # --- GeoJSON / JSON ---
    if name.endswith(".json") or name.endswith(".geojson"):
        try:
            text = content.decode("utf-8")
            traces = parse_geojson(text)
            if not traces:
                return [], "GeoJSON 文件中未找到 LineString 几何"
            return traces, None
        except Exception as e:
            return [], f"GeoJSON 解析失败: {e}"

    return [], f"不支持的文件格式: {name}（需要 .kml, .kmz, .shp, .zip, .json, .geojson）"
