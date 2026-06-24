"""Plotly-based plotting functions for the PyStrain2 web application.

All functions in this module are pure -- they accept data arrays and
return ``plotly.graph_objects.Figure`` instances.  No Streamlit
dependency.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

# Shared map layout defaults
_MAP_LAYOUT_BASE = dict(
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="closest",
    dragmode="pan",
)

_GEO_LAYOUT_BASE = dict(
    projection_type="mercator",
    showland=True,
    landcolor="rgb(240, 240, 240)",
    coastlinecolor="rgb(180, 180, 180)",
    showocean=True,
    oceancolor="rgb(230, 240, 250)",
    showcountries=True,
    countrycolor="rgb(200, 200, 200)",
    lataxis=dict(showgrid=True, gridcolor="rgba(180,180,180,0.3)"),
    lonaxis=dict(showgrid=True, gridcolor="rgba(180,180,180,0.3)"),
)


def _make_geo_figure(
    title: str = "",
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    height: int = 600,
) -> go.Figure:
    """Create a base ``Scattergeo`` figure with standard layout."""
    fig = go.Figure()
    geo = dict(**_GEO_LAYOUT_BASE)
    if lon_range is not None:
        geo["lonaxis"]["range"] = list(lon_range)
    if lat_range is not None:
        geo["lataxis"]["range"] = list(lat_range)
    fig.update_layout(
        title=title,
        geo=geo,
        height=height,
        **_MAP_LAYOUT_BASE,
    )
    return fig


def _lat_aspect(lat: np.ndarray) -> float:
    """Approximate aspect ratio for a latitude band."""
    return float(1.0 / np.cos(np.deg2rad(np.nanmean(lat))))


# ---------------------------------------------------------------------------
# Velocity field map
# ---------------------------------------------------------------------------


def plot_velocity_field(
    lon: np.ndarray,
    lat: np.ndarray,
    ve: np.ndarray,
    vn: np.ndarray,
    site_names: Optional[List[str]] = None,
    polygon: Optional[np.ndarray] = None,
    outlier_lon: Optional[np.ndarray] = None,
    outlier_lat: Optional[np.ndarray] = None,
    title: str = "GNSS Velocity Field",
    arrow_scale: float = 0.15,
    vector_color: str = "steelblue",
    height: int = 600,
) -> go.Figure:
    """Plot GPS velocity vectors as arrows on an interactive map.

    Parameters
    ----------
    lon, lat : arrays of site coordinates (degrees).
    ve, vn : east / north velocity components (mm/yr).
    site_names : optional site labels for hover tooltips.
    polygon : optional (N,2) boundary to overlay.
    outlier_lon, outlier_lat : optional outlier site positions to mark in red.
    title : figure title.
    arrow_scale : scale factor applied to velocities for arrow length.
    vector_color : colour for the velocity arrows.
    height : figure height in pixels.
    """
    pad = 0.5
    lon_min, lon_max = float(np.nanmin(lon) - pad), float(np.nanmax(lon) + pad)
    lat_min, lat_max = float(np.nanmin(lat) - pad), float(np.nanmax(lat) + pad)

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    # Velocity arrows as line segments + markers
    arrow_lon_start, arrow_lat_start = [], []
    arrow_lon_end, arrow_lat_end = [], []
    hover_texts = []
    for i in range(len(lon)):
        arrow_lon_start.append(float(lon[i]))
        arrow_lat_start.append(float(lat[i]))
        arrow_lon_end.append(float(lon[i] + ve[i] * arrow_scale))
        arrow_lat_end.append(float(lat[i] + vn[i] * arrow_scale))
        name = site_names[i] if site_names is not None else f"site_{i}"
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Lon: {lon[i]:.4f}°<br>"
            f"Lat: {lat[i]:.4f}°<br>"
            f"Ve: {ve[i]:.2f} mm/yr<br>"
            f"Vn: {vn[i]:.2f} mm/yr"
        )

    # Draw lines for arrows
    for i in range(len(lon)):
        fig.add_trace(
            go.Scattergeo(
                lon=[arrow_lon_start[i], arrow_lon_end[i]],
                lat=[arrow_lat_start[i], arrow_lat_end[i]],
                mode="lines",
                line=dict(color=vector_color, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Arrowhead markers at tips
    fig.add_trace(
        go.Scattergeo(
            lon=arrow_lon_end,
            lat=arrow_lat_end,
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=6,
                color=vector_color,
                angleref="previous",
            ),
            hoverinfo="skip",
            showlegend=False,
            name="velocity",
        )
    )

    # Site points (for hover)
    fig.add_trace(
        go.Scattergeo(
            lon=list(lon),
            lat=list(lat),
            mode="markers",
            marker=dict(size=5, color=vector_color, symbol="circle"),
            text=hover_texts,
            hoverinfo="text",
            name="GPS Sites",
        )
    )

    # Outliers in red
    if outlier_lon is not None and outlier_lat is not None and len(outlier_lon) > 0:
        fig.add_trace(
            go.Scattergeo(
                lon=list(outlier_lon),
                lat=list(outlier_lat),
                mode="markers",
                marker=dict(size=10, color="red", symbol="x"),
                name="Outliers",
            )
        )

    # Polygon boundary
    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))

    return fig


# ---------------------------------------------------------------------------
# Outlier map
# ---------------------------------------------------------------------------


def plot_outlier_map(
    lon: np.ndarray,
    lat: np.ndarray,
    outlier_mask: np.ndarray,
    site_names: Optional[List[str]] = None,
    polygon: Optional[np.ndarray] = None,
    title: str = "Outlier Detection Results",
    height: int = 600,
) -> go.Figure:
    """Scatter map distinguishing clean sites from outliers.

    Parameters
    ----------
    outlier_mask : bool array, True = outlier.
    """
    pad = 0.5
    lon_min, lon_max = float(np.nanmin(lon) - pad), float(np.nanmax(lon) + pad)
    lat_min, lat_max = float(np.nanmin(lat) - pad), float(np.nanmax(lat) + pad)

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    clean = ~outlier_mask
    # Clean sites
    hover_clean = []
    for i in np.where(clean)[0]:
        name = site_names[i] if site_names is not None else f"site_{i}"
        hover_clean.append(f"<b>{name}</b><br>Lon: {lon[i]:.4f}°<br>Lat: {lat[i]:.4f}°")
    fig.add_trace(
        go.Scattergeo(
            lon=list(lon[clean]),
            lat=list(lat[clean]),
            mode="markers",
            marker=dict(size=7, color="steelblue", symbol="circle"),
            text=hover_clean,
            hoverinfo="text",
            name=f"Clean ({clean.sum()})",
        )
    )

    # Outliers
    hover_out = []
    for i in np.where(outlier_mask)[0]:
        name = site_names[i] if site_names is not None else f"site_{i}"
        hover_out.append(f"<b>{name} (OUTLIER)</b><br>Lon: {lon[i]:.4f}°<br>Lat: {lat[i]:.4f}°")
    if outlier_mask.any():
        fig.add_trace(
            go.Scattergeo(
                lon=list(lon[outlier_mask]),
                lat=list(lat[outlier_mask]),
                mode="markers",
                marker=dict(size=12, color="red", symbol="x-thin", line=dict(width=2)),
                text=hover_out,
                hoverinfo="text",
                name=f"Outliers ({outlier_mask.sum()})",
            )
        )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))

    return fig


# ---------------------------------------------------------------------------
# Triangulation mesh
# ---------------------------------------------------------------------------


def plot_triangulation_mesh(
    tri: Delaunay,
    lon: np.ndarray,
    lat: np.ndarray,
    good_mask: Optional[np.ndarray] = None,
    polygon: Optional[np.ndarray] = None,
    ve: Optional[np.ndarray] = None,
    vn: Optional[np.ndarray] = None,
    title: str = "Delaunay Triangulation",
    height: int = 600,
) -> go.Figure:
    """Plot a Delaunay triangulation on the map.

    Parameters
    ----------
    tri : scipy Delaunay object.
    good_mask : optional bool array marking valid triangles.
    ve, vn : optional velocity components for station arrows.
    """
    pad = 0.5
    lon_min, lon_max = float(np.nanmin(lon) - pad), float(np.nanmax(lon) + pad)
    lat_min, lat_max = float(np.nanmin(lat) - pad), float(np.nanmax(lat) + pad)

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    n_good = good_mask.sum() if good_mask is not None else len(tri.simplices)
    n_bad = (len(tri.simplices) - n_good) if good_mask is not None else 0

    for i, simplex in enumerate(tri.simplices):
        pts_lon = list(lon[simplex]) + [lon[simplex[0]]]  # close polygon
        pts_lat = list(lat[simplex]) + [lat[simplex[0]]]
        is_good = good_mask is None or good_mask[i]
        color = "gray" if is_good else "red"
        width = 0.8 if is_good else 0.4
        dash = "solid" if is_good else "dot"

        fig.add_trace(
            go.Scattergeo(
                lon=pts_lon,
                lat=pts_lat,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Add good/bad legend entries (manual)
    if n_bad > 0:
        fig.add_trace(
            go.Scattergeo(
                lon=[None], lat=[None],
                mode="lines",
                line=dict(color="gray", width=0.8),
                name=f"Good ({n_good})",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[None], lat=[None],
                mode="lines",
                line=dict(color="red", width=0.4, dash="dot"),
                name=f"Bad ({n_bad})",
            )
        )

    # Station markers
    fig.add_trace(
        go.Scattergeo(
            lon=list(lon),
            lat=list(lat),
            mode="markers",
            marker=dict(size=5, color="steelblue"),
            name=f"Sites ({len(lon)})",
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))

    return fig


# ---------------------------------------------------------------------------
# Grid points overlay
# ---------------------------------------------------------------------------


def plot_grid_points_overlay(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    title: str = "Grid Points & GPS Sites",
    height: int = 600,
) -> go.Figure:
    """Show regular grid points overlaid on GPS station locations.

    Used for the Shen et al. (2015) algorithm preview.
    """
    pad = 0.5
    all_lon = np.concatenate([grid_lon, site_lon])
    all_lat = np.concatenate([grid_lat, site_lat])
    lon_min, lon_max = float(np.nanmin(all_lon) - pad), float(np.nanmax(all_lon) + pad)
    lat_min, lat_max = float(np.nanmin(all_lat) - pad), float(np.nanmax(all_lat) + pad)

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    # Grid points
    fig.add_trace(
        go.Scattergeo(
            lon=list(grid_lon),
            lat=list(grid_lat),
            mode="markers",
            marker=dict(size=4, color="orange", symbol="square", opacity=0.6),
            name=f"Grid ({len(grid_lon)} pts)",
        )
    )

    # GPS sites
    fig.add_trace(
        go.Scattergeo(
            lon=list(site_lon),
            lat=list(site_lat),
            mode="markers",
            marker=dict(size=7, color="steelblue", symbol="circle"),
            name=f"GPS ({len(site_lon)} sites)",
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))

    return fig


# ---------------------------------------------------------------------------
# Wang 2012 mesh preview
# ---------------------------------------------------------------------------


def plot_wang2012_mesh_preview(
    mesh_lon: np.ndarray,
    mesh_lat: np.ndarray,
    simplices: np.ndarray,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    title: str = "Wang (2012) Mesh Preview",
    height: int = 600,
) -> go.Figure:
    """Show a Wang-2012 triangular mesh with GPS station locations."""
    pad = 0.5
    all_lon = np.concatenate([mesh_lon, site_lon])
    all_lat = np.concatenate([mesh_lat, site_lat])
    lon_min, lon_max = float(np.nanmin(all_lon) - pad), float(np.nanmax(all_lon) + pad)
    lat_min, lat_max = float(np.nanmin(all_lat) - pad), float(np.nanmax(all_lat) + pad)

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    # Mesh edges
    for simplex in simplices:
        pts_lon = list(mesh_lon[simplex]) + [mesh_lon[simplex[0]]]
        pts_lat = list(mesh_lat[simplex]) + [mesh_lat[simplex[0]]]
        fig.add_trace(
            go.Scattergeo(
                lon=pts_lon,
                lat=pts_lat,
                mode="lines",
                line=dict(color="lightgray", width=0.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Mesh vertices
    fig.add_trace(
        go.Scattergeo(
            lon=list(mesh_lon),
            lat=list(mesh_lat),
            mode="markers",
            marker=dict(size=3, color="darkgray", symbol="circle"),
            name=f"Mesh vertices ({len(mesh_lon)})",
        )
    )

    # GPS sites
    fig.add_trace(
        go.Scattergeo(
            lon=list(site_lon),
            lat=list(site_lat),
            mode="markers",
            marker=dict(size=7, color="steelblue", symbol="circle"),
            name=f"GPS ({len(site_lon)} sites)",
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))

    return fig


# ---------------------------------------------------------------------------
# Scalar strain field
# ---------------------------------------------------------------------------


def plot_scalar_strain_field(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    field_name: str = "strain",
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar_label: str = "Strain (nstrain/yr)",
    polygon: Optional[np.ndarray] = None,
    faults=None,
    title: Optional[str] = None,
    marker_size: int = 50,
    height: int = 600,
) -> go.Figure:
    """Color-coded scatter plot of a scalar strain field.

    Parameters
    ----------
    symmetric : if True, colour range is ± percentile(95).
    faults : optional list of (N, 2) arrays of fault trace coordinates.
    """
    finite = np.isfinite(values)
    if not finite.any():
        fig = _make_geo_figure(title=title or field_name, height=height)
        return fig

    pad = 0.5
    lon_min = float(np.nanmin(lon[finite]) - pad)
    lon_max = float(np.nanmax(lon[finite]) + pad)
    lat_min = float(np.nanmin(lat[finite]) - pad)
    lat_max = float(np.nanmax(lat[finite]) + pad)

    # Expand bounds to include faults
    if faults:
        for f in faults:
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] >= 2:
                lon_min = min(lon_min, float(np.nanmin(f[:, 0]) - pad))
                lon_max = max(lon_max, float(np.nanmax(f[:, 0]) + pad))
                lat_min = min(lat_min, float(np.nanmin(f[:, 1]) - pad))
                lat_max = max(lat_max, float(np.nanmax(f[:, 1]) + pad))

    title_str = title or f"{field_name}"
    fig = _make_geo_figure(
        title=title_str,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    z = values[finite]
    if symmetric:
        vmax_val = float(np.percentile(np.abs(z), 95))
        vmin_val = -vmax_val
    else:
        vmax_val = float(np.percentile(z, 98))
        vmin_val = float(np.percentile(z, 2))

    hover_texts = [
        f"Lon: {x:.4f}°<br>Lat: {y:.4f}°<br>{field_name}: {v:.2f}"
        for x, y, v in zip(lon[finite], lat[finite], z)
    ]

    fig.add_trace(
        go.Scattergeo(
            lon=list(lon[finite]),
            lat=list(lat[finite]),
            mode="markers",
            marker=dict(
                size=marker_size,
                color=z,
                colorscale=cmap,
                cmin=vmin_val,
                cmax=vmax_val,
                colorbar=dict(title=colorbar_label),
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=field_name,
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))
    _add_fault_traces(fig, faults)

    return fig


# ---------------------------------------------------------------------------
# Principal strain crosses
# ---------------------------------------------------------------------------


def plot_principal_strain(
    lon: np.ndarray,
    lat: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    azimuth: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    faults=None,
    title: str = "Principal Strain Rate",
    scale: Optional[float] = None,
    height: int = 600,
) -> go.Figure:
    """Principal strain crosses as line-segment pairs.

    Blue = compression (negative), Red = extension (positive).
    faults : optional list of (N, 2) arrays of fault trace coordinates.
    """
    finite = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(azimuth)
    if not finite.any():
        fig = _make_geo_figure(title=title, height=height)
        return fig

    pad = 0.5
    lon_min = float(np.nanmin(lon[finite]) - pad)
    lon_max = float(np.nanmax(lon[finite]) + pad)
    lat_min = float(np.nanmin(lat[finite]) - pad)
    lat_max = float(np.nanmax(lat[finite]) + pad)

    # Expand bounds to include faults
    if faults:
        for f in faults:
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] >= 2:
                lon_min = min(lon_min, float(np.nanmin(f[:, 0]) - pad))
                lon_max = max(lon_max, float(np.nanmax(f[:, 0]) + pad))
                lat_min = min(lat_min, float(np.nanmin(f[:, 1]) - pad))
                lat_max = max(lat_max, float(np.nanmax(f[:, 1]) + pad))

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    if scale is None:
        all_vals = np.concatenate([np.abs(e1[finite]), np.abs(e2[finite])])
        max_val = np.max(all_vals)
        scale = 0.3 / max(max_val, 1.0) if max_val > 0 else 0.1

    for i in np.where(finite)[0]:
        az_rad = np.radians(azimuth[i])
        cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)
        dx1, dy1 = e1[i] * scale * cos_a, e1[i] * scale * sin_a
        dx2, dy2 = e2[i] * scale * (-sin_a), e2[i] * scale * cos_a

        color1 = "blue" if e1[i] < 0 else "red"
        color2 = "blue" if e2[i] < 0 else "red"

        # e1 line
        fig.add_trace(
            go.Scattergeo(
                lon=[lon[i] - dx1, lon[i] + dx1],
                lat=[lat[i] - dy1, lat[i] + dy1],
                mode="lines",
                line=dict(color=color1, width=1.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # e2 line
        fig.add_trace(
            go.Scattergeo(
                lon=[lon[i] - dx2, lon[i] + dx2],
                lat=[lat[i] - dy2, lat[i] + dy2],
                mode="lines",
                line=dict(color=color2, width=1.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Legend entries
    fig.add_trace(
        go.Scattergeo(
            lon=[None], lat=[None],
            mode="lines",
            line=dict(color="red", width=1.5),
            name="Extension (+)",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[None], lat=[None],
            mode="lines",
            line=dict(color="blue", width=1.5),
            name="Compression (-)",
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))
    _add_fault_traces(fig, faults)

    return fig


# ---------------------------------------------------------------------------
# Polygon on map (for time-series station display)
# ---------------------------------------------------------------------------


def plot_stations_with_polygon(
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    station_names: Optional[List[str]] = None,
    polygon: Optional[np.ndarray] = None,
    title: str = "GPS Stations & Analysis Polygon",
    height: int = 600,
) -> go.Figure:
    """Display station locations with an overlaid semi-transparent polygon."""
    pad = 0.5
    lon_min = float(np.nanmin(station_lon) - pad)
    lon_max = float(np.nanmax(station_lon) + pad)
    lat_min = float(np.nanmin(station_lat) - pad)
    lat_max = float(np.nanmax(station_lat) + pad)

    if polygon is not None:
        lon_min = min(lon_min, float(np.nanmin(polygon[:, 0]) - pad))
        lon_max = max(lon_max, float(np.nanmax(polygon[:, 0]) + pad))
        lat_min = min(lat_min, float(np.nanmin(polygon[:, 1]) - pad))
        lat_max = max(lat_max, float(np.nanmax(polygon[:, 1]) + pad))

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    hover_texts = []
    for i in range(len(station_lon)):
        name = station_names[i] if station_names is not None else f"sta_{i}"
        hover_texts.append(
            f"<b>{name}</b><br>Lon: {station_lon[i]:.4f}°<br>Lat: {station_lat[i]:.4f}°"
        )

    fig.add_trace(
        go.Scattergeo(
            lon=list(station_lon),
            lat=list(station_lat),
            mode="markers",
            marker=dict(size=8, color="steelblue", symbol="circle"),
            text=hover_texts,
            hoverinfo="text",
            name=f"Stations ({len(station_lon)})",
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon, opacity=0.25, name="Analysis Polygon"))

    return fig


# ---------------------------------------------------------------------------
# Time-series strain location map (clickable)
# ---------------------------------------------------------------------------


def plot_timeseries_locations(
    loc_lon: np.ndarray,
    loc_lat: np.ndarray,
    indices: Optional[np.ndarray] = None,
    polygon: Optional[np.ndarray] = None,
    title: str = "Strain Locations — Click to View Time Series",
    height: int = 600,
) -> go.Figure:
    """Interactive map where each strain location is a clickable marker.

    Parameters
    ----------
    loc_lon, loc_lat : strain computation locations.
    indices : custom data for each point (e.g. result index).
    """
    pad = 0.5
    lon_min = float(np.nanmin(loc_lon) - pad)
    lon_max = float(np.nanmax(loc_lon) + pad)
    lat_min = float(np.nanmin(loc_lat) - pad)
    lat_max = float(np.nanmax(loc_lat) + pad)

    fig = _make_geo_figure(
        title=title,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    customdata = list(range(len(loc_lon))) if indices is None else list(indices)
    hover_texts = [
        f"<b>Location {i}</b><br>Lon: {x:.4f}°<br>Lat: {y:.4f}°<br><i>Click to view</i>"
        for i, x, y in zip(customdata, loc_lon, loc_lat)
    ]

    fig.add_trace(
        go.Scattergeo(
            lon=list(loc_lon),
            lat=list(loc_lat),
            mode="markers",
            marker=dict(size=12, color="crimson", symbol="circle", line=dict(width=1, color="white")),
            text=hover_texts,
            hoverinfo="text",
            customdata=customdata,
            name=f"Locations ({len(loc_lon)})",
        )
    )

    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon, opacity=0.2))

    return fig


# ---------------------------------------------------------------------------
# Time-series line plot
# ---------------------------------------------------------------------------


def plot_timeseries_result(
    decyr: np.ndarray,
    values: np.ndarray,
    label: str = "",
    y_label: str = "Strain (nstrain/yr)",
    color: str = "steelblue",
    title: Optional[str] = None,
    height: int = 400,
) -> go.Figure:
    """Line plot of a single strain component vs time.

    Parameters
    ----------
    decyr : decimal year array.
    values : strain values (same length).
    """
    title_str = title or label
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=decyr,
            y=values,
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
            name=label,
            hovertemplate=f"Year: %{{x:.3f}}<br>{label}: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title_str,
        xaxis_title="Decimal Year",
        yaxis_title=y_label,
        height=height,
        margin=dict(l=40, r=10, t=40, b=30),
        hovermode="x",
    )
    return fig


def plot_timeseries_multi_component(
    decyr: np.ndarray,
    dilation: Optional[np.ndarray] = None,
    shear: Optional[np.ndarray] = None,
    e1: Optional[np.ndarray] = None,
    e2: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    sec_inv: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    height: int = 450,
) -> go.Figure:
    """Multi-component time series overlaid or faceted.

    All arrays are same length as *decyr*. Components that are None are
    skipped.
    """
    fig = go.Figure()
    components = [
        ("dilation", dilation, "#1f77b4"),
        ("shear", shear, "#d62728"),
        ("e1", e1, "#2ca02c"),
        ("e2", e2, "#ff7f0e"),
        ("omega", omega, "#9467bd"),
        ("sec_inv", sec_inv, "#8c564b"),
    ]
    for label, arr, color in components:
        if arr is None:
            continue
        mask = np.isfinite(arr)
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=decyr[mask],
                y=arr[mask],
                mode="lines+markers",
                line=dict(color=color, width=1.2),
                marker=dict(size=3),
                name=label,
                hovertemplate=f"Year: %{{x:.3f}}<br>{label}: %{{y:.4f}}<extra></extra>",
            )
        )

    title_str = title or "Strain Time Series"
    fig.update_layout(
        title=title_str,
        xaxis_title="Decimal Year",
        yaxis_title="Strain (nstrain/yr)",
        height=height,
        margin=dict(l=40, r=10, t=40, b=30),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Time-series all-components 2-column panel
# ---------------------------------------------------------------------------


def plot_timeseries_all_components_2col(
    decyr: np.ndarray,
    dilation: Optional[np.ndarray] = None,
    shear: Optional[np.ndarray] = None,
    e1: Optional[np.ndarray] = None,
    e2: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    sec_inv: Optional[np.ndarray] = None,
    exx: Optional[np.ndarray] = None,
    exy: Optional[np.ndarray] = None,
    eyy: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    height: int = 1000,
) -> go.Figure:
    """All nine strain-component time series arranged in a 5×2 subplot grid.

    Parameters
    ----------
    decyr : decimal year array (shared x-axis for all panels).
    All other parameters are optional 1-D arrays of the same length as *decyr*.
    title : overall figure title.
    height : total figure height in pixels.
    """
    from plotly.subplots import make_subplots

    components = [
        ("Dilatation (膨胀率)", dilation, "#1f77b4", "nstrain/yr"),
        ("Max Shear (最大剪应变)", shear, "#d62728", "nstrain/yr"),
        ("e1 (主应变1)", e1, "#2ca02c", "nstrain/yr"),
        ("e2 (主应变2)", e2, "#ff7f0e", "nstrain/yr"),
        ("Omega (旋转率)", omega, "#9467bd", "nrad/yr"),
        ("2nd Invariant (第二不变量)", sec_inv, "#8c564b", "nstrain²/yr²"),
        ("exx", exx, "#17becf", "nstrain/yr"),
        ("exy", exy, "#bcbd22", "nstrain/yr"),
        ("eyy", eyy, "#e377c2", "nstrain/yr"),
    ]

    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[c[0] for c in components],
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
    )

    for i, (label, arr, color, y_label) in enumerate(components):
        row = i // 2 + 1
        col = i % 2 + 1
        if arr is not None:
            mask = np.isfinite(arr)
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=decyr[mask],
                        y=arr[mask],
                        mode="markers",
                        marker=dict(size=5, color=color),
                        name=label,
                        showlegend=False,
                        hovertemplate=(
                            f"Year: %{{x:.3f}}<br>{label}: %{{y:.4f}}<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(title_text="Decimal Year", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    title_str = title or "Strain Time Series — All Components"
    fig.update_layout(
        title=dict(text=title_str, font=dict(size=16)),
        height=height,
        margin=dict(l=50, r=20, t=60, b=30),
        hovermode="x",
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _polygon_trace(
    polygon: np.ndarray,
    opacity: float = 0.15,
    name: str = "Boundary",
    line_color: str = "black",
    fill_color: str = "gray",
) -> go.Scattergeo:
    """Create a polygon overlay trace."""
    # Close the polygon
    if polygon[0, 0] != polygon[-1, 0] or polygon[0, 1] != polygon[-1, 1]:
        lon_vals = list(polygon[:, 0]) + [polygon[0, 0]]
        lat_vals = list(polygon[:, 1]) + [polygon[0, 1]]
    else:
        lon_vals = list(polygon[:, 0])
        lat_vals = list(polygon[:, 1])

    return go.Scattergeo(
        lon=lon_vals,
        lat=lat_vals,
        mode="lines",
        line=dict(color=line_color, width=1.5),
        fill="toself",
        fillcolor=f"rgba(128,128,128,{opacity})",
        name=name,
    )


def _add_fault_traces(fig: go.Figure, faults) -> None:
    """Add fault line traces to an existing Scattergeo figure.

    Parameters
    ----------
    faults : list of (N, 2) numpy arrays in (lon, lat) order, or None.
    """
    if faults is None or len(faults) == 0:
        return
    for i, fault in enumerate(faults):
        if not isinstance(fault, np.ndarray) or fault.ndim != 2 or fault.shape[0] < 2:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=list(fault[:, 0]),
                lat=list(fault[:, 1]),
                mode="lines",
                line=dict(color="black", width=1.0),
                hoverinfo="skip",
                showlegend=(i == 0),
                name="Faults" if i == 0 else None,
            )
        )


# ---------------------------------------------------------------------------
# Gridded (GMT-style) strain field
# ---------------------------------------------------------------------------


def plot_gridded_strain_field(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    field_name: str = "strain",
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar_label: str = "Strain (nstrain/yr)",
    polygon: Optional[np.ndarray] = None,
    faults=None,
    title: Optional[str] = None,
    grid_spacing: float = 0.1,
    height: int = 600,
) -> go.Figure:
    """GMT-style colour-filled grid display of a scalar strain field.

    Interpolates scattered strain values to a regular lon/lat grid and
    renders square markers on a Scattergeo map to preserve the geographic
    background (land / coastlines / countries).

    Parameters
    ----------
    grid_spacing : resolution of the output grid in degrees.
    faults : optional list of (N, 2) arrays of fault trace coordinates.
    """
    from scipy.interpolate import griddata

    finite = np.isfinite(values)
    if not finite.any():
        fig = _make_geo_figure(title=title or field_name, height=height)
        return fig

    z = values[finite]
    x = lon[finite]
    y = lat[finite]

    pad = 0.5
    lon_min = float(np.nanmin(x) - pad)
    lon_max = float(np.nanmax(x) + pad)
    lat_min = float(np.nanmin(y) - pad)
    lat_max = float(np.nanmax(y) + pad)

    # Expand bounds to include faults
    if faults:
        for f in faults:
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] >= 2:
                lon_min = min(lon_min, float(np.nanmin(f[:, 0]) - pad))
                lon_max = max(lon_max, float(np.nanmax(f[:, 0]) + pad))
                lat_min = min(lat_min, float(np.nanmin(f[:, 1]) - pad))
                lat_max = max(lat_max, float(np.nanmax(f[:, 1]) + pad))

    # Build regular grid
    grid_lon = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)
    grid_lat = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    # Interpolate
    grid_z = griddata((x, y), z, (grid_lon_2d, grid_lat_2d), method="linear")
    # Only use griddata result within the original data extent (no extrapolation)
    # NaN values outside the original data convex hull are rendered as gaps

    if symmetric:
        vmax_val = float(np.nanpercentile(np.abs(z), 95))
        vmin_val = -vmax_val
    else:
        vmax_val = float(np.nanpercentile(z, 98))
        vmin_val = float(np.nanpercentile(z, 2))

    title_str = title or f"{field_name} (GMT Grid)"

    # Use Scattergeo markers so the map background stays visible
    fig = _make_geo_figure(
        title=title_str,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    # Flatten + filter NaN
    mask = np.isfinite(grid_z)
    gz = grid_z[mask]
    glon = grid_lon_2d[mask]
    glat = grid_lat_2d[mask]

    # Approximate marker size to fill grid cells; Scattergeo sizes are in px
    # so this is an approximation that looks reasonable at default zoom
    marker_px = max(8, min(40, grid_spacing * 35))

    hover_texts = [
        f"Lon: {lo:.4f}°<br>Lat: {la:.4f}°<br>{field_name}: {val:.2f}"
        for lo, la, val in zip(glon, glat, gz)
    ]

    fig.add_trace(
        go.Scattergeo(
            lon=list(glon),
            lat=list(glat),
            mode="markers",
            marker=dict(
                size=marker_px,
                color=gz,
                colorscale=cmap,
                cmin=vmin_val,
                cmax=vmax_val,
                symbol="square",
                colorbar=dict(title=colorbar_label),
                line=dict(width=0),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=field_name,
        )
    )

    # Polygon boundary
    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))
    _add_fault_traces(fig, faults)

    return fig


# ---------------------------------------------------------------------------
# Triangle-filled strain field (Delaunay)
# ---------------------------------------------------------------------------


def plot_triangle_strain_field(
    tri,
    good_mask: np.ndarray,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    values: np.ndarray,
    field_name: str = "strain",
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar_label: str = "Strain (nstrain/yr)",
    polygon: Optional[np.ndarray] = None,
    faults=None,
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """Strain field rendered as filled Delaunay triangles coloured by value.

    Each triangle in the triangulation is drawn as a filled polygon whose
    colour encodes the scalar strain value at that triangle's centroid.
    The underlying Scattergeo map (coastlines, land) is preserved.

    Parameters
    ----------
    tri : scipy.spatial.Delaunay object.
    good_mask : bool array marking valid triangles (same length as tri.simplices).
    site_lon, site_lat : original GPS site coordinates (triangle vertices).
    values : strain values at triangle centroids (same length as tri.simplices).
    faults : optional list of (N, 2) arrays of fault trace coordinates.
    """
    from plotly.colors import sample_colorscale

    finite = np.isfinite(values)
    if not finite.any():
        fig = _make_geo_figure(title=title or field_name, height=height)
        return fig

    # Only render good, finite-valued triangles
    keep = good_mask & finite
    if not keep.any():
        fig = _make_geo_figure(title=title or field_name, height=height)
        return fig

    z = values[keep]
    vmax_val: float
    vmin_val: float
    if symmetric:
        vmax_val = float(np.nanpercentile(np.abs(z), 95))
        vmin_val = -vmax_val
    else:
        vmax_val = float(np.nanpercentile(z, 98))
        vmin_val = float(np.nanpercentile(z, 2))

    # Clamp to avoid divide-by-zero
    span = vmax_val - vmin_val
    if span == 0:
        span = 1.0
    norm_vals = np.clip((z - vmin_val) / span, 0.0, 1.0)

    # Map to RGBA colours via plotly's sample_colorscale
    colours = sample_colorscale(cmap, list(norm_vals))

    pad = 0.5
    lon_min = float(np.nanmin(site_lon) - pad)
    lon_max = float(np.nanmax(site_lon) + pad)
    lat_min = float(np.nanmin(site_lat) - pad)
    lat_max = float(np.nanmax(site_lat) + pad)

    # Expand bounds to include faults
    if faults:
        for f in faults:
            if isinstance(f, np.ndarray) and f.ndim == 2 and f.shape[0] >= 2:
                lon_min = min(lon_min, float(np.nanmin(f[:, 0]) - pad))
                lon_max = max(lon_max, float(np.nanmax(f[:, 0]) + pad))
                lat_min = min(lat_min, float(np.nanmin(f[:, 1]) - pad))
                lat_max = max(lat_max, float(np.nanmax(f[:, 1]) + pad))

    title_str = title or f"{field_name} (Triangle Fill)"
    fig = _make_geo_figure(
        title=title_str,
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        height=height,
    )

    kept_indices = np.where(keep)[0]

    # --- Colour-binned approach: group triangles into N bins to limit
    #     trace count while keeping colour fidelity.  Each bin gets one
    #     Scattergeo trace with NaN separators between polygons. ---
    N_BINS = 48
    if len(colours) <= N_BINS:
        # One trace per triangle (small dataset)
        for idx, tri_idx in enumerate(kept_indices):
            verts = tri.simplices[tri_idx]
            plon = list(site_lon[verts]) + [site_lon[verts[0]]]
            plat = list(site_lat[verts]) + [site_lat[verts[0]]]
            fig.add_trace(
                go.Scattergeo(
                    lon=plon,
                    lat=plat,
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=colours[idx],
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    else:
        # Many triangles — bin by colour
        bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
        bin_indices = np.digitize(norm_vals, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, N_BINS - 1)

        for b in range(N_BINS):
            in_bin = np.where(bin_indices == b)[0]
            if len(in_bin) == 0:
                continue
            bin_colour = sample_colorscale(cmap, [(bin_edges[b] + bin_edges[b + 1]) / 2])[0]
            lon_segs: list = []
            lat_segs: list = []
            for idx in in_bin:
                tri_idx = kept_indices[idx]
                verts = tri.simplices[tri_idx]
                for v in verts:
                    lon_segs.append(float(site_lon[v]))
                    lat_segs.append(float(site_lat[v]))
                # close the triangle
                lon_segs.append(float(site_lon[verts[0]]))
                lat_segs.append(float(site_lat[verts[0]]))
                # NaN separator
                lon_segs.append(float("nan"))
                lat_segs.append(float("nan"))
            fig.add_trace(
                go.Scattergeo(
                    lon=lon_segs,
                    lat=lat_segs,
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=bin_colour,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # --- Hidden colour-bar via a dummy scattergeo trace ---
    fig.add_trace(
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="markers",
            marker=dict(
                size=0,
                color=[vmin_val, vmax_val],
                colorscale=cmap,
                cmin=vmin_val,
                cmax=vmax_val,
                colorbar=dict(title=colorbar_label),
                showscale=True,
            ),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Polygon boundary
    if polygon is not None:
        fig.add_trace(_polygon_trace(polygon))

    return fig
