"""Visualization utilities for PyStrain2 — Cartopy-based step-by-step plots."""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path
from typing import Optional, Sequence

from pystrain.data import StrainResult, VelocityField


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_interactive():
    """Check whether to show figures interactively (blocking)."""
    return os.environ.get("PYSTRAIN2_NO_DISPLAY", "") != "1"


def _show_and_close(fig):
    """Show the figure interactively with full zoom/pan support, then close it.

    When PYSTRAIN2_NO_DISPLAY=1, the figure is saved to disk without display.
    Otherwise a blocking interactive window is shown — the user can zoom, pan,
    and inspect the data, then close the window to continue the pipeline.
    """
    if _is_interactive():
        plt.show(block=True)
    plt.close(fig)

def _compute_extent(lon: np.ndarray, lat: np.ndarray, buffer: float = 0.1):
    """Return [min_lon, max_lon, min_lat, max_lat] with buffer padding."""
    return [
        float(lon.min()) - buffer,
        float(lon.max()) + buffer,
        float(lat.min()) - buffer,
        float(lat.max()) + buffer,
    ]


def _create_map_axes(
    region: Optional[Sequence[float]] = None,
    lon: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    buffer: float = 0.1,
    figsize: tuple = (10, 8),
):
    """Create a Matplotlib figure with a PlateCarree Cartopy axes.

    Parameters
    ----------
    region : sequence of 4 floats, optional
        [min_lon, max_lon, min_lat, max_lat]. If None, auto-computed from
        *lon* and *lat* plus *buffer*.
    lon, lat : np.ndarray, optional
        Arrays used for auto-extent when *region* is None.
    buffer : float
        Degree buffer added around data when auto-computing extent.
    figsize : tuple
        Figure size passed to ``plt.figure``.

    Returns
    -------
    fig, ax : matplotlib Figure and GeoAxes
    extent : list [min_lon, max_lon, min_lat, max_lat]
    """
    if region is None:
        if lon is None or lat is None:
            raise ValueError("Either region or lon/lat must be provided.")
        extent = _compute_extent(lon, lat, buffer)
    else:
        extent = list(region)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())

    ax.add_feature(cfeature.LAND, facecolor="lightyellow", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.LAKES, facecolor="lightcyan", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--", zorder=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, zorder=1,
                      linewidth=0.4, color="gray", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax, extent


def _add_reference_arrow(ax, extent, vecsize, label="100 mm"):
    """Draw a reference arrow in the lower-left of the map."""
    min_lon, max_lon, min_lat, max_lat = extent
    ref_lon = min_lon + (max_lon - min_lon) * 0.12
    ref_lat = min_lat + (max_lat - min_lat) * 0.10
    lat_offset = (max_lat - min_lat) * 0.04
    ax.quiver(ref_lon, ref_lat, 100, 0,
              transform=ccrs.PlateCarree(), scale=vecsize, color="black", zorder=5)
    ax.text(ref_lon, ref_lat + lat_offset, label,
            transform=ccrs.PlateCarree(), fontsize=8, zorder=5)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_raw_velocity_field(
    vf: VelocityField,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    vecsize: float = 400,
    dpi: int = 150,
):
    """Plot all GPS velocity vectors in blue with a reference arrow.

    Parameters
    ----------
    vf : VelocityField
    output_path : str
        File path for the saved figure.
    region : sequence of 4 floats, optional
        [min_lon, max_lon, min_lat, max_lat]. Auto-computed if None.
    vecsize : float
        Quiver scale parameter.
    dpi : int
        Output DPI.
    """
    fig, ax, extent = _create_map_axes(region=region, lon=vf.lon, lat=vf.lat)
    ax.quiver(vf.lon, vf.lat, vf.ve, vf.vn,
              transform=ccrs.PlateCarree(), scale=vecsize, color="blue",
              width=0.003, zorder=3)
    _add_reference_arrow(ax, extent, vecsize)
    ax.set_title("Raw Velocity Field")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_outlier_result(
    vf: VelocityField,
    outlier_mask: np.ndarray,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    vecsize: float = 400,
    dpi: int = 150,
):
    """Plot velocity field with outliers highlighted in red.

    Clean sites are drawn in blue, outliers in red.

    Parameters
    ----------
    vf : VelocityField
        Full (pre-cleaned) velocity field.
    outlier_mask : np.ndarray of bool
        True where a station is an outlier.
    output_path : str
    region : sequence of 4 floats, optional
    vecsize : float
    dpi : int
    """
    fig, ax, extent = _create_map_axes(region=region, lon=vf.lon, lat=vf.lat)

    clean = ~outlier_mask
    if clean.any():
        ax.quiver(vf.lon[clean], vf.lat[clean], vf.ve[clean], vf.vn[clean],
                  transform=ccrs.PlateCarree(), scale=vecsize, color="blue",
                  width=0.003, label="Clean", zorder=3)
    if outlier_mask.any():
        ax.quiver(vf.lon[outlier_mask], vf.lat[outlier_mask],
                  vf.ve[outlier_mask], vf.vn[outlier_mask],
                  transform=ccrs.PlateCarree(), scale=vecsize, color="red",
                  width=0.003, label="Outlier", zorder=4)

    _add_reference_arrow(ax, extent, vecsize)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Outlier Detection Result")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_clean_velocity_field(
    vf: VelocityField,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    vecsize: float = 400,
    dpi: int = 150,
):
    """Plot only the clean (post-outlier-removal) velocity field.

    Parameters
    ----------
    vf : VelocityField
        Cleaned velocity field.
    output_path : str
    region : sequence of 4 floats, optional
    vecsize : float
    dpi : int
    """
    fig, ax, extent = _create_map_axes(region=region, lon=vf.lon, lat=vf.lat)
    ax.quiver(vf.lon, vf.lat, vf.ve, vf.vn,
              transform=ccrs.PlateCarree(), scale=vecsize, color="blue",
              width=0.003, zorder=3)
    _add_reference_arrow(ax, extent, vecsize)
    ax.set_title("Clean Velocity Field")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_triangulation(
    vf: VelocityField,
    tri,
    good_triangles: np.ndarray,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    dpi: int = 150,
):
    """Plot the Delaunay triangulation mesh with Cartopy basemap.

    Parameters
    ----------
    vf : VelocityField
    tri : scipy.spatial.Delaunay
        Delaunay triangulation object with ``.simplices`` attribute.
    good_triangles : np.ndarray of bool
        Boolean mask — True for triangles that pass quality checks.
    output_path : str
    region : sequence of 4 floats, optional
    dpi : int
    """
    fig, ax, extent = _create_map_axes(region=region, lon=vf.lon, lat=vf.lat)

    simplices = tri.simplices[good_triangles]
    triangulation = mtri.Triangulation(vf.lon, vf.lat, triangles=simplices)
    ax.triplot(triangulation, color="gray", linewidth=0.5,
               transform=ccrs.PlateCarree(), zorder=2)

    ax.scatter(vf.lon, vf.lat, s=12, color="black", zorder=3,
               transform=ccrs.PlateCarree(), label="GPS sites")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Delaunay Triangulation")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_grid_points(
    grid,
    vf: VelocityField,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    dpi: int = 150,
):
    """Plot grid computation points and GPS station locations.

    Grid points are shown as small gray dots; GPS stations as blue triangles.

    Parameters
    ----------
    grid : Grid
        Grid object with ``.lon`` and ``.lat`` numpy arrays.
    vf : VelocityField
    output_path : str
    region : sequence of 4 floats, optional
    dpi : int
    """
    all_lon = np.concatenate([grid.lon, vf.lon])
    all_lat = np.concatenate([grid.lat, vf.lat])
    fig, ax, extent = _create_map_axes(region=region, lon=all_lon, lat=all_lat)

    ax.scatter(grid.lon, grid.lat, s=4, color="gray", zorder=2,
               transform=ccrs.PlateCarree(), label="Grid points")
    ax.scatter(vf.lon, vf.lat, s=25, color="blue", marker="^", zorder=3,
               transform=ccrs.PlateCarree(), label="GPS stations")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Grid Points and GPS Stations")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_search_radius_map(
    grid,
    D_values: np.ndarray,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    dpi: int = 150,
):
    """Color-coded scatter plot of effective search radii (D values) at grid points.

    Parameters
    ----------
    grid : Grid
        Grid object with ``.lon`` and ``.lat`` numpy arrays.
    D_values : np.ndarray
        Effective search radius (or smoothing distance) for each grid point.
    output_path : str
    region : sequence of 4 floats, optional
    dpi : int
    """
    fig, ax, extent = _create_map_axes(region=region, lon=grid.lon, lat=grid.lat)

    sc = ax.scatter(grid.lon, grid.lat, c=D_values, cmap="plasma", s=20,
                    transform=ccrs.PlateCarree(), zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Search radius D (km)", fontsize=9)
    ax.set_title("Search Radius Map")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_scalar_field(
    result: StrainResult,
    field: str,
    output_path: str,
    cmap: str = "RdYlBu_r",
    region: Optional[Sequence[float]] = None,
    dpi: int = 150,
    symmetric: bool = False,
):
    """Scatter plot of a scalar strain field with Cartopy basemap.

    Parameters
    ----------
    result : StrainResult
    field : str
        Attribute name on *result* to plot (e.g. ``"dilation"``, ``"shear"``).
    output_path : str
    cmap : str
    region : sequence of 4 floats, optional
    dpi : int
    symmetric : bool
        If True, use symmetric vmin/vmax centered at 0 (for dilation, rotation).
        Otherwise use full data range.
    """
    values = getattr(result, field)
    mask = np.isfinite(values)
    finite_lon = result.lon[mask]
    finite_lat = result.lat[mask]
    finite_vals = values[mask]

    if len(finite_vals) == 0:
        # No valid data — skip the plot silently
        return

    fig, ax, extent = _create_map_axes(region=region, lon=finite_lon, lat=finite_lat)

    if len(finite_vals) > 1:
        if symmetric:
            vmax = np.percentile(np.abs(finite_vals), 98)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(finite_vals, [2, 98])
            if vmin >= vmax:
                span = max(abs(vmax), 1e-6)
                vmin, vmax = vmin - span * 0.1, vmax + span * 0.1
    elif len(finite_vals) == 1:
        v = float(finite_vals[0])
        span = max(abs(v), 1e-6)
        vmin, vmax = v - span * 0.1, v + span * 0.1
    else:
        vmin, vmax = 0, 1

    sc = ax.scatter(finite_lon, finite_lat, c=finite_vals, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=20,
                    transform=ccrs.PlateCarree(), zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(field, fontsize=9)
    ax.set_title(f"Strain field: {field}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_wang2012_mesh(
    result: StrainResult,
    vf: VelocityField,
    output_path: str,
    polygon: Optional[np.ndarray] = None,
    region: Optional[Sequence[float]] = None,
    dpi: int = 150,
):
    """Plot the Wang (2012) uniform triangular mesh with GPS stations.

    The mesh is a regular triangular grid (NOT at station locations).  GPS
    stations are overlaid as blue triangles.  The polygon boundary (if any)
    is drawn as a black outline.

    Parameters
    ----------
    result : StrainResult
        Must contain ``mesh_lon``, ``mesh_lat``, and ``mesh_simplices`` in
        ``result.meta``.
    vf : VelocityField
        GPS velocity field (used only for station locations).
    output_path : str
    polygon : np.ndarray, optional
        Nx2 array of polygon vertices.
    region : sequence of 4 floats, optional
    dpi : int
    """
    mesh_lon = result.meta.get("mesh_lon")
    mesh_lat = result.meta.get("mesh_lat")
    simplices = result.meta.get("mesh_simplices")

    if mesh_lon is None or simplices is None:
        return

    all_lon = np.concatenate([mesh_lon, vf.lon])
    all_lat = np.concatenate([mesh_lat, vf.lat])
    fig, ax, extent = _create_map_axes(region=region, lon=all_lon, lat=all_lat)

    # Draw mesh triangles
    triangulation = mtri.Triangulation(mesh_lon, mesh_lat, triangles=simplices)
    ax.triplot(triangulation, color="gray", linewidth=0.4,
               transform=ccrs.PlateCarree(), zorder=2)

    # GPS stations
    ax.scatter(vf.lon, vf.lat, s=30, color="blue", marker="^", zorder=3,
               transform=ccrs.PlateCarree(), label=f"GPS ({len(vf)} sites)")

    ax.legend(loc="upper right", fontsize=8)
    n_vtx = len(mesh_lon)
    n_tri = len(simplices)
    ax.set_title(f"Wang2012 Mesh ({n_vtx} vertices, {n_tri} triangles)")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


def plot_principal_strain_crosses(
    result: StrainResult,
    output_path: str,
    polygon: Optional[np.ndarray] = None,
    region: Optional[Sequence[float]] = None,
    dpi: int = 150,
):
    """Principal strain-rate crosses on a Cartopy basemap.

    Compression (negative eigenvalue) is drawn in blue; extension (positive)
    in red.  Each grid point gets two opposing arrows along the principal axes.

    Parameters
    ----------
    result : StrainResult
    output_path : str
    polygon : np.ndarray, optional
        Nx2 array of polygon vertices to overlay.
    region : sequence of 4 floats, optional
    dpi : int
    """
    fig, ax, extent = _create_map_axes(region=region, lon=result.lon, lat=result.lat)

    finite = np.concatenate([
        result.e1[np.isfinite(result.e1)],
        result.e2[np.isfinite(result.e2)],
    ])
    if len(finite) > 0 and np.max(np.abs(finite)) > 0:
        scale = 0.3 / max(float(np.max(np.abs(finite))), 1.0)
        transform = ccrs.PlateCarree()
        for i in range(len(result.lon)):
            if not (
                np.isfinite(result.e1[i])
                and np.isfinite(result.e2[i])
                and np.isfinite(result.azimuth[i])
            ):
                continue
            az_rad = np.radians(result.azimuth[i])
            cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)

            dx1 = float(result.e1[i]) * scale * cos_a
            dy1 = float(result.e1[i]) * scale * sin_a
            dx2 = float(result.e2[i]) * scale * (-sin_a)
            dy2 = float(result.e2[i]) * scale * cos_a

            c1 = "blue" if result.e1[i] < 0 else "red"
            c2 = "blue" if result.e2[i] < 0 else "red"

            lon_i, lat_i = float(result.lon[i]), float(result.lat[i])
            ax.annotate(
                "",
                xy=(lon_i + dx1, lat_i + dy1),
                xytext=(lon_i - dx1, lat_i - dy1),
                arrowprops=dict(arrowstyle="->", color=c1, lw=0.8),
                xycoords=transform._as_mpl_transform(ax),
                textcoords=transform._as_mpl_transform(ax),
                zorder=4,
            )
            ax.annotate(
                "",
                xy=(lon_i + dx2, lat_i + dy2),
                xytext=(lon_i - dx2, lat_i - dy2),
                arrowprops=dict(arrowstyle="->", color=c2, lw=0.8),
                xycoords=transform._as_mpl_transform(ax),
                textcoords=transform._as_mpl_transform(ax),
                zorder=4,
            )

    if polygon is not None:
        ax.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=5)

    ax.set_title("Principal Strain Rate (blue=compression, red=extension)")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def plot_velocity_field(
    vf: VelocityField,
    outlier_mask: Optional[np.ndarray] = None,
    polygon: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    region: Optional[Sequence[float]] = None,
    vecsize: float = 400,
    dpi: int = 150,
):
    """Backward-compatible wrapper.

    Routes to :func:`plot_outlier_result` when *outlier_mask* is provided,
    otherwise to :func:`plot_raw_velocity_field`.
    """
    if output_path is None:
        return

    if outlier_mask is not None:
        plot_outlier_result(vf, outlier_mask, output_path,
                            region=region, vecsize=vecsize, dpi=dpi)
    else:
        plot_raw_velocity_field(vf, output_path,
                                region=region, vecsize=vecsize, dpi=dpi)
