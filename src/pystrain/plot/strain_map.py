"""GMT-style strain-rate maps from gridded data.

Produces publication-quality continuous-colour maps using
:func:`matplotlib.pyplot.pcolormesh` (equivalent to GMT's ``grdimage``)
on Cartopy basemaps with coastlines, graticules, and colour bars.

Typical usage::

    from pystrain2.plot import strain_file_to_grid, plot_strain_map

    gridded = strain_file_to_grid("shen2015_strain.txt")
    plot_strain_map(
        gridded["grid_lon"], gridded["grid_lat"],
        gridded["dilation"], field="dilation",
        output_path="dilation.png",
    )
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Field metadata: default colormap, symmetric scaling, label
# ---------------------------------------------------------------------------

_FIELD_META = {
    "dilation":  {"cmap": "RdBu_r",      "symmetric": True,  "label": "Dilatation (ns/yr)"},
    "shear":     {"cmap": "YlOrRd",       "symmetric": False, "label": "Max shear (ns/yr)"},
    "sec_inv":   {"cmap": "plasma",      "symmetric": False, "label": "2nd invariant (ns/yr)"},
    "omega":     {"cmap": "RdBu_r",      "symmetric": True,  "label": "Rotation (nrad/yr)"},
    "e1":        {"cmap": "RdBu_r",      "symmetric": True,  "label": "Max principal (ns/yr)"},
    "e2":        {"cmap": "RdBu_r",      "symmetric": True,  "label": "Min principal (ns/yr)"},
    "exx":       {"cmap": "RdBu_r",      "symmetric": True,  "label": "E–W strain (ns/yr)"},
    "exy":       {"cmap": "RdBu_r",      "symmetric": True,  "label": "Shear strain (ns/yr)"},
    "eyy":       {"cmap": "RdBu_r",      "symmetric": True,  "label": "N–S strain (ns/yr)"},
    "ve":        {"cmap": "RdBu_r",      "symmetric": True,  "label": "Ve (mm/yr)"},
    "vn":        {"cmap": "RdBu_r",      "symmetric": True,  "label": "Vn (mm/yr)"},
}


def _get_field_meta(field: str):
    """Return (cmap, symmetric, label) for a given field name."""
    if field in _FIELD_META:
        m = _FIELD_META[field]
        return m["cmap"], m["symmetric"], m["label"]
    return "viridis", False, field


# ---------------------------------------------------------------------------
# Map axes helper (mirrors visualization._create_map_axes)
# ---------------------------------------------------------------------------

def _is_interactive():
    return os.environ.get("PYSTRAIN2_NO_DISPLAY", "") != "1"


def _show_and_close(fig):
    if _is_interactive():
        import matplotlib.pyplot as plt
        plt.show(block=True)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _create_map_axes(
    region: Optional[Sequence[float]] = None,
    lon: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    buffer: float = 0.1,
    figsize: tuple = (10, 8),
    projection=None,
):
    """Create a Cartopy GeoAxes with standard basemap features.

    Parameters
    ----------
    region : [slon, elon, slat, elat] or None
    lon, lat : 1D arrays used to auto-compute extent when *region* is None.
    buffer : float
        Degree padding around auto-computed extent.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, ax, extent
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if projection is None:
        projection = ccrs.PlateCarree()

    if region is None:
        if lon is None or lat is None:
            raise ValueError("Either region or lon/lat must be provided.")
        extent = [
            float(lon.min()) - buffer,
            float(lon.max()) + buffer,
            float(lat.min()) - buffer,
            float(lat.max()) + buffer,
        ]
    else:
        extent = list(region)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
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


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_strain_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    values: np.ndarray,
    field: str = "dilation",
    output_path: Optional[str] = None,
    region: Optional[Sequence[float]] = None,
    cmap: Optional[str] = None,
    symmetric: Optional[bool] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    polygon: Optional[np.ndarray] = None,
    faults: Optional[List[np.ndarray]] = None,
    vf: Optional[object] = None,
    title: Optional[str] = None,
    dpi: int = 150,
    figsize: tuple = (10, 8),
) -> object:
    """Plot a single gridded strain field as a GMT-style colour map.

    Uses :func:`~matplotlib.pyplot.pcolormesh` for continuous colour fill,
    equivalent to GMT's ``grdimage``.

    Parameters
    ----------
    grid_lon, grid_lat : 1D ndarray
        Grid coordinate arrays.
    values : 2D ndarray (nlat × nlon)
        Gridded field values.  NaN regions are left transparent.
    field : str
        Field name (e.g. ``"dilation"``, ``"shear"``).  Determines the
        default colormap, symmetric/asymmetric scaling, and colour-bar label.
    output_path : str or None
        File path for the saved PNG.  If *None*, the figure is shown
        interactively and not saved.
    region : [slon, elon, slat, elat] or None
    cmap : str or None
        Matplotlib colormap name.  Auto-selected from *field* when *None*.
    symmetric : bool or None
        Force symmetric (diverging) or asymmetric (sequential) colour
        scaling.  Auto-selected when *None*.
    vmin, vmax : float or None
        Explicit data range.  Auto-computed from percentiles when *None*.
    polygon : (N, 2) ndarray or None
        Optional boundary polygon to overlay.
    title : str or None
        Figure title.  Auto-generated when *None*.
    dpi : int
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # --- metadata ----------------------------------------------------------
    auto_cmap, auto_sym, label = _get_field_meta(field)
    if cmap is None:
        cmap = auto_cmap
    if symmetric is None:
        symmetric = auto_sym

    # --- filter valid values for auto-scaling ------------------------------
    finite = values[np.isfinite(values)]

    # --- create map axes ---------------------------------------------------
    if region is None:
        region_ = [
            float(grid_lon[0]), float(grid_lon[-1]),
            float(grid_lat[0]), float(grid_lat[-1]),
        ]
    else:
        region_ = list(region)

    fig, ax, _ = _create_map_axes(region=region_, figsize=figsize)

    # --- colour range ------------------------------------------------------
    if vmin is None or vmax is None:
        if len(finite) > 1:
            if symmetric:
                vmax_auto = float(np.percentile(np.abs(finite), 98).item())
                vmin_auto = -vmax_auto
            else:
                pcts = np.percentile(finite, [2, 98])
                vmin_auto, vmax_auto = float(pcts[0].item()), float(pcts[1].item())
                if vmin_auto >= vmax_auto:
                    span = max(abs(vmax_auto), 1e-6)
                    vmin_auto, vmax_auto = vmin_auto - span * 0.1, vmax_auto + span * 0.1
        elif len(finite) == 1:
            v = float(finite[0])
            span = max(abs(v), 1e-6)
            vmin_auto, vmax_auto = v - span * 0.1, v + span * 0.1
        else:
            vmin_auto, vmax_auto = 0, 1

        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    # --- render -----------------------------------------------------------
    mesh = ax.pcolormesh(
        grid_lon, grid_lat, values,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading="auto", zorder=2,
    )

    # --- colour bar --------------------------------------------------------
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label(label, fontsize=10)

    # --- polygon overlay ---------------------------------------------------
    if polygon is not None:
        ax.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=5)

    # --- fault line overlay ------------------------------------------------
    if faults:
        for trace in faults:
            ax.plot(trace[:, 0], trace[:, 1], "r-", linewidth=0.6,
                    transform=ccrs.PlateCarree(), zorder=5, alpha=0.8)

    # --- velocity overlay (if provided) ------------------------------------
    if vf is not None:
        _draw_velocity_arrows(ax, vf, color="black", scale=None, width=0.002)

    # --- title -------------------------------------------------------------
    if title is None:
        n_valid = int(np.sum(np.isfinite(values)))
        n_total = values.size
        title = f"{label}  ({n_valid:,} / {n_total:,} cells valid)"
    ax.set_title(title, fontsize=11)

    # --- save / show -------------------------------------------------------
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        _show_and_close(fig)
    else:
        # Return figure for interactive use / embedding
        return fig

    return fig


# ---------------------------------------------------------------------------
# Velocity field plotting
# ---------------------------------------------------------------------------

def _draw_velocity_arrows(ax, vf, color="black", scale=None, width=0.002, zorder=4):
    """Draw GPS velocity arrows on a Cartopy axes.

    Parameters
    ----------
    ax : GeoAxes
    vf : VelocityField
    color : str or array-like
        Arrow color(s).
    scale : float or None
        Quiver scale.  Auto-computed when *None*.
    width : float
        Arrow shaft width.
    zorder : int
    """
    import cartopy.crs as ccrs

    if scale is None:
        # Auto-scale: make the median speed ~0.5° long on the map
        speeds = np.sqrt(vf.ve ** 2 + vf.vn ** 2)
        ref = float(np.median(speeds[np.isfinite(speeds)])) if len(speeds) > 0 else 1.0
        scale = ref / 0.5 if ref > 0 else 400

    ax.quiver(vf.lon, vf.lat, vf.ve, vf.vn,
              transform=ccrs.PlateCarree(), scale=scale,
              color=color, width=width, zorder=zorder)


def plot_velocity_map(
    vf,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    polygon: Optional[np.ndarray] = None,
    faults: Optional[List[np.ndarray]] = None,
    color_by_speed: bool = True,
    cmap: str = "plasma",
    dpi: int = 150,
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
) -> object:
    """Plot GPS velocity field as a GMT-style map with colour-coded arrows.

    Parameters
    ----------
    vf : VelocityField
        GPS velocity field to plot.
    output_path : str
        Output PNG path.
    region : [slon, elon, slat, elat] or None
    polygon : (N,2) ndarray or None
        Boundary polygon overlay.
    faults : list of ndarray or None
        Fault traces to overlay (red lines).
    color_by_speed : bool
        If True (default), arrows are coloured by total speed magnitude.
        If False, arrows are drawn in black.
    cmap : str
        Colormap for speed-based colouring.
    dpi : int
    figsize : tuple
    title : str or None

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    speeds = np.sqrt(vf.ve ** 2 + vf.vn ** 2)

    fig, ax, extent = _create_map_axes(region=region, lon=vf.lon, lat=vf.lat,
                                       figsize=figsize)

    if color_by_speed and len(speeds) > 0:
        vmin = 0.0
        vmax = float(np.percentile(speeds[np.isfinite(speeds)], 95).item())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = plt.cm.get_cmap(cmap)(norm(speeds))

        ax.quiver(vf.lon, vf.lat, vf.ve, vf.vn,
                  transform=ccrs.PlateCarree(), scale=None,
                  color=colors, width=0.003, zorder=3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.03)
        cbar.set_label("Velocity (mm/yr)", fontsize=10)
    else:
        _draw_velocity_arrows(ax, vf, color="black", width=0.003)

    # --- reference arrow ---
    min_lon, max_lon, min_lat, max_lat = extent
    ref_lon = min_lon + (max_lon - min_lon) * 0.12
    ref_lat = min_lat + (max_lat - min_lat) * 0.08
    ref_label = f"{vmax:.0f} mm/yr" if (color_by_speed and len(speeds) > 0) else "ref"
    ax.quiver(ref_lon, ref_lat, vmax if color_by_speed else 100, 0,
              transform=ccrs.PlateCarree(), scale=400 if not color_by_speed else None,
              color="black", width=0.005, zorder=5)
    ax.text(ref_lon, ref_lat + (max_lat - min_lat) * 0.03, ref_label,
            transform=ccrs.PlateCarree(), fontsize=8, zorder=5)

    # --- overlays ---
    if polygon is not None:
        ax.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=5)
    if faults:
        for trace in faults:
            ax.plot(trace[:, 0], trace[:, 1], "r-", linewidth=0.6,
                    transform=ccrs.PlateCarree(), zorder=5, alpha=0.8)

    if title is None:
        title = f"GPS Velocity Field ({len(vf)} sites)"
    ax.set_title(title, fontsize=11)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)
    return fig


def plot_strain_overview(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    fields: dict,
    output_path: str,
    region: Optional[Sequence[float]] = None,
    polygon: Optional[np.ndarray] = None,
    faults: Optional[List[np.ndarray]] = None,
    dpi: int = 150,
    figsize: tuple = (14, 12),
) -> object:
    """4-panel overview figure: dilatation, max shear, 2nd invariant, rotation.

    Parameters
    ----------
    grid_lon, grid_lat : 1D ndarray
        Grid coordinates.
    fields : dict
        Must contain ``"dilation"``, ``"shear"``, ``"sec_inv"``, ``"omega"``
        as 2D (nlat × nlon) arrays.
    output_path : str
    region : [slon, elon, slat, elat] or None
    polygon : (N, 2) ndarray or None
    dpi : int
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    panel_fields = [
        ("dilation", "Dilatation"),
        ("shear", "Max shear"),
        ("sec_inv", "2nd invariant"),
        ("omega", "Rotation"),
    ]

    if region is None:
        region = [
            float(grid_lon[0]), float(grid_lon[-1]),
            float(grid_lat[0]), float(grid_lat[-1]),
        ]

    fig, axes = plt.subplots(2, 2, figsize=figsize,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()

    for ax, (field_name, field_title) in zip(axes, panel_fields):
        if field_name not in fields:
            ax.text(0.5, 0.5, f"{field_name}\nnot available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")
            ax.set_title(field_title)
            continue

        data = fields[field_name]
        cmap, sym, label = _get_field_meta(field_name)
        finite = data[np.isfinite(data)]

        if len(finite) > 1:
            if sym:
                vmax_ = float(np.percentile(np.abs(finite), 98).item())
                vmin_ = -vmax_
            else:
                pcts = np.percentile(finite, [2, 98])
                vmin_, vmax_ = float(pcts[0].item()), float(pcts[1].item())
                if vmin_ >= vmax_:
                    span = max(abs(vmax_), 1e-6) * 0.1
                    vmin_ -= span
                    vmax_ += span
        else:
            vmin_, vmax_ = 0, 1

        # Basemap
        ax.set_extent(region, ccrs.Geodetic())
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor="lightyellow", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", zorder=1)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, zorder=1,
                          linewidth=0.3, color="gray", alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        if ax != axes[0] and ax != axes[2]:
            gl.left_labels = False

        # Data
        mesh = ax.pcolormesh(
            grid_lon, grid_lat, data,
            cmap=cmap, vmin=vmin_, vmax=vmax_,
            transform=ccrs.PlateCarree(),
            shading="auto", zorder=2,
        )

        cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)
        cbar.set_label(label, fontsize=8)

        if polygon is not None:
            ax.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=0.8,
                    transform=ccrs.PlateCarree(), zorder=5)

        if faults:
            for trace in faults:
                ax.plot(trace[:, 0], trace[:, 1], "r-", linewidth=0.4,
                        transform=ccrs.PlateCarree(), zorder=5, alpha=0.7)

        ax.set_title(field_title, fontsize=10)

    fig.suptitle("Strain Rate Overview", fontsize=13, y=0.98)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)
    return fig


def plot_principal_crosses_overlay(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    azimuth: np.ndarray,
    output_path: str,
    background_field: Optional[np.ndarray] = None,
    background_cmap: str = "RdBu_r",
    background_label: str = "Dilatation (ns/yr)",
    region: Optional[Sequence[float]] = None,
    polygon: Optional[np.ndarray] = None,
    dpi: int = 150,
    step: int = 3,
    scale: Optional[float] = None,
) -> object:
    """Plot principal strain crosses on a gridded background.

    Similar to GMT's ``psvelo`` style: crosses indicating extension
    (red) and compression (blue) directions over a continuous colour
    background.

    Parameters
    ----------
    grid_lon, grid_lat : 1D ndarray
        Grid coordinates (same shape for all 2D fields).
    e1, e2, azimuth : 2D ndarray (nlat × nlon)
        Principal strain components and azimuth at each grid cell.
    output_path : str
    background_field : 2D ndarray or None
        Scalar field to show as background colour.  If *None*, the
        principal strains are drawn on the Cartopy basemap.
    background_cmap, background_label : str
    region : [slon, elon, slat, elat] or None
    polygon : (N, 2) ndarray or None
    dpi : int
    step : int
        Decimation factor: arrows are drawn every *step* grid cells.
    scale : float or None

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    if region is None:
        region = [
            float(grid_lon[0]), float(grid_lon[-1]),
            float(grid_lat[0]), float(grid_lat[-1]),
        ]

    fig, ax, _ = _create_map_axes(region=region)

    # --- background --------------------------------------------------------
    if background_field is not None:
        finite = background_field[np.isfinite(background_field)]
        if len(finite) > 1:
            vmax_ = float(np.percentile(np.abs(finite), 98).item())
            vmin_ = -vmax_
        else:
            vmin_, vmax_ = 0, 1

        ax.pcolormesh(
            grid_lon, grid_lat, background_field,
            cmap=background_cmap, vmin=vmin_, vmax=vmax_,
            transform=ccrs.PlateCarree(),
            shading="auto", zorder=2, alpha=0.85,
        )

    # --- auto-scale --------------------------------------------------------
    valid_e1 = e1[np.isfinite(e1)]
    valid_e2 = e2[np.isfinite(e2)]
    all_valid = np.concatenate([valid_e1, valid_e2])
    if scale is None and len(all_valid) > 0:
        max_abs = float(np.max(np.abs(all_valid)))
        scale = 0.20 / max(max_abs, 1.0)

    # --- draw crosses (decimated) ------------------------------------------
    for i in range(0, len(grid_lat), step):
        for j in range(0, len(grid_lon), step):
            if not (np.isfinite(e1[i, j]) and np.isfinite(e2[i, j]) and
                    np.isfinite(azimuth[i, j])):
                continue

            az_rad = np.radians(azimuth[i, j])
            cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)

            dx1 = float(e1[i, j]) * scale * cos_a
            dy1 = float(e1[i, j]) * scale * sin_a
            dx2 = float(e2[i, j]) * scale * (-sin_a)
            dy2 = float(e2[i, j]) * scale * cos_a

            c1 = "blue" if e1[i, j] < 0 else "red"
            c2 = "blue" if e2[i, j] < 0 else "red"

            lon_ij, lat_ij = float(grid_lon[j]), float(grid_lat[i])

            transform = ccrs.PlateCarree()._as_mpl_transform(ax)
            ax.annotate(
                "",
                xy=(lon_ij + dx1, lat_ij + dy1),
                xytext=(lon_ij - dx1, lat_ij - dy1),
                arrowprops=dict(arrowstyle="->", color=c1, lw=0.5),
                xycoords=transform,
                textcoords=transform,
                zorder=4,
            )
            ax.annotate(
                "",
                xy=(lon_ij + dx2, lat_ij + dy2),
                xytext=(lon_ij - dx2, lat_ij - dy2),
                arrowprops=dict(arrowstyle="->", color=c2, lw=0.5),
                xycoords=transform,
                textcoords=transform,
                zorder=4,
            )

    # --- polygon -----------------------------------------------------------
    if polygon is not None:
        ax.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=5)

    ax.set_title("Principal Strain Rate (blue=compression, red=extension)",
                 fontsize=11)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    _show_and_close(fig)
    return fig
