#!/usr/bin/env python3
"""Plot strain-rate distributions from PyStrain2 output files.

Usage
-----
    python examples/plot_strain_rate.py pystrain2_output --out-dir figures

The script reads all ``*_strain.txt`` files in the input directory and produces
one multi-panel figure per file showing the main strain-rate scalar fields.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Column names in PyStrain2 strain output files
STRAIN_FIELDS = {
    "exx": r"$\dot{\epsilon}_{xx}$ (nstrain/yr)",
    "exy": r"$\dot{\epsilon}_{xy}$ (nstrain/yr)",
    "eyy": r"$\dot{\epsilon}_{yy}$ (nstrain/yr)",
    "e1": r"$\dot{\epsilon}_1$ (nstrain/yr)",
    "e2": r"$\dot{\epsilon}_2$ (nstrain/yr)",
    "shear": "Max shear (nstrain/yr)",
    "dilation": "Dilation (nstrain/yr)",
    "sec_inv": "Second invariant (nstrain/yr)",
}


def read_strain_file(filepath: str):
    """Read a PyStrain2 strain output file.

    Returns
    -------
    dict
        Arrays keyed by column name.
    """
    data = np.loadtxt(filepath, comments="#")
    with open(filepath, "r", encoding="utf-8") as fid:
        header = fid.readline().lstrip("#").strip().split()

    result = {}
    for i, key in enumerate(header):
        result[key] = data[:, i]
    return result


def _set_map_aspect(lat: np.ndarray):
    """Set axes aspect ratio compensated for mean latitude."""
    lat_mean = np.mean(lat)
    aspect = 1.0 / np.cos(np.deg2rad(lat_mean))
    plt.gca().set_aspect(aspect)


def plot_strain_distribution(
    data: dict,
    title: str = "Strain rate",
    fields: list = None,
    cmap: str = "RdYlBu_r",
    size: float = 20.0,
):
    """Create a multi-panel figure of strain-rate scalar fields."""
    if fields is None:
        fields = ["exx", "exy", "eyy", "e1", "e2", "shear", "dilation", "sec_inv"]

    n_fields = len(fields)
    n_cols = 4
    n_rows = int(np.ceil(n_fields / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
    )

    lon = data["lon"]
    lat = data["lat"]

    for ax, field in zip(axes.flat, fields):
        values = data[field]
        finite = np.isfinite(values)
        if np.any(finite):
            vmin, vmax = np.percentile(values[finite], [2, 98])
        else:
            vmin, vmax = -1.0, 1.0

        sc = ax.scatter(
            lon[finite], lat[finite], c=values[finite],
            cmap=cmap, vmin=vmin, vmax=vmax, s=size,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=STRAIN_FIELDS.get(field, field))

        lat_mean = np.mean(lat[finite]) if np.any(finite) else np.mean(lat)
        aspect = 1.0 / np.cos(np.deg2rad(lat_mean))
        ax.set_aspect(aspect)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(field)

    # Hide unused subplots
    for ax in axes.flat[n_fields:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_principal_strain_axes(
    data: dict,
    title: str = "Principal strain axes",
    scale: float = 1.0,
):
    """Plot principal strain-rate axes as crosses."""
    fig, ax = plt.subplots(figsize=(10, 8))

    lon = data["lon"]
    lat = data["lat"]
    e1 = data["e1"]
    e2 = data["e2"]
    theta = np.deg2rad(data["theta"])

    finite = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(theta)
    lon = lon[finite]
    lat = lat[finite]
    e1 = e1[finite]
    e2 = e2[finite]
    theta = theta[finite]

    # Normalize axis lengths for visibility
    e1_norm = e1 / (np.nanmax(np.abs(e1)) + 1e-30)
    e2_norm = e2 / (np.nanmax(np.abs(e2)) + 1e-30)

    # e1 axis (extension, red)
    dx1 = e1_norm * np.cos(theta) * scale
    dy1 = e1_norm * np.sin(theta) * scale
    # e2 axis (compression, blue)
    dx2 = -e2_norm * np.sin(theta) * scale
    dy2 = e2_norm * np.cos(theta) * scale

    for i in range(len(lon)):
        ax.plot([lon[i] - dx1[i], lon[i] + dx1[i]],
                [lat[i] - dy1[i], lat[i] + dy1[i]], "r-", alpha=0.6, linewidth=1.0)
        ax.plot([lon[i] - dx2[i], lon[i] + dx2[i]],
                [lat[i] - dy2[i], lat[i] + dy2[i]], "b-", alpha=0.6, linewidth=1.0)

    ax.scatter(lon, lat, c="k", s=5, zorder=5)

    lat_mean = np.mean(lat)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_mean)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot PyStrain2 strain-rate output files."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing *_strain.txt files from PyStrain2.",
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="strain_figures",
        help="Directory to save figures (default: strain_figures).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        choices=list(STRAIN_FIELDS.keys()),
        help="Fields to plot (default: all).",
    )
    parser.add_argument(
        "--cmap",
        default="RdYlBu_r",
        help="Matplotlib colormap (default: RdYlBu_r).",
    )
    parser.add_argument(
        "--principal-axes",
        action="store_true",
        help="Also generate principal strain-axis plots.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strain_files = sorted(input_dir.glob("*_strain.txt"))
    if not strain_files:
        print(f"No *_strain.txt files found in {input_dir}")
        return 1

    for filepath in strain_files:
        print(f"Processing {filepath.name} ...")
        data = read_strain_file(str(filepath))
        stem = filepath.stem

        fig = plot_strain_distribution(
            data,
            title=f"Strain rate: {stem}",
            fields=args.fields,
            cmap=args.cmap,
        )
        out_path = out_dir / f"{stem}_scalars.png"
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

        if args.principal_axes:
            fig = plot_principal_strain_axes(
                data,
                title=f"Principal axes: {stem}",
            )
            out_path = out_dir / f"{stem}_axes.png"
            fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
