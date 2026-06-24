"""Polygon boundary file I/O."""

from pathlib import Path
from typing import List
import numpy as np


def read_polygon(filepath: str, tol: float = 1e-6) -> List[np.ndarray]:
    """Read a polygon file with blank-line-separated rings.

    Returns
    -------
    List[np.ndarray]
        List of (N,2) arrays of lon/lat.  The first ring is the exterior;
        subsequent rings are holes.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Polygon file not found: {filepath}")

    rings: List[List[List[float]]] = [[]]
    with open(path, "r", encoding="utf-8") as fid:
        for raw in fid:
            line = raw.strip()
            if not line or line.startswith("#"):
                if rings[-1]:
                    rings.append([])
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rings[-1].append([float(parts[0]), float(parts[1])])

    if rings and not rings[-1]:
        rings.pop()

    result = []
    for ring in rings:
        if len(ring) < 3:
            continue
        arr = np.array(ring)
        # Auto-close with tolerance
        if np.linalg.norm(arr[0] - arr[-1]) > tol:
            arr = np.vstack([arr, arr[0]])
        result.append(arr)

    if not result:
        raise ValueError(f"No valid polygon rings found in {filepath}")

    return result


def write_polygon(filepath: str, rings: List[np.ndarray]) -> None:
    """Write polygon rings to file."""
    with open(filepath, "w", encoding="utf-8") as fid:
        for ring in rings:
            for pt in ring:
                fid.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
            fid.write("\n")
