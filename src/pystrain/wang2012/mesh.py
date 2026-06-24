"""Adaptive mesh generation for Wang (2012) strain-rate estimation.

Generates a triangular mesh whose resolution adapts to GPS station density:
dense clusters get finer triangles; sparse regions stay coarse.

Two mesh-generation strategies are provided:

1. **adaptive**  — density-map + multi-level grid with small random perturbation.
   Fast and deterministic, but grid-aligned points can produce right triangles.
2. **poisson**   — density-map + Poisson disk sampling at each resolution level.
   Produces naturally distributed points and well-shaped triangles at a
   modestly higher computational cost.

Algorithm (density-map, shared by both strategies)
--------------------------------------------------
1. Build a 2D histogram of stations at min_spacing resolution.
2. Down-sample (max-pool) the histogram to coarser levels to determine
   which cells need refinement: any cell whose station count exceeds
   max_stations gets refined to min_spacing.
3. Enforce 2:1 balance via a fast two-pass rule.
4. Generate vertices (grid-based or Poisson-disk) per spacing level.
5. Collect unique vertices, clip to polygon, Delaunay-triangulate.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial import Delaunay, ConvexHull

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def _tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


# ---------------------------------------------------------------------------
# Shared: build a spacing map from station density
# ---------------------------------------------------------------------------

def _build_spacing_map(
    region: list,
    min_spacing: float,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    max_stations: int = 6,
    max_spacing: Optional[float] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Build a 2-D spacing map from GPS station density.

    Parameters
    ----------
    show_progress : bool
        Display a progress bar during the multi-level refinement pass.

    Returns
    -------
    spacing_map : (nlon_fine, nlat_fine) ndarray
        Desired mesh spacing at each fine-grid cell (degrees).
    fine_lon_edges, fine_lat_edges : ndarray
        Bin edges of the fine grid.
    levels : list of float
        Spacing levels from coarse to fine (sorted ascending).
    """
    slon, elon, slat, elat = region

    if max_spacing is None:
        max_spacing = min_spacing * 4.0

    n_levels = max(1, int(np.round(np.log2(max_spacing / min_spacing))) + 1)
    levels = [max_spacing / (2 ** k) for k in range(n_levels)]
    levels[-1] = min_spacing

    # ---- fine grid edges ----
    fine_lon_edges = np.arange(slon, elon + min_spacing / 2, min_spacing)
    fine_lat_edges = np.arange(slat, elat + min_spacing / 2, min_spacing)
    if fine_lon_edges[-1] < elon:
        fine_lon_edges = np.append(fine_lon_edges, fine_lon_edges[-1] + min_spacing)
    if fine_lat_edges[-1] < elat:
        fine_lat_edges = np.append(fine_lat_edges, fine_lat_edges[-1] + min_spacing)

    fine_nlon = len(fine_lon_edges) - 1
    fine_nlat = len(fine_lat_edges) - 1

    if fine_nlon < 1 or fine_nlat < 1:
        raise ValueError("Region too small for the given spacing.")

    # ---- station counts at finest resolution ----
    station_counts, _, _ = np.histogram2d(
        station_lon, station_lat,
        bins=[fine_lon_edges, fine_lat_edges],
    )

    # ---- multi-level refinement map ----
    spacing_map = np.full((fine_nlon, fine_nlat), levels[0])

    level_iter = _tqdm(
        range(n_levels - 1),
        desc="  Building spacing map",
        unit="level",
        disable=not (show_progress and _HAS_TQDM),
    )
    for k in level_iter:
        sp_k = levels[k]
        sp_next = levels[k + 1]
        factor = int(sp_k / min_spacing)

        nlon_k = max(1, fine_nlon // factor)
        nlat_k = max(1, fine_nlat // factor)
        counts_k = _downsample_counts(station_counts, factor, nlon_k, nlat_k)
        needs_refine = counts_k > max_stations

        # Expand to fine-grid resolution
        expanded = np.repeat(np.repeat(needs_refine, factor, axis=0), factor, axis=1)

        # Crop / pad to match fine grid
        elon_dim, elat_dim = expanded.shape
        if elon_dim > fine_nlon:
            expanded = expanded[:fine_nlon, :]
        elif elon_dim < fine_nlon:
            expanded = np.pad(expanded, ((0, fine_nlon - elon_dim), (0, 0)), mode='edge')
        if expanded.shape[1] > fine_nlat:
            expanded = expanded[:, :fine_nlat]
        elif expanded.shape[1] < fine_nlat:
            expanded = np.pad(expanded, ((0, 0), (0, fine_nlat - expanded.shape[1])), mode='edge')

        spacing_map[expanded] = np.minimum(spacing_map[expanded], sp_next)

        if _HAS_TQDM and show_progress:
            n_refined = int(np.sum(expanded))
            level_iter.set_postfix(refined_cells=n_refined)

    # ---- enforce 2:1 balance ----
    spacing_map = _enforce_balance(spacing_map, levels, show_progress=show_progress)

    return spacing_map, fine_lon_edges, fine_lat_edges, levels


# ---------------------------------------------------------------------------
# Poisson disk sampling
# ---------------------------------------------------------------------------

def _poisson_disk_sample(
    region: list,
    radius: float,
    existing_points: Optional[np.ndarray] = None,
    k: int = 30,
    show_progress: bool = True,
    desc: str = "  Poisson sampling",
) -> np.ndarray:
    """Generate points via Poisson disk sampling (bridson-style).

    Points are at least *radius* apart and distributed randomly (blue noise),
    which yields naturally shaped Delaunay triangles.

    Parameters
    ----------
    region : [slon, elon, slat, elat]
    radius : float
        Minimum separation distance.
    existing_points : (N, 2) ndarray or None
        Points already placed; new points will respect their exclusion zones.
    k : int
        Candidates to generate per active point before marking it inactive.
    show_progress : bool
        Display a progress bar tracking accepted points.
    desc : str
        Label for the progress bar.

    Returns
    -------
    new_points : (M, 2) ndarray
        Only the *newly generated* points (excludes *existing_points*).
    """
    slon, elon, slat, elat = region
    width = elon - slon
    height = elat - slat

    if width <= 0 or height <= 0:
        return np.empty((0, 2))

    cell_size = radius / np.sqrt(2)
    ncols = max(1, int(np.ceil(width / cell_size)))
    nrows = max(1, int(np.ceil(height / cell_size)))

    # Estimate total points for the progress bar
    # Poisson disk density ≈ 1 / (radius² · √3/2)
    est_total = int(width * height / (radius * radius * 0.866)) + 1

    # Spatial grid for fast neighbour look-ups
    grid = np.full((ncols, nrows), -1, dtype=int)
    points = []   # list of [x, y]
    active = []   # indices into *points*

    def _cell(x, y):
        return int((x - slon) / cell_size), int((y - slat) / cell_size)

    def _add_point(x, y):
        col, row = _cell(x, y)
        if 0 <= col < ncols and 0 <= row < nrows:
            grid[col, row] = len(points)
        points.append([x, y])
        active.append(len(points) - 1)

    # Seed existing points into the grid (they do NOT enter the active list)
    n_existing = 0
    if existing_points is not None and len(existing_points) > 0:
        n_existing = len(existing_points)
        for i, (x, y) in enumerate(existing_points):
            col, row = _cell(x, y)
            if 0 <= col < ncols and 0 <= row < nrows:
                grid[col, row] = i
            points.append([x, y])

    # If no seed yet, pick a random start
    if not points:
        x0 = slon + np.random.rand() * width
        y0 = slat + np.random.rand() * height
        _add_point(x0, y0)

    # Progress bar: track accepted points
    pbar = _tqdm(
        total=est_total,
        desc=desc,
        unit="pts",
        disable=not (show_progress and _HAS_TQDM),
    )
    pbar.update(len(points) - n_existing)

    while active:
        idx = np.random.randint(len(active))
        px, py = points[active[idx]]
        found = False

        for _ in range(k):
            angle = np.random.rand() * 2 * np.pi
            r = radius * (1.0 + np.random.rand())
            nx = px + r * np.cos(angle)
            ny = py + r * np.sin(angle)

            if not (slon <= nx <= elon and slat <= ny <= elat):
                continue

            col, row = _cell(nx, ny)
            if col < 0 or col >= ncols or row < 0 or row >= nrows:
                continue

            # Check neighbours in a 5×5 window
            ok = True
            for dc in range(-2, 3):
                for dr in range(-2, 3):
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < ncols and 0 <= nr < nrows:
                        nidx = grid[nc, nr]
                        if nidx >= 0:
                            dx = nx - points[nidx][0]
                            dy = ny - points[nidx][1]
                            if dx * dx + dy * dy < radius * radius:
                                ok = False
                                break
                if not ok:
                    break

            if ok:
                _add_point(nx, ny)
                pbar.update(1)
                found = True
                break

        if not found:
            active.pop(idx)

    pbar.close()

    # Return only newly generated points
    if n_existing == 0:
        return np.array(points)
    return np.array(points[n_existing:])


# ---------------------------------------------------------------------------
# Poisson-disk adaptive mesh
# ---------------------------------------------------------------------------

def generate_poisson_mesh(
    region: list,
    min_spacing: float,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    max_stations: int = 6,
    max_spacing: Optional[float] = None,
    show_progress: bool = True,
) -> Tuple[Delaunay, np.ndarray, np.ndarray]:
    """Generate a Poisson-disk-sampled adaptive triangular mesh.

    Station-dense cells resolve down to *min_spacing*; sparse regions
    stay at up to *max_spacing* (default: min_spacing × 4).  Vertex
    positions come from Poisson disk sampling **within each spacing
    level**, producing naturally distributed points that result in
    well-shaped (near-equilateral) Delaunay triangles — unlike the
    grid-based approach which tends to produce right triangles.

    Parameters
    ----------
    region : [slon, elon, slat, elat]   bounding box in degrees.
    min_spacing : float                  finest spacing (degrees).
    station_lon, station_lat : ndarray   GPS positions.
    polygon : ndarray or None            N×2 boundary to clip to.
    max_stations : int                   refine cells with > this count.
    max_spacing : float or None          coarsest spacing.
    show_progress : bool                 display tqdm progress bars.

    Returns
    -------
    tri : Delaunay
    mesh_lon, mesh_lat : ndarray
    """
    slon, elon, slat, elat = region

    if max_spacing is None:
        max_spacing = min_spacing * 4.0

    # ---- 1. Build spacing map (shared with grid-based adaptive mesh) ----
    spacing_map, fine_lon_edges, fine_lat_edges, levels = _build_spacing_map(
        region, min_spacing, station_lon, station_lat,
        max_stations=max_stations, max_spacing=max_spacing,
        show_progress=show_progress,
    )
    fine_nlon, fine_nlat = spacing_map.shape

    # ---- 2. Poisson-disk sample at each spacing level (coarse → fine) ----
    all_points = []   # accumulated points at coarser levels constrain finer ones
    n_levels = len(levels)

    level_list = sorted(levels, reverse=True)   # coarse → fine (coarse points constrain finer ones)
    for i, sp in enumerate(_tqdm(level_list, desc="  Poisson mesh levels", unit="level",
                                  disable=not (show_progress and _HAS_TQDM))):
        radius = sp  # minimum separation ≈ grid spacing

        pts = _poisson_disk_sample(
            region, radius,
            existing_points=np.array(all_points) if all_points else None,
            show_progress=show_progress,
            desc=f"  Poisson L{i+1}/{n_levels} (sp={sp:.3f}°)",
        )
        if len(pts) == 0:
            continue

        # Keep only points whose mapped spacing matches this level
        ilon = np.clip(
            np.searchsorted(fine_lon_edges, pts[:, 0]) - 1, 0, fine_nlon - 1
        )
        ilat = np.clip(
            np.searchsorted(fine_lat_edges, pts[:, 1]) - 1, 0, fine_nlat - 1
        )
        keep = np.isclose(spacing_map[ilon, ilat], sp, atol=1e-10)
        all_points.extend(pts[keep].tolist())

    if not all_points:
        raise ValueError("Poisson mesh generated no points — check region / spacing.")

    mesh_lon = np.array([p[0] for p in all_points])
    mesh_lat = np.array([p[1] for p in all_points])

    # ---- 3. Deduplicate ----
    if show_progress and _HAS_TQDM:
        print("  Deduplicating vertices ...", end=" ", flush=True)
    xy = np.column_stack([mesh_lon, mesh_lat])
    xy = np.unique(xy.round(decimals=10), axis=0)
    mesh_lon, mesh_lat = xy[:, 0], xy[:, 1]
    if show_progress and _HAS_TQDM:
        print(f"{len(mesh_lon)} unique.")

    # ---- 4. Clip to polygon ----
    if polygon is not None and len(mesh_lon) > 0:
        if show_progress and _HAS_TQDM:
            print("  Clipping to polygon ...", end=" ", flush=True)
        from matplotlib.path import Path as MplPath
        path = MplPath(polygon)
        inside = path.contains_points(np.column_stack([mesh_lon, mesh_lat]))
        mesh_lon = mesh_lon[inside]
        mesh_lat = mesh_lat[inside]
        if show_progress and _HAS_TQDM:
            print(f"{len(mesh_lon)} inside.")

    if len(mesh_lon) < 3:
        raise ValueError(
            "Too few mesh vertices (%d) inside polygon — "
            "reduce min_spacing or check region/polygon." % len(mesh_lon)
        )

    # ---- 5. Delaunay triangulation -----------------------------------------
    if show_progress and _HAS_TQDM:
        print("  Delaunay triangulating ...", end=" ", flush=True)
    tri = Delaunay(np.column_stack([mesh_lon, mesh_lat]))
    if show_progress and _HAS_TQDM:
        print(f"{len(tri.simplices)} triangles.")

    return tri, mesh_lon, mesh_lat


# ---------------------------------------------------------------------------
# Gmsh-based boundary-conforming mesh
# ---------------------------------------------------------------------------

def generate_gmsh_mesh(
    region: list,
    min_spacing: float,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    max_stations: int = 6,
    max_spacing: Optional[float] = None,
    show_progress: bool = True,
) -> Tuple[Delaunay, np.ndarray, np.ndarray]:
    """Generate a boundary-conforming, station-density-adaptive mesh via Gmsh.

    Mesh resolution is driven by the same station-density spacing map used
    by the ``adaptive`` and ``poisson`` methods: cells with more than
    ``max_stations`` GPS sites get finer triangles.  Gmsh's
    Frontal-Delaunay mesher produces well-shaped triangles with smooth
    size transitions and exact polygon-boundary fidelity.

    Requires ``pip install gmsh``.

    Parameters
    ----------
    region : [slon, elon, slat, elat]
    min_spacing : float
        Finest mesh spacing (degrees) in dense station clusters.
    station_lon, station_lat : ndarray
        GPS station positions — used ONLY to build the spacing map,
        NOT added as mesh nodes.
    polygon : ndarray or None
        N×2 boundary.  If None, the region box is used.
    max_stations : int
        Refine cells with more than this many stations.
    max_spacing : float or None
        Coarsest spacing (default: min_spacing × 4).
    show_progress : bool

    Returns
    -------
    tri : Delaunay
    mesh_lon, mesh_lat : ndarray
    """
    import gmsh

    if max_spacing is None:
        max_spacing = min_spacing * 4.0

    # ---- 1. Build station-density spacing map ----
    spacing_map, fine_lon_edges, fine_lat_edges, levels = _build_spacing_map(
        region, min_spacing, station_lon, station_lat,
        max_stations=max_stations, max_spacing=max_spacing,
        show_progress=show_progress,
    )
    fine_nlon, fine_nlat = spacing_map.shape
    slon, elon, slat, elat = region

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_spacing)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_spacing)
    # Limit mesh size extrapolation — don't go coarser than max_spacing
    # even far from geometry
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

    try:
        model = gmsh.model
        model.add("wang2012_gmsh")

        # ---- 2. Polygon boundary geometry ----
        if polygon is not None and len(polygon) >= 3:
            bnd = polygon
        else:
            bnd = np.array([
                [slon, slat], [elon, slat], [elon, elat], [slon, elat],
            ])

        pt_tags = []
        for blon, blat in bnd:
            # Boundary spacing from map (tends toward max_spacing at edges,
            # min_spacing near dense clusters)
            ilon = np.clip(np.searchsorted(fine_lon_edges, blon) - 1,
                           0, fine_nlon - 1)
            ilat = np.clip(np.searchsorted(fine_lat_edges, blat) - 1,
                           0, fine_nlat - 1)
            local_sp = float(spacing_map[ilon, ilat])
            tag = model.geo.addPoint(float(blon), float(blat), 0.0, local_sp)
            pt_tags.append(tag)

        line_tags = []
        for i in range(len(pt_tags)):
            tag = model.geo.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
            line_tags.append(tag)

        surf_tag = model.geo.addPlaneSurface([model.geo.addCurveLoop(line_tags)])
        model.geo.synchronize()

        # ---- 3. Embed size-control points in the surface ----
        # Sample spacing map at *max_spacing* resolution.  Each point
        # carries the local target mesh size and is embedded into the
        # 2-D surface so gmsh respects it during meshing.
        ctl_lons = np.arange(slon, elon + max_spacing / 2, max_spacing)
        ctl_lats = np.arange(slat, elat + max_spacing / 2, max_spacing)
        ctl_lon_grid, ctl_lat_grid = np.meshgrid(ctl_lons, ctl_lats)
        ctl_lon = ctl_lon_grid.ravel()
        ctl_lat = ctl_lat_grid.ravel()

        ctl_ilon = np.clip(
            np.searchsorted(fine_lon_edges, ctl_lon) - 1, 0, fine_nlon - 1
        )
        ctl_ilat = np.clip(
            np.searchsorted(fine_lat_edges, ctl_lat) - 1, 0, fine_nlat - 1
        )
        ctl_sizes = spacing_map[ctl_ilon, ctl_ilat]

        # Keep only points inside polygon
        if polygon is not None and len(polygon) >= 3:
            from matplotlib.path import Path as MplPath
            inside = MplPath(polygon).contains_points(
                np.column_stack([ctl_lon, ctl_lat])
            )
            ctl_lon = ctl_lon[inside]
            ctl_lat = ctl_lat[inside]
            ctl_sizes = ctl_sizes[inside]

        # Add as geometry points with local mesh size, then EMBED them
        # into the surface.  Unlike disconnected geo points, embedded
        # points ARE respected by the 2-D mesher.
        size_pt_tags = []
        for clon, clat, csz in zip(ctl_lon, ctl_lat, ctl_sizes):
            tag = model.geo.addPoint(float(clon), float(clat), 0.0,
                                     float(csz))
            size_pt_tags.append(tag)

        model.geo.synchronize()

        # Embed all size-control points into the surface
        model.mesh.embed(0, size_pt_tags, 2, surf_tag)

        if show_progress and _HAS_TQDM:
            print(f"  {len(size_pt_tags)} embedded size-control points,"
                  f" meshing ...", end=" ", flush=True)

        # ---- 4. Generate mesh ----
        model.mesh.generate(2)
        if show_progress and _HAS_TQDM:
            print("done.", flush=True)

        # ---- 5. Extract gmsh nodes & triangles ----
        # Gmsh has generated a boundary-conforming mesh INSIDE the
        # polygon surface.  We keep gmsh's own triangles — no Delaunay
        # re-run, no polygon clipping.  This preserves:
        #   - Exact boundary fidelity (triangles go right to the edge)
        #   - Gmsh's well-shaped elements (Frontal-Delaunay)
        #   - Smooth size transitions between refined/coarse regions
        node_tags, node_coords, _ = model.mesh.getNodes()
        all_lon = node_coords[0::3]
        all_lat = node_coords[1::3]

        # Build tag→index map
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Get 2-D elements
        elem_types, elem_tags, elem_node_tags = model.mesh.getElements(2)
        all_tris = []
        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 2:  # 3-node triangle
                arr = np.array(enodes, dtype=int)
                n_elem = len(etags)
                n_per = len(arr) // n_elem
                tris = arr.reshape(n_elem, n_per)[:, :3]  # drop repeated 4th col
                all_tris.append(tris)
        if all_tris:
            raw_simplices = np.vstack(all_tris)
        else:
            raw_simplices = np.empty((0, 3), dtype=int)

        # Map gmsh tags → compact 0-based indices
        simplices = np.zeros_like(raw_simplices)
        for k in range(raw_simplices.shape[0]):
            for m in range(3):
                simplices[k, m] = tag_to_idx[int(raw_simplices[k, m])]

        # Only keep nodes that are referenced by surface triangles
        used = np.unique(simplices)
        mesh_lon = all_lon[used]
        mesh_lat = all_lat[used]
        reindex = {old: new for new, old in enumerate(used)}
        simplices = np.array([
            [reindex[s[0]], reindex[s[1]], reindex[s[2]]]
            for s in simplices
        ], dtype=np.int32)

    finally:
        gmsh.finalize()

    if len(mesh_lon) < 3:
        raise ValueError(
            "Gmsh produced too few mesh vertices — "
            "reduce min_spacing or check region/polygon."
        )

    if show_progress and _HAS_TQDM:
        print(f"  {len(mesh_lon)} mesh nodes, {len(simplices)} triangles.",
              flush=True)

    # ---- Wrap gmsh triangulation with custom find_simplex ----
    tri_pts = np.column_stack([mesh_lon, mesh_lat])
    centroids = tri_pts[simplices].mean(axis=1)  # (n_tri, 2)
    from scipy.spatial import KDTree
    kd = KDTree(centroids)

    class _GmshTriangulation:
        """Gmsh-native triangulation wrapper with scipy-compatible API."""
        def __init__(self):
            self.simplices = simplices
            self._pts = tri_pts
            self._kd = kd
            self._centroids = centroids

        def find_simplex(self, xy):
            """Locate which gmsh triangle contains each query point.

            Uses KD-tree centroid search + barycentric test.
            """
            xy = np.atleast_2d(xy)
            n_q = xy.shape[0]
            result = np.full(n_q, -1, dtype=np.int32)

            for i in range(n_q):
                px, py = xy[i, 0], xy[i, 1]
                # Search nearest 5 centroids
                _, nn = self._kd.query([px, py], k=min(5, len(self.simplices)))
                if not hasattr(nn, '__len__'):
                    nn = [nn]
                for t_idx in nn:
                    a, b, c = self._pts[self.simplices[t_idx]]
                    # Barycentric test
                    v0x, v0y = c[0] - a[0], c[1] - a[1]
                    v1x, v1y = b[0] - a[0], b[1] - a[1]
                    v2x, v2y = px - a[0], py - a[1]
                    denom = v0x * v1y - v0y * v1x
                    if abs(denom) < 1e-20:
                        continue
                    u = (v2x * v1y - v2y * v1x) / denom
                    v = (v0x * v2y - v0y * v2x) / denom
                    if u >= -1e-12 and v >= -1e-12 and u + v <= 1 + 1e-12:
                        result[i] = t_idx
                        break
            return result

    return _GmshTriangulation(), mesh_lon, mesh_lat


# ---------------------------------------------------------------------------
# Grid-based adaptive mesh (original)


# ---------------------------------------------------------------------------
# Grid-based adaptive mesh (original)
# ---------------------------------------------------------------------------

def generate_adaptive_mesh(
    region: list,
    min_spacing: float,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    polygon: Optional[np.ndarray] = None,
    max_stations: int = 6,
    max_spacing: Optional[float] = None,
    randomize: bool = True,
    randomize_fraction: float = 0.2,
    show_progress: bool = True,
) -> Tuple[Delaunay, np.ndarray, np.ndarray]:
    """Generate a GPS-density-adaptive triangular mesh (grid-based).

    Station-dense cells resolve down to *min_spacing*; sparse regions
    stay at up to *max_spacing* (default: min_spacing × 4).

    Note: this method places vertices on regular grids at each spacing
    level, which can produce right / rectangle-shaped triangles after
    Delaunay triangulation.  For more naturally shaped triangles,
    consider :func:`generate_poisson_mesh`.

    Parameters
    ----------
    region : [slon, elon, slat, elat]   bounding box in degrees.
    min_spacing : float                  finest cell size (degrees).
    station_lon, station_lat : ndarray   GPS positions.
    polygon : ndarray or None            N×2 boundary to clip to.
    max_stations : int                   refine cells with > this count.
    max_spacing : float or None          coarsest cell size.
    randomize : bool                     perturb interior nodes.
    randomize_fraction : float           perturbation / min_spacing.
    show_progress : bool                 display tqdm progress bars.

    Returns
    -------
    tri : Delaunay
    mesh_lon, mesh_lat : ndarray
    """
    slon, elon, slat, elat = region

    # ---- 1-3. Build spacing map (shared helper) ----
    spacing_map, fine_lon_edges, fine_lat_edges, levels = _build_spacing_map(
        region, min_spacing, station_lon, station_lat,
        max_stations=max_stations, max_spacing=max_spacing,
        show_progress=show_progress,
    )
    fine_nlon, fine_nlat = spacing_map.shape

    # ---- 4. Generate grid vertices per spacing level ----
    vertices_lon = []
    vertices_lat = []

    unique_spacings = sorted(set(spacing_map.flat), reverse=True)
    for sp in _tqdm(unique_spacings, desc="  Grid mesh levels", unit="level",
                    disable=not (show_progress and _HAS_TQDM)):
        mask = spacing_map == sp
        if not mask.any():
            continue

        lons_sp = np.arange(slon, elon + sp / 2, sp)
        lats_sp = np.arange(slat, elat + sp / 2, sp)

        glon, glat = np.meshgrid(lons_sp, lats_sp)
        glon = glon.ravel()
        glat = glat.ravel()

        ilon = np.clip(
            np.searchsorted(fine_lon_edges, glon) - 1, 0, fine_nlon - 1
        )
        ilat = np.clip(
            np.searchsorted(fine_lat_edges, glat) - 1, 0, fine_nlat - 1
        )

        keep = spacing_map[ilon, ilat] == sp
        vertices_lon.append(glon[keep])
        vertices_lat.append(glat[keep])

    mesh_lon = np.concatenate(vertices_lon)
    mesh_lat = np.concatenate(vertices_lat)

    # ---- 5. Deduplicate ----
    if show_progress and _HAS_TQDM:
        print("  Deduplicating vertices ...", end=" ", flush=True)
    xy = np.column_stack([mesh_lon, mesh_lat])
    xy = np.unique(xy.round(decimals=10), axis=0)
    mesh_lon, mesh_lat = xy[:, 0], xy[:, 1]
    if show_progress and _HAS_TQDM:
        print(f"{len(mesh_lon)} unique.")

    # ---- 6. Clip to polygon ----
    if polygon is not None and len(mesh_lon) > 0:
        if show_progress and _HAS_TQDM:
            print("  Clipping to polygon ...", end=" ", flush=True)
        from matplotlib.path import Path as MplPath
        path = MplPath(polygon)
        inside = path.contains_points(np.column_stack([mesh_lon, mesh_lat]))
        mesh_lon = mesh_lon[inside]
        mesh_lat = mesh_lat[inside]
        if show_progress and _HAS_TQDM:
            print(f"{len(mesh_lon)} inside.")

    if len(mesh_lon) < 3:
        raise ValueError(
            "Too few mesh vertices (%d) inside polygon — "
            "reduce min_spacing or check region/polygon." % len(mesh_lon)
        )

    # ---- 7. Random perturbation for interior vertices ----
    if randomize and len(mesh_lon) > 3:
        if show_progress and _HAS_TQDM:
            print("  Perturbing interior vertices ...", end=" ", flush=True)
        pts = np.column_stack([mesh_lon, mesh_lat])
        hull = ConvexHull(pts)
        boundary = set(hull.vertices)
        interior = np.array([i for i in range(len(mesh_lon)) if i not in boundary])
        if len(interior) > 0:
            pert = randomize_fraction * min_spacing
            mesh_lon[interior] += pert * (np.random.rand(len(interior)) - 0.5)
            mesh_lat[interior] += pert * (np.random.rand(len(interior)) - 0.5)
        if show_progress and _HAS_TQDM:
            print(f"{len(interior)} perturbed.")

    # ---- 8. Delaunay triangulation ----
    if show_progress and _HAS_TQDM:
        print("  Delaunay triangulating ...", end=" ", flush=True)
    tri = Delaunay(np.column_stack([mesh_lon, mesh_lat]))
    if show_progress and _HAS_TQDM:
        print(f"{len(tri.simplices)} triangles.")

    return tri, mesh_lon, mesh_lat


# ---------------------------------------------------------------------------
# Helpers (shared)
# ---------------------------------------------------------------------------

def _downsample_counts(
    fine_counts: np.ndarray, factor: int, nlon: int, nlat: int
) -> np.ndarray:
    """Sum fine-grid counts into coarse cells of size *factor*."""
    nlon_fine, nlat_fine = fine_counts.shape
    nlon_use = min(nlon_fine, nlon * factor)
    nlat_use = min(nlat_fine, nlat * factor)
    trimmed = fine_counts[:nlon_use, :nlat_use]

    pad_lon = nlon * factor - nlon_use
    pad_lat = nlat * factor - nlat_use
    if pad_lon > 0 or pad_lat > 0:
        trimmed = np.pad(trimmed, ((0, pad_lon), (0, pad_lat)), mode='constant')

    reshaped = trimmed.reshape(nlon, factor, nlat * factor)
    summed = reshaped.sum(axis=1)
    result = summed.reshape(nlon, nlat, factor).sum(axis=2)
    return result


def _enforce_balance(
    spacing_map: np.ndarray,
    levels: list,
    show_progress: bool = True,
) -> np.ndarray:
    """Enforce 2:1 rule: no cell is more than one level coarser than neighbours.

    Iteratively refines cells that abut finer neighbours, up to ~20 passes.
    """
    nlon, nlat = spacing_map.shape
    result = spacing_map.copy()
    level_values = sorted(set(levels), reverse=True)

    sp_to_idx = {sp: i for i, sp in enumerate(sorted(set(levels)))}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}
    idx_map = np.full_like(result, 0, dtype=int)
    for sp, idx in sp_to_idx.items():
        idx_map[result == sp] = idx
    max_idx = max(idx_to_sp.keys())

    changed = True
    pbar = _tqdm(
        range(20), desc="  Enforcing 2:1 balance", unit="pass",
        disable=not (show_progress and _HAS_TQDM),
    )
    for _ in pbar:
        if not changed:
            break
        changed = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                shifted = np.full_like(idx_map, -1)
                i_src_start = max(0, -di)
                i_src_end = nlon - max(0, di)
                j_src_start = max(0, -dj)
                j_src_end = nlat - max(0, dj)
                i_dst_start = max(0, di)
                i_dst_end = nlon + min(0, di)
                j_dst_start = max(0, dj)
                j_dst_end = nlat + min(0, dj)

                src_slice = idx_map[i_src_start:i_src_end, j_src_start:j_src_end]
                shifted[i_dst_start:i_dst_end, j_dst_start:j_dst_end] = src_slice

                violation = (shifted >= 0) & (shifted > idx_map + 1)
                if violation.any():
                    idx_map[violation] = np.minimum(
                        idx_map[violation] + 1, max_idx
                    )
                    changed = True

        n_violations = int(np.sum(idx_map > 0))
        if _HAS_TQDM and show_progress:
            pbar.set_postfix(violations=n_violations)
    pbar.close()

    for idx, sp in idx_to_sp.items():
        result[idx_map == idx] = sp

    return result


# ---------------------------------------------------------------------------
# Uniform mesh (backward-compatible)
# ---------------------------------------------------------------------------

def generate_mesh(
    region: list,
    spacing: float,
    polygon: Optional[np.ndarray] = None,
    randomize: bool = True,
    randomize_fraction: float = 0.2,
    show_progress: bool = True,
) -> Tuple[Delaunay, np.ndarray, np.ndarray]:
    """Generate a **uniform** triangular mesh.

    Prefer :func:`generate_adaptive_mesh` for station-density-aware meshing.
    """
    slon, elon, slat, elat = region

    if show_progress and _HAS_TQDM:
        print("  Generating uniform grid ...", end=" ", flush=True)
    lons = np.arange(slon, elon + spacing / 2.0, spacing)
    lats = np.arange(slat, elat + spacing / 2.0, spacing)

    grid_lon, grid_lat = np.meshgrid(lons, lats)
    mesh_lon = grid_lon.ravel()
    mesh_lat = grid_lat.ravel()
    if show_progress and _HAS_TQDM:
        print(f"{len(mesh_lon)} vertices.", flush=True)

    if polygon is not None:
        if show_progress and _HAS_TQDM:
            print("  Clipping to polygon ...", end=" ", flush=True)
        from matplotlib.path import Path as MplPath
        path = MplPath(polygon)
        inside = path.contains_points(np.column_stack([mesh_lon, mesh_lat]))
        mesh_lon = mesh_lon[inside]
        mesh_lat = mesh_lat[inside]
        if show_progress and _HAS_TQDM:
            print(f"{len(mesh_lon)} inside.")

    if len(mesh_lon) < 3:
        raise ValueError("Too few mesh nodes — check region/polygon/spacing.")

    if randomize and len(mesh_lon) > 3:
        if show_progress and _HAS_TQDM:
            print("  Perturbing interior vertices ...", end=" ", flush=True)
        pts = np.column_stack([mesh_lon, mesh_lat])
        hull = ConvexHull(pts)
        boundary = set(hull.vertices)
        interior = np.array(
            [i for i in range(len(mesh_lon)) if i not in boundary]
        )
        if len(interior) > 0:
            perturbation = randomize_fraction * spacing
            mesh_lon[interior] += perturbation * (
                np.random.rand(len(interior)) - 0.5
            )
            mesh_lat[interior] += perturbation * (
                np.random.rand(len(interior)) - 0.5
            )
        if show_progress and _HAS_TQDM:
            print(f"{len(interior)} perturbed.")

    if show_progress and _HAS_TQDM:
        print("  Delaunay triangulating ...", end=" ", flush=True)
    tri = Delaunay(np.column_stack([mesh_lon, mesh_lat]))
    if show_progress and _HAS_TQDM:
        print(f"{len(tri.simplices)} triangles.")

    return tri, mesh_lon, mesh_lat
