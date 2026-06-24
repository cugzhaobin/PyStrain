# PyStrain

**PyStrain** is a Python package for estimating GNSS crustal strain rates and strain time series from GPS velocity data. It is a refactored and enhanced version of the original PyStrain2, offering multiple well-established strain estimation methodologies, outlier detection, uncertainty propagation, and rich visualization — all accessible through a unified command-line interface, a YAML-driven pipeline, and an interactive web application.

## Features

### Strain Rate Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| **Shen et al. (2015)** | Grid-based optimal interpolation with distance weighting (Gaussian / quadratic) and spatial-coverage weighting (azimuthal or Voronoi-cell area). Automatically searches for the optimal smoothing distance *D* via Brent's method. | Shen et al., *BSSA*, 2015 |
| **Delaunay Triangulation** | Quality-filtered Delaunay triangulation with weighted least-squares strain estimation at each triangle centroid. Supports minimum angle, maximum edge length, and area-ratio quality controls. | — |
| **Wang (2012)** | Finite-element mesh (adaptive grid, Poisson-disk, or Gmsh-based) with Laplacian smoothing regularization and L-curve optimal smoothing selection. | Wang et al., 2012; England & Molnar, 2005 |
| **User-defined points** | Estimate strain rate at arbitrary user-specified longitude/latitude locations. | — |

### Additional Capabilities

- **Outlier Detection** — Two methods: KNN + MAD pre-screening with iterative IQR removal (fast, for dense networks), and leave-one-out (LOO) strain-based detection (rigorous, for velocity-field inconsistencies).
- **Uncertainty Propagation** — Monte Carlo simulation that perturbs velocities by their Cholesky-decomposed covariance to estimate strain-rate standard deviations.
- **Strain Time Series** — Per-epoch strain at grid points, triangle centroids, or user-defined site groups using aligned multi-station position time series.
- **Visualization** — Cartopy-based pipeline plots: raw velocity, outliers, clean velocity, triangulation, grid points, search radius, strain scalars (dilatation, max shear), Wang mesh, and principal strain crosses.
- **Post-processing** — Scattered-to-regular-grid interpolation with NetCDF and GeoTIFF output, GMT-style continuous-colour pcolormesh maps, fault trace overlays (KML/SHP), and velocity overlay maps.
- **Web Interface** — Streamlit-based GUI for interactive strain-rate and time-series computation (bilingual: English / Chinese).

## Installation

### Requirements

- Python ≥ 3.7
- pip

### From source

```bash
cd /simulation/geodesy/PyStrain
pip install -e .
```

### With optional dependencies

```bash
# For the Streamlit web application
pip install -e ".[gui]"

# For development and testing
pip install -e ".[dev]"

# Everything
pip install -e ".[gui,dev]"
```

### Verify installation

```bash
pystrain --help
```

## Quick Start

### 1. Shen et al. (2015) — Grid-based strain rate

```bash
pystrain shen2015 \
  --vel-file examples/wangmin_velo.gmtvec \
  --region 70 110 15 55 \
  --spacing 1 1 \
  --output-dir ./output_shen2015 \
  --plot
```

### 2. Delaunay Triangulation — Triangle-based strain rate

```bash
pystrain delaunay \
  --vel-file examples/wangmin_velo.gmtvec \
  --poly-file examples/poly.dat \
  --output-dir ./output_delaunay \
  --plot
```

### 3. Wang (2012) — Mesh-based strain rate

```bash
pystrain wang2012 \
  --vel-file examples/wangmin_velo.gmtvec \
  --poly-file examples/poly.dat \
  --mesh-method adaptive \
  --mesh-spacing 0.5 \
  --output-dir ./output_wang2012 \
  --plot
```

### 4. Full pipeline via configuration file

For full control over all parameters, use a YAML configuration file:

```bash
pystrain compute --config examples/config_default.yaml
```

### 5. Post-processing: gridding and mapping

Grid existing strain results to a regular grid and generate publication-quality maps:

```bash
# Generate NetCDF + strain maps with fault overlays
pystrain plot output_shen2015/shen2015_strain.txt \
  --output-dir ./maps \
  --region 70 110 20 50 \
  --format both \
  --fields dilation shear e1 e2

# Add GPS velocity vectors and fault traces to the maps
pystrain plot output_shen2015/shen2015_strain.txt \
  --output-dir ./maps \
  --vel-file examples/wangmin_velo.gmtvec \
  --faults faults.kml
```

### 6. Strain time series

```bash
pystrain timeseries --config examples/config_timeseries.yaml
```

### 7. Launch the web application

```bash
pystrain web --port 8501
```

Then open **http://localhost:8501** in your browser.

## Input Format

### Velocity File

GMT-style columnar text format, auto-detected by column count:

**GMT8** (8 columns, preferred):
```
lon  lat  Ve  Vn  Se  Sn  Rho  SiteName
```

**GMT7** (7 columns, no correlation coefficient):
```
lon  lat  Ve  Vn  Se  Sn  SiteName
```

**GLOBK** (13 columns):
Standard GLOBK `.org` velocity format.

Units: `lon`/`lat` in decimal degrees, `Ve`/`Vn` in mm/yr, `Se`/`Sn` in mm/yr.

### Polygon File (optional)

A boundary file containing one or more polygon rings separated by blank lines:
```
lon1 lat1
lon2 lat2
...
lonN latN

lon1 lat1   ← start of second ring
...
```

### Time Series Input

- **PBO `.pos` files** — Standard UNR/PBO position time-series format.
- **PyTsfit `.dat` files** — Output from the PyTsfit time-series analysis package.

## Configuration File

The YAML configuration file controls every aspect of the pipeline. An annotated example is provided at `examples/config_default.yaml`. Key sections:

```yaml
data:
  vel_file: my_data.gmtvec      # input velocity file
  poly_file: boundary.dat       # optional boundary polygon
  output_dir: ./results         # output directory
  format: auto                  # auto | gmt7 | gmt8 | globk

algorithms:
  method: shen2015              # shen2015 | delaunay | wang2012
  shen2015: { ... }             # method-specific parameters
  delaunay: { ... }
  wang2012: { ... }

outlier_detection:
  enable: true
  method: knn_iqr               # knn_iqr | loo_strain

uncertainty:
  enable: true
  mc_iterations: 200

visualization:
  save_figures: true
  dpi: 150
```

## Output

The pipeline produces the following files in the output directory:

| File | Description |
|------|-------------|
| `{method}_strain.txt` | Strain results: lon, lat, exx, exy, eyy, dilation, shear, second invariant, omega, principal strains (e1, e2), azimuth, with uncertainties if enabled |
| `outliers.txt` | List of detected outlier stations with reasons |
| `loo_strain_residuals.txt` | Per-station LOO residuals (when using LOO outlier detection) |
| `wang2012_vertex_velocities.txt` | Estimated mesh vertex velocities (Wang 2012 method) |
| `figures/` | Visualization outputs: raw velocity, outliers, triangulation, strain scalars, principal crosses, etc. |

## Web Application

The Streamlit web app provides two interactive workflows:

1. **Strain Rate** — Upload velocity data, configure the method, run computation, and explore results with interactive Plotly maps.
2. **Strain Time Series** — Upload time-series data, select a method, and visualize per-epoch strain evolution.

Launch with:
```bash
pystrain web
# or directly:
streamlit run src/pystrain/web/app.py
```

## CLI Reference

```
pystrain {command} [options]

Commands:
  compute      Full strain-rate pipeline via YAML config file
  shen2015     Shen et al. (2015) grid-based strain rate
  delaunay     Delaunay triangulation strain rate
  wang2012     Wang (2012) mesh-based strain rate
  timeseries   Strain time-series estimation (via YAML config)
  plot         Grid existing strain output, generate GMT-style maps
  web          Launch the Streamlit web application
```

## License

MIT License

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
