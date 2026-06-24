# PyStrain2

A refactored Python package for estimating GNSS crustal strain rates and strain time series.

## Features

- **Grid-based strain rate**: Shen et al. (2015) BSSA optimal interpolation with distance and spatial-coverage weighting.
- **Triangulation strain rate**: Delaunay-based least-squares strain estimation.
- **Velmap-style strain rate**: Linear triangular finite-element shape functions.
- **User-defined points**: Strain rate at arbitrary lon/lat points.
- **Outlier detection**: KNN pre-screening + iterative residual IQR removal.
- **Uncertainty**: Analytic covariance and Monte Carlo propagation.

## Installation

```bash
cd /simulation/geodesy/PyStrain2
pip install -e .
```

## Quick start

```bash
pystrain2 grid --vel-file data.gmtvec --region 70 140 20 50 --spacing 2 2 --output-dir out
```

## Input format

GMT 8-column velocity file:
```
lon lat Ve Vn Se Sn Rho SiteName
```

## License

MIT
