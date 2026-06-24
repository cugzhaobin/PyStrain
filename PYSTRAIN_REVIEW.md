# PyStrain Code Review — Prioritized Refactoring Recommendations

**Date:** 2026-06-22  
**Scope:** `/simulation/geodesy/PyStrain` — PyStrain GNSS strain-rate package  
**Deliverable:** Prioritized issue register, cross-cutting themes, and refactoring roadmap

---

## 1. Executive Summary

PyStrain is a geodesy package with **two overlapping implementations**: a legacy monolith (`src/pystrain/PyStrain.py`) and a newer modular pipeline (`src/pystrain/gnss_strain/`). The modern pipeline is better organized, but the review found **several critical correctness and packaging issues** that can produce wrong scientific results or prevent the package from being installed/used.

**Highest-impact findings:**

1. **Broken iterative outlier detection in the modern pipeline.** `iterative_outlier_removal()` mixes global site arrays with a triangulation built on a site subset, so `tri.simplices` (local indices) are interpreted as global indices. The IQR stage therefore scans the wrong sites and can silently miss real outliers or misidentify valid ones.
2. **Legacy `Strain.strainrate()` distance weighting is inverted** (`exp(+D²/R²)`), so far-away sites receive higher weight than nearby sites when `gridweight` is used.
3. **Package cannot be installed** because `pyproject.toml` declares `logging` and `sklearn` as dependencies; `logging` is stdlib and the PyPI `logging` package is Python-2-only and incompatible with Python 3.11.
4. **`UserStrainRate.py` is syntactically invalid** (empty `with` block) and references undefined names, so the class cannot be imported.
5. **Legacy CLI entry point `do_pystrain.py` is broken** because it uses `Config` without importing it.
6. **No automated tests exist**, and the README is essentially empty.

---

## 2. Methodology

The review was performed in read-only mode and focused on correctness, maintainability, duplication, and packaging.

Tools used:

- `python -m py_compile` on all `.py` files under `src/`
- `python -m pyflakes src/pystrain` for undefined names / syntax errors
- `python -m flake8 src/pystrain --max-line-length=120 --statistics` for style and basic lint
- `pip install -e .` to verify installability
- Manual code reading of critical modules:
  - `src/pystrain/PyStrain.py`
  - `src/pystrain/UserStrainRate.py`
  - `src/pystrain/scripts/do_pystrain.py`
  - `src/pystrain/gnss_strain/{gnss_strain.py,config.py,io_utils.py,triangulation.py,strain_rate.py,outlier_detection.py,uncertainty.py,visualization.py,gnss_strain_app.py}`
  - `pyproject.toml`, `requirement.txt`, `README.md`

Cross-checks performed:

- Compared legacy vs. modern strain-rate solvers (design matrix, weighting, unit conversion, shear/azimuth conventions).
- Traced site-index mapping through the modern outlier-removal loop.
- Compared default parameter values across `config.py`, `config_default.yaml`, `gnss_strain.py`, and `gnss_strain_app.py`.

---

## 3. Issue Register

### P0 — Critical (wrong results or crashes)

| ID | Category | File / Line | Issue | Impact | Recommended Action | Effort |
|---|---|---|---|---|---|---|
| P0-1 | Crash / Syntax | `src/pystrain/UserStrainRate.py:117-119` | Empty `with open(...) as fid:` block causes `IndentationError`; file also uses undefined names (`logging`, `Grid_Site_DistAizm`, `llh2localxy`, `debug`) and a typo (`usecvols` instead of `usecols`). | `UserStrainRate` cannot be imported; legacy user-mesh workflow is completely broken. | Fix the `with` block, add missing imports, correct the typo, or remove the class if it is no longer supported. | Small |
| P0-2 | Crash / Import | `src/pystrain/scripts/do_pystrain.py:11` | `Config` is used but never imported. | Legacy CLI entry point crashes immediately. | Add `from pystrain.PyStrain import Config` or deprecate `do_pystrain.py` and point users to the modern CLI. | Small |
| P0-3 | Packaging | `pyproject.toml:18-26`, `requirement.txt` | Declares `logging` and `sklearn` as dependencies. `logging` is in the standard library; the PyPI `logging` package is Python-2-only and breaks install on Python 3. `sklearn` is the wrong PyPI name (should be `scikit-learn`). `pyyaml` and `streamlit` are used but not declared. | `pip install -e .` fails with a syntax error inside the downloaded `logging` package. | Remove `logging`; replace `sklearn` with `scikit-learn`; add `pyyaml>=5.1` and `streamlit` (optional) to dependencies. | Small |
| P0-4 | Numerical correctness | `src/pystrain/gnss_strain/outlier_detection.py:184-291` and `gnss_strain.py:34-45` | `iterative_outlier_removal` passes full `ve/vn/se/sn` arrays to `compute_residuals_triangulation`, but the triangulation `tri` is built on a site subset, so `tri.simplices` contains **local indices** (0..m-1). The residual function treats these as global indices and iterates `site_idx in range(n_sites)`. | IQR outlier detection scans the wrong sites. Real outliers can be missed; valid sites can be flagged. Results are silently incorrect. | Pass a `site_indices` array through the pipeline and map local triangulation indices back to global site indices before computing residuals. Add a regression test with known outliers. | Medium |
| P0-5 | Numerical correctness | `src/pystrain/PyStrain.py:398` | Legacy `Strain.strainrate()` computes distance weight as `w = exp(D[i]**2 / gridweight**2)`. The exponent is **positive**, so weight grows with distance. | When `gridweight` is used, distant GPS sites dominate the least-squares fit and strain rates are wrong. | Change to `exp(-D[i]**2 / gridweight**2)` and verify against a simple case where the nearest site should dominate. | Small |
| P0-6 | Unit convention | `src/pystrain/PyStrain.py:403-414` | Design matrix divides `x[i]` and `y[i]` by `1e3` even though the docstring says the inputs are already in km. The subsequent `*1e-9` scaling is then inconsistent with the documented output units. | Legacy strain-rate magnitudes may be off by orders of magnitude depending on the actual input units callers pass. | Clarify the expected input units in the docstring, fix the scaling, and add a unit test comparing a constant velocity gradient to the analytic strain value. | Medium |

### P1 — Important (maintainability, robustness, hidden errors)

| ID | Category | File / Line | Issue | Impact | Recommended Action | Effort |
|---|---|---|---|---|---|---|
| P1-1 | Consistency | `src/pystrain/gnss_strain/config.py:40`, `gnss_strain.py:56`, `gnss_strain_app.py:107`, `config_default.yaml:55` | Smoothing iteration defaults are inconsistent: `config.py` and `config_default.yaml` use `3`; `gnss_strain.py` and the Streamlit app use `2`. | Same package produces different results depending on which interface is used. | Pick one default, centralize it in `config.py`, and make all callers read from config. | Small |
| P1-2 | Consistency | `src/pystrain/gnss_strain/uncertainty.py:15`, `config.py:43` | Monte Carlo default is `500` in `uncertainty.py` but `200` in `config.py` and the Streamlit app. | Default uncertainty estimates differ between direct API use and CLI/GUI use. | Use `config.py` as the single source of truth; remove hard-coded defaults from function signatures or keep them aligned. | Small |
| P1-3 | Numerical correctness | `src/pystrain/gnss_strain/outlier_detection.py:51-52` | KNN pre-screening builds the KD-tree in `(lon, lat)` degree space using Euclidean distance. Longitude degrees are not length-invariant with latitude. | At high latitudes the neighborhood search is anisotropic and can falsely flag valid sites or miss outliers. | Project to a local metric (e.g., UTM km) before building the KD-tree, or use geodesic distance. | Small |
| P1-4 | Error handling | `src/pystrain/gnss_strain/io_utils.py:54-108` | `read_velocity_file()` silently skips malformed rows with `except ValueError: continue` and rows with too few columns. | Data-quality problems are hidden; the user does not know how many rows were dropped or why. | Log/print warnings with line numbers and reasons; optionally add a strict mode that raises on parse errors. | Small |
| P1-5 | Testing | Repository root / `test/` | No unit, integration, or regression tests exist. | Bugs like P0-4 and P0-5 are not caught; every change is risky. | Add a minimal test suite (pytest) covering projection, strain tensor, outlier detection, and an end-to-end regression on `test/camp_eura.vel`. | Medium |
| P1-6 | Code organization / duplication | `src/pystrain/PyStrain.py` vs `src/pystrain/gnss_strain/` | Two independent implementations of projection, strain-rate solve, and triangulation. The legacy file is ~1420 lines and mixes I/O, algorithms, and plotting. | Maintenance burden is doubled; fixes in one implementation may not propagate. | Decide which implementation is canonical. Gradually deprecate legacy classes or refactor them to wrap the modern modules. | Large |
| P1-7 | Dead code | `src/pystrain/gnss_strain/gnss_strain.py:23` | `compute_site_residuals` is imported but never used. | Confusing API; the unused residual implementation may also have index bugs. | Remove the import or replace `compute_residuals_triangulation` with the barycentric-based `compute_site_residuals` after fixing index mapping. | Small |
| P1-8 | Incomplete feature | `src/pystrain/gnss_strain/strain_rate.py:306-307` | `interpolate_to_site()` returns placeholder `ve_pred=0.0, vn_pred=0.0` and is not used. | Misleading docstring; unusable for velocity residual computation. | Implement velocity interpolation with shape functions or remove the function. | Small |
| P1-9 | Missing output | `src/pystrain/gnss_strain/uncertainty.py:68-75` | Monte Carlo uncertainty does not propagate `omega` (rotation rate), although `strain_rate.py` computes it as a core output. | Users cannot assess uncertainty on the rotation/vorticity field. | Add `omega` to the MC loop output (`omega_std`, `omega_mean`). | Small |
| P1-10 | API design | `src/pystrain/__init__.py` | Package `__init__.py` is empty, but `PyStrain.py` defines `__all__`. Nothing is exported at package top level. | `import pystrain` gives no usable symbols; users must know internal module paths. | Populate `__init__.py` with canonical public imports (modern pipeline first) and a clear public API. | Small |
| P1-11 | Side effects | `src/pystrain/PyStrain.py:17-20` | `logging.basicConfig(level=logging.DEBUG)` runs at module import time. | Importing pystrain reconfigures the root logger, potentially overriding application logging settings. | Remove the global config; let applications configure logging, or use a package-specific logger with `NullHandler`. | Small |
| P1-12 | Packaging | `pyproject.toml:28-30` | Console script names end in `.py` (`do_pystrain.py`, `plot_strain_ts.py`), which is unconventional and confusing. | Generated executables are literally named `do_pystrain.py`. | Rename to `pystrain` and `pystrain-plot-ts` (or similar) and update docs. | Small |
| P1-13 | Documentation | `README.md` | README contains only the project title. | New users have no setup, usage, or citation information. | Write a README with install instructions, quick-start CLI example, input-file format, and output description. | Small |
| P1-14 | Error handling | `src/pystrain/gnss_strain/io_utils.py:180-181` | Polygon closure check uses exact float equality (`poly[0,0] != poly[-1,0]`). | Slightly different start/end coordinates (e.g., from formatting) cause an extra closing vertex to be appended unnecessarily, or fail to close if they differ significantly. | Use a tolerance-based comparison or always force closure. | Small |

### P2 — Nice-to-have (style, performance, minor ergonomics)

| ID | Category | File / Line | Issue | Impact | Recommended Action | Effort |
|---|---|---|---|---|---|---|
| P2-1 | Style | All Python files | ~1000 flake8 warnings (trailing whitespace, multiple spaces around operators, long lines, missing whitespace). | Code is hard to read and review. | Apply a formatter (black/ruff) and add a lint check to CI. | Small |
| P2-2 | Performance | `src/pystrain/gnss_strain/triangulation.py:179-188`, `198-220`, `236-240`, etc. | Quality filters and area computations loop over triangles in pure Python. | Slow for large networks; vectorized NumPy/SciPy operations would be much faster. | Vectorize edge-length, angle, and area filters; use `np.cross` on whole simplex arrays. | Medium |
| P2-3 | Performance | `src/pystrain/gnss_strain/uncertainty.py:76-104` | Monte Carlo loop recomputes strain triangle-by-triangle in pure Python. | With 500 iterations and many triangles this dominates runtime. | Vectorize the inner triangle loop with NumPy broadcasting or use a small Cython/Numba kernel. | Medium |
| P2-4 | API ergonomics | `src/pystrain/gnss_strain/gnss_strain.py:285` | `all_outlier_mask` is computed but never used. | Minor dead code. | Remove or use it in the plotting call. | Trivial |
| P2-5 | CLI ergonomics | `src/pystrain/scripts/plot_strain_ts.py:15` | `--pltdiff` uses `type=bool`, which makes any non-empty string truthy; argparse best practice is `action='store_true'`. | Users cannot reliably disable the flag. | Change to `action='store_true'`. | Trivial |
| P2-6 | Type safety | All modules | No type hints; many functions accept/return opaque dicts. | Hard to reason about data shapes and catch bugs. | Add type hints to public functions; use `typing.NamedTuple` or dataclasses for `SiteData` and result dicts. | Medium |

---

## 4. Cross-Cutting Themes

### 4.1 Global vs. local site-index confusion
The modern pipeline repeatedly subsets sites (`site_indices_initial`, `site_indices_final`, `sub_global_idx` inside `iterative_outlier_removal`) but then passes global-length arrays to functions that expect matching local arrays. `tri.simplices` is always local to the points fed to `Delaunay`, so every downstream consumer must know which index space it lives in. This is the root cause of P0-4 and a likely source of future bugs.

**Recommendation:** Introduce an explicit `site_indices` array and helper functions `local_to_global(tri, site_indices)` and `global_to_local(tri, global_idx)` so that index translation is explicit and auditable.

### 4.2 Legacy / modern duplication
Almost every capability exists twice: polyconic vs. UTM projection, least-squares grid strain vs. triangular finite-element strain, matplotlib triangulation vs. SciPy Delaunay, etc. The legacy code has known bugs (P0-5, P0-6); the modern code is cleaner but still immature. Maintaining both is unsustainable.

**Recommendation:** Declare the `gnss_strain/` pipeline the canonical implementation. Refactor legacy classes to delegate to it, or remove legacy entry points and clearly document the migration path.

### 4.3 Configuration drift
Defaults are scattered across `config.py`, `gnss_strain.py`, `gnss_strain_app.py`, and `config_default.yaml`. Smoothing iterations and MC iterations already disagree.

**Recommendation:** Make `config.py` the single source of truth. Function signatures should default to `None` and then fall back to config defaults, or read directly from a validated config object.

### 4.4 Silent failures
Malformed input rows are skipped silently, IQR outlier detection fails silently because of index confusion, and `sys.exit()` calls inside `Config.__init__` can terminate an importing application. The code is forgiving to a fault.

**Recommendation:** Adopt a "fail loudly at boundaries" policy: validate inputs, log warnings with context, and raise exceptions for unrecoverable errors instead of swallowing them.

---

## 5. Suggested Refactoring Roadmap

### PR 1 — Fix P0 correctness and packaging (urgent)
- Fix `pyproject.toml` dependencies and console-script names.
- Fix or remove `UserStrainRate.py` and `do_pystrain.py` legacy entry points.
- Fix the index-mapping bug in `iterative_outlier_removal` / `compute_residuals_triangulation`.
- Fix legacy `Strain.strainrate()` weighting and unit scaling.
- Add a minimal regression test that reproduces P0-4 and P0-5.

### PR 2 — Unify configuration and input validation
- Centralize all defaults in `config.py`.
- Align CLI, Streamlit, and YAML defaults.
- Add input validation to `io_utils.read_velocity_file` (warnings/strict mode).
- Improve polygon closure logic with tolerance.

### PR 3 — Reconcile or remove legacy code
- Decide the fate of `PyStrain.py` classes.
- If kept, refactor them to call `gnss_strain/` functions.
- If removed, update `pyproject.toml` entry points and document migration.

### PR 4 — Add tests and continuous integration
- Set up `pytest` with tests for:
  - Projection round-trip / known values
  - Strain tensor from a synthetic linear velocity field
  - Delaunay quality filters
  - Outlier detection with planted outliers
  - End-to-end regression on `test/camp_eura.vel`
- Add GitHub Actions or equivalent to run tests and flake8 on every PR.

### PR 5 — Documentation, packaging polish, and performance
- Write a real `README.md`.
- Remove module-level `logging.basicConfig`.
- Add `omega` to Monte Carlo uncertainty output.
- Vectorize triangulation filters and MC loop where profiling shows benefit.
- Apply code formatting and add type hints to public APIs.

---

## 6. Appendix

### A. Linter / static-check summary

| Tool | Result |
|---|---|
| `python -m py_compile src/pystrain/**/*.py` | Fails on `UserStrainRate.py` with `IndentationError` at line 117. |
| `python -m pyflakes src/pystrain` | 1 syntax error, 1 undefined name (`Config` in `do_pystrain.py`), several unused imports/variables. |
| `python -m flake8 --statistics` | ~1000 warnings; dominant categories: `E221` (multiple spaces before operator), `E231` (missing whitespace after comma), `W293` (blank line contains whitespace). |
| `pip install -e .` | Fails while trying to build PyPI `logging==0.4.9.6` (Python 2 syntax error). |

### B. Dependency issues

| Declared | Problem | Fix |
|---|---|---|
| `logging` in `pyproject.toml` and `requirement.txt` | Standard-library module; PyPI package is incompatible. | Remove. |
| `sklearn` in `pyproject.toml` | Wrong package name; use `scikit-learn`. | Replace. |
| `pyyaml` | Used by `config.py` but not declared. | Add. |
| `streamlit` | Used by `gnss_strain_app.py` but not declared. | Add as optional dependency (`[app]` extra). |

### C. Default-value matrix

| Parameter | `config.py` | `config_default.yaml` | `gnss_strain.py` | `gnss_strain_app.py` |
|---|---|---|---|---|
| `smooth_iter` / `smoothing.iterations` | 3 | 3 | 2 | 2 |
| `mc_iterations` / `uncertainty.mc_iterations` | 200 | 200 | 200 | 200 |

Note: `uncertainty.py` internally defaults to `500`, so direct API calls differ from CLI/GUI.

### D. Files with the highest issue density

1. `src/pystrain/UserStrainRate.py` — syntax error, undefined names.
2. `src/pystrain/scripts/do_pystrain.py` — undefined `Config`.
3. `src/pystrain/gnss_strain/outlier_detection.py` — index-mapping bug.
4. `src/pystrain/PyStrain.py` — inverted weighting, unit confusion, dead code.
5. `pyproject.toml` / `requirement.txt` — invalid dependencies.
6. `src/pystrain/gnss_strain/io_utils.py` — silent parse failures.
7. `src/pystrain/gnss_strain/uncertainty.py` — missing `omega` uncertainty, default mismatch.
