"""Plotting and gridding tools for PyStrain2 strain-rate results.

Provides:
- ``scattered_to_grid()`` — interpolate scattered strain data to a regular grid
  with distance-based NaN masking for data-void preservation.
- ``save_netcdf()`` / ``save_geotiff()`` — write gridded fields to disk.
- ``strain_file_to_grid()`` — end-to-end: read strain output → grid → save.
- ``plot_strain_map()`` — GMT-style continuous-colour map of a single field.
- ``plot_strain_overview()`` — 4-panel overview (dilatation, shear, 2nd invariant,
  rotation).
"""

from pystrain2.plot.gridding import (
    scattered_to_grid,
    save_netcdf,
    save_geotiff,
    strain_file_to_grid,
)

from pystrain2.plot.strain_map import (
    plot_strain_map,
    plot_strain_overview,
    plot_velocity_map,
)

__all__ = [
    "scattered_to_grid",
    "save_netcdf",
    "save_geotiff",
    "strain_file_to_grid",
    "plot_strain_map",
    "plot_strain_overview",
    "plot_velocity_map",
]
