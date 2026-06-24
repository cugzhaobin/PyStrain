"""Grid generation utilities."""

from typing import Tuple
import numpy as np


class Grid:
    """Regular longitude/latitude grid."""

    def __init__(
        self,
        slon: float,
        elon: float,
        slat: float,
        elat: float,
        dlon: float,
        dlat: float,
        stagger: bool = True,
    ):
        self.slon = slon
        self.elon = elon
        self.slat = slat
        self.elat = elat
        self.dlon = dlon
        self.dlat = dlat
        self.stagger = stagger
        self.lon, self.lat = self._generate()

    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        lons = np.arange(self.slon, self.elon + self.dlon / 2, self.dlon)
        lats = np.arange(self.slat, self.elat + self.dlat / 2, self.dlat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        if self.stagger:
            # Offset every other row by half grid spacing
            offset = self.dlon / 2.0
            lon_grid[1::2, :] += offset

        return lon_grid.ravel(), lat_grid.ravel()

    def __len__(self):
        return len(self.lon)
