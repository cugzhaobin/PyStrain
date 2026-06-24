"""Configuration management for PyStrain2."""

from typing import Any, Dict, Optional
import yaml


DEFAULTS = {
    "data": {
        "vel_file": None,
        "poly_file": None,
        "output_dir": "./pystrain_output",
        "format": "auto",
    },
    "algorithms": {
        "method": "shen2015",
        "shen2015": {
            "region": None,
            "spacing": None,
            "stagger": True,
            "distance_weight": "gaussian",
            "spatial_weight": "voronoi",
            "weight_threshold_Wt": 24.0,
            "distance_cutoff_L0": 0.01,
            "min_sites": 6,
            "maxdist_km": None,
            "auto_search_radius": True,
            "D_min_km": 1.0,
            "D_max_km": 1000.0,
            "fault_barriers": [],
        },
        "delaunay": {
            "min_angle_deg": 10.0,
            "max_edge_pctl": 95.0,
            "max_edge_factor": 1.5,
            "min_area_ratio": 0.1,
            "max_edge_km": None,
            "projection": "utm",
            "remove_hanging_sites": True,
        },
        "wang2012": {
            "poly_file": None,
            "mesh_region": None,
            "mesh_method": "adaptive",
            "mesh_spacing": 0.25,
            "mesh_randomize": True,
            "mesh_randomize_fraction": 0.2,
            "max_stations_per_cell": 6,
            "max_spacing": None,
            "mesh_outline_file": None,
            "smooth_factor": 0.01,
            "smooth_search": True,
            "smooth_range": [-2.2, -0.8],
            "smooth_step": 0.2,
            "smooth_method": "lcurve",
            "smooth_boundary": True,
            "min_area_ratio": 0.1,
        },
    },
    "outlier_detection": {
        "enable": True,
        # Method: "knn_iqr" | "loo_strain"
        "method": "knn_iqr",
        # ---- KNN + triangulation IQR ----
        "knn_iqr": {
            "k_neighbors": 8,
            "mad_factor": 3.5,
            "iqr_factor": 1.5,
            "max_iterations": 5,
            "min_residual_mm": 0.5,
            "metric": "utm",
            "velocity_threshold_mm": None,
            "report_details": True,
        },
        # ---- LOO strain cross-validation ----
        "loo_strain": {
            "maxdist_km": 200.0,
            "min_sites": 8,
            "min_residual_mm": 3.0,   # direct threshold (mm/yr)
        },
    },
    "uncertainty": {
        "enable": False,
        "mc_iterations": 200,
        "seed": 42,
    },
    "visualization": {
        "dpi": 150,
        "save_figures": True,
        "figure_dir": "figures",
        "show_raw_velocity": True,
        "show_outliers": True,
        "show_clean_velocity": True,
        "show_triangulation": True,
        "show_grid_points": True,
        "show_search_radius": True,
        "show_strain_scalars": True,
        "show_wang2012_mesh": True,
        "show_principal_crosses": True,
    },
    "timeseries": {
        "method": "grid",
        "gps_info_file": None,
        "ts_type": "pos",
        "ts_path": None,
        "sepoch": None,
        "eepoch": None,
        "output_dir": "./pystrain_output",
        "grid": {
            "slon": None,
            "elon": None,
            "slat": None,
            "elat": None,
            "dn": None,
            "de": None,
            "stagger": True,
            "maxdist_km": 300.0,
            "min_sites": 6,
            "check_azimuth": True,
        },
        "tri": {
            "min_angle_deg": 10.0,
            "max_edge_pctl": 95.0,
            "max_edge_factor": 1.5,
            "min_area_ratio": 0.1,
            "max_edge_km": None,
            "projection": "utm",
        },
        "user": {
            "site_groups_file": None,
            "max_sigma_mm": 5.0,
        },
    },
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Configuration container with validated defaults."""

    def __init__(self, path: Optional[str] = None, overrides: Optional[Dict] = None):
        self.cfg = deep_merge({}, DEFAULTS)
        if path is not None:
            with open(path, "r", encoding="utf-8") as fid:
                user_cfg = yaml.safe_load(fid)
            if user_cfg:
                self.cfg = deep_merge(self.cfg, user_cfg)
        if overrides:
            self.cfg = deep_merge(self.cfg, overrides)

    def __getitem__(self, key: str) -> Any:
        return self.cfg[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.cfg.get(key, default)

    def to_dict(self) -> Dict:
        return dict(self.cfg)
