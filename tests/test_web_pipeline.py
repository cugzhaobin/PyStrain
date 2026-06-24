"""Tests for the PyStrain2 web pipeline runner."""

import numpy as np
import pytest

from pystrain2.web.pipeline import run_pystrain2_pipeline


def test_pipeline_delaunay(gmt8_text):
    result = run_pystrain2_pipeline(
        vel_path=gmt8_text,
        poly_path=None,
        fmt="auto",
        algorithm="delaunay",
        outlier_enable=True,
        outlier_kwargs={
            "k_neighbors": 3,
            "mad_factor": 3.5,
            "iqr_factor": 1.5,
            "max_iterations": 2,
        },
        tri_kwargs={
            "min_angle_deg": 10.0,
            "max_edge_pctl": 95.0,
            "max_edge_factor": 1.5,
            "min_area_ratio": 0.1,
            "max_edge_km": None,
        },
        mc_iterations=10,
        mc_seed=42,
    )

    assert result["algorithm"] == "delaunay"
    assert result["n_sites_total"] >= result["n_sites_used"]
    assert len(result["centroids_lon"]) == result["n_good_triangles"]
    assert np.all(np.isfinite(result["dilatation"]))
    assert np.all(np.isfinite(result["max_shear"]))
    assert "dilatation_std" in result
    assert result["dilatation_std"] is not None


def test_pipeline_velmap(gmt8_text):
    result = run_pystrain2_pipeline(
        vel_path=gmt8_text,
        poly_path=None,
        fmt="auto",
        algorithm="velmap",
        outlier_enable=False,
        outlier_kwargs={},
        tri_kwargs={
            "min_angle_deg": 10.0,
            "max_edge_pctl": 95.0,
            "max_edge_factor": 1.5,
            "min_area_ratio": 0.1,
            "max_edge_km": None,
        },
        velmap_kwargs={"smooth_weight": 0.3, "smooth_iter": 1},
        mc_iterations=0,
    )

    assert result["algorithm"] == "velmap"
    assert result["n_outliers"] == 0
    assert len(result["centroids_lon"]) > 0


def test_pipeline_shenwang(gmt8_text):
    result = run_pystrain2_pipeline(
        vel_path=gmt8_text,
        poly_path=None,
        fmt="auto",
        algorithm="shenwang",
        outlier_enable=False,
        outlier_kwargs={},
        tri_kwargs={
            "min_angle_deg": 10.0,
            "max_edge_pctl": 95.0,
            "max_edge_factor": 1.5,
            "min_area_ratio": 0.1,
            "max_edge_km": None,
        },
        grid_kwargs={
            "region": [112.0, 113.5, 40.0, 41.5],
            "spacing": [0.3, 0.3],
            "Wt": 12.0,
            "L0": 0.01,
            "min_sites": 4,
            "maxdist_km": 200.0,
        },
        mc_iterations=0,
    )

    assert result["algorithm"] == "shenwang"
    assert len(result["centroids_lon"]) > 0


def test_unknown_algorithm_raises(gmt8_text):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_pystrain2_pipeline(
            vel_path=gmt8_text,
            poly_path=None,
            fmt="auto",
            algorithm="notreal",
            outlier_enable=False,
            outlier_kwargs={},
            tri_kwargs={
                "min_angle_deg": 10.0,
                "max_edge_pctl": 95.0,
                "max_edge_factor": 1.5,
            },
            mc_iterations=0,
        )
