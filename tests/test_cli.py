"""Tests for CLI entry point."""

import pytest

from pystrain2.cli.main import main


def test_cli_help():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_cli_grid_command(tmp_path, gmt8_text):
    output_dir = tmp_path / "out"
    code = main([
        "grid",
        "--vel-file", gmt8_text,
        "--region", "112.0", "113.5", "40.0", "41.0",
        "--spacing", "0.5", "0.5",
        "--output-dir", str(output_dir),
        "--Wt", "10.0",
    ])
    assert code == 0
    # Should produce output files
    assert (output_dir / "grid_strain.txt").exists()


def test_cli_compute_command(tmp_path, gmt8_text):
    config = tmp_path / "config.yaml"
    config.write_text(f"""
data:
  vel_file: {gmt8_text}
  poly_file: null
  output_dir: {tmp_path / "out2"}
  format: auto

outlier_detection:
  enable: false

algorithms:
  grid:
    activate: true
    region: [112.0, 113.5, 40.0, 41.0]
    spacing: [0.5, 0.5]
    weight_threshold_Wt: 10.0
  delaunay:
    activate: true
  velmap:
    activate: false
""")
    code = main(["compute", "--config", str(config)])
    assert code == 0
    assert (tmp_path / "out2" / "grid_strain.txt").exists()
    assert (tmp_path / "out2" / "delaunay_strain.txt").exists()
