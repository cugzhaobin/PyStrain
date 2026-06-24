"""Tests for GPS time utilities, time-series I/O, loader, and strain TS."""

import os
import numpy as np
import pytest

from pystrain2.data import StrainTimeSeriesResult, TimeSeries, TimeSeriesCollection
from pystrain2.gps_time import (
    daily_epoch_grid,
    decyrs_to_jd,
    decyrs_to_mjd,
    jd_to_decyrs,
    jd_to_ymdhms,
    ymd_to_decyrs,
    ymdhms_to_jd,
)
from pystrain2.io.timeseries import DatData, PosData


# ======================================================================
# gps_time tests
# ======================================================================

class TestGPSTime:
    """Test GPS time conversion utilities against known values."""

    def test_ymdhms_to_jd_known(self):
        """2020-01-01 00:00:00 → JD 2458849.5"""
        jd = ymdhms_to_jd([2020, 1, 1, 0, 0], 0.0)
        assert jd == pytest.approx(2458849.5, abs=0.1)

    def test_ymdhms_to_jd_roundtrip(self):
        """JD → calendar → JD should be identity."""
        jd_in = 2458849.5
        date, sec, _ = jd_to_ymdhms(jd_in)
        jd_out = ymdhms_to_jd(date, sec)
        assert jd_out == pytest.approx(jd_in, abs=1e-4)

    def test_jd_to_decyrs_scalar(self):
        """2020-01-01 ≈ 2020.0"""
        jd = ymdhms_to_jd([2020, 1, 1, 0, 0], 0.0)
        decyr = jd_to_decyrs(jd)
        assert decyr == pytest.approx(2020.0, abs=0.01)

    def test_jd_to_decyrs_array(self):
        """Array of JDs → array of decimal years."""
        jd1 = ymdhms_to_jd([2020, 1, 1, 0, 0], 0.0)
        jd2 = ymdhms_to_jd([2020, 7, 2, 0, 0], 0.0)  # mid-year
        result = jd_to_decyrs([jd1, jd2])
        assert len(result) == 2
        assert result[0] == pytest.approx(2020.0, abs=0.01)
        assert result[1] == pytest.approx(2020.5, abs=0.01)

    def test_decyrs_to_jd_roundtrip(self):
        """decyr → JD → decyr should be identity."""
        decyr_in = 2020.5
        jd = decyrs_to_jd(decyr_in)
        decyr_out = jd_to_decyrs(jd)
        assert decyr_out == pytest.approx(decyr_in, abs=0.01)

    def test_decyrs_to_mjd(self):
        """MJD = JD - 2400000.0"""
        decyr = 2020.0
        mjd = decyrs_to_mjd(decyr)
        jd = decyrs_to_jd(decyr)
        assert mjd == pytest.approx(jd - 2400000.0, abs=0.01)

    def test_ymd_to_decyrs(self):
        """2020-07-02 ≈ 2020.5"""
        decyr = ymd_to_decyrs(2020, 7, 2)
        assert decyr == pytest.approx(2020.5, abs=0.01)

    def test_daily_epoch_grid(self):
        """One-year grid should have ~365 days."""
        grid = daily_epoch_grid(2020.0, 2021.0)
        assert 360 < len(grid) < 370
        assert grid[0] >= 2020.0
        assert grid[-1] < 2021.0


# ======================================================================
# io/timeseries tests
# ======================================================================

class TestPosData:
    """Test PBO .pos file parsing."""

    def _make_pos_data(self, mjd, dn_val, de_val, du_val=0.0):
        """Build one PBO data line (25 space-delimited columns)."""
        # Cols: YYYYMMDD HHMMSS MJD X Y Z Sx Sy Sz Rxy Rxz Ryz
        #       Nlat Elong Height dN dE dU Sn Se Su Rne Rnu Reu Soln
        return (
            f" 20200101 000000 {mjd:.4f} "
            f"0.0 0.0 0.0 "
            f"0.001 0.001 0.001 0.0 0.0 0.0 "
            f"45.678 -123.456 0.0 "
            f"{dn_val:.6f} {de_val:.6f} {du_val:.6f} "
            f"0.0001 0.0001 0.0001 "
            f"0.0 0.0 0.0 final"
        )

    def test_pos_parsing(self, tmp_path):
        """Create a minimal PBO v1.1 .pos file and parse it."""
        posfile = tmp_path / "TEST.NA12.pos"

        header_lines = [
            "PBO Station Position Time Series. Reference Frame : ITRF2014",
            "Format Version: 1.1.1",
            "4-character ID: TEST",
            "Station name  : TESTSTA",
            "First Epoch   : 20200101 000000",
            "Last Epoch    : 20200103 000000",
            "Release Date  : 20210101 000000",
            "XYZ Reference position :  0.0  0.0  0.0 (ITRF2014)",
            "NEU Reference position :  45.678  -123.456  1234.567 (ITRF2014/WGS84)",
            "Start Field Description",
            "...",
            "End Field Description",
            "*YYYYMMDD HHMMSS JJJJJ.JJJJ         X             Y             Z            Sx        Sy       Sz     Rxy   Rxz    Ryz            NLat         Elong         Height         dN        dE        dU         Sn       Se       Su      Rne    Rnu    Reu  Soln",
        ]

        lines = header_lines + [
            self._make_pos_data(58849.0, 0.0010, -0.0020),
            self._make_pos_data(58850.0, 0.0015, -0.0015),
            self._make_pos_data(58851.0, 0.0012, -0.0018),
        ]
        posfile.write_text("\n".join(lines), encoding="utf-8")

        pd = PosData(str(posfile))
        assert pd.site == "TEST"
        assert pd.lat == pytest.approx(45.678)
        assert pd.lon == pytest.approx(-123.456)

        ts = pd.to_timeseries()
        assert isinstance(ts, TimeSeries)
        assert ts.site == "TEST"

    def test_missing_file(self):
        """Non-existent file returns empty PosData."""
        pd = PosData("/nonexistent/file.pos")
        assert len(pd.decyr) == 0


class TestDatData:
    """Test PyTsfit .dat file parsing."""

    def test_dat_parsing(self, tmp_path):
        """Create a .dat file and parse it."""
        datfile = tmp_path / "TEST_obs.dat"
        # decyr N E U SN SE SU (7 columns, mm)
        content = (
            "2020.0000  1.0  2.0  3.0  0.1  0.1  0.1\n"
            "2020.0027  1.1  2.1  3.1  0.1  0.1  0.1\n"
            "2020.0055  1.2  2.2  3.2  0.1  0.1  0.1\n"
        )
        datfile.write_text(content, encoding="utf-8")

        dd = DatData(str(datfile))
        assert len(dd.decyr) == 3
        assert dd.N[0] == pytest.approx(1.0)
        assert dd.E[0] == pytest.approx(2.0)
        assert dd.U[0] == pytest.approx(3.0)

        ts = dd.to_timeseries()
        assert isinstance(ts, TimeSeries)

    def test_dat_with_nans(self, tmp_path):
        """Rows with NaN in N/E/U should be filtered out."""
        datfile = tmp_path / "NAN_obs.dat"
        content = (
            "2020.0000  1.0  2.0  3.0  0.1  0.1  0.1\n"
            "2020.0027  NaN  2.1  3.1  0.1  0.1  0.1\n"
            "2020.0055  1.2  NaN  3.2  0.1  0.1  0.1\n"
        )
        datfile.write_text(content, encoding="utf-8")
        dd = DatData(str(datfile))
        # Only first row should survive
        assert len(dd.decyr) == 1

    def test_missing_file(self):
        """Non-existent file returns empty DatData."""
        dd = DatData("/nonexistent/file.dat")
        assert len(dd.decyr) == 0


# ======================================================================
# TimeSeriesLoader tests (synthetic data)
# ======================================================================

class TestTimeSeriesLoader:
    """Test multi-site loading and alignment."""

    def test_loader_pos(self, tmp_path):
        """Full loader pipeline with synthetic .pos data."""
        from pystrain2.timeseries.loader import TimeSeriesLoader

        # Create GPS info file
        info_file = tmp_path / "gps_info.txt"
        info_file.write_text(
            "112.5 40.5 100.0 SITE\n"
            "112.6 40.6 100.0 SIT2\n",
            encoding="utf-8",
        )

        # Create .pos files for each site
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()

        for site in ["SITE", "SIT2"]:
            posfile = ts_dir / f"{site}.NA12.pos"
            header_lines = [
                "PBO Station Position Time Series. Reference Frame : ITRF2014",
                "Format Version: 1.1.1",
                f"4-character ID: {site}",
                "Station name  : TESTSTA",
                "First Epoch   : 20200101 000000",
                "Last Epoch    : 20200110 000000",
                "Release Date  : 20210101 000000",
                "XYZ Reference position :  0.0  0.0  0.0 (ITRF2014)",
                f"NEU Reference position :  40.5  112.5  100.0 (ITRF2014/WGS84)",
                "Start Field Description",
                "...",
                "End Field Description",
                "*YYYYMMDD HHMMSS JJJJJ.JJJJ         X             Y             Z            Sx        Sy       Sz     Rxy   Rxz    Ryz            NLat         Elong         Height         dN        dE        dU         Sn       Se       Su      Rne    Rnu    Reu  Soln",
            ]
            data_rows = []
            for day_offset in range(10):
                mjd = 58849.0 + day_offset
                data_rows.append(
                    f" 20200101 000000 {mjd:.4f} "
                    f"0.0 0.0 0.0 0.001 0.001 0.001 0.0 0.0 0.0 "
                    f"40.5 112.5 0.0 "
                    f"{0.001*day_offset:.6f} {0.002*day_offset:.6f} 0.0 "
                    f"0.0001 0.0001 0.0001 0.0 0.0 0.0 final"
                )
            posfile.write_text(
                "\n".join(header_lines) + "\n" + "\n".join(data_rows),
                encoding="utf-8",
            )

        loader = TimeSeriesLoader(
            gps_info_file=str(info_file),
            ts_type="pos",
            ts_path=str(ts_dir),
            sepoch=2020.0,
            eepoch=2020.1,
        )
        tsc = loader.load()
        assert isinstance(tsc, TimeSeriesCollection)
        assert len(tsc.sites) >= 1
        assert tsc.E.shape[1] == len(tsc.sites)
        assert tsc.N.shape[1] == len(tsc.sites)

    def test_loader_dat(self, tmp_path):
        """Full loader pipeline with synthetic .dat data."""
        from pystrain2.timeseries.loader import TimeSeriesLoader

        info_file = tmp_path / "gps_info.txt"
        info_file.write_text(
            "112.5 40.5 100.0 S001\n"
            "112.6 40.6 100.0 S002\n"
            "112.7 40.7 100.0 S003\n",
            encoding="utf-8",
        )

        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()

        for i, site in enumerate(["S001", "S002", "S003"]):
            datfile = ts_dir / f"{site}_obs.dat"
            lines = []
            for t in np.arange(2020.0, 2021.0, 1 / 365.25):
                n = 0.1 * np.sin(2 * np.pi * t) + 0.01 * i
                e = 0.1 * np.cos(2 * np.pi * t) + 0.01 * i
                lines.append(f"{t:.6f}  {n:.6f}  {e:.6f}  0.0  0.1  0.1  0.1")
            datfile.write_text("\n".join(lines), encoding="utf-8")

        loader = TimeSeriesLoader(
            gps_info_file=str(info_file),
            ts_type="dat",
            ts_path=str(ts_dir),
            sepoch=2020.0,
            eepoch=2020.1,  # short window
        )
        tsc = loader.load()
        assert isinstance(tsc, TimeSeriesCollection)
        assert len(tsc.sites) == 3
        assert tsc.E.shape == (len(tsc.decyr), 3)

    def test_loader_site_list(self, tmp_path):
        """Only requested sites are loaded."""
        from pystrain2.timeseries.loader import TimeSeriesLoader

        info_file = tmp_path / "gps_info.txt"
        info_file.write_text(
            "112.5 40.5 100.0 S001\n"
            "112.6 40.6 100.0 S002\n"
            "112.7 40.7 100.0 S003\n",
            encoding="utf-8",
        )

        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()

        for site in ["S001", "S002", "S003"]:
            datfile = ts_dir / f"{site}_obs.dat"
            lines = []
            for t in np.arange(2020.0, 2021.0, 1 / 365.25):
                lines.append(f"{t:.6f}  1.0  2.0  3.0  0.1  0.1  0.1")
            datfile.write_text("\n".join(lines), encoding="utf-8")

        loader = TimeSeriesLoader(
            gps_info_file=str(info_file),
            ts_type="dat",
            ts_path=str(ts_dir),
            sepoch=2020.0,
            eepoch=2020.1,
            site_list=["S001"],
        )
        tsc = loader.load()
        assert len(tsc.sites) == 1
        assert tsc.sites[0] == "S001"


# ======================================================================
# StrainTimeSeriesResult tests
# ======================================================================

class TestStrainTimeSeriesResult:
    """Test the StrainTimeSeriesResult dataclass."""

    def test_valid_creation(self):
        n = 10
        decyr = np.linspace(2020, 2021, n)
        arr = np.random.randn(n)
        result = StrainTimeSeriesResult(
            lon=112.5, lat=40.5, decyr=decyr,
            exx=arr, exy=arr.copy(), eyy=arr.copy(),
            omega=arr.copy(), e1=arr.copy(), e2=arr.copy(),
            azimuth=arr.copy(), shear=arr.copy(), dilation=arr.copy(),
            sec_inv=arr.copy(), ve=arr.copy(), vn=arr.copy(),
            condition_number=arr.copy(),
        )
        assert result.lon == 112.5
        assert len(result.exx) == n

    def test_mismatched_lengths_raise(self):
        decyr = np.linspace(2020, 2021, 10)
        arr = np.random.randn(5)  # wrong length
        with pytest.raises(ValueError):
            StrainTimeSeriesResult(
                lon=0, lat=0, decyr=decyr,
                exx=arr, exy=arr, eyy=arr,
                omega=arr, e1=arr, e2=arr,
                azimuth=arr, shear=arr, dilation=arr,
                sec_inv=arr, ve=arr, vn=arr,
                condition_number=arr,
            )


# ======================================================================
# Strain time series computation tests (synthetic)
# ======================================================================

class TestGridStrainTimeSeries:
    """Test grid-based strain time series computation."""

    def test_synthetic_grid(self):
        """Compute strain TS on a tiny grid with synthetic aligned data."""
        from pystrain2.grid.grid import Grid
        from pystrain2.timeseries.strain_ts import GridStrainTimeSeries

        # Create a TimeSeriesCollection with 5 sites, 20 epochs
        n_sites = 5
        n_epochs = 20
        decyr = np.linspace(2020.0, 2020.05, n_epochs)
        sites = [f"S{i:02d}" for i in range(n_sites)]
        lon = np.array([112.0, 112.1, 112.2, 112.3, 112.4])
        lat = np.array([40.0, 40.05, 40.1, 40.15, 40.2])

        # Pure E-W extension: ve grows with lon, vn=0
        E = np.zeros((n_epochs, n_sites))
        N = np.zeros((n_epochs, n_sites))
        for j in range(n_epochs):
            scale = 1.0 + 0.01 * np.sin(2 * np.pi * j / n_epochs)
            for i in range(n_sites):
                E[j, i] = (lon[i] - 112.2) * 5.0 * scale
                N[j, i] = (lat[i] - 40.1) * 0.5 * scale

        SE = np.full((n_epochs, n_sites), 0.1)
        SN = np.full((n_epochs, n_sites), 0.1)
        U_arr = np.zeros((n_epochs, n_sites))
        SU_arr = np.zeros((n_epochs, n_sites))

        tsc = TimeSeriesCollection(
            sites=sites, decyr=decyr,
            E=E, N=N, U=U_arr, SE=SE, SN=SN, SU=SU_arr,
            lon=lon, lat=lat,
        )

        # One grid point in the center
        grid = Grid(112.15, 112.25, 40.05, 40.15, 0.1, 0.1, stagger=False)

        estimator = GridStrainTimeSeries(
            tsc, grid,
            maxdist_km=500.0,
            min_sites=3,
            check_azimuth=False,
            output_dir="/tmp/pystrain2_test_ts",
        )
        results = estimator.compute()
        assert len(results) > 0
        assert len(results[0].exx) == n_epochs
        # Should have solved at least some epochs
        n_solved = np.sum(np.isfinite(results[0].exx))
        assert n_solved > 0


class TestTriStrainTimeSeries:
    """Test triangle-based strain time series computation."""

    def test_synthetic_tri(self):
        """Compute strain TS on Delaunay triangles with synthetic data."""
        from pystrain2.timeseries.strain_ts import TriStrainTimeSeries

        n_sites = 6
        n_epochs = 10
        decyr = np.linspace(2020.0, 2020.02, n_epochs)
        sites = [f"T{i:02d}" for i in range(n_sites)]
        lon = np.array([112.0, 112.1, 112.05, 112.15, 112.08, 112.12])
        lat = np.array([40.0, 40.0, 40.1, 40.1, 40.05, 40.08])

        E = np.zeros((n_epochs, n_sites))
        N = np.zeros((n_epochs, n_sites))
        for j in range(n_epochs):
            for i in range(n_sites):
                E[j, i] = (lon[i] - 112.08) * 3.0
                N[j, i] = (lat[i] - 40.05) * 0.5

        SE = np.full((n_epochs, n_sites), 0.1)
        SN = np.full((n_epochs, n_sites), 0.1)
        U_arr = np.zeros((n_epochs, n_sites))
        SU_arr = np.zeros((n_epochs, n_sites))

        tsc = TimeSeriesCollection(
            sites=sites, decyr=decyr,
            E=E, N=N, U=U_arr, SE=SE, SN=SN, SU=SU_arr,
            lon=lon, lat=lat,
        )

        estimator = TriStrainTimeSeries(
            tsc,
            min_angle_deg=5.0,
            max_edge_pctl=100.0,
            max_edge_factor=10.0,
            min_area_ratio=0.01,
            output_dir="/tmp/pystrain2_test_ts",
        )
        results = estimator.compute()
        assert len(results) > 0
        assert results[0].exx.shape[0] == n_epochs


class TestUserStrainTimeSeries:
    """Test user-defined site-group strain time series computation."""

    def test_synthetic_user(self, tmp_path):
        """Compute strain TS for user-defined groups."""
        from pystrain2.timeseries.strain_ts import UserStrainTimeSeries

        # Site groups file
        groups_file = tmp_path / "groups.txt"
        groups_file.write_text("S01 S02 S03\nS04 S05 S06\n", encoding="utf-8")

        n_sites = 6
        n_epochs = 10
        decyr = np.linspace(2020.0, 2020.02, n_epochs)
        sites = ["S01", "S02", "S03", "S04", "S05", "S06"]
        lon = np.array([112.0, 112.1, 112.05, 112.15, 112.2, 112.18])
        lat = np.array([40.0, 40.0, 40.1, 40.1, 40.0, 40.1])

        E = np.zeros((n_epochs, n_sites))
        N = np.zeros((n_epochs, n_sites))
        for j in range(n_epochs):
            for i in range(n_sites):
                E[j, i] = (lon[i] - 112.1) * 2.0
                N[j, i] = (lat[i] - 40.05) * 0.5

        SE = np.full((n_epochs, n_sites), 0.1)
        SN = np.full((n_epochs, n_sites), 0.1)
        U_arr = np.zeros((n_epochs, n_sites))
        SU_arr = np.zeros((n_epochs, n_sites))

        tsc = TimeSeriesCollection(
            sites=sites, decyr=decyr,
            E=E, N=N, U=U_arr, SE=SE, SN=SN, SU=SU_arr,
            lon=lon, lat=lat,
        )

        estimator = UserStrainTimeSeries(
            tsc,
            site_groups_file=str(groups_file),
            max_sigma_mm=10.0,
            output_dir="/tmp/pystrain2_test_ts",
        )
        results = estimator.compute()
        # Should have 2 groups (both have ≥3 sites)
        assert len(results) == 2
        for r in results:
            assert r.exx.shape[0] == n_epochs
            n_solved = np.sum(np.isfinite(r.exx))
            assert n_solved > 0


# ======================================================================
# Distance weight test (bug fix)
# ======================================================================

def test_distance_weight_fix():
    """Verify the distance weight bug fix: use exp(-d²/D²), not exp(+d²/D²)."""
    from pystrain2.timeseries.strain_ts import _distance_weight

    d = np.array([0.0, 10.0, 100.0, 300.0])
    D = 100.0
    w = _distance_weight(d, D)

    # At d=0, weight should be 1.0
    assert w[0] == pytest.approx(1.0)
    # At d=D (index 2 = 100.0 km), weight should be exp(-1) ≈ 0.368
    assert w[2] == pytest.approx(np.exp(-1.0), abs=0.01)
    # At d >> D, weight should be near 0
    assert w[3] < 0.01


# ======================================================================
# Azimuth coverage test
# ======================================================================

def test_azimuth_coverage():
    """Test four-quadrant azimuth check."""
    from pystrain2.timeseries.strain_ts import _check_azimuth_coverage

    # Good coverage: all 4 quadrants
    assert _check_azimuth_coverage(np.array([45, 135, -135, -45]))

    # Poor coverage: only 2 quadrants
    assert not _check_azimuth_coverage(np.array([45, 80, 30, 60]))

    # Edge case: exactly at boundaries
    assert _check_azimuth_coverage(np.array([1, 91, -179, -1]))
