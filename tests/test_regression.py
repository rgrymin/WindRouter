"""
Regression tests for WindRouter — specify correct behaviour after each bug is fixed.

Each test is marked xfail(strict=True):
  - While the bug exists: test is expected to fail (shows as 'xfail' in CI — OK).
  - After the bug is fixed: test turns green automatically.
  - If the bug is re-introduced: strict=True makes the suite fail (shows as 'XPASS' error).

Bugs covered:
  B-01, B-04, B-08, B-18, B-20, B-21, B-24, B-26
"""
import json
import math
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

import grib as grib_module
from grib import (
    calculate_distance_nm,
    find_shortest_path_dijkstra,
    load_grib_to_memory,
    print_route_summary,
    save_graph_to_json,
    save_to_gpx,
    save_waypoints_gpx,
    simulate_vmg_route,
)
from conftest import make_weather_cache

import gpxpy


class TestBugFixes:
    """
    Each test here specifies what the correct behaviour SHOULD be after the
    corresponding bug is fixed. While the bug exists the test is xfail
    (expected to fail). When fixed it turns green automatically. If the bug
    is re-introduced, strict=True causes the suite to fail.
    """

    @pytest.mark.slow
    def test_b04_southbound_vmg_returns_non_empty_route(self):
        """After B-04 is fixed, a southbound target must produce a real route."""
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5, 53.75, 54.0],
            lons_1d=[2.0, 2.25, 2.5, 2.75],
            u_speed=0.0, v_speed=5.0,  # northerly wind — southbound sailing viable
        )
        result = simulate_vmg_route(
            cache,
            start_lat=54.0, start_lon=2.25,
            target_lat=53.0, target_lon=2.25,
            start_time=cache["dates"][0],
        )
        assert len(result) > 0, "Southbound route must not be empty after B-04 fix"
        assert result[-1]["lat"] < 54.0, "Route must make southward progress"

    def test_b08_zone_gpx_written_as_waypoints(self, tmp_path):
        """B-08 fixed: save_waypoints_gpx writes <wpt> elements readable by Visualiser."""
        points = [{"lat": 53.0, "lon": 2.0}]
        gpx_path = str(tmp_path / "zone.gpx")
        save_waypoints_gpx(points, gpx_path)
        with open(gpx_path) as fh:
            gpx = gpxpy.parse(fh)
        assert len(gpx.waypoints) == 1, "Zone files must use <wpt>"

    def test_b08_save_waypoints_gpx_does_not_write_trkpt(self, tmp_path):
        """B-08: save_waypoints_gpx must write <wpt>, NOT <trkpt>.

        Regression: reverting B-08 (removing save_waypoints_gpx and routing zone
        files through save_to_gpx) would produce <trkpt> elements.  Visualiser
        reads gpx.waypoints; if <wpt> is absent, zones never render.
        """
        points = [{"lat": 53.0, "lon": 2.0}, {"lat": 54.0, "lon": 3.0}]
        gpx_path = str(tmp_path / "zones.gpx")
        save_waypoints_gpx(points, gpx_path)
        raw = open(gpx_path).read()
        assert "<wpt" in raw, "B-08: save_waypoints_gpx must produce <wpt> elements"
        assert "<trkpt" not in raw, (
            "B-08: save_waypoints_gpx must NOT produce <trkpt> — "
            "Visualiser reads waypoints, not track points"
        )
        with open(gpx_path) as fh:
            gpx = gpxpy.parse(fh)
        assert len(gpx.waypoints) == 2, (
            "B-08: both zone points must be readable as waypoints by gpxpy"
        )


    def test_b01_graph_json_uses_graph_key(self, tmp_path):
        """After B-01 is fixed, the JSON key must be 'graph' so Visualiser can read it."""
        nodes = {(0, 0), (0, 1)}
        adj = {(0, 0): [{"target": (0, 1), "cost": 1.0}], (0, 1): []}
        spm = {
            (0, 0): {"lat": 53.0, "lon": 2.0,  "max_speed": 5.0},
            (0, 1): {"lat": 53.0, "lon": 2.25, "max_speed": 5.0},
        }
        json_path = str(tmp_path / "graph.json")
        save_graph_to_json(nodes, adj, spm, json_path)
        data = json.loads(open(json_path).read())
        assert "graph" in data, "After B-01 fix, key must be 'graph' not 'edges'"
        assert "edges" not in data

    def test_b18_file_handle_closed_even_on_iteration_exception(self, tmp_path, monkeypatch):
        """After B-18 is fixed, grbs.close() must be called even when iteration raises."""
        dummy = tmp_path / "bad.grib"
        dummy.write_bytes(b"x")

        grbs = MagicMock()
        grbs.__iter__ = MagicMock(side_effect=RuntimeError("corrupt mid-iteration"))
        monkeypatch.setattr(grib_module.pygrib, "open", MagicMock(return_value=grbs))

        load_grib_to_memory(str(dummy))

        grbs.close.assert_called_once()  # must be called from finally block

    @pytest.mark.xfail(strict=True, reason="B-21: flat-earth formula breaks at the antimeridian — lon crossing gives ~21000 nm instead of ~120 nm")
    def test_b21_distance_handles_antimeridian_crossing(self):
        """After B-21 is fixed with proper Haversine, antimeridian crossing must give ~120 nm.
        The flat formula uses raw lon2-lon1 = -358° which inflates to ~21000 nm."""
        result = calculate_distance_nm(0.0, 179.0, 0.0, -179.0)
        # Real great-circle distance at equator between lon 179° and lon -179°
        real_nm = 2.0 * 60.0  # 2° of longitude = 120 nm at equator
        assert result < 200.0, (
            f"B-21: antimeridian crossing gives {result:.1f} nm (should be ~{real_nm:.0f} nm). "
            "Fix: use Haversine with modular lon diff."
        )

    def test_b24_dijkstra_2d_handles_leaf_nodes_without_keyerror(self):
        """After B-24 is fixed, a leaf node (target-only, no outgoing edges) must not raise KeyError."""
        # (0,1) is a target node but has no entry as a source in adj
        adj = {
            (0, 0): [{"target": (0, 1), "cost": 1.0}],
            # (0,1) intentionally absent as a source key — leaf node
        }
        spm = {
            (0, 0): {"lat": 53.0,  "lon": 2.0,  "max_speed": 5.0},
            (0, 1): {"lat": 53.25, "lon": 2.25, "max_speed": 5.0},
        }
        # After fix this must not raise — currently raises KeyError
        path, cost = find_shortest_path_dijkstra((0, 0), adj, spm, 53.25, 2.25)
        assert len(path) >= 1
        assert cost >= 0.0

    def test_b26_print_route_summary_none_time_hours_does_not_crash(self):
        """After B-26 is fixed, summary must print 'N/A' instead of crashing."""
        points = [
            {"lat": 53.0,  "lon": 2.0},
            {"lat": 53.25, "lon": 2.25},
        ]
        # No 'time' key in points, no time_hours argument → time_hours stays None
        # Currently raises TypeError: unsupported format character
        try:
            print_route_summary(points, "Test route")
        except TypeError as e:
            pytest.fail(f"B-26: print_route_summary crashed with TypeError: {e}")

    @pytest.mark.slow
    def test_b20_vmg_lon_update_uses_pre_advance_latitude(self, monkeypatch):
        """B-20 fixed: lon update uses old_lat (before lat increment), not new_lat.

        Strategy: intercept the heading and boat-speed chosen by VMG on the first step
        by monkeypatching grib.calculate_distance_nm so the loop exits after exactly
        one advance. Then independently compute expected_lon using old_lat and
        expected_lon_buggy using new_lat. Assert actual lon matches expected_lon.
        """
        import math as _math
        import grib as grib_mod
        from grib import get_scampi_30_polars, calculate_bearing

        cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75, 55.0],
            lons_1d=[2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5],
            u_speed=5.0, v_speed=5.0,   # SW wind ~13.6 kt, TWD=225°
        )
        start_lat, start_lon = 53.0, 2.0
        target_lat, target_lon = 54.75, 3.25
        t0 = cache["dates"][0]

        # Intercept the best_hdg chosen at step 0 by capturing math.cos calls.
        # Simpler: run the full route and inspect result[1], but compute expected lon
        # from scratch using get_scampi_30_polars + the known weather at (53.0, 2.0).
        from grib import get_weather_from_cache, calculate_distance_nm
        weather = get_weather_from_cache(cache, start_lat, start_lon, t0)
        u, v = weather["wind_u"], weather["wind_v"]
        # True wind speed and direction
        tws_kt = _math.sqrt(u**2 + v**2) * 1.94384
        twd = (_math.degrees(_math.atan2(-u, -v)) + 360) % 360

        polars = get_scampi_30_polars()
        time_step_min = 10.0

        # Find the best heading VMG would choose (mirrors the inner loop logic)
        best_vmg, best_hdg, best_bs = -1.0, None, None
        for hdg in range(0, 360, 2):
            twa = (hdg - twd + 360) % 360
            if twa < 32 or twa > 328:
                continue
            twa_sym = twa if twa <= 180 else 360 - twa
            bs = polars([twa_sym, tws_kt])[0]
            if bs <= 0:
                continue
            dist_step = bs * (time_step_min / 60.0)
            next_lat = start_lat + (dist_step * _math.cos(_math.radians(hdg))) / 60.0
            next_lon = start_lon + (dist_step * _math.sin(_math.radians(hdg))) / (
                60.0 * _math.cos(_math.radians(start_lat))  # old_lat — correct formula
            )
            vmg = calculate_distance_nm(start_lat, start_lon, target_lat, target_lon) - \
                  calculate_distance_nm(next_lat, next_lon, target_lat, target_lon)
            if vmg > best_vmg:
                best_vmg, best_hdg, best_bs = vmg, hdg, bs

        assert best_hdg is not None, "Test setup: VMG must find a valid heading"

        # Compute expected p1.lon analytically using old_lat (the correct formula)
        dist = best_bs * (time_step_min / 60.0)
        expected_lon_correct = start_lon + (dist * _math.sin(_math.radians(best_hdg))) / (
            60.0 * _math.cos(_math.radians(start_lat))  # old_lat
        )
        # And what the buggy formula (new_lat) would have produced
        new_lat = start_lat + (dist * _math.cos(_math.radians(best_hdg))) / 60.0
        expected_lon_buggy = start_lon + (dist * _math.sin(_math.radians(best_hdg))) / (
            60.0 * _math.cos(_math.radians(new_lat))   # new_lat — the bug
        )

        # Sanity: the two formulas must give different values for this to be a real test
        assert not _math.isclose(expected_lon_correct, expected_lon_buggy, rel_tol=1e-9), (
            "Test setup error: old_lat and new_lat formulas give identical lon — "
            "heading has no E-W component, choose a different scenario"
        )

        # Run the actual implementation
        result = simulate_vmg_route(
            cache, start_lat=start_lat, start_lon=start_lon,
            target_lat=target_lat, target_lon=target_lon,
            start_time=t0,
        )
        assert len(result) >= 2, "B-20: route must have at least 2 points"

        actual_lon = result[1]["lon"]
        assert actual_lon == pytest.approx(expected_lon_correct, abs=1e-9), (
            f"B-20: p1.lon={actual_lon:.10f} does not match old_lat formula "
            f"({expected_lon_correct:.10f}); buggy new_lat formula gives {expected_lon_buggy:.10f}"
        )
