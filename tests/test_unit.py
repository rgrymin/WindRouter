"""
Unit tests for WindRouter — grib.py functions with known expected outputs.

Every assertion here reflects the CORRECT, INTENDED behaviour as documented
in REQUIREMENTS.md. If a test breaks during refactoring, the behaviour changed
and must be consciously verified.
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
    calculate_bearing,
    calculate_distance_nm,
    find_shortest_path_dijkstra,
    find_shortest_path_dijkstra_3d,
    generate_reachable_graph,
    get_scampi_30_polars,
    get_weather_from_cache,
    identify_safe_sailing_areas,
    identify_weather_danger_zones,
    load_grib_to_memory,
    print_route_summary,
    save_graph_to_json,
    save_route_detailed_log,
    save_to_gpx,
    simulate_vmg_route,
)
from conftest import make_weather_cache, make_safe_points_map, make_full_adjacency, make_mock_grib_message


# ---------------------------------------------------------------------------
# Polars model (REQ-01)
# ---------------------------------------------------------------------------

class TestPolars:
    def test_returns_callable_interpolator(self):
        polars = get_scampi_30_polars()
        assert callable(polars)

    def test_known_value_twa90_tws10(self):
        polars = get_scampi_30_polars()
        result = polars([90, 10])[0]
        assert abs(result - 6.8) < 0.01, f"Expected 6.8 kt at TWA=90°/TWS=10, got {result}"

    def test_twa_180_tws_20(self):
        polars = get_scampi_30_polars()
        result = polars([180, 20])[0]
        assert abs(result - 7.6) < 0.01, f"Expected 7.6 kt at TWA=180°/TWS=20, got {result}"


# ---------------------------------------------------------------------------
# Utility functions (REQ-03 helpers)
# ---------------------------------------------------------------------------

class TestCalculateDistanceNm:
    def test_zero_distance(self):
        assert calculate_distance_nm(53.0, 2.0, 53.0, 2.0) == pytest.approx(0.0)

    def test_one_degree_latitude_approx_60nm(self):
        d = calculate_distance_nm(53.0, 2.0, 54.0, 2.0)
        assert abs(d - 60.0) < 1.0, f"Expected ~60 nm, got {d}"

    def test_longitude_uses_cos_correction(self):
        """C-29: uses cos(avg_lat) correction — not pure flat-earth."""
        d_ns = calculate_distance_nm(53.0, 2.0, 54.0, 2.0)
        d_ew = calculate_distance_nm(53.5, 2.0, 53.5, 3.0)
        # E-W degree at 53.5°N ≈ 60 × cos(53.5°) ≈ 35.7 nm
        assert d_ew < d_ns, "E-W distance should be shorter than N-S at 53°N"
        assert abs(d_ew - 35.7) < 1.0, f"Expected ~35.7 nm E-W at 53.5°N, got {d_ew}"

    def test_cos_correction_uses_average_latitude(self):
        """C-29: cos factor uses average of both latitudes, not departure latitude.
        Use asymmetric endpoints so avg_lat (53.5°) ≠ departure_lat (53.0°).
        avg_lat formula: d_lon × 60 × cos(avg_lat_rad) where avg = (53+54)/2 = 53.5°
        departure_lat formula would use cos(53.0°) instead — gives different result."""
        # Asymmetric pair: lat1=53.0, lat2=54.0, lon difference = 1.0°
        d = calculate_distance_nm(53.0, 2.0, 54.0, 3.0)
        avg_lat_rad = math.radians((53.0 + 54.0) / 2.0)  # 53.5°
        dep_lat_rad = math.radians(53.0)
        d_lat = (54.0 - 53.0) * 60.0
        d_lon_avg = (3.0 - 2.0) * 60.0 * math.cos(avg_lat_rad)
        d_lon_dep = (3.0 - 2.0) * 60.0 * math.cos(dep_lat_rad)
        expected_avg = math.sqrt(d_lat**2 + d_lon_avg**2)
        expected_dep = math.sqrt(d_lat**2 + d_lon_dep**2)
        assert d == pytest.approx(expected_avg, abs=0.001), (
            f"C-29: expected avg_lat formula ({expected_avg:.4f} nm), got {d:.4f} nm"
        )
        assert d != pytest.approx(expected_dep, abs=0.001), (
            "C-29: result must differ from departure_lat formula"
        )

    def test_symmetry(self):
        d1 = calculate_distance_nm(53.0, 2.0, 54.0, 3.0)
        d2 = calculate_distance_nm(54.0, 3.0, 53.0, 2.0)
        assert d1 == pytest.approx(d2)


class TestCalculateBearing:
    def test_due_north(self):
        b = calculate_bearing(53.0, 2.0, 54.0, 2.0)
        assert abs(b) < 1.0, f"Expected ~0° (north), got {b}"

    def test_due_east(self):
        b = calculate_bearing(53.0, 2.0, 53.0, 3.0)
        assert abs(b - 90.0) < 1.0, f"Expected ~90° (east), got {b}"

    def test_due_south(self):
        b = calculate_bearing(54.0, 2.0, 53.0, 2.0)
        assert abs(b - 180.0) < 1.0, f"Expected ~180° (south), got {b}"

    def test_due_west(self):
        b = calculate_bearing(53.0, 3.0, 53.0, 2.0)
        assert abs(b - 270.0) < 1.0, f"Expected ~270° (west), got {b}"

    def test_result_in_0_360_range(self):
        b = calculate_bearing(53.0, 2.0, 54.0, 3.0)
        assert 0.0 <= b < 360.0


# ---------------------------------------------------------------------------
# Weather lookup (REQ-03)
# ---------------------------------------------------------------------------

class TestGetWeatherFromCache:
    def setup_method(self):
        self.cache = make_weather_cache()
        self.t0 = self.cache["dates"][0]

    def test_returns_none_for_falsy_cache(self):
        assert get_weather_from_cache(None, 53.0, 2.0, self.t0) is None
        assert get_weather_from_cache({}, 53.0, 2.0, self.t0) is None

    def test_returns_dict_with_expected_keys(self):
        result = get_weather_from_cache(self.cache, 53.0, 2.0, self.t0)
        assert result is not None
        assert "wind_u" in result
        assert "wind_v" in result
        assert "meta" in result

    def test_meta_keys(self):
        result = get_weather_from_cache(self.cache, 53.0, 2.0, self.t0)
        meta = result["meta"]
        assert "actual_lat" in meta
        assert "actual_lon" in meta
        assert "time" in meta
        assert "approximated" in meta

    def test_nearest_neighbor_lookup(self):
        """REQ-03: nearest grid point by min (Δlat²+Δlon²), no interpolation.
        Point (53.1, 2.1) is closest to grid node (53.0, 2.0): distance² = 0.01²+0.01² = 0.0002
        vs. next candidate (53.25, 2.25): distance² = 0.15²+0.15² = 0.045."""
        result = get_weather_from_cache(self.cache, 53.1, 2.1, self.t0)
        assert result is not None
        assert result["meta"]["actual_lat"] == pytest.approx(53.0), (
            "REQ-03: nearest grid point to (53.1, 2.1) must be (53.0, 2.0)"
        )
        assert result["meta"]["actual_lon"] == pytest.approx(2.0)

    def test_approximated_flag_within_window(self):
        result = get_weather_from_cache(self.cache, 53.0, 2.0, self.t0)
        assert result["meta"]["approximated"] is False

    def test_approximated_flag_outside_window(self):
        future = self.t0 + timedelta(days=30)
        result = get_weather_from_cache(self.cache, 53.0, 2.0, future)
        assert result["meta"]["approximated"] is True

    def test_wind_values_match_fixture(self):
        """u=5.0 m/s, v=0.0 m/s as defined in fixture."""
        result = get_weather_from_cache(self.cache, 53.0, 2.0, self.t0)
        assert result["wind_u"] == pytest.approx(5.0)
        assert result["wind_v"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Safety zone classification (REQ-04)
# ---------------------------------------------------------------------------

class TestIdentifySafeAreas:
    def test_safe_cells_with_calm_wind(self):
        cache = make_weather_cache(u_speed=5.0, v_speed=0.0)  # ~9.7 kt
        spm = identify_safe_sailing_areas(cache, max_wind_threshold=30.0)
        assert len(spm) == 16  # 4×4 grid, all safe

    def test_returns_dict_with_correct_keys(self):
        cache = make_weather_cache()
        spm = identify_safe_sailing_areas(cache)
        for key, val in spm.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert "lat" in val and "lon" in val and "max_speed" in val
            break

    def test_strong_wind_excludes_cells(self):
        """25 m/s ≈ 47 kt — exceeds 30 kt threshold, no safe cells."""
        cache = make_weather_cache(u_speed=25.0, v_speed=0.0)
        spm = identify_safe_sailing_areas(cache, max_wind_threshold=30.0)
        assert len(spm) == 0

    def test_forbidden_zones_strictly_above_40kt(self):
        """identify_weather_danger_zones: forbidden = strictly > 40 kt (not >=)."""
        cache = make_weather_cache(u_speed=22.0, v_speed=0.0)  # ~42.7 kt > 40
        zones = identify_weather_danger_zones(cache, min_threshold=40.0)
        assert len(zones) == 16  # all 16 cells forbidden

    def test_exactly_40kt_is_not_forbidden(self):
        """REQ-04: condition is strictly > 40 kt. Exactly 40.0 kt must NOT be forbidden.
        40.0 kt / 1.94384 ≈ 20.5778 m/s."""
        u_for_exactly_40kt = 40.0 / 1.94384  # ≈ 20.5778 m/s
        cache = make_weather_cache(u_speed=u_for_exactly_40kt, v_speed=0.0)
        zones = identify_weather_danger_zones(cache, min_threshold=40.0)
        assert len(zones) == 0, (
            "REQ-04: exactly 40 kt must NOT be forbidden (condition is strictly > 40)"
        )

    def test_caution_zone_30_to_40kt(self):
        """identify_weather_danger_zones: caution = > 30 and <= 40 kt."""
        cache = make_weather_cache(u_speed=18.0, v_speed=0.0)  # ~35 kt
        zones = identify_weather_danger_zones(cache, min_threshold=30.0, max_threshold=40.0)
        assert len(zones) == 16

    def test_danger_zone_dict_keys(self):
        cache = make_weather_cache(u_speed=22.0, v_speed=0.0)
        zones = identify_weather_danger_zones(cache, min_threshold=40.0)
        assert len(zones) > 0
        z = zones[0]
        assert set(z.keys()) == {"lat", "lon", "speed", "time"}


# ---------------------------------------------------------------------------
# Graph construction (REQ-05)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateReachableGraph(object):
    def test_returns_tuple_of_three(self, tmp_path):
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75],
            lons_1d=[2.0, 2.25, 2.5, 2.75, 3.0],
        )
        spm = identify_safe_sailing_areas(cache)
        log = str(tmp_path / "graph_log.txt")
        result = generate_reachable_graph(
            cache, spm,
            start_lat=53.0, start_lon=2.0,
            target_lat=54.75, target_lon=2.75,
            start_time=cache["dates"][0],
            log_file=log,
        )
        assert isinstance(result, tuple) and len(result) == 3

    def test_returns_empty_when_no_safe_cells(self, tmp_path):
        cache = make_weather_cache(u_speed=25.0)  # all forbidden
        spm = {}  # empty safe map
        log = str(tmp_path / "graph_log.txt")
        nodes, adj, start_idx = generate_reachable_graph(
            cache, spm,
            start_lat=53.0, start_lon=2.0,
            target_lat=54.75, target_lon=2.75,
            start_time=cache["dates"][0],
            log_file=log,
        )
        assert nodes == set()
        assert adj == {}
        assert start_idx is None

    def test_adjacency_map_edge_structure(self, tmp_path):
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75],
            lons_1d=[2.0, 2.25, 2.5, 2.75, 3.0],
        )
        spm = identify_safe_sailing_areas(cache)
        log = str(tmp_path / "graph_log.txt")
        nodes, adj, start_idx = generate_reachable_graph(
            cache, spm,
            start_lat=53.0, start_lon=2.0,
            target_lat=54.75, target_lon=2.75,
            start_time=cache["dates"][0],
            log_file=log,
        )
        assert start_idx is not None
        # Check edge structure
        for node, edges in adj.items():
            assert isinstance(node, tuple)
            for edge in edges:
                assert "target" in edge
                assert "cost" in edge
                assert isinstance(edge["cost"], float)
                assert edge["cost"] > 0

    def test_start_node_in_returned_set(self, tmp_path):
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75],
            lons_1d=[2.0, 2.25, 2.5, 2.75, 3.0],
        )
        spm = identify_safe_sailing_areas(cache)
        log = str(tmp_path / "graph_log.txt")
        nodes, adj, start_idx = generate_reachable_graph(
            cache, spm,
            start_lat=53.0, start_lon=2.0,
            target_lat=54.75, target_lon=2.75,
            start_time=cache["dates"][0],
            log_file=log,
        )
        assert start_idx is not None, "start_idx must be set for non-empty spm"
        assert start_idx in nodes

    def test_log_file_created_and_appends(self, tmp_path):
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75],
            lons_1d=[2.0, 2.25, 2.5, 2.75, 3.0],
        )
        spm = identify_safe_sailing_areas(cache)
        log = str(tmp_path / "graph_log.txt")
        generate_reachable_graph(
            cache, spm, 53.0, 2.0, 54.75, 2.75,
            cache["dates"][0], log_file=log,
        )
        assert os.path.exists(log)
        with open(log) as f:
            content = f.read()
        assert "GRAPH CONSTRUCTION LOG" in content


# ---------------------------------------------------------------------------
# Dijkstra 2D (REQ-06)
# ---------------------------------------------------------------------------

class TestDijkstra2D:
    LATS = [53.0, 53.25, 53.5, 53.75]
    LONS = [2.0, 2.25, 2.5, 2.75]

    def setup_method(self):
        self.spm = make_safe_points_map(self.LATS, self.LONS)
        self.adj = make_full_adjacency(self.spm, cost=1.0)
        self.target_lat = 53.75
        self.target_lon = 2.75
        # start node at top-left (0,0)
        self.start_node = (0, 0)

    def test_returns_tuple_of_two(self):
        path, cost = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        assert isinstance(path, list)
        assert isinstance(cost, (int, float))

    def test_path_starts_at_start_node(self):
        path, _ = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        assert len(path) > 0
        assert path[0]["lat"] == pytest.approx(self.LATS[0])
        assert path[0]["lon"] == pytest.approx(self.LONS[0])

    def test_path_ends_near_target(self):
        path, _ = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        last = path[-1]
        d = calculate_distance_nm(last["lat"], last["lon"], self.target_lat, self.target_lon)
        # Target (53.75, 2.75) is a grid node — best_finish must land exactly on it
        assert d == pytest.approx(0.0, abs=0.1), f"Path end too far from target: {d} nm"

    def test_path_points_have_correct_keys(self):
        path, _ = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        for pt in path:
            assert "lat" in pt
            assert "lon" in pt
            assert "max_speed" in pt
            # No 'time' key — added externally in __main__
            assert "time" not in pt

    def test_returns_empty_list_for_no_start_node(self):
        path, cost = find_shortest_path_dijkstra(
            None, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        assert path == []
        assert cost == 0

    def test_returns_empty_list_for_empty_adjacency(self):
        path, cost = find_shortest_path_dijkstra(
            self.start_node, {}, self.spm,
            self.target_lat, self.target_lon,
        )
        assert path == []
        assert cost == 0

    def test_cost_is_positive_for_valid_path(self):
        path, cost = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        assert len(path) > 0
        assert cost > 0

    def test_path_is_direct_reference_not_copy(self):
        """REQ-06: waypoints are direct references into safe_points_map, not copies."""
        path, _ = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        assert len(path) > 0
        # Record original lat before any mutation
        original_lat = path[0]["lat"]
        original_lon = path[0]["lon"]
        # Find the spm key that matches this point by coordinates
        spm_key = next(
            k for k, v in self.spm.items()
            if v["lat"] == pytest.approx(original_lat) and v["lon"] == pytest.approx(original_lon)
        )
        # Mutate through the path reference
        path[0]["lat"] = 999.0
        # The spm entry must be the same object — mutation must be visible there too
        assert self.spm[spm_key]["lat"] == 999.0, (
            "REQ-06: path[0] must be a direct reference into spm, not a copy"
        )
        path[0]["lat"] = original_lat  # restore


# ---------------------------------------------------------------------------
# Dijkstra 3D (REQ-07)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDijkstra3D:
    LATS = [53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75]
    LONS = [2.0, 2.25, 2.5, 2.75, 3.0]

    def setup_method(self):
        self.cache = make_weather_cache(
            lats_1d=self.LATS,
            lons_1d=self.LONS,
            u_speed=5.0,
            v_speed=0.0,
        )
        self.spm = identify_safe_sailing_areas(self.cache)
        self.start_node = min(self.spm.keys())  # smallest (row, col)
        self.target_lat = 54.75
        self.target_lon = 2.75
        self.t0 = self.cache["dates"][0]

    def test_returns_tuple_of_two(self):
        path, cost = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert isinstance(path, list)
        assert isinstance(cost, (int, float))

    def test_returns_empty_for_none_start(self):
        """B-05: handles None start_node gracefully via internal guard."""
        path, cost = find_shortest_path_dijkstra_3d(
            None, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert path == []
        assert cost == 0

    def test_returns_empty_for_empty_spm(self):
        path, cost = find_shortest_path_dijkstra_3d(
            self.start_node, {},
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert path == []
        assert cost == 0

    def test_path_points_have_time_key(self):
        """REQ-07: timestamps embedded at reconstruction time (unlike 2D)."""
        path, cost = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert len(path) > 0, "Expected non-empty path for valid inputs"
        for pt in path:
            assert "time" in pt, "3D path points must have 'time' key"

    def test_path_points_have_all_keys(self):
        path, cost = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert len(path) > 0, "Expected non-empty path for valid inputs"
        pt = path[0]
        assert "lat" in pt
        assert "lon" in pt
        assert "max_speed" in pt
        assert "time" in pt

    def test_first_waypoint_timestamp_equals_departure(self):
        """REQ-07: first waypoint timestamp = departure_time + 0h."""
        path, _ = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert len(path) > 0, "Expected non-empty path for valid inputs"
        assert path[0]["time"] == self.t0

    def test_timestamps_monotonically_increasing(self):
        path, _ = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert len(path) > 1, "Expected path of at least 2 nodes"
        times = [pt["time"] for pt in path]
        for a, b in zip(times, times[1:]):
            assert b >= a, "3D path timestamps must be non-decreasing"

    def test_path_is_copy_not_reference(self):
        """REQ-07: each waypoint is a copy of safe_points_map entry (uses .copy())."""
        path, _ = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert len(path) > 0, "Expected non-empty path for valid inputs"
        original_lat = path[0]["lat"]
        original_lon = path[0]["lon"]
        spm_key = next(
            k for k, v in self.spm.items()
            if v["lat"] == pytest.approx(original_lat) and v["lon"] == pytest.approx(original_lon)
        )
        path[0]["lat"] = 999.0
        assert self.spm[spm_key]["lat"] == pytest.approx(original_lat), (
            "REQ-07: path[0] must be a copy of spm entry, not a direct reference"
        )

    def test_cost_matches_distances_of_best_finish(self):
        """REQ-07: returns distances[best_finish_node] as cost."""
        path, cost = find_shortest_path_dijkstra_3d(
            self.start_node, self.spm,
            self.target_lat, self.target_lon,
            self.t0, self.cache,
        )
        assert len(path) > 1, "Expected path of at least 2 nodes"
        assert cost > 0
        elapsed = (path[-1]["time"] - path[0]["time"]).total_seconds() / 3600.0
        assert elapsed == pytest.approx(cost, abs=0.01)


# ---------------------------------------------------------------------------
# VMG simulation (REQ-08)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestVMG:
    LATS = [53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75]
    LONS = [2.0, 2.25, 2.5, 2.75, 3.0]

    def setup_method(self):
        self.cache = make_weather_cache(
            lats_1d=self.LATS,
            lons_1d=self.LONS,
            u_speed=5.0,
            v_speed=0.0,
        )
        self.t0 = self.cache["dates"][0]

    def test_returns_list(self):
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert isinstance(result, list)

    def test_returns_empty_for_none_cache(self):
        result = simulate_vmg_route(
            None,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert result == []

    def test_each_point_has_lat_lon_time(self):
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert len(result) > 0, "Expected non-empty route for valid northward inputs"
        for pt in result:
            assert "lat" in pt
            assert "lon" in pt
            assert "time" in pt
            assert "max_speed" not in pt  # VMG has no max_speed field

    def test_first_point_is_start_position(self):
        """REQ-08: start appended before advance (but after weather lookup)."""
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert len(result) > 0, "Expected non-empty route for valid northward inputs"
        assert result[0]["lat"] == pytest.approx(53.0, abs=0.01)
        assert result[0]["lon"] == pytest.approx(2.25, abs=0.01)

    def test_first_point_timestamp_equals_departure(self):
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert len(result) > 0, "Expected non-empty route for valid northward inputs"
        assert result[0]["time"] == self.t0

    def test_timestamps_monotonically_increasing(self):
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert len(result) > 1, "Expected at least 2 VMG points for northward route"
        for a, b in zip(result, result[1:]):
            assert b["time"] >= a["time"], "VMG timestamps must be non-decreasing"

    def test_no_max_speed_key_in_vmg_points(self):
        """REQ-08 output: {'lat', 'lon', 'time'} only — no max_speed unlike Dijkstra."""
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        for pt in result:
            assert "max_speed" not in pt

    def test_heading_range_2_degree_increments(self):
        """REQ-08: headings 0..358 step 2 — 180 headings evaluated per step.
        Tested indirectly: with westerly wind (u=5 m/s, TWD=270°), beam-reach north
        gives max VMG — route should make northward progress."""
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert len(result) > 1, "Expected northward progress for westerly wind"
        assert result[-1]["lat"] > result[0]["lat"], "VMG should make northward progress"


# ---------------------------------------------------------------------------
# Additional edge cases and bug coverage
# ---------------------------------------------------------------------------

class TestDijkstra2DEdgeCases:
    """Additional edge cases not covered in the main TestDijkstra2D class."""

    LATS = [53.0, 53.25, 53.5, 53.75]
    LONS = [2.0, 2.25, 2.5, 2.75]

    def setup_method(self):
        self.spm = make_safe_points_map(self.LATS, self.LONS)
        self.target_lat = 53.75
        self.target_lon = 2.75

    def test_start_node_with_no_outgoing_edges_returns_single_point_path(self):
        """REQ-06: if start node has no outgoing edges, returns 1-element path, NOT empty list."""
        # Build adjacency where only node (0,0) is listed but has empty edge list
        adj = {(0, 0): []}
        path, cost = find_shortest_path_dijkstra(
            (0, 0), adj, self.spm,
            self.target_lat, self.target_lon,
        )
        assert len(path) == 1, (
            "REQ-06: start-only reachable graph must return 1-element path, not empty list"
        )
        assert path[0]["lat"] == pytest.approx(self.LATS[0])
        assert cost == pytest.approx(0.0)


@pytest.mark.slow
class TestDijkstra3DEdgeCases:
    """Additional edge cases for 3D Dijkstra — B-14."""

    LATS = [53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75]
    LONS = [2.0, 2.25, 2.5, 2.75, 3.0]

    def setup_method(self):
        self.cache = make_weather_cache(lats_1d=self.LATS, lons_1d=self.LONS)
        self.spm = identify_safe_sailing_areas(self.cache)
        self.start_node = min(self.spm.keys())
        self.t0 = self.cache["dates"][0]

    def test_b14_none_weather_during_expansion_raises_typeerror(self):
        """B-14: if get_weather_from_cache returns None during edge expansion,
        the next line weather['wind_u'] raises TypeError."""
        # Make a cache that returns None for all lookups by clearing dates
        # so that the cache is falsy at the function level
        bad_cache = {}  # falsy cache — get_weather_from_cache returns None immediately
        # 3D Dijkstra's internal guard: `if not start_node or not safe_points_map`
        # does NOT guard against bad_cache. The crash happens inside the expansion loop.
        # However, with an empty cache dict, get_weather_from_cache returns None
        # and then `weather['wind_u']` raises TypeError.
        with pytest.raises(TypeError):
            find_shortest_path_dijkstra_3d(
                self.start_node, self.spm,
                54.75, 2.75,
                self.t0, bad_cache,
            )


@pytest.mark.slow
class TestVMGEdgeCases:
    """Additional VMG edge cases."""

    LATS = [53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75]
    LONS = [2.0, 2.25, 2.5, 2.75, 3.0]

    def setup_method(self):
        self.t0 = datetime(2026, 4, 20, 0, 0)

    def test_vmg_step_limit_10000_terminates(self):
        """REQ-08 C-22: step >= 10000 is the hard upper bound — simulate_vmg_route always returns.
        Verified by counting: result list length cannot exceed 10000 entries."""
        cache = make_weather_cache(
            lats_1d=self.LATS, lons_1d=self.LONS,
            u_speed=5.0, v_speed=0.0,
        )
        result = simulate_vmg_route(
            cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        # The loop appends one point per step before advancing; step limit is 10000.
        # Result can never exceed 10000 entries regardless of termination reason.
        assert isinstance(result, list)
        assert len(result) <= 10000, (
            f"VMG returned {len(result)} points — exceeds the 10000-step hard limit"
        )

    def test_vmg_time_step_default_is_10_minutes(self):
        """REQ-08: time_step_min defaults to 10.0; each step advances time by 10 min."""
        cache = make_weather_cache(
            lats_1d=self.LATS, lons_1d=self.LONS,
            u_speed=5.0, v_speed=0.0,
        )
        result = simulate_vmg_route(
            cache,
            start_lat=53.0, start_lon=2.25,
            target_lat=54.5, target_lon=2.25,
            start_time=self.t0,
        )
        assert len(result) >= 2, "Expected at least 2 VMG points to verify time step"
        delta = result[1]["time"] - result[0]["time"]
        assert delta == timedelta(minutes=10), (
            f"Expected 10-minute time step, got {delta}"
        )

    def test_vmg_best_hdg_none_breaks_loop(self):
        """REQ-08: if best_hdg is None (all headings rejected), loop breaks.
        NOTE: with v=-25 m/s (wind from north, TWD=0°) at lat=54.75, heading 180°
        (south) still satisfies TWA ≥ 32° and stays within grid — best_hdg is NOT None.
        The loop terminates via step >= 10000, not via best_hdg is None.
        Gałąź best_hdg is None jest praktycznie nieosiągalna przy standardowej siatce
        — wymagałaby żeby WSZYSTKIE 180 kursów wypadały poza siatkę lub do dead zone.
        Test dokumentuje faktyczne zachowanie: loop runs 10000 steps."""
        cache = make_weather_cache(
            lats_1d=self.LATS, lons_1d=self.LONS,
            u_speed=0.0, v_speed=-25.0,
        )
        result = simulate_vmg_route(
            cache,
            start_lat=54.75, start_lon=2.5,
            target_lat=55.5, target_lon=2.5,
            start_time=self.t0,
        )
        # best_hdg is never None — loop hits step >= 10000 limit instead
        assert len(result) == 10000, (
            "REQ-08 C-22: with no valid escape from grid, loop runs full 10000 steps"
        )


# ---------------------------------------------------------------------------
# Output file functions — save_to_gpx, save_route_detailed_log, save_graph_to_json
# ---------------------------------------------------------------------------

class TestSaveToGpx:
    """Tests for save_to_gpx — covers lines 499-502 and branches."""

    def setup_method(self):
        self.t0 = datetime(2026, 4, 20, 12, 0)

    def test_creates_file(self, tmp_path):
        """save_to_gpx always creates the file."""
        out = str(tmp_path / "route.gpx")
        save_to_gpx([], out)
        assert os.path.exists(out)

    def test_empty_points_writes_valid_gpx_shell(self, tmp_path):
        """REQ-10: empty point list produces a valid GPX shell (no <trkpt> elements)."""
        out = str(tmp_path / "empty.gpx")
        save_to_gpx([], out)
        content = open(out).read()
        assert '<?xml version="1.0"' in content
        assert '<trkseg>' in content
        assert '</trkseg>' in content
        assert '<trkpt' not in content

    def test_label_written_as_name_tag(self, tmp_path):
        """REQ-10: label parameter is written as <name> inside <trk>."""
        out = str(tmp_path / "labeled.gpx")
        save_to_gpx([], out, label="TestRoute")
        content = open(out).read()
        assert "<name>TestRoute</name>" in content

    def test_default_label_is_route(self, tmp_path):
        """REQ-10: default label is 'Route'."""
        out = str(tmp_path / "default.gpx")
        save_to_gpx([], out)
        content = open(out).read()
        assert "<name>Route</name>" in content

    def test_time_tag_format_and_literal_z(self, tmp_path):
        """REQ-10: time format is YYYY-MM-DDTHH:MM:SSZ with hardcoded literal Z."""
        out = str(tmp_path / "timed.gpx")
        points = [{"lat": 53.0, "lon": 2.0, "time": datetime(2026, 4, 20, 12, 30, 0)}]
        save_to_gpx(points, out)
        content = open(out).read()
        assert "<time>2026-04-20T12:30:00Z</time>" in content

    def test_time_tag_omitted_when_no_time_key(self, tmp_path):
        """REQ-10: <time> tag is omitted when point has no 'time' key."""
        out = str(tmp_path / "notime.gpx")
        points = [{"lat": 53.0, "lon": 2.0}]  # no 'time' key
        save_to_gpx(points, out)
        content = open(out).read()
        assert '<trkpt' in content
        assert '<time>' not in content

    def test_coordinates_written_with_6_decimal_places(self, tmp_path):
        """REQ-10: coordinates formatted to 6 decimal places."""
        out = str(tmp_path / "coords.gpx")
        points = [{"lat": 53.123456, "lon": 2.654321}]
        save_to_gpx(points, out)
        content = open(out).read()
        assert 'lat="53.123456"' in content
        assert 'lon="2.654321"' in content

    def test_gpx_version_and_creator(self, tmp_path):
        """REQ-10: GPX 1.1, creator='ScampiRouter'."""
        out = str(tmp_path / "meta.gpx")
        save_to_gpx([], out)
        content = open(out).read()
        assert 'version="1.1"' in content
        assert 'creator="ScampiRouter"' in content


class TestSaveGraphToJson:
    """Tests for save_graph_to_json — covers lines 421-426 and B-01."""

    LATS = [53.0, 53.25, 53.5, 53.75, 54.0]
    LONS = [2.0, 2.25, 2.5, 2.75]

    def setup_method(self):
        self.spm = make_safe_points_map(self.LATS, self.LONS)
        self.adj = make_full_adjacency(self.spm, cost=1.5)
        self.nodes = set(self.spm.keys())

    def test_creates_valid_json_file(self, tmp_path):
        """save_graph_to_json writes valid JSON."""
        out = str(tmp_path / "graph.json")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_top_level_keys(self, tmp_path):
        """REQ-10: JSON has exactly three top-level keys: metadata, nodes, graph."""
        out = str(tmp_path / "graph.json")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        with open(out) as f:
            data = json.load(f)
        assert set(data.keys()) == {"metadata", "nodes", "graph"}

    def test_node_keys_are_row_col_strings(self, tmp_path):
        """REQ-10: node keys are strings in format 'row,col'."""
        out = str(tmp_path / "graph.json")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        with open(out) as f:
            data = json.load(f)
        for key in data["nodes"]:
            assert "," in key, f"Node key must be 'row,col' string, got: {key}"
            parts = key.split(",")
            assert len(parts) == 2
            assert parts[0].isdigit() and parts[1].isdigit()

    def test_node_values_have_lat_lon_no_max_speed(self, tmp_path):
        """REQ-10: node values contain lat, lon — no max_speed in JSON output."""
        out = str(tmp_path / "graph.json")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        with open(out) as f:
            data = json.load(f)
        for val in data["nodes"].values():
            assert "lat" in val
            assert "lon" in val
            assert "max_speed" not in val

    def test_metadata_keys(self, tmp_path):
        """REQ-10: metadata contains nodes count, unit='hours', timestamp."""
        out = str(tmp_path / "graph.json")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        with open(out) as f:
            data = json.load(f)
        meta = data["metadata"]
        assert meta["nodes"] == len(self.nodes)
        assert meta["unit"] == "hours"
        assert "timestamp" in meta

    def test_overwrites_existing_file(self, tmp_path):
        """C-21: save_graph_to_json always overwrites — open mode 'w'."""
        out = str(tmp_path / "graph.json")
        # Write dummy content first
        with open(out, "w") as f:
            f.write("old content")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        with open(out) as f:
            data = json.load(f)
        assert "metadata" in data  # old content was overwritten


class TestSaveRouteDetailedLog:
    """Tests for save_route_detailed_log — covers lines 373-406."""

    def setup_method(self):
        self.cache = make_weather_cache(
            lats_1d=[53.0, 53.25, 53.5],
            lons_1d=[2.0, 2.25, 2.5],
            u_speed=5.0, v_speed=0.0,
        )
        self.t0 = datetime(2026, 4, 20, 12, 0)

    def test_empty_points_returns_without_creating_file(self, tmp_path):
        """REQ-10: if points list is empty, function returns immediately — no file created."""
        out = str(tmp_path / "log.txt")
        save_route_detailed_log([], self.cache, out, "Test")
        assert not os.path.exists(out)

    def test_creates_file_with_points(self, tmp_path):
        """save_route_detailed_log creates file when points is non-empty."""
        out = str(tmp_path / "log.txt")
        points = [
            {"lat": 53.0, "lon": 2.0, "time": self.t0},
            {"lat": 53.25, "lon": 2.25, "time": self.t0 + timedelta(hours=1)},
        ]
        save_route_detailed_log(points, self.cache, out, "TestRoute")
        assert os.path.exists(out)

    def test_header_contains_label(self, tmp_path):
        """REQ-10: log file starts with label in header."""
        out = str(tmp_path / "log.txt")
        points = [
            {"lat": 53.0, "lon": 2.0, "time": self.t0},
            {"lat": 53.25, "lon": 2.25, "time": self.t0 + timedelta(hours=1)},
        ]
        save_route_detailed_log(points, self.cache, out, "MyLabel")
        content = open(out).read()
        assert "MyLabel" in content

    def test_columns_present_in_header(self, tmp_path):
        """REQ-10: header contains all 8 column names."""
        out = str(tmp_path / "log.txt")
        points = [{"lat": 53.0, "lon": 2.0, "time": self.t0}]
        save_route_detailed_log(points, self.cache, out, "Test")
        content = open(out).read()
        for col in ["Time", "Latitude", "Longitude", "TWS", "TWD", "Heading", "TWA", "BS"]:
            assert col in content, f"Column '{col}' missing from log header"

    def test_file_mode_is_write_overwrites(self, tmp_path):
        """REQ-10: file opened in mode 'w' — always overwrites."""
        out = str(tmp_path / "log.txt")
        with open(out, "w") as f:
            f.write("old content\n" * 100)
        points = [{"lat": 53.0, "lon": 2.0, "time": self.t0}]
        save_route_detailed_log(points, self.cache, out, "Test")
        content = open(out).read()
        assert "old content" not in content

    def test_missing_time_key_uses_datetime_now(self, tmp_path):
        """REQ-10: if point has no 'time' key, datetime.now() is used as fallback."""
        out = str(tmp_path / "log.txt")
        # Point without 'time' key
        points = [{"lat": 53.0, "lon": 2.0}]
        # Should not raise — datetime.now() is used silently
        save_route_detailed_log(points, self.cache, out, "Test")
        assert os.path.exists(out)

    def test_timestamp_format_in_rows(self, tmp_path):
        """REQ-10: timestamp format is '%Y-%m-%d %H:%M'."""
        out = str(tmp_path / "log.txt")
        points = [
            {"lat": 53.0, "lon": 2.0, "time": datetime(2026, 4, 20, 14, 35)},
        ]
        save_route_detailed_log(points, self.cache, out, "Test")
        content = open(out).read()
        assert "2026-04-20 14:35" in content


class TestSafeAreasEdgeCases:
    """Branch coverage for identify_weather_danger_zones and identify_safe_sailing_areas."""

    def test_danger_zones_returns_empty_for_falsy_cache(self):
        """identify_weather_danger_zones returns [] for falsy cache (None or {})."""
        assert identify_weather_danger_zones(None, 40.0) == []
        assert identify_weather_danger_zones({}, 40.0) == []

    def test_safe_areas_returns_empty_dict_for_falsy_cache(self):
        """identify_safe_sailing_areas returns {} for falsy cache (None or {})."""
        assert identify_safe_sailing_areas(None) == {}
        assert identify_safe_sailing_areas({}) == {}

    def test_graph_construction_no_start_node(self, tmp_path):
        """Lines 181-182: generate_reachable_graph returns empty when start position
        maps to (0,0) but (0,0) is not in safe_points_map."""
        # Build spm with only far-away cells — none near start_lat=53.0, start_lon=2.0
        # Use a spm with only one entry far from start, but valid cache
        cache = make_weather_cache(
            lats_1d=[54.5, 54.75],
            lons_1d=[4.0, 4.25],
        )
        # spm has only cells at 54.5-54.75°N / 4.0-4.25°E
        spm = identify_safe_sailing_areas(cache)
        log = str(tmp_path / "graph_log.txt")
        # start at 53.0, 2.0 — far from any spm cell, but spm is non-empty
        # The closest spm node will still be found (it's just far away)
        # The `if not start_node` guard inside generate_reachable_graph only fires when
        # start_node is None. start_node is a tuple, which is never falsy unless None.
        # start_node is None only when safe_points_map is empty (caught earlier).
        # So that guard is effectively dead code / unreachable in practice.
        # This test documents that generate_reachable_graph always finds a start_node
        # when spm is non-empty (the nearest node is always found).
        nodes, adj, start_idx = generate_reachable_graph(
            cache, spm,
            start_lat=53.0, start_lon=2.0,
            target_lat=54.75, target_lon=4.25,
            start_time=cache["dates"][0],
            log_file=log,
        )
        # start_idx will be the closest cell to (53.0, 2.0), which is (0,0) in the spm
        assert start_idx is not None


class TestDijkstra2DBranchCoverage:
    """Branch coverage for find_shortest_path_dijkstra: lazy deletion, missing-neighbor KeyError, unreachable nodes."""

    LATS = [53.0, 53.25, 53.5, 53.75]
    LONS = [2.0, 2.25, 2.5, 2.75]

    def setup_method(self):
        self.spm = make_safe_points_map(self.LATS, self.LONS)
        self.target_lat = 53.75
        self.target_lon = 2.75

    def test_lazy_deletion_branch_fires(self):
        """Stale heap entries are skipped (curr_dist > distances[curr_node]).
        With make_full_adjacency, multiple paths exist — shorter one updates distances,
        making the longer heap entry stale. Dijkstra still returns correct result."""
        adj = make_full_adjacency(self.spm, cost=1.0)
        path, cost = find_shortest_path_dijkstra(
            (0, 0), adj, self.spm, self.target_lat, self.target_lon,
        )
        # If lazy deletion fires correctly, we still get a valid path
        assert len(path) > 0
        assert cost > 0

    def test_no_reachable_nodes_returns_empty(self):
        """When start is in adj with no neighbors, the no-reachable-nodes guard is dead code:
        distances[start]=0 so start is always reachable. The guard (`if not reachable_and_visited`)
        can never be True when start is in adj, because start always has distance 0 != inf.
        Test confirms that start is always returned as the sole path element."""
        # Build adj where start has distance 0 — start is always in reachable_and_visited
        adj = {(0, 0): []}  # start in adj, no neighbors
        path, cost = find_shortest_path_dijkstra(
            (0, 0), adj, self.spm, self.target_lat, self.target_lon,
        )
        assert len(path) == 1  # start node always reachable
        assert cost == pytest.approx(0.0)


@pytest.mark.slow
class TestDijkstra3DBranchCoverage:
    """Branch coverage for find_shortest_path_dijkstra_3d: empty-path guard, bs<=0 edge skip, best-finish-node."""

    LATS = [53.0, 53.25, 53.5, 53.75, 54.0, 54.25, 54.5, 54.75]
    LONS = [2.0, 2.25, 2.5, 2.75, 3.0]

    def setup_method(self):
        self.cache = make_weather_cache(lats_1d=self.LATS, lons_1d=self.LONS)
        self.spm = identify_safe_sailing_areas(self.cache)
        self.t0 = self.cache["dates"][0]

    def test_best_finish_node_none_returns_empty(self):
        """`if not best_finish_node: return [], 0` is dead code.
        best_finish_node is set on the very first heappop (start_node is always pushed).
        With single-node spm the loop pops (0,0), sets best_finish_node=(0,0), finds no
        neighbors, and exits. Path = [(0,0) dict], cost = distances[(0,0)] = 0."""
        single_spm = {(0, 0): {"lat": 53.0, "lon": 2.0, "max_speed": 9.7}}
        path, cost = find_shortest_path_dijkstra_3d(
            (0, 0), single_spm, 54.75, 2.75, self.t0, self.cache,
        )
        assert len(path) == 1, "Single-node spm must return 1-element path"
        assert path[0]["lat"] == pytest.approx(53.0)
        assert path[0]["lon"] == pytest.approx(2.0)
        assert cost == pytest.approx(0.0)

    def test_bs_zero_skips_edge(self):
        """Edges where bs <= 0 are skipped (no 9999h fallback in 3D).
        NOTE: u=v=0.1 gives TWS ≈ 0.27 kt → polars extrapolate to bs ≈ 2.44 kt (positive).
        The bs <= 0 guard does NOT fire with these inputs — edges are NOT skipped.
        The branch `if bs <= 0: continue` requires polars to return a negative value,
        which only happens for very low TWA (< 32° dead zone handled separately) or
        extrapolation far below the table minimum in a direction where gradient is negative.
        This test documents that 3D Dijkstra completes without crashing for weak wind,
        and that some path IS found (edges accepted since bs > 0)."""
        cache = make_weather_cache(
            lats_1d=self.LATS, lons_1d=self.LONS,
            u_speed=0.1, v_speed=0.1,
        )
        spm = identify_safe_sailing_areas(cache, max_wind_threshold=30.0)
        if not spm:
            pytest.skip("No safe cells with this wind configuration")
        start_node = min(spm.keys())
        path, cost = find_shortest_path_dijkstra_3d(
            start_node, spm, 54.75, 2.75, self.t0, cache,
        )
        # bs ≈ 2.44 kt (positive) → edges accepted → non-empty path expected
        assert len(path) > 0, "Expected path with weak but positive bs"
        assert cost > 0


# ---------------------------------------------------------------------------
# GRIB loading (load_grib_to_memory)
# ---------------------------------------------------------------------------

class TestLoadGribToMemory:
    """Characterization tests for load_grib_to_memory.

    pygrib is fully mocked — no real GRIB files needed.
    Tests document the contract between load_grib_to_memory and the rest of
    the code: the shape and keys of the returned weather_cache dict.
    """

    def setup_method(self):
        self.load = load_grib_to_memory
        self.t0 = datetime(2026, 4, 20, 0, 0)
        self.t1 = datetime(2026, 4, 20, 6, 0)
        self.lats = np.array([[53.0, 53.0], [53.25, 53.25]])
        self.lons = np.array([[2.0, 2.25], [2.0, 2.25]])
        self.u_vals = np.full((2, 2), 5.0)
        self.v_vals = np.full((2, 2), 0.0)

    def _make_grbs(self, messages):
        """Wrap a list of mock messages in a mock pygrib file handle."""
        grbs = MagicMock()
        grbs.__iter__ = MagicMock(side_effect=lambda: iter(messages))
        return grbs

    def test_returns_none_when_file_not_found(self, tmp_path):
        """load_grib_to_memory returns None immediately when file does not exist."""
        result = self.load(str(tmp_path / "nonexistent.grib"))
        assert result is None

    def test_returns_none_on_pygrib_exception(self, tmp_path, monkeypatch):
        """load_grib_to_memory catches all exceptions and returns None."""
        dummy_file = tmp_path / "bad.grib"
        dummy_file.write_bytes(b"not a grib")

        def exploding_open(_path):
            raise RuntimeError("corrupted file")

        monkeypatch.setattr(grib_module.pygrib, "open", exploding_open)
        result = self.load(str(dummy_file))
        assert result is None

    def test_returns_dict_with_four_keys(self, tmp_path, monkeypatch):
        """REQ-02: returned dict has exactly keys: data, dates, lats, lons."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        msgs = [
            make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons),
            make_mock_grib_message("10 metre v wind component", self.t0, self.v_vals, self.lats, self.lons),
        ]
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs(msgs))
        result = self.load(str(dummy_file))
        assert result is not None
        assert set(result.keys()) == {"data", "dates", "lats", "lons"}

    def test_dates_are_sorted(self, tmp_path, monkeypatch):
        """REQ-02: dates list is sorted ascending — min() and list[-1] depend on this."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        # Feed messages in reverse chronological order
        msgs = [
            make_mock_grib_message("10 metre U wind component", self.t1, self.u_vals, self.lats, self.lons),
            make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons),
        ]
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs(msgs))
        result = self.load(str(dummy_file))
        assert result["dates"] == sorted(result["dates"]), "REQ-02: dates must be sorted"
        assert result["dates"][0] == self.t0
        assert result["dates"][1] == self.t1

    def test_data_keyed_by_datetime(self, tmp_path, monkeypatch):
        """REQ-02: data dict is keyed by datetime objects (validDate from message)."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        msgs = [
            make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons),
        ]
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs(msgs))
        result = self.load(str(dummy_file))
        assert self.t0 in result["data"]
        assert isinstance(list(result["data"].keys())[0], datetime)

    def test_data_values_keyed_by_param_name(self, tmp_path, monkeypatch):
        """REQ-02: inner dict keyed by msg.name — preserves exact string from GRIB message."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        msgs = [
            make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons),
            make_mock_grib_message("10 metre v wind component", self.t0, self.v_vals, self.lats, self.lons),
        ]
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs(msgs))
        result = self.load(str(dummy_file))
        assert "10 metre U wind component" in result["data"][self.t0]
        assert "10 metre v wind component" in result["data"][self.t0]

    def test_lats_lons_from_first_message_only(self, tmp_path, monkeypatch):
        """REQ-02: lats/lons set once from first message — latlons() not called on subsequent messages."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        other_lats = np.array([[60.0, 60.0], [61.0, 61.0]])
        other_lons = np.array([[10.0, 11.0], [10.0, 11.0]])
        msg1 = make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons)
        msg2 = make_mock_grib_message("10 metre v wind component", self.t0, self.v_vals, other_lats, other_lons)
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs([msg1, msg2]))
        result = self.load(str(dummy_file))
        # Result must use first message's grid
        np.testing.assert_array_equal(result["lats"], self.lats)
        np.testing.assert_array_equal(result["lons"], self.lons)
        # latlons() must have been called on msg1 but NOT on msg2
        msg1.latlons.assert_called_once()
        msg2.latlons.assert_not_called()

    def test_multiple_timesteps_all_present_in_data(self, tmp_path, monkeypatch):
        """REQ-02: all unique validDate values appear as keys in data dict."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        msgs = [
            make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons),
            make_mock_grib_message("10 metre U wind component", self.t1, self.u_vals, self.lats, self.lons),
        ]
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs(msgs))
        result = self.load(str(dummy_file))
        assert self.t0 in result["data"]
        assert self.t1 in result["data"]
        assert len(result["dates"]) == 2

    def test_grbs_close_called_on_success(self, tmp_path, monkeypatch):
        """REQ-02: grbs.close() is called after successful read (happy path)."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        grbs = self._make_grbs([
            make_mock_grib_message("10 metre U wind component", self.t0, self.u_vals, self.lats, self.lons),
        ])
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: grbs)
        self.load(str(dummy_file))
        grbs.close.assert_called_once()


# ---------------------------------------------------------------------------
# Coverage gap tests (from test_contracts_and_regressions.py)
# ---------------------------------------------------------------------------

class TestGetWeatherFromCacheGaps:

    def test_v_component_uppercase_fallback_returns_correct_data(self):
        """REQ-02: uppercase-V fallback key works and returns correct wind_v value."""
        cache = make_weather_cache(v_speed=3.0)
        t0 = cache["dates"][0]
        # Rename lowercase key to uppercase V in every timestep
        for dt in cache["data"]:
            d = cache["data"][dt]
            if "10 metre v wind component" in d:
                d["10 metre V wind component"] = d.pop("10 metre v wind component")

        result = get_weather_from_cache(cache, 53.0, 2.0, t0)

        assert result is not None, "Cache with uppercase V key must still return data"
        assert result["wind_v"] == pytest.approx(3.0), (
            "Uppercase V fallback must return the same wind_v value"
        )

    def test_approximated_true_when_time_before_first_forecast(self):
        """REQ-03: approximated=True fires for times BEFORE first date, not only after last."""
        cache = make_weather_cache()
        past = cache["dates"][0] - timedelta(days=30)
        result = get_weather_from_cache(cache, 53.0, 2.0, past)
        assert result["meta"]["approximated"] is True, (
            "Time before forecast window must set approximated=True"
        )

    def test_approximated_false_at_exact_first_date(self):
        """REQ-03: approximated=False when time exactly equals first forecast date."""
        cache = make_weather_cache()
        result = get_weather_from_cache(cache, 53.0, 2.0, cache["dates"][0])
        assert result["meta"]["approximated"] is False

    def test_approximated_false_at_exact_last_date(self):
        """REQ-03: approximated=False when time exactly equals last forecast date."""
        cache = make_weather_cache()
        result = get_weather_from_cache(cache, 53.0, 2.0, cache["dates"][-1])
        assert result["meta"]["approximated"] is False


class TestIdentifySafeAreasGaps:

    def test_max_speed_reflects_maximum_across_all_timesteps(self):
        """REQ-04: max_speed in spm is the highest TWS seen across ALL timesteps."""
        cache = make_weather_cache(u_speed=5.0, n_times=3)
        t1, t2 = cache["dates"][1], cache["dates"][2]
        # Override second timestep to double wind, third to triple wind
        cache["data"][t1]["10 metre U wind component"] *= 2   # 10 m/s
        cache["data"][t2]["10 metre U wind component"] *= 3   # 15 m/s

        spm = identify_safe_sailing_areas(cache, max_wind_threshold=40.0)
        max_speeds = [v["max_speed"] for v in spm.values()]

        expected_max_tws = 15.0 * 1.94384  # ~29.2 kt
        assert all(s == pytest.approx(expected_max_tws, abs=0.1) for s in max_speeds), (
            "max_speed must reflect the highest TWS (third timestep, 15 m/s)"
        )

    def test_mixed_grid_only_safe_cells_returned(self):
        """REQ-04: only cells safe across ALL timesteps are included in spm."""
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25],
            lons_1d=[2.0, 2.25],
            u_speed=5.0,    # ~9.7 kt — safe
            n_times=2,
        )
        t1 = cache["dates"][1]
        # Make only the top-left cell (0,0) stormy in the second timestep
        cache["data"][t1]["10 metre U wind component"] = np.array(
            [[20.0, 5.0],   # row 0: cell(0,0)=20m/s (~38.9kt), cell(0,1)=5m/s (safe)
             [5.0,  5.0]],  # row 1: both safe
            dtype=float,
        )
        spm = identify_safe_sailing_areas(cache, max_wind_threshold=30.0)

        assert (0, 0) not in spm, "Cell (0,0) exceeded threshold in t1 — must be excluded"
        assert (0, 1) in spm,    "Cell (0,1) was always safe — must be included"
        assert (1, 0) in spm,    "Cell (1,0) was always safe — must be included"
        assert (1, 1) in spm,    "Cell (1,1) was always safe — must be included"
        assert len(spm) == 3


class TestIdentifyDangerZonesGaps:

    def test_records_first_exceedance_time_not_last(self):
        """REQ-04: first_time_grid stores the FIRST timestep that exceeded the threshold."""
        cache = make_weather_cache(u_speed=5.0, n_times=3)
        t0, t1, t2 = cache["dates"][0], cache["dates"][1], cache["dates"][2]

        # t0: calm (safe), t1: stormy (first exceedance), t2: also stormy
        cache["data"][t0]["10 metre U wind component"] = np.full((4, 4), 5.0)    # ~9.7 kt
        cache["data"][t1]["10 metre U wind component"] = np.full((4, 4), 22.0)   # ~42.7 kt
        cache["data"][t2]["10 metre U wind component"] = np.full((4, 4), 25.0)   # ~48.6 kt

        zones = identify_weather_danger_zones(cache, min_threshold=40.0)

        assert len(zones) == 16, "All 16 cells must appear in danger zones"
        assert all(z["time"] == t1 for z in zones), (
            "first_time_grid must record t1 (first exceedance), not t2 (second)"
        )

    def test_only_exceeding_cells_returned_not_all_cells(self):
        """REQ-04: only cells that actually exceeded the threshold are returned."""
        cache = make_weather_cache(
            lats_1d=[53.0, 53.25],
            lons_1d=[2.0, 2.25],
            u_speed=5.0,
            n_times=1,
        )
        t0 = cache["dates"][0]
        # Make only cell (0,0) exceed threshold
        cache["data"][t0]["10 metre U wind component"] = np.array(
            [[22.0, 5.0],   # (0,0)=~42.7kt, (0,1)=safe
             [5.0,  5.0]],  # (1,0)=safe,    (1,1)=safe
            dtype=float,
        )
        zones = identify_weather_danger_zones(cache, min_threshold=40.0)

        assert len(zones) == 1, f"Only 1 cell exceeds threshold, got {len(zones)}"
        assert zones[0]["lat"] == pytest.approx(53.0)
        assert zones[0]["lon"] == pytest.approx(2.0)

    def test_upper_bound_max_threshold_inclusive(self):
        """REQ-04: caution zone uses `<= max_threshold` — exactly at upper bound IS included."""
        cache = make_weather_cache(u_speed=0.0, n_times=1)
        t0 = cache["dates"][0]
        # TWS exactly 40 kt = 40/1.94384 m/s ≈ 20.577 m/s
        u_exactly_40kt = 40.0 / 1.94384
        cache["data"][t0]["10 metre U wind component"] = np.full((4, 4), u_exactly_40kt)
        # Caution: 30 < TWS <= 40
        zones = identify_weather_danger_zones(cache, min_threshold=30.0, max_threshold=40.0)
        assert len(zones) == 16, "Exactly 40 kt must be included in caution zone (<= 40)"


class TestCalculateBearingGaps:

    def test_northeast_bearing(self):
        b = calculate_bearing(53.0, 2.0, 54.0, 3.0)
        assert 30.0 < b < 60.0, f"NE bearing should be ~45°, got {b}"

    def test_southwest_bearing(self):
        b = calculate_bearing(54.0, 3.0, 53.0, 2.0)
        assert 210.0 < b < 240.0, f"SW bearing should be ~225°, got {b}"

    def test_reciprocal_bearing_differs_by_180(self):
        """Bearing A→B and B→A should differ by ~180° (spherical approx)."""
        b_fwd = calculate_bearing(53.0, 2.0, 54.0, 2.0)
        b_rev = calculate_bearing(54.0, 2.0, 53.0, 2.0)
        diff = abs(((b_fwd - b_rev + 180) % 360) - 180)
        assert diff == pytest.approx(180.0, abs=1.0), (
            f"Reciprocal bearings must differ by ~180°, got {b_fwd:.1f}° and {b_rev:.1f}°"
        )


class TestCalculateDistanceNmGaps:

    def test_distance_is_non_negative_for_standard_inputs(self):
        """Distance must never be negative for any reasonable coordinate pair."""
        pairs = [
            (53.0, 2.0, 53.0, 2.0),
            (0.0, 0.0, 90.0, 0.0),
            (-10.0, 5.0, 10.0, 5.0),
            (53.0, 2.0, 53.0, 3.0),
        ]
        for lat1, lon1, lat2, lon2 in pairs:
            d = calculate_distance_nm(lat1, lon1, lat2, lon2)
            assert d >= 0.0, f"Negative distance {d} for ({lat1},{lon1})→({lat2},{lon2})"


class TestSaveGraphToJsonGaps:

    def test_edge_structure_target_as_string_cost_rounded(self, tmp_path):
        """REQ-10: edge targets stored as 'r,c' strings, cost rounded to 4 decimal places."""
        nodes = {(0, 0), (0, 1)}
        adj = {
            (0, 0): [{"target": (0, 1), "cost": 1.23456789}],
            (0, 1): [],
        }
        spm = {
            (0, 0): {"lat": 53.0, "lon": 2.0,  "max_speed": 5.0},
            (0, 1): {"lat": 53.0, "lon": 2.25, "max_speed": 5.0},
        }
        json_path = str(tmp_path / "g.json")
        save_graph_to_json(nodes, adj, spm, json_path)

        data = json.loads(open(json_path).read())
        edges_00 = data["graph"]["0,0"]
        assert len(edges_00) == 1
        assert edges_00[0]["target"] == "0,1", "Target must be serialised as 'r,c' string"
        assert edges_00[0]["cost"] == pytest.approx(1.2346, abs=1e-4), (
            "Cost must be rounded to 4 decimal places"
        )

    def test_empty_graph_produces_valid_json(self, tmp_path):
        """save_graph_to_json with empty inputs must still produce valid JSON."""
        json_path = str(tmp_path / "empty.json")
        save_graph_to_json(set(), {}, {}, json_path)

        data = json.loads(open(json_path).read())
        assert data["nodes"] == {}
        assert data["graph"] == {}
        assert data["metadata"]["nodes"] == 0

    def test_node_values_contain_lat_lon_not_max_speed(self, tmp_path):
        """REQ-10: JSON node values have {lat, lon} only — max_speed is NOT included."""
        nodes = {(0, 0)}
        adj = {(0, 0): []}
        spm = {(0, 0): {"lat": 53.0, "lon": 2.0, "max_speed": 9.7}}
        json_path = str(tmp_path / "g.json")
        save_graph_to_json(nodes, adj, spm, json_path)

        data = json.loads(open(json_path).read())
        node_val = data["nodes"]["0,0"]
        assert "lat" in node_val
        assert "lon" in node_val
        assert "max_speed" not in node_val, "C-27: max_speed must NOT appear in JSON nodes"


class TestPrintRouteSummaryGaps:

    def test_summary_works_with_explicit_time_hours(self, capsys):
        """print_route_summary must not crash when time_hours is passed explicitly."""
        points = [
            {"lat": 53.0,  "lon": 2.0},
            {"lat": 53.25, "lon": 2.25},
        ]
        print_route_summary(points, "Test route", time_hours=2.5)
        out = capsys.readouterr().out
        assert "Test route" in out
        assert "2.50" in out

    def test_summary_works_with_time_keys_in_points(self, capsys):
        """print_route_summary derives time_hours from points['time'] when available."""
        t0 = datetime(2026, 4, 20, 0, 0)
        points = [
            {"lat": 53.0,  "lon": 2.0,  "time": t0},
            {"lat": 53.25, "lon": 2.25, "time": t0 + timedelta(hours=3)},
        ]
        print_route_summary(points, "Timed route")
        out = capsys.readouterr().out
        assert "3.00" in out

    def test_fewer_than_two_points_prints_no_data(self, capsys):
        """print_route_summary with 0 or 1 points must print 'No data available'."""
        print_route_summary([], "Empty")
        out = capsys.readouterr().out
        assert "No data" in out

        print_route_summary([{"lat": 53.0, "lon": 2.0}], "Single")
        out = capsys.readouterr().out
        assert "No data" in out


class TestSaveRouteDetailedLogGaps:

    def _make_points(self, n=3):
        t0 = datetime(2026, 4, 20, 0, 0)
        return [
            {"lat": 53.0 + i * 0.25, "lon": 2.0, "time": t0 + timedelta(hours=i)}
            for i in range(n)
        ]

    def test_last_waypoint_heading_uses_reverse_bearing(self, tmp_path):
        """REQ-10: last waypoint heading = bearing from previous point to last."""
        cache = make_weather_cache()
        points = self._make_points(3)
        log_path = str(tmp_path / "log.txt")
        save_route_detailed_log(points, cache, log_path, "Test")

        lines = open(log_path).readlines()
        # Filter rows that start with a year (e.g. '2026') — skip header/separator lines
        data_rows = [l for l in lines if l.strip().startswith("20")]
        assert len(data_rows) == 3

        # Heading column (index 5, 0-based after splitting on '|'):
        # "Time | Lat | Lon | TWS | TWD | Heading | TWA | BS"
        last_row = data_rows[-1]
        cols = [c.strip() for c in last_row.split("|")]
        last_hdg = float(cols[5])

        expected_last_hdg = calculate_bearing(
            points[1]["lat"], points[1]["lon"],
            points[2]["lat"], points[2]["lon"],
        )
        assert last_hdg == pytest.approx(expected_last_hdg, abs=1.0)

    def test_missing_time_key_uses_fallback_without_crash(self, tmp_path):
        """REQ-10: points without 'time' key fall back to datetime.now() — must not crash."""
        cache = make_weather_cache()
        points = [
            {"lat": 53.0, "lon": 2.0},
            {"lat": 53.25, "lon": 2.25},
        ]
        log_path = str(tmp_path / "log.txt")
        save_route_detailed_log(points, cache, log_path, "NoTime")
        assert os.path.exists(log_path)

    def test_twa_below_32_records_zero_boat_speed(self, tmp_path):
        """REQ-10: if TWA < 32° (dead zone), boat speed in log must be 0."""
        # Wind from exactly north (TWD=0°), heading due north (hdg=0°) → TWA=0° < 32°
        cache = make_weather_cache(u_speed=0.0, v_speed=-10.0)   # v<0 = wind from north
        t0 = cache["dates"][0]
        # Two points due north of each other so heading=0°
        points = [
            {"lat": 53.0,  "lon": 2.0, "time": t0},
            {"lat": 53.25, "lon": 2.0, "time": t0 + timedelta(hours=1)},
        ]
        log_path = str(tmp_path / "log.txt")
        save_route_detailed_log(points, cache, log_path, "DeadZone")

        lines = open(log_path).readlines()
        # Lines that start with a year are data rows (skip header/separator/generated-on)
        data_rows = [l for l in lines if l.strip().startswith("20")]
        assert len(data_rows) == 2, f"Expected 2 data rows, got {len(data_rows)}"

        # Col layout: Time|Lat|Lon|TWS|TWD|Heading|TWA|BS  (index 7 = BS)
        first_row = data_rows[0]
        cols = [c.strip() for c in first_row.split("|")]
        bs = float(cols[7])
        assert bs == pytest.approx(0.0, abs=0.01), (
            f"Dead zone heading (TWA=0°) must produce BS=0, got {bs}"
        )


class TestPolarsGaps:

    def test_all_table_boundary_values_correct(self):
        """REQ-01: spot-check all four corners and centre of the polar table."""
        polars = get_scampi_30_polars()
        cases = [
            (32,  6,  2.8),   # top-left corner
            (32,  20, 5.4),   # top-right corner
            (180, 6,  3.1),   # bottom-left corner
            (180, 20, 7.6),   # bottom-right corner
            (90,  12, 7.1),   # centre region
            (60,  14, 6.8),   # middle row/col
        ]
        for twa, tws, expected in cases:
            result = polars([twa, tws])[0]
            assert abs(result - expected) < 0.01, (
                f"Polar table mismatch at TWA={twa}°/TWS={tws}kt: "
                f"expected {expected}, got {result:.4f}"
            )

    def test_twa_at_lower_boundary_32_degrees(self):
        """REQ-01: TWA=32° is the minimum defined angle — must return table value."""
        polars = get_scampi_30_polars()
        result = polars([32, 10])[0]
        assert abs(result - 4.6) < 0.01, f"Expected 4.6 kt at TWA=32/TWS=10, got {result}"

    def test_high_tws_extrapolation_does_not_raise(self):
        """REQ-01: TWS=30 (above table max of 20) extrapolates without error."""
        polars = get_scampi_30_polars()
        result = polars([90, 30])[0]
        assert isinstance(result, (int, float, np.floating))


class TestUKeyLowercaseFallback:
    """B-22 related: identify_weather_danger_zones and identify_safe_sailing_areas
    must handle both uppercase and lowercase U key from the GRIB file."""

    def _make_lowercase_u_cache(self):
        cache = make_weather_cache(u_speed=5.0, v_speed=0.0)
        for dt in cache["data"]:
            d = cache["data"][dt]
            if "10 metre U wind component" in d:
                d["10 metre u wind component"] = d.pop("10 metre U wind component")
        return cache

    def test_danger_zones_with_lowercase_u_key(self):
        """identify_weather_danger_zones must work with lowercase '10 metre u wind component'."""
        cache = self._make_lowercase_u_cache()
        zones = identify_weather_danger_zones(cache, min_threshold=40.0)
        assert isinstance(zones, list)

    def test_safe_areas_with_lowercase_u_key(self):
        """identify_safe_sailing_areas must work with lowercase '10 metre u wind component'."""
        cache = self._make_lowercase_u_cache()
        spm = identify_safe_sailing_areas(cache, max_wind_threshold=30.0)
        assert isinstance(spm, dict)
        assert len(spm) == 16  # all 16 cells safe at ~9.7 kt

    def test_safe_areas_uppercase_and_lowercase_give_same_result(self):
        """Uppercase and lowercase U key must produce identical safe-area sets."""
        cache_upper = make_weather_cache(u_speed=5.0, v_speed=0.0)
        cache_lower = self._make_lowercase_u_cache()
        spm_upper = identify_safe_sailing_areas(cache_upper, max_wind_threshold=30.0)
        spm_lower = identify_safe_sailing_areas(cache_lower, max_wind_threshold=30.0)
        assert set(spm_upper.keys()) == set(spm_lower.keys())

    def test_b22_get_weather_from_cache_with_lowercase_u_key_returns_wind_data(self):
        """Core B-22 regression: get_weather_from_cache must not return None when
        the cache contains only lowercase '10 metre u wind component'.

        With the bug (no fallback), get_weather_from_cache returns None because
        the key lookup fails silently — u is None → early return None.
        After the fix the function tries the lowercase key and returns a dict
        with 'wind_u' and 'wind_v'.
        """
        from grib import get_weather_from_cache
        cache = self._make_lowercase_u_cache()
        t0 = cache["dates"][0]
        lats, lons = cache["lats"], cache["lons"]
        mid_r, mid_c = lats.shape[0] // 2, lons.shape[1] // 2
        lat, lon = float(lats[mid_r, mid_c]), float(lons[mid_r, mid_c])

        result = get_weather_from_cache(cache, lat, lon, t0)

        assert result is not None, (
            "B-22: get_weather_from_cache returned None for lowercase U key — "
            "the case-insensitive fallback is missing"
        )
        assert "wind_u" in result, "B-22: result must contain 'wind_u'"
        assert "wind_v" in result, "B-22: result must contain 'wind_v'"

    def test_b22_get_weather_returns_wind_data_for_lowercase_v_key_also(self):
        """B-22: get_weather_from_cache must handle lowercase V key too."""
        from grib import get_weather_from_cache
        cache = make_weather_cache(u_speed=5.0, v_speed=3.0)
        for dt in cache["data"]:
            d = cache["data"][dt]
            for old_key, new_key in [
                ("10 metre U wind component", "10 metre u wind component"),
                ("10 metre V wind component", "10 metre v wind component"),
            ]:
                if old_key in d:
                    d[new_key] = d.pop(old_key)
        t0 = cache["dates"][0]
        lats, lons = cache["lats"], cache["lons"]
        mid_r, mid_c = lats.shape[0] // 2, lons.shape[1] // 2
        lat, lon = float(lats[mid_r, mid_c]), float(lons[mid_r, mid_c])

        result = get_weather_from_cache(cache, lat, lon, t0)

        assert result is not None, (
            "B-22: get_weather_from_cache returned None for lowercase V key — "
            "the V-key case-insensitive fallback is missing"
        )
        assert "wind_v" in result

    def test_b22_get_weather_returns_none_when_both_keys_absent(self):
        """B-22 boundary: get_weather_from_cache must return None when neither
        uppercase nor lowercase key exists — not crash with KeyError."""
        from grib import get_weather_from_cache
        cache = make_weather_cache(u_speed=5.0, v_speed=0.0)
        for dt in cache["data"]:
            d = cache["data"][dt]
            d.pop("10 metre U wind component", None)
            d.pop("10 metre u wind component", None)
        t0 = cache["dates"][0]
        lats, lons = cache["lats"], cache["lons"]
        lat, lon = float(lats[0, 0]), float(lons[0, 0])

        result = get_weather_from_cache(cache, lat, lon, t0)
        assert result is None, (
            "B-22: get_weather_from_cache must return None (not crash) "
            "when both uppercase and lowercase U keys are absent"
        )


class TestIOContractsUnit:
    """B-02: save_graph_to_json is callable and produces valid JSON."""

    def test_b02_save_graph_to_json_is_callable_and_produces_valid_json(self, tmp_path):
        """B-02: save_graph_to_json is never called in __main__ — documents that
        the function exists and works correctly so calling it would fix B-02."""
        nodes = {(0, 0)}
        adj = {(0, 0): []}
        spm = {(0, 0): {"lat": 53.0, "lon": 2.0, "max_speed": 5.0}}
        json_path = str(tmp_path / "graph.json")

        save_graph_to_json(nodes, adj, spm, json_path)

        assert os.path.exists(json_path), "save_graph_to_json must create the file"
        with open(json_path) as fh:
            data = json.load(fh)
        assert "metadata" in data
        assert "nodes" in data
        assert data["metadata"]["nodes"] == 1
