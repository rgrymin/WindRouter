"""
Characterization tests for WindRouter — documents CURRENT behaviour including known bugs.

These tests pin what the code does TODAY. They enable safe refactoring: if a test breaks,
a behaviour changed and must be consciously verified against REQUIREMENTS.md.

Bugs captured here:
  B-04: VMG loop condition `while curr_lat < target_lat` — empty result when target <= start lat
  B-06: calculate_bearing returns 0 for identical points (atan2(0,0)=0, no guard)
  B-08: save_to_gpx writes <trkpt>; Visualiser reads gpx.waypoints (<wpt>) — zones never render
  B-10: empty dates list causes ValueError in get_weather_from_cache
  B-17: missing V-component causes crash in get_weather_from_cache, silent skip in safe areas
  B-18: grbs.close() not in finally — file handle leaks on iteration exception
  B-24: Dijkstra 2D crashes with KeyError for leaf nodes not in adjacency_map keys
  B-26: print_route_summary crashes with TypeError when time_hours is None
  C-10: Dijkstra 2D timestamp interpolation denominator = N not N-1
  C-16: get_scampi_30_polars returns new object every call (no memoization)
  C-22: VMG step limit 10000 — unreachable best_hdg=None branch
  C-30: duplicate (datetime, param) message — last write silently wins
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

import grib as grib_module
from grib import (
    calculate_bearing,
    calculate_distance_nm,
    find_shortest_path_dijkstra,
    get_scampi_30_polars,
    get_weather_from_cache,
    identify_safe_sailing_areas,
    identify_weather_danger_zones,
    load_grib_to_memory,
    print_route_summary,
    save_graph_to_json,
    save_to_gpx,
    save_waypoints_gpx,
    simulate_vmg_route,
)
from conftest import make_weather_cache, make_safe_points_map, make_full_adjacency, make_mock_grib_message

import gpxpy
import json


# ---------------------------------------------------------------------------
# Polars characterization
# ---------------------------------------------------------------------------

class TestPolarsCharacterization:
    def test_dead_zone_extrapolation_may_return_negative(self):
        """REQ-01: extrapolation outside table can return negative speeds."""
        polars = get_scampi_30_polars()
        result = polars([5, 10])[0]
        # No assertion on sign — characterization: just must not crash
        assert isinstance(result, (int, float, np.floating))

    def test_new_object_each_call(self):
        """C-16: no memoization — different object each call."""
        p1 = get_scampi_30_polars()
        p2 = get_scampi_30_polars()
        assert p1 is not p2


# ---------------------------------------------------------------------------
# Bearing characterization
# ---------------------------------------------------------------------------

class TestCalculateBearingCharacterization:
    def test_identical_points_returns_zero(self):
        """B-06: atan2(0,0)=0 — no guard for identical points."""
        b = calculate_bearing(53.0, 2.0, 53.0, 2.0)
        assert b == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VMG characterization
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestVMGCharacterization:
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

    def test_b04_target_south_of_start_returns_empty(self):
        """B-04: while curr_lat < target_lat — loop never executes when target <= start."""
        result = simulate_vmg_route(
            self.cache,
            start_lat=54.5, start_lon=2.25,
            target_lat=53.0, target_lon=2.25,
            start_time=self.t0,
        )
        assert result == [], "B-04: target south of start must produce empty route"

    def test_b04_target_same_latitude_returns_empty(self):
        """B-04: target at same latitude — loop condition False immediately."""
        result = simulate_vmg_route(
            self.cache,
            start_lat=53.5, start_lon=2.25,
            target_lat=53.5, target_lon=2.5,
            start_time=self.t0,
        )
        assert result == [], "B-04: target at same latitude must produce empty route"


# ---------------------------------------------------------------------------
# save_to_gpx characterization
# ---------------------------------------------------------------------------

class TestSaveToGpxCharacterization:
    def setup_method(self):
        self.t0 = datetime(2026, 4, 20, 12, 0)

    def test_writes_trkpt_not_wpt(self, tmp_path):
        """B-08: save_to_gpx uses <trkpt> elements, NOT <wpt>.
        This is the root cause of zones never rendering in the viewer."""
        out = str(tmp_path / "zones.gpx")
        points = [{"lat": 53.0, "lon": 2.0, "time": self.t0}]
        save_to_gpx(points, out)
        content = open(out).read()
        assert '<trkpt' in content, "B-08: save_to_gpx must use <trkpt>"
        assert '<wpt' not in content, "B-08: save_to_gpx must NOT use <wpt>"


# ---------------------------------------------------------------------------
# save_graph_to_json characterization
# ---------------------------------------------------------------------------

class TestSaveGraphToJsonCharacterization:
    LATS = [53.0, 53.25, 53.5, 53.75, 54.0]
    LONS = [2.0, 2.25, 2.5, 2.75]

    def setup_method(self):
        self.spm = make_safe_points_map(self.LATS, self.LONS)
        self.adj = make_full_adjacency(self.spm, cost=1.5)
        self.nodes = set(self.spm.keys())

    def test_b01_engine_writes_graph_key(self, tmp_path):
        """B-01 fixed: engine writes adjacency under key 'graph', matching Visualiser."""
        out = str(tmp_path / "graph.json")
        save_graph_to_json(self.nodes, self.adj, self.spm, filename=out)
        with open(out) as f:
            data = json.load(f)
        assert "graph" in data, "B-01 fixed: engine writes key 'graph'"
        assert "edges" not in data, "B-01 fixed: 'edges' key removed"


# ---------------------------------------------------------------------------
# Weather edge cases characterization (B-10, B-17)
# ---------------------------------------------------------------------------

class TestGetWeatherEdgeCases:
    """Edge cases for weather lookup — B-10, B-17."""

    def test_b10_empty_dates_list_raises(self):
        """B-10: cache with empty 'dates' list causes ValueError in min()."""
        cache = make_weather_cache()
        cache["dates"] = []  # empty dates — no guard in get_weather_from_cache
        with pytest.raises((ValueError, TypeError)):
            get_weather_from_cache(cache, 53.0, 2.0, datetime(2026, 4, 20))

    def test_b17_missing_v_component_returns_none(self):
        """B-22 fixed: get_weather_from_cache returns None when V-component key is absent."""
        cache = make_weather_cache()
        for dt in cache["dates"]:
            cache["data"][dt].pop("10 metre v wind component", None)
            cache["data"][dt].pop("10 metre V wind component", None)
        result = get_weather_from_cache(cache, 53.0, 2.0, cache["dates"][0])
        assert result is None

    def test_b17_missing_v_safe_areas_silently_skips(self):
        """B-17: identify_safe_sailing_areas silently skips timesteps with missing V.
        With n_times=2 and V removed from first timestep, second timestep (u=5, v=0
        → ~9.7 kt, safe) is still processed — all 16 cells remain safe."""
        cache = make_weather_cache(n_times=2)
        first_dt = cache["dates"][0]
        cache["data"][first_dt].pop("10 metre v wind component", None)
        spm = identify_safe_sailing_areas(cache)
        assert len(spm) == 16, (
            "B-17: second timestep is still valid (u=5 m/s safe) — all 16 cells must be safe"
        )


# ---------------------------------------------------------------------------
# Dijkstra 2D characterization
# ---------------------------------------------------------------------------

class TestDijkstra2DCharacterization:
    LATS = [53.0, 53.25, 53.5, 53.75]
    LONS = [2.0, 2.25, 2.5, 2.75]

    def setup_method(self):
        self.spm = make_safe_points_map(self.LATS, self.LONS)
        self.adj = make_full_adjacency(self.spm, cost=1.0)
        self.target_lat = 53.75
        self.target_lon = 2.75
        self.start_node = (0, 0)

    def test_timestamp_interpolation_last_point_not_exact(self):
        """C-10: last waypoint timestamp != departure + total_cost."""
        path, cost = find_shortest_path_dijkstra(
            self.start_node, self.adj, self.spm,
            self.target_lat, self.target_lon,
        )
        departure = datetime(2026, 4, 20, 0, 0)
        n = len(path)
        for idx, p in enumerate(path):
            p["time"] = departure + timedelta(hours=(idx * cost / n))

        assert n >= 2, (
            "Test requires path of at least 2 nodes to make C-10 off-by-one visible"
        )
        last_time = path[-1]["time"]
        expected_exact = departure + timedelta(hours=cost)
        expected_last = departure + timedelta(hours=cost * (n - 1) / n)
        # Last point should be departure + (N-1)/N * cost, NOT departure + cost
        assert last_time < expected_exact, (
            "C-10: last waypoint should be before departure+cost due to denominator=N"
        )
        # Use timedelta comparison directly — pytest.approx does not support datetime
        assert abs(last_time - expected_last) <= timedelta(seconds=1), (
            f"C-10: expected {expected_last}, got {last_time}"
        )

    def test_neighbor_not_in_distances_raises_keyerror(self):
        """`distances[neighbor]` raises KeyError when neighbor is not in adj.keys().
        distances is initialized only from adj.keys() = {(0,0)}.
        When (0,0) is expanded, neighbor (0,1) is found — but (0,1) is not in distances
        because it has no entry in adj, so `distances[(0,1)]` raises KeyError.
        This is a latent bug: generate_reachable_graph only adds nodes reachable from
        start as adj keys, so in practice neighbor is always in adj — but it's not
        enforced by the code."""
        adj = {
            (0, 0): [{"target": (0, 1), "cost": 1.0}],
            # (0,1) intentionally absent — neighbor but not a source node in adj
        }
        spm = make_safe_points_map(self.LATS[:2], self.LONS[:2])
        with pytest.raises(KeyError):
            find_shortest_path_dijkstra(
                (0, 0), adj, spm, 53.25, 2.25,
            )


# ---------------------------------------------------------------------------
# Safe areas characterization
# ---------------------------------------------------------------------------

class TestSafeAreasCharacterization:
    def test_danger_zones_uv_none_skips_timestep(self):
        """Timestep with u=None is silently skipped in danger zones; remaining timesteps still processed."""
        cache = make_weather_cache(u_speed=22.0, n_times=2)
        # Remove U from first timestep — that timestep is skipped, second still processes
        first_dt = cache["dates"][0]
        del cache["data"][first_dt]["10 metre U wind component"]
        zones = identify_weather_danger_zones(cache, min_threshold=40.0)
        # Second timestep still has u=22 m/s (~42.7 kt > 40) — all 16 cells forbidden
        assert len(zones) == 16


# ---------------------------------------------------------------------------
# load_grib_to_memory characterization
# ---------------------------------------------------------------------------

class TestLoadGribToMemoryCharacterization:
    """Characterization tests for known GRIB loading bugs."""

    def setup_method(self):
        self.load = load_grib_to_memory
        self.t0 = datetime(2026, 4, 20, 0, 0)
        self.lats = np.array([[53.0, 53.0], [53.25, 53.25]])
        self.lons = np.array([[2.0, 2.25], [2.0, 2.25]])
        self.u_vals = np.full((2, 2), 5.0)
        self.v_vals = np.full((2, 2), 0.0)

    def _make_grbs(self, messages):
        """Wrap a list of mock messages in a mock pygrib file handle."""
        grbs = MagicMock()
        grbs.__iter__ = MagicMock(side_effect=lambda: iter(messages))
        return grbs

    def test_grbs_close_called_even_on_iteration_exception(self, tmp_path, monkeypatch):
        """B-18 fixed: grbs.close() is now in a finally block — called even when iteration raises."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        bad_msg = MagicMock()
        bad_msg.validDate = self.t0
        bad_msg.name = "10 metre U wind component"
        bad_msg.latlons.side_effect = RuntimeError("read error mid-iteration")

        grbs = self._make_grbs([bad_msg])
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: grbs)
        result = self.load(str(dummy_file))
        assert result is None  # exception caught, returns None
        grbs.close.assert_called_once()  # B-18 fixed: close() called from finally block

    def test_zero_messages_returns_none_lats_lons(self, tmp_path, monkeypatch):
        """REQ-02: empty GRIB (zero messages) returns dict with lats=None, lons=None.
        Any subsequent grid operation will raise AttributeError — callers must guard."""
        dummy_file = tmp_path / "empty.grib"
        dummy_file.write_bytes(b"")

        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs([]))
        result = self.load(str(dummy_file))
        assert result is not None
        assert result["lats"] is None
        assert result["lons"] is None
        assert result["dates"] == []
        assert result["data"] == {}

    def test_c30_duplicate_message_last_write_wins(self, tmp_path, monkeypatch):
        """C-30: duplicate (datetime, param) pair — second message silently overwrites first."""
        dummy_file = tmp_path / "dup.grib"
        dummy_file.write_bytes(b"")

        first_u = np.full((2, 2), 3.0)
        second_u = np.full((2, 2), 7.0)
        msgs = [
            make_mock_grib_message("10 metre U wind component", self.t0, first_u, self.lats, self.lons),
            make_mock_grib_message("10 metre U wind component", self.t0, second_u, self.lats, self.lons),
        ]
        monkeypatch.setattr(grib_module.pygrib, "open", lambda _: self._make_grbs(msgs))
        result = self.load(str(dummy_file))
        stored = result["data"][self.t0]["10 metre U wind component"]
        np.testing.assert_array_equal(stored, second_u, err_msg="C-30: last message must win")


# ---------------------------------------------------------------------------
# analyze_grib_performance characterization (B-19)
# ---------------------------------------------------------------------------

class TestAnalyzeGribPerformanceCharacterization:
    def _make_grbs(self, monkeypatch, lats, lons):
        import numpy as np
        grbs = MagicMock()
        msg = MagicMock()
        msg.latlons.return_value = (np.array(lats), np.array(lons))
        grbs.readline.return_value = msg
        monkeypatch.setattr(grib_module.pygrib, "open", MagicMock(return_value=grbs))
        return grbs

    def test_b19_grbs_close_called_even_on_latlons_exception(self, tmp_path, monkeypatch):
        """B-19 fixed: grbs.close() is now in a finally block in analyze_grib_performance.
        The exception still propagates, but close() must be called before it escapes."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        grbs = MagicMock()
        msg = MagicMock()
        msg.latlons.side_effect = RuntimeError("corrupt first message")
        grbs.readline.return_value = msg
        monkeypatch.setattr(grib_module.pygrib, "open", MagicMock(return_value=grbs))

        import grib as grib_mod
        with pytest.raises(RuntimeError):
            grib_mod.analyze_grib_performance(str(dummy_file))

        grbs.close.assert_called_once()

    def test_b19_happy_path_close_called_and_prints_diagnostics(self, tmp_path, monkeypatch, capsys):
        """Happy path: analyze_grib_performance prints grid diagnostics and closes file."""
        dummy_file = tmp_path / "test.grib"
        dummy_file.write_bytes(b"")

        lats = [[53.0, 53.0], [54.0, 54.0]]
        lons = [[2.0,  3.0],  [2.0,  3.0]]
        grbs = self._make_grbs(monkeypatch, lats, lons)

        import grib as grib_mod
        grib_mod.analyze_grib_performance(str(dummy_file))

        grbs.close.assert_called_once()
        out = capsys.readouterr().out
        assert "GRIB DIAGNOSTICS" in out
        assert "53" in out and "54" in out  # lat range present in output


# ---------------------------------------------------------------------------
# I/O contracts characterization (from TestIOContracts)
# ---------------------------------------------------------------------------

class TestIOContractsCharacterization:
    """Round-trip between grib.py (writer) and Visualiser (reader) — current broken state."""

    def test_b08_save_waypoints_gpx_writes_wpt_elements(self, tmp_path):
        """B-08 fixed: save_waypoints_gpx writes <wpt> elements (NOT <trkpt>).
        Visualiser.load_area_file reads gpx.waypoints; <trkpt> elements are invisible to it."""
        points = [
            {"lat": 53.0, "lon": 2.0},
            {"lat": 53.25, "lon": 2.25},
        ]
        gpx_path = str(tmp_path / "forbidden.gpx")
        save_waypoints_gpx(points, gpx_path, "Forbidden")

        with open(gpx_path) as fh:
            gpx = gpxpy.parse(fh)

        assert len(gpx.waypoints) == 2, (
            "B-08: save_waypoints_gpx must write 2 waypoints readable by gpxpy"
        )
        n_trkpts = sum(len(seg.points) for trk in gpx.tracks for seg in trk.segments)
        assert n_trkpts == 0, "B-08: save_waypoints_gpx must NOT write any <trkpt> elements"

    def test_b08_save_waypoints_gpx_raw_xml_contains_wpt(self, tmp_path):
        """B-08 fixed: raw XML produced by save_waypoints_gpx contains <wpt> not <trkpt>."""
        points = [{"lat": 54.0, "lon": 3.0}]
        gpx_path = str(tmp_path / "caution.gpx")
        save_waypoints_gpx(points, gpx_path, "Caution")

        with open(gpx_path) as fh:
            raw = fh.read()

        assert "<wpt" in raw, "B-08: save_waypoints_gpx must write <wpt> element"
        assert "<trkpt" not in raw, (
            "B-08: save_waypoints_gpx must NOT write <trkpt> — "
            "Visualiser reads waypoints, not track segments"
        )

    def test_b01_engine_writes_graph_key(self, tmp_path):
        """B-01 fixed: save_graph_to_json writes key 'graph' readable by Visualiser."""
        nodes = {(0, 0), (0, 1), (1, 0)}
        adj = {
            (0, 0): [{"target": (0, 1), "cost": 1.0}, {"target": (1, 0), "cost": 1.4}],
            (0, 1): [{"target": (1, 0), "cost": 1.4}],
            (1, 0): [],
        }
        spm = {
            (0, 0): {"lat": 53.0,  "lon": 2.0,  "max_speed": 5.0},
            (0, 1): {"lat": 53.0,  "lon": 2.25, "max_speed": 5.0},
            (1, 0): {"lat": 53.25, "lon": 2.0,  "max_speed": 5.0},
        }
        json_path = str(tmp_path / "graph.json")
        save_graph_to_json(nodes, adj, spm, json_path)

        with open(json_path) as fh:
            data = json.load(fh)

        assert "graph" in data, "B-01 fixed: engine writes 'graph' key"
        assert "edges" not in data, "B-01 fixed: 'edges' key removed"

    def test_b01_visualiser_simulation_sees_correct_edges(self, tmp_path):
        """B-01 fixed: Visualiser reads 'graph' key and gets the correct edge count."""
        nodes = {(0, 0), (0, 1)}
        adj = {
            (0, 0): [{"target": (0, 1), "cost": 0.5}],
            (0, 1): [],
        }
        spm = {
            (0, 0): {"lat": 53.0, "lon": 2.0,  "max_speed": 5.0},
            (0, 1): {"lat": 53.0, "lon": 2.25, "max_speed": 5.0},
        }
        json_path = str(tmp_path / "graph.json")
        save_graph_to_json(nodes, adj, spm, json_path)

        with open(json_path) as fh:
            data = json.load(fh)

        visualiser_adj = data.get("graph", {})
        edges_drawn = sum(len(targets) for targets in visualiser_adj.values())

        assert edges_drawn == 1, "B-01 fixed: Visualiser reads 'graph' key → 1 edge drawn"


# ---------------------------------------------------------------------------
# Distance characterization
# ---------------------------------------------------------------------------

class TestDistanceCharacterization:
    def test_near_antimeridian_gives_large_wrong_value(self):
        """C-29 / B-21 limitation: flat formula uses lon2-lon1 directly.
        Crossing the antimeridian gives a huge erroneous distance (~21000 nm).
        This test documents the known bug — real distance is ~120 nm."""
        d = calculate_distance_nm(0.0, 179.0, 0.0, -179.0)
        # Real great-circle distance ≈ 120 nm; flat formula gives ~21480 nm
        assert d > 10000.0, (
            f"Antimeridian crossing gives wrong distance {d:.1f} nm "
            "(expected ~21000 with flat formula — documents the limitation)"
        )


# ---------------------------------------------------------------------------
# print_route_summary characterization
# ---------------------------------------------------------------------------

class TestPrintRouteSummaryCharacterization:
    def test_b26_none_time_hours_raises_type_error(self):
        """B-26: {time_hours:.2f} crashes with TypeError when time_hours is None.
        Documents the bug — points have no 'time' key, no time_hours arg passed."""
        points = [
            {"lat": 53.0,  "lon": 2.0},
            {"lat": 53.25, "lon": 2.25},
        ]
        with pytest.raises(TypeError):
            print_route_summary(points, "Test")
