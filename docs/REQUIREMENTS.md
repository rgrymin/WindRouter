# WindRouter — Requirements

> Reverse-engineered from source code. Describes **what the system currently does**.
> Edit this file to capture intended changes before implementing them.

---

## System Overview

WindRouter is a two-component desktop application for weather-based sailing route optimization for a **Scampi 30** yacht. The routing engine (`grib.py`) computes optimal routes through a GRIB2 weather forecast and writes results to files. The map viewer (`Visualiser.py`) displays those results on an interactive map.

---

## Dependencies

### Routing Engine (`grib.py`)

| Package | Purpose | Required |
|---------|---------|---------|
| `pygrib` | Reading GRIB2 weather files | Yes |
| `numpy` | Array operations, grid math | Yes |
| `scipy` (`RegularGridInterpolator`) | Bilinear interpolation of polar table | Yes |

### Map Viewer (`Visualiser.py`)

| Package | Purpose | Required |
|---------|---------|---------|
| `tkintermapview` | Interactive map widget (OSM/satellite tiles) | Yes |
| `gpxpy` | Parsing GPX route and area files | Yes |
| `global_land_mask` | Land detection for grey overlay | No — silently skipped if absent; land mask feature disabled |

Both components use Python stdlib only beyond the above (`os`, `json`, `math`, `heapq`, `collections`, `datetime`, `tkinter`).

---

## Component 1: Routing Engine (`grib.py`)

### REQ-01: Vessel Performance Model

**The system shall model the sailing performance of a Scampi 30 yacht using a polar curve lookup table.**

Rules:
- The polar table covers TWA values: 32, 36, 40, 45, 52, 60, 70, 80, 90, 100, 110, 120, 135, 150, 180 degrees (15 points, non-uniform spacing).
- The polar table covers TWS values: 6, 8, 10, 12, 14, 16, 20 knots (7 points, non-uniform spacing).
- Boat speed values (knots), rows = TWA, columns = TWS (6→20 kt):

  | TWA \ TWS | 6   | 8   | 10  | 12  | 14  | 16  | 20  |
  |-----------|-----|-----|-----|-----|-----|-----|-----|
  | 32°       | 2.8 | 3.8 | 4.6 | 5.1 | 5.3 | 5.4 | 5.4 |
  | 36°       | 3.2 | 4.3 | 5.1 | 5.5 | 5.7 | 5.8 | 5.8 |
  | 40°       | 3.6 | 4.7 | 5.4 | 5.8 | 6.0 | 6.1 | 6.2 |
  | 45°       | 4.0 | 5.1 | 5.7 | 6.1 | 6.3 | 6.4 | 6.5 |
  | 52°       | 4.4 | 5.5 | 6.0 | 6.4 | 6.6 | 6.7 | 6.9 |
  | 60°       | 4.8 | 5.8 | 6.3 | 6.6 | 6.8 | 7.0 | 7.2 |
  | 70°       | 5.1 | 6.0 | 6.5 | 6.8 | 7.1 | 7.3 | 7.5 |
  | 80°       | 5.2 | 6.1 | 6.6 | 7.0 | 7.3 | 7.5 | 7.8 |
  | 90°       | 5.3 | 6.3 | 6.8 | 7.1 | 7.4 | 7.7 | 8.1 |
  | 100°      | 5.4 | 6.4 | 6.9 | 7.3 | 7.6 | 7.9 | 8.4 |
  | 110°      | 5.3 | 6.4 | 7.0 | 7.4 | 7.8 | 8.1 | 8.6 |
  | 120°      | 5.0 | 6.2 | 6.9 | 7.3 | 7.7 | 8.1 | 8.7 |
  | 135°      | 4.3 | 5.5 | 6.4 | 7.0 | 7.4 | 7.8 | 8.5 |
  | 150°      | 3.6 | 4.7 | 5.6 | 6.4 | 7.0 | 7.4 | 8.1 |
  | 180°      | 3.1 | 4.1 | 5.0 | 5.8 | 6.4 | 6.9 | 7.6 |

- Boat speed for any TWA/TWS combination is computed by bilinear interpolation using `RegularGridInterpolator`.
- `bounds_error=False, fill_value=None` — queries outside the table range are extrapolated, not clamped or errored. Extrapolation can return **negative boat speeds** (e.g. very low TWS or TWA < 32°) — callers must guard against this (the `bs <= 0` checks in BFS and Dijkstra 3D handle it; Dijkstra 2D uses a 9999h fallback).
- TWA below 32° (dead zone) is enforced by caller logic, not by the interpolator itself.
- The polar interpolator object is recreated on every call to `get_scampi_30_polars()` — no memoization. See C-16.
- The vessel model is hardcoded and fixed to a single boat class.

---

### REQ-02: Weather Data Loading

**The system shall load wind forecast data from a GRIB2 file into memory.**

Rules:
- The GRIB2 file must contain 10-metre U and V wind component fields.
- All forecast timesteps are loaded into RAM as a nested structure `{'data': {datetime: {param: values}}, 'dates': [sorted datetimes], 'lats': array, 'lons': array}`. `weather_cache['dates']` is an explicitly sorted list of all unique forecast datetimes.
- If a GRIB file contains duplicate messages for the same (datetime, parameter) pair, the second silently overwrites the first. See C-30.
- Wind speed in knots is derived as `sqrt(U² + V²) × 1.94384` (converts m/s to knots).
- If the GRIB file does not exist or fails to load, the function returns `None` and processing stops.
- The V-component parameter name uses a hardcoded two-option check: tries `'10 metre v wind component'` (lowercase v) first; falls back to `'10 metre V wind component'` (uppercase V) if the first is absent. This is NOT a general case-insensitive lookup — any other capitalisation (e.g. `'10 Metre V Wind Component'`) would not be found and would cause a crash.
- Non-wind parameters (temperature, pressure, etc.) are loaded into the cache and occupy RAM but are never read. See C-28.
- If the GRIB file contains zero messages, `lats` and `lons` remain `None` — any subsequent grid operation will raise `AttributeError`.
- If a timestep has U but no V (or vice versa), behaviour differs by function: `get_weather_from_cache` will crash with `TypeError`; `identify_safe_sailing_areas` silently skips the timestep.

---

### REQ-03: Weather Lookup

**The system shall return wind conditions for any geographic position and time.**

Rules:
- The nearest GRIB grid point is found by minimum Euclidean distance in lat/lon degrees (`(lat-lat)² + (lon-lon)²`, no cos(lat) correction).
- The nearest forecast timestep is found by minimum absolute time difference.
- No spatial or temporal interpolation is performed — nearest-neighbor only.
- The result includes a metadata flag `approximated: true` when the requested time falls outside the forecast window (before first or after last timestep). Even when `approximated=True`, full weather data (`wind_u`, `wind_v`) is still returned — the nearest available timestep is used. There is no special early return for out-of-window times.
- The result is a dict with keys: `wind_u`, `wind_v` (raw m/s components) and `meta` containing `actual_lat`, `actual_lon`, `time` (matched datetime), `approximated` (bool).

---

### REQ-04: Safety Zone Classification

**The system shall classify every GRIB grid cell as safe, caution, or forbidden based on wind speed across the entire forecast period.**

> **Terminology note:** The "caution" zone is labelled **"not recommended"** in the viewer UI (`"Not Recommended: Waiting..."`) and the viewer monitors the file `not_recommended.gpx`, but the engine writes the file as `caution_areas.gpx`. These three names — *caution*, *not recommended*, *caution_areas* — all refer to the same 30–40 kt wind zone. See B-03.

Rules:
- A cell is **forbidden** if wind speed exceeds 40 kt at any forecast timestep (strictly `> 40`, no upper bound — `max_threshold=inf`).
- A cell is **caution** ("not recommended") if wind speed is between 30 and 40 kt at any timestep (exclusive lower, inclusive upper: `> 30 and <= 40`).
- A cell is **safe** if wind speed never exceeds 30 kt across the entire forecast (inclusive: `<= 30`).
- Only safe cells are eligible for routing.
- For each danger zone cell, the system records the wind speed and forecast time at which the threshold was **first** exceeded.

---

### REQ-05: Routing Graph Construction

**The system shall build a graph of reachable safe nodes using BFS from the departure position.**

Rules:
- The graph is **directed** — edge A→B does not imply B→A. The bearing cone filter typically prevents reverse edges.
- The graph is restricted to safe cells only (see REQ-04).
- From each node, only the 8 immediately adjacent GRIB grid cells (Moore neighborhood, ±1 row/column) are considered as neighbors.
- An edge is created only if:
  - The neighbor is a safe cell.
  - The bearing to the neighbor is within ±80° of the bearing to the final destination (filter: `angle_diff <= 80.0` — exactly 80.0° is permitted, strictly above is rejected).
  - The resulting TWA is ≥ 32° (no sailing into the dead zone).
- TWA is computed as `abs(((twd - X + 180) % 360) - 180)` — same algebraic formula used in BFS, Dijkstra 3D, VMG, and the detailed log, but the second variable differs by context: `bearing_to_neighbor` (BFS, Dijkstra 3D), `h` — candidate heading integer 0–358° (VMG), `hdg` — bearing to next waypoint (detailed log).
- Edge cost = travel time in hours = distance (nm) / boat speed (knots).
- If boat speed from polars is 0 or negative, edge cost defaults to 9999.0 hours (effectively unreachable).
- The starting node is the safe cell that minimises `(lat - start_lat)² + (lon - start_lon)²` (raw squared Euclidean in degrees, same formula as REQ-03). `calculate_distance_nm` is **not** used here — no cos(lat) correction, no nm conversion.
- Neighbor bounds are implicitly enforced: a neighbor `(row±1, col±1)` that falls outside the GRIB grid simply does not exist in `safe_points_map` and is skipped — no explicit index bounds check.
- The starting node may end up with an empty adjacency list (no outgoing edges) if all 8 neighbours fail the bearing cone or TWA filter — Dijkstra will then return only the start node.
- Only nodes **reachable from the start node** appear in `adjacency_map` — disconnected safe islands are not included.
- If no safe starting node exists, graph construction returns `(empty set, empty dict, None)`.
- Returns a tuple: `(reachable_nodes: set, adjacency_map: dict, start_node: tuple|None)`. Edge entries in `adjacency_map` are `{"target": (row, col), "cost": float}`.
- `safe_points_map` entries include a `max_speed` field (maximum observed TWS across the forecast) that is never read by any routing algorithm. See C-27.
- Each run appends to `graph_log.txt` (the file is deleted once at the very start of `__main__`, **only when `weather_cache` is valid** — if GRIB loading fails, the old `graph_log.txt` is not deleted). All 4 departure iterations append to the same fresh file. Each call makes **3 `log.write()` calls** producing 4 visible lines: the first write prefixes its string with `\n` (blank line before the header), then writes the header with start time; the second write contains the start node coordinates; the third write contains `"-"×80` — no per-edge or per-node detail is logged.
- Graph construction uses weather at the static departure time snapshot for all edges. (Note: Dijkstra 2D in REQ-06 performs **no weather lookups** during traversal — it consumes the pre-computed edge costs baked in here.)

---

### REQ-06: Static Dijkstra Routing (2D)

**The system shall find the fastest route using Dijkstra on a pre-built static graph.**

Rules:
- Weather is sampled once at departure time for all edges (static snapshot).
- The algorithm uses a min-heap (priority queue) on cumulative travel time. Heap entries are 2-tuples `(cost, node)` — no tie-breaking counter. The queue is initialized as the list literal `[(0, start_node)]` (not via `heappush`). All nodes in `adjacency_map` are initialized to `float('inf')` in `distances`; the start node is then overridden to `0`. `predecessors` for all nodes in `adjacency_map` is pre-initialized to `None`.
- The algorithm has **no early exit** — it runs until the priority queue is exhausted (full graph traversal every time).
- Uses lazy-deletion to skip stale heap entries: if a popped node's cost strictly exceeds the recorded distance (`curr_dist > distances[curr_node]`), the entry is discarded. Nodes are never explicitly marked visited — the distance check serves that purpose.
- Path is reconstructed via a `predecessors` dict (back-pointers). Each waypoint is a **direct reference** into `safe_points_map` (not a copy): `{'lat': ..., 'lon': ..., 'max_speed': ...}`. Timestamps are **not** added at reconstruction time — they are appended externally in `__main__` by mutating each dict in-place (which also mutates the `safe_points_map` entry).
- The destination is selected post-hoc: after full traversal, the function filters all nodes reachable from start (`distances[n] != inf`) and picks the one geographically closest to the target via `calculate_distance_nm` — **not** the lowest-cost node.
- If no node is reachable (all distances remain `inf`), returns `([], 0)`.
- If the start node has no outgoing edges, only the start node is reachable. `best_finish` is the start node and the function returns a 1-element path `([start_node_dict], 0.0)` — not an empty list.
- Waypoint timestamps use exact code: `p['time'] = departure_time + timedelta(hours=(idx * d_cost_2d / len(path)))` where `idx` is 0-based. The denominator is `len(path)` (not `len-1`), so the last waypoint gets timestamp `departure_time + (N-1)/N × total_cost` — it does **not** reach exactly `departure_time + total_cost`. See C-10.
- Reconstructed path includes the start node as the first waypoint (idx=0, timestamp = `departure_time`).
- Returns a 2-tuple `(path, total_cost_hours)`. On failure: `([], 0)`.
- Output: after timestamp injection in `__main__`, each dict has `{'lat', 'lon', 'max_speed', 'time'}`.

---

### REQ-07: Time-Dynamic Dijkstra Routing (3D)

**The system shall find the fastest route using time-aware Dijkstra where weather updates as the yacht advances.**

Rules:
- Edges are not pre-built; they are evaluated on-the-fly during node expansion.
- Weather at each edge is looked up at `departure_time + elapsed_hours` at that node.
- The same ±80° bearing filter and TWA ≥ 32° constraint apply (same as REQ-05).
- In 3D Dijkstra, edges with `bs <= 0` are skipped entirely (no 9999h fallback — unlike 2D).
- The bearing cone filter uses the bearing from **current node to final target** (recalculated at each node), not a fixed bearing from start. Note: the source code contains a misleading comment ("We use a fixed target bearing from start for consistency") and dead variables `start_lat_val`/`start_lon_val` that are initialized but never used — the actual behaviour is dynamic.
- Uses lazy-deletion: `if curr_cost > distances.get(curr_node, float('inf')): continue`. Uses `.get(..., inf)` rather than direct dict access — unlike 2D which pre-initializes all nodes and uses `distances[curr_node]` directly. No explicit visited set.
- Path is reconstructed via a `predecessors` dict. Each waypoint is a **dict** copied from `safe_points_map` with a `time` key added: `{'lat': ..., 'lon': ..., 'max_speed': ..., 'time': ...}`. Timestamps are embedded at reconstruction time (unlike 2D).
- The algorithm tracks the node closest to the target seen so far (`best_finish_node`) — updated each time a node is popped from the heap if it is closer to the target than the current best. Returns this node even if the 1 nm termination condition is never met. If the priority queue empties with no node ever set (`best_finish_node` remains `None`), returns `([], 0)`. On success returns `(path, distances[best_finish_node])` — a 2-tuple.
- Timestamps per waypoint are computed as `departure_time + timedelta(hours=distances[node])` — exact, not interpolated.
- Routing terminates when the yacht is within 1 nm of the target. The distance check runs **after** lazy-deletion — stale heap entries are never checked against the 1 nm condition.
- Reconstructed path includes the start node as the first waypoint (timestamp = `departure_time + 0h`).
- Output: ordered list of dicts `{'lat', 'lon', 'max_speed', 'time'}` with total travel time in hours.

---

### REQ-08: VMG Greedy Simulation

**The system shall simulate a route by greedily maximizing Velocity Made Good (VMG) at each step.**

Rules:
- Time step is 10 minutes.
- At each step, all headings from 0° to 358° (in 2° increments) are evaluated.
- VMG = boat_speed × cos(heading − bearing_to_target).
- Wind direction (TWD) is computed as `(degrees(atan2(-U, -V)) + 360) % 360` — meteorological convention (direction wind blows *from*). Identical formula used in VMG, Dijkstra 3D, BFS, and the detailed log.
- The heading with maximum VMG is chosen.
- Headings within the dead zone (TWA < 32°) are skipped.
- A predicted next position outside the GRIB grid extent is rejected (the heading is skipped, not the whole step).
- Position is advanced using flat-earth approximation with cos(lat) longitude correction.
- The longitude correction for position update uses the **new** (already advanced) latitude, not the departure latitude of that step — minor inconsistency with the boundary check which uses `curr_lat` before advance.
- `bearing_to_target` is recalculated from the current position at every step.
- The outer loop condition is `while curr_lat < target_lat` — only latitude progress is checked. All other termination conditions are `break` statements inside the loop: no valid heading (`best_hdg is None`), within 1 nm of target (`calculate_distance_nm < 1.0`), step limit reached (`step >= 10000`), and no weather (`if not weather: break`).
- The termination condition "northward progress stalls" from the summary is actually "no valid heading found" (`best_hdg is None`).
- If `get_weather_from_cache` returns `None` (e.g. empty forecast), the simulation loop breaks immediately and returns whatever points were collected so far.
- If TWS = 0 (below the polar table minimum of 6 kt), the interpolator extrapolates and may return near-zero boat speed. The yacht does not move but the step counter still increments — simulation runs until the 10,000-step limit. There is no explicit guard for TWS = 0.
- The simulation can return a single-point list (start position only) if all headings are invalid on the very first step (`best_hdg is None` immediately).
- When the outer `while curr_lat < target_lat` loop executes at least once, the start position is the first point in `route_points` — it is appended **before advancing but after the weather lookup and heading evaluation** (i.e. not at the very top of the loop body). If the loop never executes (target south of or at same latitude as start, see B-04), `route_points` is empty.
- The final position after termination is **not** appended to `route_points` — the last recorded point is the position at the start of the last successful step.
- `time_step_min` is a parameter with default `10.0`; it is passed explicitly as `10.0` from `__main__`.
- Output: ordered list of dicts `{'lat', 'lon', 'time'}` — no `max_speed` field (unlike Dijkstra routes).

---

### REQ-09: Multi-Departure Comparison

**The system shall run all three routing algorithms for multiple departure times and save results for each.**

Rules:
- Default: 4 departure times, staggered by 5 hours, starting from the first forecast timestep.
- Start position: 53.25°N, 2.6°E (southern North Sea).
- Target: northernmost latitude of the GRIB grid at the longitudinal midpoint.
- All departure times and positions are hardcoded.
- Safety zone classification (`identify_weather_danger_zones`, `identify_safe_sailing_areas`) is computed **once, before the departure loop**, and reused across all 4 departures. See C-11.
- The routing graph (BFS + adjacency) is rebuilt independently for each departure time.
- Dijkstra 3D reuses the `start_idx` returned by `generate_reachable_graph` in the **same iteration**. Unlike Dijkstra 2D (which is inside `if nodes:`), `find_shortest_path_dijkstra_3d` is called **unconditionally** — even when graph construction failed and `start_idx` is `None`. It handles `None` gracefully via its own internal guard (`if not start_node: return [], 0`). See B-05.
- In `__main__`, route saving is guarded: `save_to_gpx` and `save_route_detailed_log` are called only when the result list is non-empty. The zone GPX saves (`forbidden_areas.gpx`, `caution_areas.gpx`) have no such guard — they are always written when weather data is valid. Empty routes produce no output files for that departure.
- `save_graph_to_json` is **not called** in the main block — `sailing_graph.json` is stale from a previous manual run. See B-02.

---

### REQ-10: Output Files

**The system shall write results to files in standard formats.**

Rules:
- Each route is saved as a GPX 1.1 track file (`xmlns="http://www.topografix.com/GPX/1/1"`, creator `"ScampiRouter"`) using `<trkpt>` elements inside `<trk><trkseg>`, with per-waypoint `<time>` tags in format `YYYY-MM-DDTHH:MM:SSZ` (UTC suffix `Z` hardcoded as a literal, not derived from tzinfo). Time tag is omitted when the point has no `'time'` key.
- All output files are written to the **current working directory** (CWD) — no path management, no `os.path.dirname(__file__)`.
- One full run (4 departures) produces **up to 14 GPX files**: 4 × Dijkstra 2D + 4 × Dijkstra 3D + 4 × VMG + 1 × forbidden zones + 1 × caution zones. The 2 zone files are always written; the 12 route files are only written when the algorithm produces a non-empty result. Similarly, **up to 12 text logs** (3 per departure × 4) are written — each log is guarded by `if result:`, so failed routes produce no log. The `graph_log.txt` is always written (once per BFS call, 4 times per run).
- If the route point list is empty, `save_route_detailed_log` returns immediately without creating the file (silent no-op).
- `print_route_summary` is called only when the algorithm produced a non-empty result — all three calls in `__main__` are inside their respective `if result:` guards. If an algorithm returns an empty list, its summary is silently skipped. For Dijkstra 2D and 3D, `time_hours` is passed explicitly; for VMG it is derived from `points[0]['time']` and `points[-1]['time']` inside `print_route_summary`. When called with 1 point (which can occur), `print_route_summary` prints `"No data available."` and returns — no crash.
- Each route is saved as a human-readable tabular log. Function signature: `save_route_detailed_log(points, cache, filename, label)` — the weather cache parameter is named `cache` (not `weather_cache`). Columns in order: `Time | Latitude | Longitude | TWS [kt] | TWD [°] | Heading [°] | TWA [°] | BS [kt]` (Time field left-padded to 20 chars). Timestamp format: `%Y-%m-%d %H:%M`. File opened in mode `'w'` — always overwritten.
- In the detailed log, heading for all waypoints except the last is computed **forward** (bearing to next point); heading for the last waypoint is computed **backward** (bearing from previous point). For a single-point route, heading defaults to `0°`.
- In the detailed log, if a point has no `'time'` key, `datetime.now()` is used as a fallback — the timestamp in the log will reflect wall-clock time of the log write, not the route time.
- In the detailed log, TWA and TWS are recomputed fresh from weather at each waypoint's position and time (via `get_weather_from_cache`) — not carried over from the routing phase. TWA uses the computed heading to the next waypoint as the reference angle. If TWA < 32°, boat speed is recorded as 0.
- The routing graph is saved as JSON with top-level keys: `metadata` (`nodes` count, `unit`: `"hours"`, `timestamp` ISO string), `nodes` (keyed by `"row,col"` string, each value `{"lat": ..., "lon": ...}` — no `max_speed`), `edges` (keyed by `"row,col"`, each value a list of `{"target": "row,col", "cost": float}` objects).
- `save_to_gpx` always creates the file even when the point list is empty — the result is a valid but track-less GPX shell (header + empty `<trkseg>` + footer). The `label` parameter (default `"Route"`) is written as `<name>` inside `<trk>`.
- Safety zone files (forbidden, caution) are written by the same `save_to_gpx` function used for routes — they contain `<trkpt>` elements, **not** `<wpt>` elements. The viewer reads them via `gpx.waypoints`, which parses `<wpt>` only — see B-08.
- Zone GPX saves (`forbidden_areas.gpx`, `caution_areas.gpx`) are called **unconditionally** (no empty-result guard) — if no cells exceed the threshold, `save_to_gpx` still writes an empty-track GPX file. This is unlike route saves which are guarded by `if result:`.
- File naming convention: `fastest_path_start_N.gpx`, `fastest_path_3d_start_N.gpx`, `scampi_vmg_start_N.gpx`, `log_dijkstra_start_N.txt`, `log_dijkstra_3d_start_N.txt`, `log_vmg_start_N.txt` where N = departure index 1–4.
- Caution areas are written **only** to `caution_areas.gpx` (never to `not_recommended.gpx`). The viewer monitors `not_recommended.gpx` — caution zones will not appear unless the file is manually renamed or copied. See B-03.

---

## Component 2: Map Viewer (`Visualiser.py`)

### REQ-11: Map Display

**The system shall display routes and weather zones on an interactive map.**

Rules:
- Window title: `"Scampi Navigation - Multi-Route Monitor Full (3D Enabled)"`.
- Default map center: 58.0°N, 10.0°E, zoom level 6 (North Sea / Scandinavia).
- Supported map tile sources: OpenStreetMap (default), Google Satellite, OpenSeaMap. Tile source is changed via `set_tile_server()` — no explicit map redraw is triggered; reload behaviour depends on the `tkintermapview` library.
- Window size: 1550×950 pixels.
- All GPX and JSON files are read from the **current working directory** (CWD) — same assumption as the engine.
- The viewer supports exactly 4 departures per algorithm — slot counts are hardcoded. There is no mechanism to display a 5th departure.

---

### REQ-12: Route Rendering

**The system shall render routes with distinct colors per algorithm and departure index.**

Rules:
- VMG routes (blue shades): `#1F77B4`, `#3498DB`, `#5DADE2`, `#85C1E9` (departures 1–4).
- Dijkstra 2D routes (red shades): `#C0392B`, `#E74C3C`, `#EC7063`, `#F1948A` (departures 1–4).
- Dijkstra 3D routes (yellow/gold shades): `#F1C40F`, `#F39C12`, `#D4AC0D`, `#B7950B` (departures 1–4).
- Each route is drawn as a single polyline (`set_path`, width=3) via the tkintermapview API. Graph edges are drawn as individual two-point polylines (width=1, colour `#2ECC71`).
- Old route objects are deleted (`.delete()`) before redrawing — but only when the new file contains at least one point. If a file becomes empty, the old route stays on the map.
- GPX timestamps (`<time>` tags) are parsed by gpxpy but **discarded** — only `(lat, lon)` coordinates are stored and rendered. No time information is shown on the map.
- `load_track` iterates `gpx.tracks → segments → points` (`<trkpt>` only). A GPX file containing only `<wpt>` elements will produce an empty route and nothing is rendered.
- Departure 1 always gets the darkest shade of each colour group; departure 4 gets the lightest.

---

### REQ-13: Weather Zone Rendering

**The system shall render forbidden and caution areas as colored rectangles on the map.**

Rules:
- Each waypoint in a zone GPX file is rendered as a filled rectangle of 0.12° half-side. In the latitude direction this is ≈7.2 nm; in the longitude direction at North Sea latitudes (~57°N) it is ≈3.9 nm — the rectangle is **not square** in nautical miles.
- Forbidden areas (wind > 40 kt): red fill.
- Caution/not-recommended areas (wind 30–40 kt): orange fill.

---

### REQ-14: Graph Grid Rendering

**The system shall optionally render the routing graph edges as a green grid overlay.**

Rules:
- Graph data is read from `sailing_graph.json`.
- Each edge is drawn as a thin green line.
- Toggled via "Show graph grid" checkbox. When the checkbox is unchecked, `load_sailing_graph` clears any existing graph segments from the map (segments are actively removed, not just hidden).
- `check_files_loop` checks the graph file's mtime unconditionally (not gated by the checkbox). The checkbox is checked inside `load_sailing_graph` — if unchecked, segments are cleared and the function returns early.
- Node and edge keys in JSON are strings `"row,col"` (e.g. `"42,17"`). The viewer uses these strings directly as dict keys — it does not parse them back into tuples.
- The viewer reads the `"graph"` key from the JSON for adjacency (not `"edges"`). **Note: the engine writes the key as `"edges"` but the viewer reads `"graph"` — currently no edges are rendered. This is a bug (B-01).**

---

### REQ-15: Land Mask

**The system shall optionally render a grey land overlay.**

Rules:
- Requires `global-land-mask` library (optional dependency — silently skipped if absent).
- Land is detected on a 0.25-degree grid step with 0.125-degree half-side polygons over the bounding box of all loaded data (routes + graph + zones), expanded by ±1° in all directions. Bounding box is accumulated by `update_bounds`, which is called for every loaded point across all file types.
- Each land cell is rendered as a grey polygon (`#555555` fill, `#333333` outline).
- The land mask is redrawn every time any file update is detected (if the checkbox is on).
- Land mask is only drawn if at least one route or area file has been loaded (bounding box must be valid: `min_lat <= max_lat`).
- Toggled via "Show land mask (Grey)" checkbox.
- "Filter land in areas" checkbox hides danger/caution markers that fall on land.
- Land detection uses a single center-point check per cell (`globe.is_land(lat, lon)`) — no area sampling.

---

### REQ-16: Live File Monitoring

**The system shall automatically reload and redraw when output files are updated.**

Rules:
- `check_files_loop` is called **immediately on startup** (synchronously in `__init__`, before the Tk event loop starts) — existing files are loaded before the window appears.
- On startup and every 10 seconds, the viewer checks modification timestamps of all monitored files.
- If any file has changed since the last check, it is reloaded and redrawn.
- Monitored files: all VMG, Dijkstra 2D, Dijkstra 3D GPX files (4 each), `forbidden_areas.gpx`, `not_recommended.gpx`, and `sailing_graph.json`.
- The 10-second timer is re-scheduled via `root.after(10000, check_files_loop)` at the **end** of each non-forced check — the timer is **not** rescheduled when `force=True` (e.g. triggered by "Filter land in areas" checkbox).
- "Filter land in areas" checkbox triggers `force_reload_areas()` which resets all mtimes to 0 and calls `check_files_loop(force=True)` — this reloads **all** files unconditionally, not just area files.
- Files that do not exist are silently skipped (no error shown to user).
- There is no `try/except` around `os.path.getmtime` — if a file is deleted between the `exists()` check and `getmtime()` call (TOCTOU race condition), an unhandled `FileNotFoundError` will crash `check_files_loop`, and the `root.after(10000, ...)` rescheduling at the end will never execute, permanently stopping all file monitoring for the session. See B-12.

---

### REQ-17: Status Panel

**The system shall display a summary of loaded data.**

Rules:
- Shows point counts for each of the 12 routes (4 VMG, 4 Dijkstra 2D, 4 Dijkstra 3D).
- Shows graph node count and edge count — edge count is the number of lines actually drawn during rendering, not a value read from JSON metadata.
- Shows forbidden area count and caution area count (waypoint counts from the respective GPX files).
- There is no per-route show/hide toggle — all loaded routes are always rendered. The only rendering controls are the three global checkboxes (land mask, graph grid, filter land in areas).
- Errors during GPX parsing are silently printed to stdout only — no dialog shown. See C-14.

---

### REQ-18: Manual GPX Loading

**The system shall allow the user to manually load any GPX file.**

Rules:
- Triggered by "Load other GPX" button.
- The loaded file replaces the VMG slot 1 route.
- The file path replaces `vmg_files[0]` and `last_vmg_mtimes[0]` is reset to 0. **Side-effect:** the next `check_files_loop` tick (within 10 s) will detect `mtime > 0` for the new file and reload it again, which is harmless but redundant. However, if the original `scampi_vmg_start_1.gpx` also exists, it may overwrite the manually loaded file on the next cycle — `vmg_files[0]` points to the new file, so this does not happen.
- Map centering on first point happens because `load_track` is called with `force=True` and `index=0` — the centering logic is shared with auto-reload, not specific to manual loading.
- There is no "fit bounds" / zoom-to-route functionality. See C-13.
- The file dialog filters for `*.gpx` files only.

---

## Future Extensibility (from documentation)

These are not implemented but documented as intended directions:

- **Tidal currents**: Adding U_current / V_current parameters would enable COG and SOG calculations.
- **Fuel consumption**: A fuel-per-hour coefficient could be added to the cost function for motor-sailing.
- **Time-dynamic isochrone routing**: The 3D Dijkstra is a step toward this; a full isochrone approach would require a (Lat, Lon, Time) graph.

---

## Known Constraints and Design Decisions

| # | Constraint |
|---|------------|
| C-01 | Vessel model is hardcoded to Scampi 30. No other boat class is supported. |
| C-02 | Routing space is restricted to GRIB grid nodes (0.25° grid). N-S spacing ≈15 nm; E-W spacing ≈8 nm at 57°N (cos(57°) correction) — not isotropic. No sub-grid positions. |
| C-03 | Weather lookup uses nearest-neighbor only — no spatial or temporal interpolation. |
| C-04 | The ±80° bearing cone filter prevents backward-detouring routes but also prevents tacking strategies. |
| C-05 | A cell unsafe at any forecast timestep is excluded from routing entirely, even if calm at departure. |
| C-06 | Start position, target, departure times, safety thresholds, and the GRIB filename (`"test.grb2"`) are all hardcoded in `__main__`. There is no file picker, directory scan, or command-line argument. |
| C-07 | The viewer and engine are fully decoupled — no IPC, no subprocess launch, no shared state. The viewer only reads files written by the engine; it never calls any engine function. |
| C-08 | Distance calculation uses flat-earth approximation (not Haversine). Accurate enough for short legs at North Sea latitudes but accumulates error over long distances. |
| C-09 | Bearing calculation uses full spherical trigonometry (atan2 formula), but distance uses flat-earth — these two are inconsistent. |
| C-10 | Dijkstra 2D assigns timestamps by linear interpolation over the path, not by actual per-leg travel time computation. The last waypoint gets `departure_time + (N-1)/N × total_cost`, not exactly `departure_time + total_cost`. |
| C-11 | Safety zone classification is computed once before the departure loop, covering the entire forecast period — not re-evaluated per departure time. A cell stormy at any timestep is excluded for all 4 departures equally. |
| C-12 | The `analyze_grib_performance` diagnostic function is the **first** thing called in `__main__` (before data loading), always prints **5 lines** to stdout via 2 `print()` calls: first call prints `"--- GRIB DIAGNOSTICS ---"` then the grid lat/lon extent (embedded `\n`); second call prints the S-N and E-W distances in nm, then a blank line, then a `"-"×50` separator (embedded `\n` before the dashes, plus `print`'s own trailing newline). Cannot be suppressed. It reads only the first GRIB message — it does not inspect timestep count, resolution, or available parameters. |
| C-13 | The viewer has no "fit bounds" / zoom-to-route feature. After loading, the map stays at the current zoom level; centering only moves to the first waypoint of VMG slot 1. |
| C-14 | GPX parse errors in the viewer are silently printed to stdout only — no dialog or status label is shown to the user. |
| C-15 | Wind speed conversion factor `1.94384` (m/s → knots) is duplicated in 6 separate places in `grib.py` with no shared constant. |
| C-16 | The polar interpolator object (`RegularGridInterpolator`) is recreated on every call to `get_scampi_30_polars()` — no memoization. |
| C-17 | The viewer documentation (`Project Documentation Map Display.md`) does not mention the 3D Dijkstra routes — it was written before that feature was added. |
| C-18 | `safe_points_map` (used as the routing domain) is built from a whole-forecast safety check (`tws <= 30` across all timesteps). Dijkstra 3D then routes through this static safe domain using dynamic weather per node visit — but never re-verifies safety at visit time. A node that was calm at forecast start but stormy later is still treated as safe by the router. |
| C-19 | If `weather_cache` is `None` (GRIB file missing or unreadable), the engine prints `"GRIB file not loaded."` to stdout and exits silently — no exception, no non-zero exit code, no output files are written. |
| C-20 | There is no `try/except` around routing algorithm calls in `__main__`. An unhandled exception inside any algorithm (e.g. B-05, B-14) aborts the entire departure loop — remaining departures and their output files are not produced. |
| C-21 | `save_graph_to_json` always overwrites the output file (open mode `'w'`). There is no versioning, backup, or existence check. |
| C-22 | All threshold values are magic numbers with no named constants: dead zone `32°`, bearing filter `80.0°`, conversion `1.94384`, safe wind `30 kt`, unreachable cost `9999.0 h`, arrival radius `1.0 nm`, VMG step limit `10000`. A change to any threshold requires a manual grep across the codebase. |
| C-23 | `grib.py` is safely importable — all engine logic is inside `if __name__ == "__main__":`. Functions can be imported and called independently. |
| C-24 | The map widget (`tkintermapview.TkinterMapView`) is created with no offline cache or tile database parameters — it relies on default in-memory tile caching provided by the library. No offline mode is available. |
| C-25 | Status labels in the viewer default to `"VMG X: None"`, `"Fastest X: None"`, `"3D X: None"`, `"Graph: Waiting..."`, `"Forbidden: Waiting..."`, `"Not Recommended: Waiting..."`. Labels are **not updated** when a file is missing or loads 0 points — the previous text (or the default) persists. |
| C-26 | `grib.py` contains numerous diagnostic `print` statements throughout. Algorithm functions use tagged prefixes: `[SAFE AREA DIAG]`, `[GRAPH DIAG]`, `[DIJKSTRA 3D]`, `[VMG DIAG]`, `[LOG]`. Additionally, `__main__` prints untagged progress lines (e.g. `"--- STARTING SIMULATION..."`, `">>> SIMULATION NO {i}"`, `"Simulations complete."`) and `analyze_grib_performance` prints its own 4-line header. None of these can be suppressed without editing the source. |
| C-27 | `safe_points_map` stores a `max_speed` field per node (max observed TWS) that is computed but never consumed by any routing algorithm. It occupies memory and represents dead code. |
| C-28 | Non-wind GRIB parameters (temperature, pressure, etc.) are fully loaded into `weather_cache` and occupy RAM proportional to the number of extra parameters × grid size × timestep count. There is no filtering on load. |
| C-29 | `calculate_distance_nm` uses a flat-earth formula with a `cos(avg_lat)` correction on longitude only: `d_lat = Δlat × 60`, `d_lon = Δlon × 60 × cos(avg_lat_rad)`, result = `sqrt(d_lat² + d_lon²)` nm. The `cos` factor uses the **average** of the two latitudes (in radians), not the departure latitude. This is more accurate than a pure flat-earth but less accurate than the full Haversine formula. |
| C-30 | Duplicate GRIB messages (same datetime + parameter name) are silently overwritten during loading — the later message replaces the earlier one in `weather_cache['data']` with no warning. This is a silent last-write-wins behaviour. |

---

## Known Bugs

| # | Bug | Affected REQ |
|---|-----|--------------|
| B-01 | Engine writes graph adjacency under JSON key `"edges"`; viewer reads key `"graph"` — graph grid is **never rendered**. Must fix both `save_graph_to_json` (key name) and ensure the function is called in `__main__` (see B-02). | REQ-14 |
| B-02 | `save_graph_to_json` is defined but **never called** in `__main__` — `sailing_graph.json` in the repo is stale from a previous manual run. Even after fixing B-01, the file will not regenerate automatically. Both B-01 and B-02 must be fixed together. | REQ-14 |
| B-03 | Engine writes caution areas to `caution_areas.gpx`; viewer monitors `not_recommended.gpx` — caution zones **never appear** unless the file is manually renamed or copied. | REQ-10, REQ-13 |
| B-04 | VMG simulation loop condition is `while curr_lat < target_lat`. If the target is at the same latitude or south of the start, the loop never executes and VMG produces zero waypoints. Latent with current hardcoded positions (target is always north of 53.25°N start). | REQ-08 |
| B-05 | `find_shortest_path_dijkstra_3d` is called **unconditionally** in `__main__` — it is outside the `if nodes:` guard that wraps the 2D Dijkstra call. `start_idx` is assigned by tuple unpacking from `generate_reachable_graph()` and is `None` when graph construction fails. `find_shortest_path_dijkstra_3d(None, ...)` hits its own guard (`if not start_node: return [], 0`) so it does not crash — it silently returns an empty route. The real consequence is that 3D Dijkstra always runs even when graph construction failed, wasting time and potentially producing a misleading empty result. | REQ-07, REQ-09 |
| B-06 | `calculate_bearing` returns `0°` (north) when called with two identical points (`atan2(0,0) = 0`). No guard is raised. Affects the BFS ±80° bearing filter if a node is already at the target. | REQ-05 |
| B-07 | `reachable_safe_points.gpx` exists in the repo but `identify_safe_sailing_areas` returns a dict and never writes a GPX file — the file is a stale artifact from a previous run. | REQ-10 |
| B-08 | `save_to_gpx` writes safety zone files using `<trkpt>` elements. The viewer reads them via `gpx.waypoints`, which only parses `<wpt>` elements — `gpx.waypoints` is always empty, so **forbidden and caution zones are never rendered** even when the files exist and have the correct name. Fix: change `save_to_gpx` to emit `<wpt>` for zone files, or use a separate writer function. | REQ-10, REQ-13 |
| B-09 | Dead code in `find_shortest_path_dijkstra_3d`: variables `start_lat_val` and `start_lon_val` are assigned but never used in the loop. The accompanying comment ("We use a fixed target bearing from start for consistency in the search cone") is factually wrong — the actual bearing is dynamic (recalculated per node to the final target). | REQ-07 |
| B-10 | `get_weather_from_cache` guards against `cache=None` but not against an empty `dates` list. If `cache['dates']` is empty, `min(all_dates, ...)` raises `ValueError`. If a required wind component key is missing from the cache dict, subsequent `None[min_idx]` raises `TypeError`. Both cases crash the engine with no user-facing message. | REQ-03 |
| B-11 | When a reloaded GPX file is empty (zero track points), the viewer skips the redraw entirely — the previously rendered route stays on the map indefinitely. There is no way for the user to clear a route without restarting the viewer. | REQ-16, REQ-12 |
| B-12 | `check_files_loop` uses `os.path.exists()` + `os.path.getmtime()` without a `try/except`. A TOCTOU race (file deleted between the two calls) raises an unhandled `FileNotFoundError`, crashes the method, and prevents the `root.after(10000, ...)` reschedule — permanently stopping all live file monitoring for the session. | REQ-16 |
| B-13 | `tkintermapview` and `gpxpy` are imported at module level without `try/except`. If either library is missing, the viewer crashes on import with no user-friendly message (unlike `global_land_mask`, which is guarded). | Dependencies |
| B-14 | In `find_shortest_path_dijkstra_3d`, if `get_weather_from_cache` returns `None` during edge expansion (e.g. cache is stale or dates exhausted), the next line `weather['wind_u']` raises `TypeError: 'NoneType' object is not subscriptable` — crashing the algorithm with no recovery. VMG handles this correctly (`if not weather: break`); 3D Dijkstra does not. | REQ-07 |
| B-15 | Zones are drawn **after** routes in `check_files_loop` — in `tkintermapview`, later-drawn objects appear on top. Forbidden/caution zone rectangles may visually cover route polylines at intersections. There is no z-order control. | REQ-12, REQ-13 |
| B-16 | When `sailing_graph.json` contains invalid JSON, `json.load` raises an exception caught by the `except` block. The `for seg in self.graph_segments: seg.delete()` cleanup is placed **after** `json.load` in the `try` block — so when `json.load` fails, the cleanup is skipped and stale graph segments from the previous successful load remain on the map indefinitely. | REQ-14 |
| B-17 | If a GRIB timestep has U-component but no V-component (or vice versa), `identify_safe_sailing_areas` silently skips the timestep (treats it as "no data" = safe) — it guards with `if u is not None and v is not None:`. `get_weather_from_cache` has no such guard: missing U causes `None[min_idx]` → `TypeError` on line 91; missing V causes the same on line 93. The two functions disagree on how to handle missing wind components. | REQ-02, REQ-03, REQ-04 |
| B-18 | `load_grib_to_memory` calls `grbs.close()` inside `try` with no `finally` block. If any exception occurs during message iteration (e.g. corrupt message, `msg.latlons()` fails), `grbs.close()` is skipped and the GRIB file handle leaks. Fix: move `grbs.close()` to a `finally` block or use a context manager. | REQ-02 |
| B-19 | `analyze_grib_performance` opens a GRIB file handle via `pygrib.open()` with no `finally` block — same class of bug as B-18. If `msg.latlons()` or any arithmetic raises, `grbs.close()` on the last line is never reached and the handle leaks. On Windows this can lock the file and prevent subsequent opens. | REQ-02 |
| B-20 | `simulate_vmg_route` updates `curr_lat` on one line and then uses the already-updated `curr_lat` to compute the cos(lat) longitude correction on the very next line. The longitude step should be corrected by the latitude at the *start* of the step, not after it has been advanced. Fix: save `old_lat = curr_lat` before the latitude update and use `old_lat` in the longitude formula. | REQ-08 |
| B-21 | `calculate_distance_nm` uses a flat-earth formula (see C-29) instead of the full Haversine formula. Error reaches 3–8% on distances over 200 nm, affecting route cost accuracy for all three routing algorithms. Fix: replace with Haversine using Earth radius 3440.065 nm. | REQ-05, REQ-06, REQ-07, REQ-08 |
| B-22 | `get_weather_from_cache` has a two-option fallback for the V-component key (`'10 metre v wind component'` / `'10 metre V wind component'`) but no equivalent fallback for the U-component — it always looks up `'10 metre U wind component'` literally. A GRIB file using any other capitalisation for U causes `.get()` to return `None`, and the subsequent subscript `None[min_idx]` raises `TypeError`. Fix: add the same case heuristic for U as already exists for V. | REQ-03 |
| B-23 | In `generate_reachable_graph`, the return value of `get_weather_from_cache` is never checked for `None` before unpacking into `(u, v)`. If the cache is empty or dates are exhausted, `weather` is `None` and `weather['wind_u']` raises `TypeError`, crashing the entire BFS with no recovery. | REQ-05 |
| B-24 | In `find_shortest_path_dijkstra_3d`, `distances` and `predecessors` are initialised only from `adjacency_map` keys in the 2D variant, but in the 3D variant they grow lazily. However, `distances[neighbor]` is accessed via `.get(..., inf)` for the relaxation check — if `neighbor` has never been seen, the key is absent and the `.get` returns `inf` correctly. This is safe in 3D. In the **2D** variant (`find_shortest_path_dijkstra`), `distances` and `predecessors` are pre-seeded only from `adjacency_map.keys()`. Any edge target that is not itself a key in `adjacency_map` (a leaf node — safe node that was added as a target but has no outgoing edges) is absent from `distances`, so `distances[neighbor]` raises `KeyError` during relaxation. Fix: initialise missing neighbors lazily or seed `distances` from all edge targets as well as sources. | REQ-06 |
| B-25 | Timestamps on the Dijkstra 2D path are assigned by linear interpolation: `departure_time + timedelta(hours=(idx * total_cost / len(path)))`. This assumes equal time spacing between waypoints, which is not true — edge costs are not uniform. Additionally, the denominator is `len(path)` not `len(path)-1`, so the final waypoint gets `(N-1)/N × total_cost` and never reaches the actual arrival time. Every timestamp in the Dijkstra 2D GPX and its detailed log is therefore inaccurate. Fix: accumulate actual edge costs along the reconstructed path to derive per-waypoint timestamps. | REQ-06, REQ-10 |
| B-26 | `print_route_summary` formats `time_hours` with `{time_hours:.2f}` unconditionally. If neither the `'time'` key is present in the first/last point nor an explicit `time_hours` argument was passed, `time_hours` remains `None` and the format call raises `TypeError: unsupported format character`. The normal call paths do not trigger this, but any caller omitting both conditions will crash. Fix: format conditionally — use `"N/A"` when `time_hours is None`. | REQ-10 |
| B-27 | The "Filter land in areas" checkbox calls `force_reload_areas()` → `check_files_loop(force=True)`. Inside `check_files_loop`, the `root.after(10000, self.check_files_loop)` reschedule is guarded by `if not force:` — so when `force=True`, the periodic timer is **not rescheduled**. After a single checkbox toggle, all live file monitoring stops permanently for the rest of the session. Fix: always reschedule `root.after` at the end of `check_files_loop`, unconditionally. | REQ-16 |
| B-28 | `manual_load_gpx` permanently overwrites `self.vmg_files[0]` with the user-selected path. From that point on, `check_files_loop` monitors the manually loaded file as VMG slot 1 rather than `scampi_vmg_start_1.gpx`. The original auto-generated file is lost for the session with no way to restore it without restarting. Fix: use a separate display slot for manually loaded files instead of hijacking slot 0. | REQ-18 |
| B-29 | `load_area_file` clears `shapes_list` and redraws inside the `try` block, but if the GPX source file is deleted after being loaded once, the `if not os.path.exists(file_path): return` early exit fires *before* clearing shapes — stale forbidden/caution zone polygons remain on the map indefinitely after their source file disappears. Fix: clear existing shapes regardless of whether the file exists. | REQ-13, REQ-16 |
| B-30 | `np.empty(lats.shape, dtype=object)` in `identify_weather_danger_zones` leaves `first_time_grid` with uninitialised memory for cells that never exceed the threshold. The return comprehension correctly filters to `found_mask` cells only, but relies on `found_mask` being exactly correct — any edge-case bug in mask logic could return cells with garbage timestamps. Fix: use `np.full(lats.shape, None, dtype=object)` to initialise explicitly. | REQ-04 |
