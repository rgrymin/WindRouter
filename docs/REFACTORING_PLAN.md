# WindRouter Refactoring Plan

## Context

WindRouter is a hobby/educational weather routing tool for a single yacht (Scampi 30). The entire codebase was written in GitHub Copilot without an architectural plan. The result is a single 567-line `grib.py` file that handles GRIB I/O, marine geometry, yacht polars, routing algorithms, and output formatting all at once. All files — source code, GRIB data, logs, and GPX outputs — live in the same directory.

A full code audit identified **30 bugs and 19 architectural violations**, including two cases where entire features never worked correctly. The complete bug list with locations and affected requirements is in `REQUIREMENTS.md`.

---

## Architectural Principle Violations

### grib.py

**A-01 — DRY: TWS/TWD conversion inlined in 4 functions**
`generate_reachable_graph`, `find_shortest_path_dijkstra_3d`, `save_route_detailed_log`, `simulate_vmg_route` all independently compute:
```python
tws = np.sqrt(u**2 + v**2) * 1.94384
twd = (math.degrees(math.atan2(-u, -v)) + 360) % 360
```
The meteorological sign convention and unit factor is a domain concept that belongs in one place. Fix: extract `uv_to_tws_twd(u, v) -> (float, float)` as a pure function in `geo.py`.

**A-02 — DRY + SoC: TWA formula inlined in 4 functions with divergent dead-zone semantics**
`abs(((twd - bearing + 180) % 360) - 180)` appears in four functions. Worse: three use `if twa < 32: continue` to skip the dead zone while `save_route_detailed_log` uses `max(32, twa)` — a silent semantic change (clamps instead of skips). Fix: extract `calculate_twa(twd, bearing) -> float` as a pure function; make the dead-zone decision explicit at each callsite.

**A-03 — DIP: Routing algorithms hard-wired to Scampi 30 polars**
`generate_reachable_graph`, `find_shortest_path_dijkstra_3d`, `simulate_vmg_route`, and `save_route_detailed_log` all call `get_scampi_30_polars()` directly. High-level routing policy depends on a low-level vessel detail. Fix: inject polars as a parameter `polars_fn=None` with a Scampi 30 default — already captured in Stage 3, but the root cause is DIP.

**A-04 — SRP: `save_route_detailed_log` does three things**
(1) Fetches weather from cache for every waypoint. (2) Computes TWS, TWD, TWA, and boat speed per leg. (3) Formats and writes a text file with a hard-coded column layout. These are three distinct responsibilities. Fix: split into `compute_route_weather_profile(points, cache) -> list[dict]` (pure computation) and `write_route_log(profile, filename, label)` (I/O only). The profile becomes reusable for JSON export without re-running weather lookups.

**A-05 — CQS: Query functions with print side effects**
`identify_safe_sailing_areas` returns a dict (query) but also calls `print(f"[SAFE AREA DIAG]...")`. `find_shortest_path_dijkstra_3d` returns a path (query) but prints diagnostics including `nodes_visited` — a value that is tracked internally but never returned to the caller. Fix: remove prints from queries (covered by Stage 5's logging migration); return `nodes_visited` in the result if it is useful.

**A-06 — Fail Fast: `get_weather_from_cache` gives a misleading TypeError on missing U key**
`data_at_time.get('10 metre U wind component')[min_idx]` — when the key is absent, `.get()` returns `None` and `None[min_idx]` raises `TypeError: 'NoneType' object is not subscriptable` with no hint about which file, timestep, or key was involved. Fix: explicit guard with a `KeyError` that names the missing key and lists available keys.

**A-07 — Law of Demeter: cache dict accessed by raw key strings everywhere**
Every routing and analysis function reaches directly into `cache['data'][dt]['10 metre U wind component']`. All callers know the internal nesting structure. If the cache schema changes, every function must change. Fix: encapsulate in a `WeatherCache` dataclass with accessor methods — already implied by the `weather.py` module split in Stage 6, but the LoD violation is the design reason.

**A-08 — OCP: `polars` interpolator reconstructed on every call**
`get_scampi_30_polars()` allocates a full `np.array` and `RegularGridInterpolator` every single time it is called — at least 4 times per routing run. Fix: cache with `@functools.lru_cache(maxsize=1)` or a module-level singleton.

---

### Visualiser.py

**A-09 — SRP + OCP: `load_track` branches on `type_key` string**
The method handles VMG, Dijkstra-2D, and Dijkstra-3D routes via an `if/elif` chain on a `type_key` string. Adding a fourth route type requires modifying the method body. Fix: replace `type_key` with a `RouteSlot` dataclass carrying the object list, color palette, and label list. `load_track` becomes route-type-agnostic.

**A-10 — SRP: `GPXViewerApp.__init__` launches live monitoring**
The constructor calls `self.check_files_loop()`, starting a recurring `root.after` callback immediately on construction. This makes the object untestable in isolation — constructing it starts I/O side effects. Fix: expose `start_monitoring()` as an explicit lifecycle method; call it from `__main__` after construction.

**A-11 — OCP: `change_map_type` uses an if/elif chain**
Adding a new tile provider requires editing the method body. Fix: replace with a `MAP_TILE_URLS` dict — mapping becomes data, not code.

**A-12 — SoC: `check_files_loop` decides to redraw the land mask**
File monitoring and rendering decisions are two separate concerns in one method. Fix: `check_files_loop` sets a dirty flag or calls `on_data_updated()`; rendering consequences are handled separately.

**A-13 — DRY: `force_reload_areas` duplicates `__init__` mtime reset**
Both `__init__` and `force_reload_areas` set all mtime trackers to zero. Adding a new tracked file requires editing both in sync. Fix: extract `_reset_mtimes()` called from both.

**A-14 — YAGNI + SoC: mutable bounding box accumulates but never shrinks**
`update_bounds` accumulates `min_lat/max_lat/min_lon/max_lon` monotonically across all loaded data. Bounds grow forever — even after routes are deleted from the map. The land mask then covers an ever-growing area. Fix: recompute bounds fresh at render time from currently active route objects.

**A-15 — Law of Demeter: `is_square_land` ignores its `delta` parameter**
The method signature promises a spatial tolerance check (`delta` = half-side of cell) but the body calls `globe.is_land(lat, lon)` — a point check only. The `delta` is silently discarded. Every callsite passes a value that has no effect. Fix: either implement the corner-sampling check the signature promises, or remove `delta` and rename to `is_point_land`.

---

### Summary Table

| ID | Principle | Location | Description |
|----|-----------|----------|-------------|
| A-01 | DRY | 4 routing functions | TWS/TWD conversion inlined 4× |
| A-02 | DRY, SoC | 4 routing functions | TWA formula inlined 4×; divergent dead-zone semantics |
| A-03 | DIP | 4 routing functions | Routing algorithms hard-wired to Scampi 30 polars |
| A-04 | SRP | `save_route_detailed_log` | Weather fetch + perf calc + file write in one function |
| A-05 | CQS | `identify_safe_sailing_areas`, `find_shortest_path_dijkstra_3d` | Query functions with print side effects |
| A-06 | Fail Fast | `get_weather_from_cache` | Missing U key gives unhelpful TypeError |
| A-07 | Law of Demeter | All routing functions | Cache dict accessed by raw key strings everywhere |
| A-08 | OCP | `get_scampi_30_polars` | Interpolator reconstructed on every call |
| A-09 | SRP, OCP | `load_track` | String `type_key` branch; adding route type requires modifying method |
| A-10 | SRP | `GPXViewerApp.__init__` | Constructor launches live file monitoring loop |
| A-11 | OCP | `change_map_type` | if/elif chain for tile providers |
| A-12 | SoC | `check_files_loop` | File monitoring coupled to land mask rendering decision |
| A-13 | DRY | `force_reload_areas` | Duplicates `__init__` mtime reset |
| A-14 | YAGNI, SoC | `update_bounds` | Bounding box accumulates monotonically, never recomputed |
| A-15 | LoD, KISS | `is_square_land` | `delta` parameter accepted but silently ignored |

---

## Architectural Decision: Package or Flat Layout

**Recommendation: Python package (`windrouter/`).**

The project has documentation and a refactoring plan — it is not a proof-of-concept. Cost: one `pyproject.toml` and an `__init__.py`. Benefits: clean inter-module imports, `pip install -e .`, a named CLI entrypoint, and straightforward support for a second yacht in the future. Flat layout (all files in the root) produces messy imports and will require `sys.path` hacks in the test suite (see `test/suite` branch).

---

## Target File Structure

```
WindRouter/
├── pyproject.toml               # package metadata, dependencies, CLI entrypoint
├── README.md
├── REQUIREMENTS.md
├── REFACTORING_PLAN.md
├── .github/
│   └── workflows/tests.yml
│
├── windrouter/                  # package — all source code
│   ├── __init__.py
│   ├── geo.py                   # bearing, distance_nm (Haversine), wind_components
│   ├── polars.py                # PolarTable, get_scampi_30_polars
│   ├── weather.py               # load_grib_to_memory, get_weather_from_cache
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── zones.py             # identify_danger_zones, identify_safe_areas
│   │   ├── graph.py             # generate_reachable_graph
│   │   ├── dijkstra.py          # dijkstra_2d, dijkstra_3d
│   │   └── vmg.py               # simulate_vmg_route
│   ├── output.py                # save_track_gpx, save_waypoints_gpx, save_route_log, save_graph_to_json
│   └── cli.py                   # argparse + entrypoint
│
├── data/                        # GRIB files — separated from source (.gitignore *.grb2)
│
├── output/                      # generated GPX, logs, JSON (.gitignore *)
│   └── .gitignore
│
├── tests/
│   └── test_routing_characterization.py
│
└── Visualiser.py                # GUI — stays in root (independent of the package)
```

---

## Execution Plan

### Stage 0 — Project Layout

Before touching any source code. No logic changes.

- Move `*.grb2` into `data/`
- Create `output/` with a `.gitignore` containing `*`
- Add to the root `.gitignore`: `*.grb2`, `*.gpx`, `*.txt`, `sailing_graph.json`, `graph_log.txt`

---

### Stage 1 — Bug Fixes

One commit per bug. Tests must pass after every commit.

**1a. B-18 + B-19** — `finally` in `load_grib_to_memory` and `analyze_grib_performance`

```python
# load_grib_to_memory
try:
    grbs = pygrib.open(file_path)
    try:
        for msg in grbs:
            ...
    finally:
        grbs.close()
except Exception as e:
    ...
```

**1b. B-08** — Split `save_to_gpx` into two functions

```python
def save_track_gpx(points, filename, label="Route"):
    # writes <trkpt> elements — for routing paths

def save_waypoints_gpx(points, filename, label="Areas"):
    # writes <wpt> elements — for danger and caution zones
```

Use `save_waypoints_gpx` for `identify_weather_danger_zones` output.

**1c. B-01 + B-02** — Fix JSON key and add the missing call

```python
# save_graph_to_json: rename "edges" -> "graph"
data = {
    "metadata": {...},
    "nodes": {...},
    "graph": {...}   # was "edges"
}

# in __main__: add the call
save_graph_to_json(reachable_nodes, adj, safe_map, "output/sailing_graph.json")
```

**1d. B-22** — Add U-component fallback key and None guard

```python
u_key = '10 metre U wind component'  # add case heuristic analogous to V
u = data_at_time.get(u_key)
if u is None or v is None:
    return None
```

**1e. B-23 + B-14** — Guard against `None` after `get_weather_from_cache`

```python
weather = get_weather_from_cache(...)
if weather is None:
    continue
```

**1f. B-24** — Initialise `distances` and `predecessors` lazily at push time

```python
# Instead of pre-seeding only from adjacency_map.keys():
if neighbor not in distances:
    distances[neighbor] = float('inf')
    predecessors[neighbor] = None
```

**1g. B-25** — Accumulate Dijkstra 2D timestamps from edge costs

Replace the linear `idx * total_cost / len(path)` assignment with a cumulative sum of individual edge costs along the reconstructed path.

**1h. B-04** — VMG loop condition independent of direction

```python
# Instead of: while curr_lat < target_lat:
while calculate_distance_nm(curr_lat, curr_lon, target_lat, target_lon) > 1.0:
```

**1i. B-26** — Handle `None` in `print_route_summary`

```python
time_str = f"{time_hours:.2f} h" if time_hours is not None else "N/A"
```

**1j. B-20** — Preserve `old_lat` before update in VMG

```python
old_lat = curr_lat
curr_lat += (dist * math.cos(math.radians(best_hdg))) / 60.0
curr_lon += (dist * math.sin(math.radians(best_hdg))) / (60.0 * math.cos(math.radians(old_lat)))
```

**1k. B-21** — Haversine in `calculate_distance_nm`

```python
def calculate_distance_nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # mean Earth radius in nautical miles
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon/2)**2
    return R * 2 * math.asin(math.sqrt(a))
```

**1l. B-27** — Auto-refresh must not stop after a forced reload

```python
def check_files_loop(self, force=False):
    ...
    self.root.after(10000, self.check_files_loop)  # always, not only when not force
```

---

### Stage 2 — Eliminate Duplication (A-01, A-02, A-13)

Zero behaviour changes. Add private helper functions:

```python
def _wind_components(u, v):
    """Return (tws_kt, twd_deg) from GRIB U/V wind components."""
    tws = float(np.sqrt(u**2 + v**2)) * 1.94384
    twd = (math.degrees(math.atan2(-u, -v)) + 360) % 360
    return tws, twd

def _get_v_key(data):
    """Return the correct dict key for the V wind component (case varies by GRIB source)."""
    return '10 metre v wind component' if '10 metre v wind component' in data \
           else '10 metre V wind component'

def _calculate_twa(twd, bearing):
    """Return True Wind Angle (0–180°) for a given TWD and heading/bearing."""
    return abs(((twd - bearing + 180) % 360) - 180)
```

Replace all 5 + 3 + 4 inline copies with calls to these functions. Also extract `_reset_mtimes()` in `Visualiser.py` (A-13) and add `@functools.lru_cache(maxsize=1)` to `get_scampi_30_polars` (A-08).

---

### Stage 3 — Decouple Polars from Routing (A-03)

Change the signature of the 4 routing functions to accept an injected polar interpolator:

```python
def generate_reachable_graph(..., polars=None):
    if polars is None:
        polars = get_scampi_30_polars()

def find_shortest_path_dijkstra_3d(..., polars=None): ...

def simulate_vmg_route(..., polars=None): ...

def save_route_detailed_log(..., polars=None): ...
```

This makes it possible to inject any yacht's polars in tests and in production without touching the routing logic.

---

### Stage 4 — Remove Side Effects and Apply CQS (A-05, A-04)

```python
def generate_reachable_graph(..., log_file=None):
    ...
    if log_file:
        with open(log_file, "a", encoding="utf-8") as log:
            ...
```

`log_file=None` means no file is written. Tests no longer need `tmp_path` just to call the function.

Also split `save_route_detailed_log` (A-04) into:
- `compute_route_weather_profile(points, cache, polars_fn) -> list[dict]` — pure computation
- `write_route_log(profile, filename, label)` — I/O only

And fix `find_shortest_path_dijkstra_3d` to return `nodes_visited` in its result tuple so callers can access it without relying on stdout (A-05).

---

### Stage 5 — Replace `print` with `logging`

```python
import logging
logger = logging.getLogger(__name__)

# Instead of: print(f"[SAFE AREA DIAG] Found {n} safe points")
logger.info("Found %d safe grid points (TWS <= %.1f kt)", n, threshold)
```

Default level `WARNING` — silence in tests, optional `--verbose` flag in the CLI for production use.

---

### Stage 6 — Create the `windrouter` Package (A-07)

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "windrouter"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy", "scipy", "pygrib"]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.scripts]
windrouter = "windrouter.cli:main"
```

Move modules in **dependency order** (bottom of the tree first):

1. `geo.py` — no external dependencies beyond `math`
2. `polars.py` — depends on `numpy`, `scipy`
3. `weather.py` — depends on `geo`
4. `routing/zones.py` — depends on `weather`, `geo`
5. `routing/graph.py` — depends on `zones`, `polars`, `geo`
6. `routing/dijkstra.py` — depends on `polars`, `geo`, `weather`
7. `routing/vmg.py` — depends on `polars`, `geo`, `weather`
8. `output.py` — depends on `geo`, `polars`
9. `cli.py` — `argparse` + calls everything above

Run `pytest` after moving each module. Update `tests.yml`: `--cov=windrouter` instead of `--cov=grib`.

---

### Stage 7 — Refactor Visualiser.py (A-09 through A-15)

- **B-28**: `manual_load_gpx` appends to a separate list instead of overwriting `vmg_files[0]`
- **B-29**: Clear zone shapes when their source GPX file no longer exists
- **B-30**: Replace `np.empty(..., dtype=object)` with `np.full(..., None)` in `identify_weather_danger_zones`
- `draw_global_land_mask` defers rendering via `root.after_idle` instead of blocking the UI thread
- Reset bounding box at the start of `force_reload_areas`
- `is_square_land` checks cell corners, not only the centre point
- Map re-centres only once per forced reload, not on every `index==0` track
- **A-09**: Replace `type_key` string in `load_track` with a `RouteSlot` dataclass
- **A-10**: Move `self.check_files_loop()` out of `__init__` into explicit `start_monitoring()` method
- **A-11**: Replace `change_map_type` if/elif chain with a `MAP_TILE_URLS` dict
- **A-12**: Decouple land mask redraw from `check_files_loop` — use `on_data_updated()` callback
- **A-14**: Recompute bounding box fresh at render time instead of accumulating mutable state
- **A-15**: Implement actual corner-sampling in `is_square_land` or remove `delta` and rename
- Collapse the three identical loops in `check_files_loop` into one loop over a route-type config list
- Read GPX output filenames from a shared constant in `output.py` instead of hard-coding them

---

## PR Breakdown

| PR | Stage | Estimated diff size |
|----|-------|---------------------|
| `chore/project-layout` | 0 | `.gitignore`, file moves |
| `fix/critical-bugs` | 1a–1l | ~120 lines |
| `refactor/eliminate-duplication` | 2–3 | ~80 lines |
| `refactor/clean-api` | 4–5 | ~40 lines |
| `refactor/package-structure` | 6 | large, mechanical, no logic changes |
| `refactor/visualiser` | 7 | ~100 lines, independent of the rest |

Every PR: `pytest` passes, CI green.

---

## Out of Scope

- Changes to routing algorithms (Dijkstra, VMG) — a separate design decision
- Support for additional yachts — the `PolarTable` structure will make this easy, but the polar data itself is separate work
- Tidal current support — requires GRIB files with additional parameters
- Visualiser UI expansion beyond bug fixes and duplication removal
