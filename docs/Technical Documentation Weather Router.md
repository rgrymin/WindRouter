# Technical Documentation: Scampi 30 GRIB Router & Performance Analyzer

This document describes the technical implementation of the `grib.py` routing engine and the `Visualiser.py` map viewer. It is intended for developers maintaining or extending the codebase.

For the full bug list and refactoring roadmap see `REQUIREMENTS.md` and `REFACTORING_PLAN.md`.

---

## 1. System Architecture

The application follows a linear processing pipeline:

1. **Data Ingestion** — `load_grib_to_memory`: parses a GRIB2 file once into a RAM cache.
2. **Performance Modelling** — `get_scampi_30_polars`: initialises a 2D interpolator for the Scampi 30 polar table (TWA × TWS → Boat Speed).
3. **Spatial Analysis** — `identify_safe_sailing_areas` / `identify_weather_danger_zones`: classifies grid cells by TWS threshold across the full forecast window.
4. **Routing engines** (three independent implementations):
   - **Dijkstra 2D** (`find_shortest_path_dijkstra`): shortest path on a static weather snapshot at departure time. Fast, ignores time evolution.
   - **Dijkstra 3D** (`find_shortest_path_dijkstra_3d`): time-aware Dijkstra; weather is re-sampled at each node based on accumulated elapsed time. Slower but more accurate on long passages.
   - **VMG simulation** (`simulate_vmg_route`): greedy iterative solver; scans all headings every 10 minutes and picks the best VMG towards the target.
5. **Export** — `save_route_detailed_log`, `save_to_gpx`, `save_graph_to_json`: TXT log, GPX track, and JSON graph for the Visualiser.

---

## 2. Core Components

### 2.1 Weather Data Handling

**Cache structure** returned by `load_grib_to_memory`:

```python
{
    "data":  {datetime: {"10 metre U wind component": np.ndarray,
                         "10 metre v wind component": np.ndarray}},
    "dates": [datetime, ...],   # sorted ascending
    "lats":  np.ndarray,        # 2-D grid
    "lons":  np.ndarray,
}
```

`get_weather_from_cache` finds the nearest grid point by minimum Euclidean distance (no interpolation) and the nearest date. Sets `meta["approximated"] = True` when the requested time falls outside the forecast window.

**V-component key**: GRIB files from different sources use `"10 metre v wind component"` (lowercase v) or `"10 metre V wind component"` (uppercase V). The code checks both. There is no equivalent fallback for the U-component (B-22).

### 2.2 Yacht Performance (Polars)

`get_scampi_30_polars` returns a `scipy.interpolate.RegularGridInterpolator` fitted to the Scampi 30 polar table. Lookup: `polars([TWA, TWS])` → Boat Speed in knots.

**Dead zone**: TWA < 32° → edge/heading skipped (`continue`) in all three routing engines. The `9999` hour fallback in Dijkstra 2D applies only when `bs <= 0` due to polar extrapolation outside the table, not to the dead zone itself.

**Note**: `get_scampi_30_polars` constructs a new interpolator on every call — there is no memoization (C-16).

### 2.3 Reachable Graph (Dijkstra 2D & 3D)

`generate_reachable_graph` builds the adjacency map with BFS:

- **Safe points only**: cells where TWS never exceeds the threshold across all forecast times.
- **Angle filter**: a neighbour is connected only if the bearing to it is within ±80° of the bearing to the final target (ensures forward progress).
- **Dead zone**: TWA < 32° → edge skipped entirely (`continue`) in both Dijkstra 2D and 3D.
- **Cost**: `distance_nm / boat_speed_kt` in hours. If `bs <= 0` (extrapolation artefact outside dead zone), cost falls back to `9999.0` in Dijkstra 2D; edge is skipped in Dijkstra 3D.

**Known issue — B-24**: `distances` and `predecessors` are pre-seeded only from `adjacency_map.keys()`. Leaf nodes that appear only as edge targets (not as source keys) raise `KeyError` during relaxation.

**Known issue — B-25**: Dijkstra 2D timestamps are distributed linearly (`idx / len * total_cost`) instead of accumulated from edge costs — every timestamp in the GPX and log is wrong.

### 2.4 VMG Simulation

Iterative greedy solver in `simulate_vmg_route`:

1. **Heading scan**: every 2° from 0–358°, compute potential BS from polars.
2. **VMG**: `VMG = BS × cos(heading − bearing_to_target)`. Best heading = max VMG.
3. **Boundary guard**: predicted next position is checked against the GRIB grid extent; out-of-bounds headings are discarded.
4. **Termination**: within 1 nm of target, or 10 000 steps reached.

**Known issue — B-04**: loop condition `while curr_lat < target_lat` hard-codes a northward destination. Any southbound or same-latitude target produces an empty route immediately.

**Known issue — B-20**: the longitude update uses `cos(new_lat)` (after the latitude has already been incremented) instead of `cos(old_lat)`.

---

## 3. Function Reference

### Utility

| Function | Description |
|---|---|
| `calculate_bearing(lat1, lon1, lat2, lon2)` | Initial bearing using spherical trigonometry. Returns 0–360°. Returns 0 for identical points (B-06, no guard). |
| `calculate_distance_nm(lat1, lon1, lat2, lon2)` | Flat-earth formula with cos(avg\_lat) correction. Accurate to ~0.1% for short distances at mid-latitudes; breaks at the antimeridian (B-21). |

### Data Processing

| Function | Description |
|---|---|
| `load_grib_to_memory(file_path)` | Parses GRIB2, returns cache dict or `None` on error. `grbs.close()` is not in a `finally` block — file handle leaks on exception (B-18). |
| `get_weather_from_cache(cache, lat, lon, time)` | Nearest-neighbour lookup. Returns `None` for falsy cache. |
| `identify_safe_sailing_areas(cache, max_wind_threshold=30)` | Returns `{(row, col): {lat, lon, max_speed}}` for cells safe across all timesteps. |
| `identify_weather_danger_zones(cache, min_threshold, max_threshold=inf)` | Returns list of `{lat, lon, speed, time}` for cells that exceeded the threshold. Records first exceedance time. |

### Routing

| Function | Description |
|---|---|
| `generate_reachable_graph(cache, spm, ...)` | BFS graph construction. Always writes `graph_log.txt` as a side effect. Returns `(nodes, adjacency_map, start_node)`. |
| `find_shortest_path_dijkstra(start, adj, spm, target_lat, target_lon)` | Dijkstra 2D on static weather snapshot. Path points are **direct references** into `spm` (not copies). |
| `find_shortest_path_dijkstra_3d(start, spm, target_lat, target_lon, t0, cache)` | Time-aware Dijkstra. Path points are **copies** of `spm` entries with `time` key added. |
| `simulate_vmg_route(cache, start_lat, start_lon, target_lat, target_lon, start_time)` | Greedy VMG solver. Northbound only (B-04). |

### Output

| Function | Description |
|---|---|
| `save_route_detailed_log(points, cache, filename, label)` | Writes a pipe-delimited TXT table: Time \| Lat \| Lon \| TWS \| TWD \| Heading \| TWA \| BS. |
| `print_route_summary(points, label, time_hours=None)` | Prints total distance and elapsed time to stdout. |
| `save_to_gpx(points, filename, label)` | GPX 1.1 track using `<trkpt>` elements. |
| `save_graph_to_json(nodes, adj, spm, filename)` | Writes JSON with keys `metadata`, `nodes`, `edges`. |
| `analyze_grib_performance(file_path)` | Prints GRIB grid diagnostics. |

For known bugs in these functions (B-01, B-02, B-08, B-19, B-26, and others) see `REQUIREMENTS.md`.

---

## 4. Engine ↔ Visualiser Contracts

Two file formats are produced by `grib.py` and consumed by `Visualiser.py`. Both contracts are currently broken:

| File | Written by | Key written | Read by | Key read | Status |
|---|---|---|---|---|---|
| `sailing_graph.json` | `save_graph_to_json` | `"edges"` | `load_sailing_graph` | `"graph"` | **Broken — B-01** |
| `*.gpx` (zones) | `save_to_gpx` | `<trkpt>` | `load_area_file` | `gpx.waypoints` (`<wpt>`) | **Broken — B-08** |

---

## 5. Maintenance Notes

### Adjusting safety thresholds

The forbidden-zone limit defaults to **40 kt** and the caution-zone limit to **30 kt**. Both are passed as arguments to `identify_weather_danger_zones` and `identify_safe_sailing_areas` in `__main__`.

### Changing start/target coordinates

Hard-coded in `__main__`. GRIB filename: `test.grb2`. Start: `(53.25N, 2.6E)`. Target: northern edge of the grid at the longitudinal midpoint. Runs 4 simulations at 5-hour departure intervals. Update directly until the CLI is added (Stage 6 of REFACTORING_PLAN.md).

### Performance

If graph construction is slow:
- Tighten the `angle_diff` constraint (default ±80°).
- Reduce GRIB grid resolution at source.
- Use Dijkstra 2D instead of 3D for a first approximation.

### Running tests

See `README.md` for test commands once the test suite is added.
