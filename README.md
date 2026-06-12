# WindRouter — Scampi 30 Weather Routing Engine

A Python weather routing tool for the **Scampi 30** yacht. Reads GRIB forecast files, builds a safe-sailing graph, and finds the fastest route using Dijkstra (2D/3D) and VMG optimisation. Results are exported as GPX tracks and a detailed log.

---

## Project structure

```
WindRouter/
├── grib.py            ← routing engine (polars, routing algorithms, I/O)
├── Visualiser.py      ← Tkinter map viewer
├── data/              ← GRIB forecast files (gitignored)
├── output/            ← generated GPX, logs, JSON (gitignored)
├── docs/
│   ├── REQUIREMENTS.md         ← bug list (B-01 … B-30) and feature requirements
│   ├── REFACTORING_PLAN.md     ← staged plan for fixing bugs and splitting the codebase
│   ├── Technical Documentation Weather Router.md
│   └── Project Documentation Map Display.md
└── tests/
    ├── conftest.py            ← shared fixtures (no real GRIB files needed)
    ├── test_unit.py           ← unit tests with known expected values
    ├── test_characterization.py  ← pins current behaviour including known bugs
    ├── test_regression.py     ← xfail tests: correct behaviour after bug fixes
    └── test_contracts.py      ← I/O round-trip between engine and Visualiser
```

---

## Running the engine

```bash
# place your forecast file in the project root, then:
python grib.py
```

Hard-coded defaults in `__main__`: start/target coordinates, GRIB filename (`data/test.grb2`), output paths. Edit these directly until the CLI is added (Stage 6 of the refactoring plan).

---

## Running the visualiser

```bash
python Visualiser.py
```

Requires Tkinter (standard library). Watches the output directory and reloads GPX/JSON files automatically.

---

## Running tests

### Install test dependencies

```bash
pip install pytest pytest-cov scipy numpy gpxpy
```

pygrib is **not** required for tests — it is stubbed out in `conftest.py`.

### Quick feedback (< 3 seconds)

Skips routing optimisation tests (VMG, Dijkstra 3D, graph construction):

```bash
pytest tests/ -m "not slow"
```

Expected result: **118 passed, 6 xfailed** in ~2 seconds.

### Full suite (~2 minutes)

Includes all slow routing tests:

```bash
pytest tests/
```

Expected result: **148 passed, 8 xfailed**.

### With branch coverage (mirrors CI)

```bash
pytest tests/ --cov=grib --cov-branch --cov-report=term-missing
```

Expected coverage: ≥ 86%.

### Run a single suite

```bash
pytest tests/test_unit.py                        # unit tests
pytest tests/test_unit.py -m "not slow"          # unit tests, fast only
pytest tests/test_characterization.py            # known-bug characterization tests
pytest tests/test_regression.py                  # xfail regression tests
pytest tests/test_contracts.py                   # I/O contract tests
```

### What each test file means

| File | Purpose |
|------|---------|
| `test_unit.py` | Known correct behaviour — these are the refactoring safety net. A failure means behaviour changed and must be verified against REQUIREMENTS.md. |
| `test_characterization.py` | Current (possibly buggy) behaviour pinned. A failure means a bug was fixed — promote the test to unit or delete it. |
| `test_regression.py` | `xfail(strict=True)` — specifies correct behaviour *after* each bug is fixed. Turns green when the bug is fixed; turns red if re-introduced. |
| `test_contracts.py` | Round-trip between `grib.py` (writer) and `Visualiser.py` (reader). Catches silent I/O mismatches like B-01 and B-08. |

---

## Known bugs

30 documented bugs are listed with locations, descriptions, and affected requirements in `docs/REQUIREMENTS.md`. The staged fix plan is in `docs/REFACTORING_PLAN.md`.

---

## Dependencies

| Package | Use |
|---------|-----|
| `pygrib` | Read GRIB2 forecast files (requires ecCodes system library) |
| `numpy` | Grid arithmetic |
| `scipy` | `RegularGridInterpolator` for polar table lookup |
| `gpxpy` | Parse GPX files in tests |

### Install ecCodes (required for pygrib)

**Ubuntu/Debian:**
```bash
sudo apt-get install libeccodes-dev
pip install pygrib numpy scipy
```

**macOS (Homebrew):**
```bash
brew install eccodes
pip install pygrib numpy scipy
```

**Windows:** Use WSL or a conda environment with `conda install -c conda-forge pygrib`.
