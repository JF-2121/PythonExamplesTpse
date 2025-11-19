# TPMS / hybridization example — quick run

This repository contains example scripts that generate, blend and export TPMS (triply periodic minimal surface) meshes.

Top-level notes
- A small local shim for `microgen` has been added at `microgen/surface_functions.py` which delegates to `utils/tpms_unitcell_library.py`. This makes the example scripts runnable without an external `microgen` package. If you prefer the real `microgen`, install it and remove or ignore the shim.
- Several scripts auto-add the repo `utils/` directory onto `sys.path` so `isocaps_structured` is importable. Some scripts still require running from the repo root or setting PYTHONPATH (examples below).

## Quickstart (recommended)

1. Create and activate a virtual environment (from project root):

```bash
cd /home/jflinux/Downloads/download/python_examples
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the hybridization example (generates TPMS meshes):

```bash
# From project root
python hybridization_example/hybridization_01.py
```

The hybridization run will generate (in the project root):
- `reduced_tpms_mesh.stl` — reduced/smoothed TPMS shell suitable for remeshing
- `tpms_boolean_result.stl` — boolean/combined result prior to reduction

## Running the cap extractor (per-face caps & triangulation)

The cap extraction script `tpms_generator_3rd_attempt_best_so_far/extract_and_mesh_caps_02.py` reads TPMS fields and produces triangulated caps for each boundary face.

Run it from the repo root so local packages are importable (or set PYTHONPATH):

```bash
# Option A: run from repo root (preferred)
python tpms_generator_3rd_attempt_best_so_far/extract_and_mesh_caps_02.py

# Option B: explicitly add project root to PYTHONPATH (works from anywhere)
PYTHONPATH=$(pwd) python tpms_generator_3rd_attempt_best_so_far/extract_and_mesh_caps_02.py
```

On success the script prints per-face summaries like:

	[x0] pieces=25 selected=13 tri_meshes=13

You can enable/disable 2D/3D previews via the booleans at the top of that script (SHOW_2D_PLOTS, SHOW_3D_PREVIEW). For headless runs either set `SHOW_3D_PREVIEW=False` or enable off-screen PV rendering (see below).

## Remeshing the exported STL with Gmsh

There's an example remesher at `smoothing_and_remeshing/gmsh_t13_stl_remesh_01.py`. I updated it to default to the hybridization-generated `reduced_tpms_mesh.stl` and to write outputs to `smoothing_and_remeshing/output`.

Run it non-interactively (no GUI) from the repo root:

```bash
python smoothing_and_remeshing/gmsh_t13_stl_remesh_01.py -nopopup
```

If you prefer the interactive ONELAB panel, omit `-nopopup` and run in an environment with a display.

## Headless rendering notes
- PyVista opens an interactive window by default. For headless servers set off-screen mode before creating a Plotter in scripts that use PyVista:

```python
import pyvista as pv
pv.OFF_SCREEN = True
```

## Common troubleshooting
- ModuleNotFoundError: install missing packages with `pip install -r requirements.txt`.
- `microgen` not found: the repository contains a local shim under `microgen/`. If you have an external `microgen`, install it and optionally remove the local shim.
- `isocaps_structured` import errors: run scripts from the project root or set `PYTHONPATH=$(pwd)` so `utils/` is on the import path.
- Gmsh warnings: partitioning and topology warnings are common for dense/small triangles. If Gmsh stalls or is slow, try coarser settings in `smoothing_and_remeshing/gmsh_t13_stl_remesh_01.py` (reduce smoothing iterations, change mesh_algorithm_2d, set use_netgen_optimize=False).

## Files created by the example runs
- `reduced_tpms_mesh.stl` — reduced TPMS shell (created by `hybridization_example/hybridization_01.py`)
- `tpms_boolean_result.stl` — boolean mesh before reduction
- `smoothing_and_remeshing/output/remeshed_surface.stl` & `.msh` — output from the Gmsh remesher (when the remesher run completes)
