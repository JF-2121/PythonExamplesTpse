#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Gmsh surface remeshing of an STL (no CAD volume)
#  - Discrete STL -> reparametrize -> surface remesh
#  - Configurable input/output, sizing, smoothing, and patching
#  - Boundary-preserving mode (only adapt inside)
#  - ONELAB sliders for live tweaking (GUI)
# ------------------------------------------------------------------------------

import gmsh
import math
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# USER CONFIGURATION
# ------------------------------------------------------------------------------
# --- Paths ---
# Default to the example reduced mesh produced by `hybridization_example` and
# write outputs into a local `smoothing_and_remeshing/output` folder.
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
input_stl = PROJECT_ROOT / "reduced_tpms_mesh.stl"
output_dir = _Path(__file__).resolve().parent / "output"
output_basename = "remeshed_surface"

# --- I/O behavior ---
enable_export    = True     # write .msh and .stl
export_ascii     = False    # True = ASCII STL, False = binary
use_file_dialogs = False    # True = choose paths via Tk dialogs at runtime

# --- Remeshing defaults (also exposed in ONELAB) ---
default_angle_deg        = 90   # feature detection angle
default_curve_split_deg  = 0    # extra splitting of discrete curves (smaller => more patches)
default_force_param      = 1    # 1 => enforce parametrizable patches (robust)
default_funny_field      = 0    # 1 => enable demo background field (overrides global sizes)

# --- Meshing / smoothing options ---
mesh_smoothing_iters     = 20
use_netgen_optimize      = True
recombine_quads          = False

# --- 2D meshing algorithm ---
# 1=MeshAdapt, 5=Delaunay, 6=Frontal-Delaunay, 8=Frontal-Quad
mesh_algorithm_2d        = 1

# --- Size controls ---
# If you want these to control sizing, keep `default_funny_field = 0`
characteristic_min       = None   # set None to skip
characteristic_max       = None  # set e.g. 4.0, or None to skip
use_curvature_sizing     = False # curvature-based sizing (global)
size_from_points         = True
size_from_curves         = True
size_from_surfaces       = True
extend_from_boundary     = True

# --- Boundary preservation (this is what you asked for) ---
keep_boundaries_fixed    = True  # sets Mesh.AdaptInside=1 and Mesh.BoundaryPreserving=1

# ------------------------------------------------------------------------------
# OPTIONAL Tk FILE PICKERS
# ------------------------------------------------------------------------------
if use_file_dialogs:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        stl_path = filedialog.askopenfilename(
            title="Select input STL",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )
        if stl_path:
            input_stl = stl_path
        out_dir = filedialog.askdirectory(title="Select output directory")
        if out_dir:
            output_dir = out_dir
    except Exception:
        print("[WARN] Tkinter dialogs failed; falling back to configured paths.")

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------
input_stl  = Path(input_stl)
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if not input_stl.exists():
    raise FileNotFoundError(f"Input STL not found:\n  {input_stl}")

gmsh.initialize()

_ALGO2D_NAME = {
    1: "MeshAdapt",
    2: "Automatic",
    3: "Initial mesh only",
    5: "Delaunay",
    6: "Frontal-Delaunay",
    7: "BAMG",
    8: "Frontal-Delaunay for Quads",
}

def safe_set_number(name: str, value: float):
    """Set a numerical option only if supported by this build."""
    try:
        opt_type = gmsh.option.getType(name)  # 0=number, 1=string, 2=color, -1=unknown
    except Exception:
        opt_type = -1
    if opt_type in (0, -1):
        try:
            gmsh.option.setNumber(name, float(value))
        except Exception:
            print(f"[NOTE] Option not available in this build: {name}")
    else:
        print(f"[NOTE] Option exists but is not numeric: {name} (type={opt_type})")

def _apply_common_options():
    """Apply global mesh options in a build-safe way."""
    # Export behavior
    safe_set_number("Mesh.SaveAll", 1)
    safe_set_number("Mesh.Binary", 0 if export_ascii else 1)

    # Recombine & smoothing
    safe_set_number("Mesh.RecombineAll", 1 if recombine_quads else 0)
    safe_set_number("Mesh.Smoothing", int(mesh_smoothing_iters))

    # Global size controls (legacy + new aliases)
    if characteristic_min is not None:
        safe_set_number("Mesh.CharacteristicLengthMin", float(characteristic_min))
        safe_set_number("Mesh.MeshSizeMin",             float(characteristic_min))
    if characteristic_max is not None:
        safe_set_number("Mesh.CharacteristicLengthMax", float(characteristic_max))
        safe_set_number("Mesh.MeshSizeMax",             float(characteristic_max))

    # Curvature sizing
    safe_set_number("Mesh.CharacteristicLengthFromCurvature", 1 if use_curvature_sizing else 0)
    safe_set_number("Mesh.MeshSizeFromCurvature",            1 if use_curvature_sizing else 0)

    # Entity-based sizing toggles (may be unsupported for discrete-only workflows)
    safe_set_number("Mesh.CharacteristicLengthFromPoints",    1 if size_from_points else 0)
    safe_set_number("Mesh.CharacteristicLengthFromCurves",    1 if size_from_curves else 0)
    safe_set_number("Mesh.CharacteristicLengthFromSurfaces",  1 if size_from_surfaces else 0)

    # Extend sizes from boundary inward
    safe_set_number("Mesh.CharacteristicLengthExtendFromBoundary", 1 if extend_from_boundary else 0)

    # --- Boundary preservation (your request) ---
    if keep_boundaries_fixed:
        safe_set_number("Mesh.AdaptInside", 1)         # Only adapt interior
        safe_set_number("Mesh.BoundaryPreserving", 1)  # Keep boundary nodes/edges

def _print_quality_summary():
    try:
        mq = gmsh.model.mesh.getQualityType()
        if mq[0] == "":
            gmsh.model.mesh.setQualityType("Gamma")
        qualities = gmsh.model.mesh.getElementQualities()
        if qualities:
            qmin = min(qualities)
            qmax = max(qualities)
            qavg = sum(qualities)/len(qualities)
            print(f"[QUALITY] min={qmin:.4f}  avg={qavg:.4f}  max={qmax:.4f}  (n={len(qualities)})")
    except Exception:
        pass

def createSurfaceGeometryAndMesh():
    gmsh.clear()
    gmsh.logger.start()

    print(f"[INFO] Loading STL: {input_stl}")
    gmsh.merge(str(input_stl))

    # --- ONELAB parameters ---
    angle_deg       = gmsh.onelab.getNumber("Parameters/Angle for surface detection")[0]
    force_param     = gmsh.onelab.getNumber("Parameters/Create surfaces guaranteed to be parametrizable")[0]
    funny_field     = gmsh.onelab.getNumber("Parameters/Apply funny mesh size field?")[0]
    curve_angle_deg = gmsh.onelab.getNumber("Parameters/Curve split angle (extra patching)")[0]

    includeBoundary = True

    # --- 1) Classify into discrete patches ---
    gmsh.model.mesh.classifySurfaces(angle_deg * math.pi / 180.0,
                                     includeBoundary,
                                     int(force_param),
                                     curve_angle_deg * math.pi / 180.0)

    # --- 2) Create discrete geometry from classified mesh ---
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # --- 3) Global options (sizes, curvature, boundary-preserving, etc.) ---
    _apply_common_options()

    # --- 4) OPTIONAL background field (overrides global sizes if enabled) ---
    if int(funny_field) == 1:
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", "2*Sin((x+y)/5) + 3")
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        print("[INFO] Background field enabled (overrides global size controls).")
    else:
        try:
            gmsh.model.mesh.field.removeBackgroundMesh()
        except Exception:
            pass

    # --- 4b) Explicitly freeze boundary curves and points ---
    # This ensures boundary edges remain fixed in position.
    print("[INFO] Freezing boundary curves and corner points...")
    try:
        for dim, tag in gmsh.model.getEntities(1):  # curves
            gmsh.model.mesh.setTransfiniteCurve(tag, 2)
        for dim, tag in gmsh.model.getEntities(0):  # points
            gmsh.model.mesh.setSize([(dim, tag)], 1e-7)
    except Exception as e:
        print(f"[WARN] Could not freeze boundaries explicitly: {e}")

    # --- 5) Set meshing algorithm and generate 2D mesh ---
    algo_name = _ALGO2D_NAME.get(mesh_algorithm_2d, "Custom/Unknown")
    safe_set_number("Mesh.Algorithm", float(mesh_algorithm_2d))
    print(f"[INFO] 2D meshing algorithm: {mesh_algorithm_2d} ({algo_name})")
    print("[INFO] Generating surface mesh...")
    gmsh.model.mesh.generate(2)

    # --- 6) Optional post-optimization (Netgen) ---
    if use_netgen_optimize:
        gmsh.model.mesh.optimize("Netgen")

    # --- 7) Quality summary ---
    _print_quality_summary()
    print("[INFO] Surface remeshing complete.")

    # --- 8) Optional export ---
    if enable_export:
        msh_path = output_dir / f"{output_basename}.msh"
        stl_path = output_dir / f"{output_basename}.stl"
        gmsh.write(str(msh_path))
        gmsh.write(str(stl_path))
        print(f"[INFO] Exported to:\n  {msh_path}\n  {stl_path}")

    # --- 9) Log (warnings, notes) ---
    logs = gmsh.logger.get()
    if logs:
        print("[GMSH LOG]")
        print("\n".join(logs))
    gmsh.logger.stop()

# ------------------------------------------------------------------------------
# ONELAB PANEL (GUI sliders)
# ------------------------------------------------------------------------------
gmsh.onelab.set(f"""[
  {{
    "type":"number",
    "name":"Parameters/Angle for surface detection",
    "values":[{default_angle_deg}],
    "min":0, "max":120, "step":1
  }},
  {{
    "type":"number",
    "name":"Parameters/Create surfaces guaranteed to be parametrizable",
    "values":[{default_force_param}],
    "choices":[0,1]
  }},
  {{
    "type":"number",
    "name":"Parameters/Curve split angle (extra patching)",
    "values":[{default_curve_split_deg}],
    "min":0, "max":180, "step":1
  }},
  {{
    "type":"number",
    "name":"Parameters/Apply funny mesh size field?",
    "values":[{default_funny_field}],
    "choices":[0,1]
  }}
]""")

# ------------------------------------------------------------------------------
# RUN + GUI LOOP
# ------------------------------------------------------------------------------
createSurfaceGeometryAndMesh()

def checkForEvent():
    action = gmsh.onelab.getString("ONELAB/Action")
    if action and action[0] == "check":
        gmsh.onelab.setString("ONELAB/Action", [""])
        createSurfaceGeometryAndMesh()
        gmsh.graphics.draw()
    return True

if "-nopopup" not in sys.argv:
    gmsh.fltk.initialize()
    while gmsh.fltk.isAvailable() and checkForEvent():
        gmsh.fltk.wait()

gmsh.finalize()
