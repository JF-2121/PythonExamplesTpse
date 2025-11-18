#!/usr/bin/env python3
# tpms_shell_boolean_demo_updated.py
# Python 3.8+

import sys
import os
import logging
import numpy as np

# Compatibility shims for libs expecting NumPy < 2.0 aliases
if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "int_"):     np.int_     = np.int64
if not hasattr(np, "complex_"): np.complex_ = np.complex128

import pyvista as pv
import trimesh
from pathlib import Path

# --- Your modules ---
from microgen import surface_functions as sf
from isocaps_structured import isocaps_structured

# (Optional) Abaqus exporter
abaqus_script_dir = Path(r"path_to_esportMeshtoAbaqus")
if str(abaqus_script_dir) not in sys.path:
    sys.path.append(str(abaqus_script_dir))
try:
    from exportMeshtoAbaqus import exportMeshtoAbaqus
except Exception:
    exportMeshtoAbaqus = None  # safe if you don't export .inp

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def pv_to_trimesh(poly: pv.PolyData) -> trimesh.Trimesh:
    """Triangulate PyVista mesh and convert to trimesh."""
    tri = poly.triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    tm = trimesh.Trimesh(vertices=tri.points, faces=faces, process=False)
    tm_cleanup_inplace(tm)
    return tm

def tm_cleanup_inplace(tm: trimesh.Trimesh, nondeg_height=None) -> None:
    """
    Replace deprecated remove_* calls with modern trimesh API.
    Cleans faces, welds, fixes normals, fills tiny holes.
    """
    # Remove duplicate & degenerate faces
    tm.update_faces(tm.unique_faces())
    if nondeg_height is None:
        try:
            scale_ref = float(tm.scale) if hasattr(tm, 'scale') else 1.0
        except Exception:
            scale_ref = 1.0
        nondeg_height = max(1e-12, 1e-9 * scale_ref)
    tm.update_faces(tm.nondegenerate_faces(height=nondeg_height))

    # Housekeeping
    tm.remove_unreferenced_vertices()
    tm.merge_vertices()
    tm.rezero()
    try:
        trimesh.repair.fix_inversion(tm)
        trimesh.repair.fill_holes(tm)
    except Exception:
        pass
    tm.fix_normals()

# ------------------------------------------------------------------------------
# Thickness gradation
# ------------------------------------------------------------------------------

def compute_thickness_gradation(
    z_values: np.ndarray,
    t_min: float,
    t_max: float,
    a_percent: float,
    b_percent: float,
    gradation_type: str = 'linear',
    fixed_band: float = 0.25,
    interpret_as_half_thickness: bool = True,
) -> np.ndarray:
    """
    Thickness field with fixed plateau bands at both ends (default 25% each).
    Middle region is split by 'a_percent' and 'b_percent':
      - Decrease from t_max at L to t_min at X
      - Hold t_min from X to Y
      - Increase from t_min at Y to t_max at U

    L = fixed_band
    U = 1 - fixed_band
    X = L + (U - L) * (a_percent / 100)
    Y = L + (U - L) * (1 - b_percent / 100)

    If interpret_as_half_thickness=True, returns half-thickness values
    (useful when applying ±isosurface offsets around the midsurface).
    """
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    if z_max == z_min:
        z_norm = np.zeros_like(z_values, dtype=float)
    else:
        z_norm = (z_values - z_min) / (z_max - z_min)

    # Fixed plateaus
    L = np.clip(float(fixed_band), 0.0, 0.5)
    U = 1.0 - L

    # Transition anchors in the middle region
    a = np.clip(a_percent / 100.0, 0.0, 1.0)
    b = np.clip(b_percent / 100.0, 0.0, 1.0)
    X = L + (U - L) * a
    Y = L + (U - L) * (1.0 - b)

    # Ensure ordering if extreme values provided
    X = np.clip(X, L, U)
    Y = np.clip(Y, L, U)
    if Y < X:
        X, Y = Y, X

    T = np.empty_like(z_norm, dtype=float)

    # Outside middle: fixed t_max
    T[z_norm <= L] = t_max
    T[z_norm >= U] = t_max

    # Middle zone
    mid_mask = (z_norm > L) & (z_norm < U)
    z_mid = z_norm[mid_mask]
    T_mid = np.empty_like(z_mid, dtype=float)

    m1 = z_mid <= X
    m2 = (z_mid > X) & (z_mid < Y)
    m3 = z_mid >= Y

    gt = (gradation_type or 'linear').lower()
    if gt == 'linear':
        if X > L:
            T_mid[m1] = t_max - (t_max - t_min) * ((z_mid[m1] - L) / (X - L))
        else:
            T_mid[m1] = t_min
        T_mid[m2] = t_min
        if U > Y:
            T_mid[m3] = t_min + (t_max - t_min) * ((z_mid[m3] - Y) / (U - Y))
        else:
            T_mid[m3] = t_max

    elif gt in ('sin', 'sinus', 'sinusoidal', 'cos', 'cosine'):
        if X > L:
            T_mid[m1] = t_max - (t_max - t_min) * 0.5 * (1.0 - np.cos(np.pi * (z_mid[m1] - L) / (X - L)))
        else:
            T_mid[m1] = t_min
        T_mid[m2] = t_min
        if U > Y:
            T_mid[m3] = t_min + (t_max - t_min) * 0.5 * (1.0 - np.cos(np.pi * (z_mid[m3] - Y) / (U - Y)))
        else:
            T_mid[m3] = t_max
    else:
        raise ValueError("gradation_type must be 'linear' or 'sinusoidal'")

    T[mid_mask] = T_mid

    if interpret_as_half_thickness:
        T *= 0.5

    return T

# ------------------------------------------------------------------------------
# Core generator
# ------------------------------------------------------------------------------

def tpms_two_iso_shell(
    tpms_func=sf.gyroid,
    offset=0.30,
    n=23,
    cell_size=10.0,
    nx=1, ny=1, nz=1,
    k1=1.0, k2=1.0, sym=1,
    export_path=None,
    gradation_mode='none',
    t_min=0.0, t_max=0.0, a_percent=0.0, b_percent=0.0,
    fixed_band=0.25,
    show_plot=True,
    plot_mode: str = "overlap",           # NEW: "separate" | "overlap" | "both"
    color_inner: str = "blue",            # NEW
    color_outer: str = "red"              # NEW
) -> trimesh.Trimesh:
    """Build inner/outer isosurfaces with optional z-gradation and boolean-intersect them."""

    def create_grid_and_field(tpms_fn, nx, ny, nz, n, k1, k2, sym):
        x = np.linspace(-np.pi * nx * k1, np.pi * nx * k2 * sym, n * nx)
        y = np.linspace(-np.pi * ny * k1, np.pi * ny * k2 * sym, n * ny)
        z = np.linspace(-np.pi * nz * k1, np.pi * nz * k2, n * nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        F = tpms_fn(X, Y, Z)
        return x, y, z, X, Y, Z, F

    # Build once (reuse for inner/outer)
    x_in, y_in, z_in, XIN, YIN, ZIN, FIN = create_grid_and_field(tpms_func, nx, ny, nz, n, k1, 1.0, sym)
    x_out, y_out, z_out, XOUT, YOUT, ZOUT, FOUT = create_grid_and_field(tpms_func, nx, ny, nz, n, 1.0, k2, sym)

    gm = (gradation_mode or 'none').lower()
    if gm != 'none':
        inner_delta = compute_thickness_gradation(
            ZIN, t_min, t_max, a_percent, b_percent,
            gradation_type=gm, fixed_band=fixed_band, interpret_as_half_thickness=True
        )
        outer_delta = compute_thickness_gradation(
            ZOUT, t_min, t_max, a_percent, b_percent,
            gradation_type=gm, fixed_band=fixed_band, interpret_as_half_thickness=True
        )
    else:
        # Constant half-offset around midsurface
        inner_delta = np.full_like(FIN, float(offset) * 0.5)
        outer_delta = np.full_like(FOUT, float(offset) * 0.5)

    # Structured grid with shifted scalars
    grid_in = pv.StructuredGrid()
    grid_out = pv.StructuredGrid()
    grid_in.points = np.c_[XIN.ravel(order='F'), YIN.ravel(order='F'), ZIN.ravel(order='F')]
    grid_in.dimensions = XIN.shape
    grid_in['F'] = FIN.ravel(order='F')
    grid_out.points = np.c_[XOUT.ravel(order='F'), YOUT.ravel(order='F'), ZOUT.ravel(order='F')]
    grid_out.dimensions = XOUT.shape
    grid_out['F'] = FOUT.ravel(order='F')
    grid_in['F_inner'] = (FIN - inner_delta).ravel(order='F')  # inner shell: F - δ = 0
    grid_out['F_outer'] = (FOUT + outer_delta).ravel(order='F')  # outer shell: F + δ = 0

    # Extract isosurfaces & caps
    iso_inner = grid_in.contour([0.0], scalars='F_inner')
    iso_outer = grid_out.contour([0.0], scalars='F_outer')
    caps_inner = isocaps_structured(x_in, y_in, z_in, FIN, +inner_delta, which_caps='below')
    caps_outer = isocaps_structured(x_out, y_out, z_out, FOUT, -outer_delta, which_caps='above')

    shell_pos_pv = iso_inner.merge(caps_inner)
    shell_neg_pv = iso_outer.merge(caps_outer)

    # --- visualization options ---
    if show_plot:
        mode = (plot_mode or "overlap").lower()
        if mode not in ("separate", "overlap", "both"):
            mode = "overlap"

        if mode == "separate":
            p = pv.Plotter(shape=(1, 2), title="TPMS Shells (separate)")
            # Left: Inner shell
            p.subplot(0, 0)
            p.add_text("Inner shell (F - δ = 0)", font_size=12)
            p.add_mesh(shell_pos_pv, color=color_inner, show_edges=True, smooth_shading=False)
            p.add_axes(); p.view_isometric()
            # Right: Outer shell
            p.subplot(0, 1)
            p.add_text("Outer shell (F + δ = 0)", font_size=12)
            p.add_mesh(shell_neg_pv, color=color_outer, show_edges=True, smooth_shading=False)
            p.add_axes(); p.view_isometric()
            p.show()

        elif mode == "overlap":
            p = pv.Plotter(title="TPMS Shells (overlap)")
            p.add_mesh(shell_pos_pv, color=color_inner, opacity=0.55, smooth_shading=False, name="Inner")
            p.add_mesh(shell_neg_pv, color=color_outer, opacity=0.55, smooth_shading=False, name="Outer")
            p.add_legend([[f"Inner (F - δ = 0)", color_inner],
                          [f"Outer (F + δ = 0)", color_outer]])
            p.add_axes(); p.view_isometric()
            p.show()

        else:  # mode == "both"
            p = pv.Plotter(shape=(1, 3), title="TPMS Shells (separate + overlap)")
            # 0: Inner
            p.subplot(0, 0)
            p.add_text("Inner", font_size=12)
            p.add_mesh(shell_pos_pv, color=color_inner, show_edges=True, smooth_shading=False)
            p.add_axes(); p.view_isometric()
            # 1: Outer
            p.subplot(0, 1)
            p.add_text("Outer", font_size=12)
            p.add_mesh(shell_neg_pv, color=color_outer, show_edges=True, smooth_shading=False)
            p.add_axes(); p.view_isometric()
            # 2: Overlap
            p.subplot(0, 2)
            p.add_text("Overlap", font_size=12)
            p.add_mesh(shell_pos_pv, color=color_inner, opacity=0.55, smooth_shading=False)
            p.add_mesh(shell_neg_pv, color=color_outer, opacity=0.55, smooth_shading=False)
            p.add_axes(); p.view_isometric()
            p.show()

    # Convert to trimesh & clean
    tm_pos = pv_to_trimesh(shell_pos_pv)
    tm_neg = pv_to_trimesh(shell_neg_pv)

    # Boolean intersection with fallbacks
    shell_tm = None
    for engine in ('manifold', 'scad', 'blender'):
        try:
            shell_tm = trimesh.boolean.intersection([tm_pos, tm_neg], engine=engine)
            if shell_tm is not None and hasattr(shell_tm, 'faces') and shell_tm.faces.size > 0:
                break
        except Exception:
            continue
    if shell_tm is None or shell_tm.faces.size == 0:
        raise RuntimeError("Boolean intersection failed on all engines (manifold, scad, blender).")

    # Final clean & scale to physical cell size
    tm_cleanup_inplace(shell_tm)
    shell_tm.apply_scale(cell_size / (2.0 * np.pi))

    # Export STL if requested
    if export_path:
        try:
            Path(os.path.dirname(export_path)).mkdir(parents=True, exist_ok=True)
            shell_tm.export(export_path)
            print(f"✅ Exported STL to: {export_path}")
        except Exception as e:
            print(f"❌ Failed to export STL: {e}")

    return shell_tm

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # === User settings ===
    tpms_func = sf.gyroid
    nx, ny, nz = 1, 1, 1
    k1 = 1.0+1e-4
    k2 = 1.0+1e-4
    sym = 1             # 0 apply symmetry, 1 disable symmetry (as in your original)
    n = 27
    cell_size = 10.0
    offset = 0.5
    export_stl = True
    export_inp = False   # requires exportMeshtoAbaqus + Abaqus element choices

    # === File & Label setup ===
    tpms_name = tpms_func.__name__
    offset_str = str(offset).replace('.', '')
    filename = f"{tpms_name}_{nx}x{ny}x{nz}_offset{offset_str}.stl"
    export_dir = r"C:\Users\nm40ralo\Documents\Python\stl_output"
    export_path = os.path.join(export_dir, filename) if export_stl else None
    label = filename.rsplit('.', 1)[0]

    # === Generate Boolean shell ===
    boolean_tm = tpms_two_iso_shell(
        tpms_func=tpms_func,
        offset=offset,
        n=n,
        cell_size=cell_size,
        nx=nx, ny=ny, nz=nz,
        k1=k1, k2=k2, sym=sym,
        export_path=export_path,
        gradation_mode='none',   # 'linear' or 'Sinusoidal' (case-insensitive) or none if no gradation needed
        t_min=offset,                  # full thickness min
        t_max=1.75 * offset/offset,           # full thickness max
        a_percent=25.0,
        b_percent=25.0,
        fixed_band=0.05,               # 25% plateaus at both ends
        show_plot=False,
        plot_mode="overlap",     # "separate" | "overlap" | "both"
        color_inner="blue",
        color_outer="red"
    )

    print("Boolean shell → watertight:", getattr(boolean_tm, "is_watertight", None))
    try:
        print("Euler number  :", boolean_tm.euler_number)
    except Exception:
        pass

    # === Preview the result ===
    p = pv.Plotter()
    p.add_mesh(pv.wrap(boolean_tm), show_edges=True, smooth_shading=False)
    p.add_axes()
    p.show(title=f"{tpms_name}: Boolean shell")

    # === Convert mesh to Abaqus format (triangles only) ===
    mesh = boolean_tm
    if hasattr(mesh, "vertices"):  # Trimesh
        nodes = mesh.vertices
        if not hasattr(mesh, "faces"):
            raise ValueError("Mesh has no faces.")
        elements = mesh.faces
    elif hasattr(mesh, "points"):  # PyVista
        nodes = mesh.points
        raw_faces = mesh.faces
        if np.all(raw_faces[::4] == 3):
            faces = raw_faces.reshape(-1, 4)
            elements = faces[:, 1:]
        else:
            raise ValueError("Only triangle faces are supported for Abaqus export.")
    else:
        raise ValueError("Unsupported mesh type for export.")

    if np.any(elements >= len(nodes)) or np.any(elements < 0):
        raise ValueError("Invalid element connectivity in Boolean mesh.")

    # === Export to Abaqus if enabled ===
    if export_inp:
        if exportMeshtoAbaqus is None:
            raise RuntimeError("exportMeshtoAbaqus is not available/imported.")
        element_type = "S3R"
        out_dir = Path(r"C:\Users\nm40ralo\Documents\Python\stl_output")
        out_dir.mkdir(exist_ok=True)
        inp_path = out_dir / f"{label}.inp"

        exportMeshtoAbaqus(
            nodes=nodes,
            elements=elements,
            filename=str(inp_path),
            element_type=element_type,
            materials=[{"name": f"{label}Material", "E": 2e5, "nu": 0.3}],
            sections=[{"elset": "EALL", "material": f"{label}Material"}],
            steps=[{"name": "Step-1", "nlgeom": "NO", "stepType": "Static"}]
        )
        print(f"✅ Exported Abaqus INP: {inp_path}")
