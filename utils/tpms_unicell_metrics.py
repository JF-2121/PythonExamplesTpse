# tpms_metrics.py
# Python 3.8+
# Metrics + STL + batch sweeps + plots. Uses tpms_unitcell_library.TPMS_FUNCS.
# Optimized for speed at high RES (ImageData + float32 + FlyingEdges + no normals + parallel batch).
# User-facing OFFSET := full wall thickness. Internally we use δ_half = OFFSET/2.

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from tpms_unitcell_library import TPMS_FUNCS  # <<— φ(u,v,w) and ∇φ(u,v,w)

# ========================= CONFIG — EDIT & RUN =========================
MODE = "single"          # "single" or "batch"

# Common settings
L        = 10.0          # unit cell size [mm]
RES      = 333           # grid resolution per axis (↑ smoother, slower)
PCT      = 5.0           # percentile for P%-averaged min/max (e.g. 5%)

# Single-case settings (used when MODE="single")
TPMS_SINGLE   = "splitp" # choose from tpms_library.TPMS_FUNCS keys
PREVIEW       = True      # preview triangulated shell |φ| ≤ δ_half
EXPORT_STL    = False     # export STL for single case
STL_OUT       = None      # None -> auto name; or provide path string

# Choose EXACTLY ONE driver for single case (set others to None)
OFFSET  = None           # FULL wall thickness [mm-equivalent via φ-band]
DENSITY = None           # relative density (0..1)
TMEAN   = None           # mean thickness [mm]
TMIN    = 0.12          # P%-averaged min thickness [mm]
TMAX    = None          # P%-averaged max thickness [mm]

# Batch settings (used when MODE="batch")
STRUCTURES = ["gyroid", "schwarz_p", "diamond"]  # list of tpms names
DRIVER     = "density"    # "offset","density","tmean","tmin","tmax"
RANGE      = (0.05, 0.25, 9)  # (start, stop, num points) for the chosen driver
PLOTS      = True
# =====================================================================

# Global numeric dtype tweak (performance): keep float32 unless 64-bit is needed
_DTYPE = np.float32

# ------------------------------ Helpers ------------------------------
def _to_half_delta_from_offset_full(offset_full: float) -> float:
    """User-facing OFFSET (full thickness) -> internal half-band δ_half."""
    return float(offset_full) * 0.5

# ------------------------------ Utilities ------------------------------
def build_image_grid(L: float, res: int) -> Tuple[pv.ImageData, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Regular grid as ImageData (faster + lighter than StructuredGrid).
    Returns (ImageData, u, v, w, SFact). u,v,w are 1D axes in param space.
    """
    spacing = L / (res - 1)
    img = pv.ImageData(dimensions=(res, res, res),
                       spacing=(spacing, spacing, spacing),
                       origin=(0.0, 0.0, 0.0))
    SFact = L / (2.0*np.pi)
    ax = np.linspace(0.0, L, res, dtype=_DTYPE)
    u = (ax / SFact).astype(_DTYPE, copy=False)
    # v and w reuse the same values; callers broadcast them
    return img, u, u, u, SFact

def pyvista_tri_faces(poly: pv.PolyData) -> np.ndarray:
    faces = poly.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        poly = poly.triangulate()
        faces = poly.faces.reshape(-1, 4)
        assert np.all(faces[:, 0] == 3)
    return faces[:, 1:4]

def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0, p1, p2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return (0.5*np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)).astype(_DTYPE, copy=False)

def weighted_tail_mean(values: np.ndarray, weights: np.ndarray, frac: float, tail: str = "low") -> float:
    assert 0.0 < frac < 1.0
    order = np.argsort(values)
    if tail == "high":
        order = order[::-1]
    v, w = values[order], weights[order]
    target = frac * float(np.sum(w, dtype=np.float64))
    acc = 0.0
    wacc = 0.0
    for vi, wi in zip(v, w):
        wi = float(wi)
        if wacc + wi <= target:
            acc += float(vi) * wi
            wacc += wi
        else:
            remain = target - wacc
            if remain > 0:
                acc += float(vi) * remain
                wacc = target
            break
    return acc / target if target > 0 else float("nan")

# ------------------------------ Core precompute ------------------------------
@dataclass
class Precomp:
    t1_per_tri: np.ndarray  # 2*SFact/|∇_u φ| per triangle (thickness per unit δ_half)
    areas:      np.ndarray  # triangle areas
    A_total:    float
    K_vol:      float       # 2*SFact * sum(A * invgrad)
    K_tmean:    float       # (2*SFact/A_total) * sum(A * invgrad)
    grid:       pv.ImageData
    phi_grid:   np.ndarray  # φ on grid (res³)
    res:        int
    L:          float
    SFact:      float

def precompute_on_midsurface(tpms: str, L: float, res: int) -> Tuple[Precomp, pv.PolyData]:
    if tpms not in TPMS_FUNCS:
        raise ValueError(f"Unknown TPMS '{tpms}'. Choices: {list(TPMS_FUNCS.keys())}")
    phi_fn, grad_fn = TPMS_FUNCS[tpms]

    img, u, v, w, SFact = build_image_grid(L, res)

    # Build φ(u,v,w) via broadcasting in float32
    uu = u[:, None, None]
    vv = v[None, :, None]
    ww = w[None, None, :]
    phi = phi_fn(uu, vv, ww).astype(_DTYPE, copy=False)  # shape (res,res,res)

    # Attach to ImageData (VTK wants Fortran/column-major flatten)
    img.point_data.clear()
    img["phi"] = np.asfortranarray(phi).ravel(order="F")

    # Fast iso-surface at φ=0 (FlyingEdges); skip normals/gradients/scalars
    iso = img.contour(
        isosurfaces=[0.0],
        scalars="phi",
        compute_normals=False,
        compute_gradients=False,
        compute_scalars=False,
    )
    if iso.n_cells == 0:
        raise RuntimeError("φ=0 produced no triangles; increase RES or check φ.")

    if not iso.is_all_triangles:
        iso = iso.triangulate()

    faces = pyvista_tri_faces(iso)
    pts   = iso.points.astype(_DTYPE, copy=False)
    areas = triangle_areas(pts, faces)
    A_total = float(np.sum(areas, dtype=np.float64))

    # Gradient magnitude in (u,v,w) at triangle vertices (analytic, vectorized)
    uvw = (pts / _DTYPE(SFact)).astype(_DTYPE, copy=False)
    du, dv, dw = grad_fn(uvw[:, 0], uvw[:, 1], uvw[:, 2])
    grad_mag_u = np.sqrt(du*du + dv*dv + dw*dw).astype(_DTYPE, copy=False)
    grad_mag_u = np.maximum(grad_mag_u, _DTYPE(1e-12))

    invg_tri = (1.0 / grad_mag_u)[faces].mean(axis=1).astype(_DTYPE, copy=False)
    t1_per_tri = (2.0 * SFact * invg_tri).astype(_DTYPE, copy=False)  # thickness per unit δ_half
    sum_A_invg = float(np.sum(areas * invg_tri, dtype=np.float64))
    K_vol   = 2.0 * SFact * sum_A_invg
    K_tmean = (2.0 * SFact / A_total) * sum_A_invg

    pre = Precomp(
        t1_per_tri=t1_per_tri,
        areas=areas,
        A_total=A_total,
        K_vol=K_vol,
        K_tmean=K_tmean,
        grid=img,
        phi_grid=phi,
        res=res,
        L=L,
        SFact=SFact,
    )
    return pre, iso

# ------------------------------ Metrics & inversion ------------------------------
def metrics_from_delta(delta_half: float, pre: Precomp, pct: float) -> Dict[str, float]:
    """
    Return all metrics using internal half-band δ_half; report 'offset' as FULL thickness (2*δ_half).
    """
    delta_half = float(delta_half)
    t_per_tri = (delta_half * pre.t1_per_tri).astype(_DTYPE, copy=False)  # local thickness
    t_mean = float(np.sum(t_per_tri * pre.areas, dtype=np.float64) / pre.A_total)
    frac = pct / 100.0
    t_min_pct = float(weighted_tail_mean(t_per_tri, pre.areas, frac=frac, tail="low"))
    t_max_pct = float(weighted_tail_mean(t_per_tri, pre.areas, frac=frac, tail="high"))
    t_abs_min = float(np.min(t_per_tri))
    t_abs_max = float(np.max(t_per_tri))
    volume_shell = delta_half * pre.K_vol
    rho = float(volume_shell / (pre.L**3))
    return {
        "offset": 2.0 * delta_half,  # report FULL thickness to user
        "density": rho,
        "t_mean": t_mean,
        "t_min": t_min_pct,
        "t_max": t_max_pct,
        "t_abs_min": t_abs_min,
        "t_abs_max": t_abs_max
    }

def invert_delta_from_target(kind: str, target: float, pre: Precomp, pct: float) -> float:
    """
    Returns INTERNAL δ_half for the given target.
      kind ∈ {"density","tmean","tmin","tmax"}.
    NOTE: For user-facing OFFSET (full thickness), convert via δ_half = OFFSET/2 before using.
    """
    if kind == "density":
        if pre.K_vol <= 0: raise RuntimeError("K_vol <= 0")
        return target * (pre.L**3) / pre.K_vol
    if kind == "tmean":
        if pre.K_tmean <= 0: raise RuntimeError("K_tmean <= 0")
        return target / pre.K_tmean
    frac = pct / 100.0
    if kind == "tmin":
        base = weighted_tail_mean(pre.t1_per_tri, pre.areas, frac=frac, tail="low")
        if base <= 0: raise RuntimeError("Base tmin <= 0")
        return target / base
    if kind == "tmax":
        base = weighted_tail_mean(pre.t1_per_tri, pre.areas, frac=frac, tail="high")
        if base <= 0: raise RuntimeError("Base tmax <= 0")
        return target / base
    raise ValueError(f"Unknown kind '{kind}'")

# ------------------------------ STL export & preview ------------------------------
def export_stl_shell(pre: Precomp, delta_half: float, out_path: Path) -> None:
    g = pre.grid.copy()
    # reattach φ if missing
    if "phi" not in g.point_data:
        g["phi"] = np.asfortranarray(pre.phi_grid).ravel(order="F")
    g["absphi"] = np.abs(g["phi"]).astype(_DTYPE, copy=False)
    # IMPORTANT: use half-band inside |φ| ≤ δ_half
    shell = g.threshold(value=[0.0, float(delta_half)], scalars="absphi").extract_surface()
    if not shell.is_all_triangles:
        shell = shell.triangulate()
    shell.save(str(out_path), binary=True)

def preview_shell(pre: Precomp, delta_half: float, title: str = "TPMS shell preview"):
    g = pre.grid.copy()
    if "phi" not in g.point_data:
        g["phi"] = np.asfortranarray(pre.phi_grid).ravel(order="F")
    g["absphi"] = np.abs(g["phi"]).astype(_DTYPE, copy=False)
    # IMPORTANT: use half-band inside |φ| ≤ δ_half
    shell = g.threshold(value=[0.0, float(delta_half)], scalars="absphi").extract_surface()
    if not shell.is_all_triangles:
        shell = shell.triangulate()
    p = pv.Plotter()
    p.add_mesh(shell, opacity=1.0, show_edges=False)
    p.add_axes()
    p.show(title=title)

# ------------------------------ Batch sweep + reporting ------------------------------
def _sweep_one(args):
    tpms, L, res, driver, xs, pct = args
    pre, _ = precompute_on_midsurface(tpms, L=L, res=res)
    series = {"driver": [], "density": [], "t_mean": [], "t_min": [], "t_max": [],
              "t_abs_min": [], "t_abs_max": [], "offset": []}  # 'offset' stored as FULL thickness
    for x in xs:
        x = float(x)
        # Compute INTERNAL δ_half for the given driver value x
        if driver == "offset":
            # x is user-facing full thickness; convert to δ_half
            delta_half = _to_half_delta_from_offset_full(x)
        elif driver == "density":
            delta_half = invert_delta_from_target("density", x, pre, pct)
        elif driver == "tmean":
            delta_half = invert_delta_from_target("tmean", x, pre, pct)
        elif driver == "tmin":
            delta_half = invert_delta_from_target("tmin", x, pre, pct)
        elif driver == "tmax":
            delta_half = invert_delta_from_target("tmax", x, pre, pct)
        else:
            raise ValueError("driver must be one of: offset, density, tmean, tmin, tmax")

        m = metrics_from_delta(delta_half, pre, pct)  # returns offset as FULL thickness
        series["driver"].append(x)
        for k in ("offset", "density", "t_mean", "t_min", "t_max", "t_abs_min", "t_abs_max"):
            series[k].append(m[k])

    for k in series.keys():
        series[k] = np.asarray(series[k], dtype=_DTYPE)
    return tpms, series

def batch_sweep(tpms_list: List[str], L: float, res: int,
                driver: str, start: float, stop: float, num: int,
                pct: float) -> Dict[str, Dict[str, np.ndarray]]:
    xs = np.linspace(start, stop, num, dtype=_DTYPE)
    tasks = [(tp, L, res, driver, xs, pct) for tp in tpms_list]
    results = {}
    # cap workers to avoid RAM spikes; 2–4 is often ideal for RES≈333
    max_workers = min(len(tasks), max(2, (os.cpu_count() or 4)//2))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_sweep_one, t) for t in tasks]
        for fu in as_completed(futs):
            tpms, series = fu.result()
            results[tpms] = series
    return results

def print_batch_summary(tpms: str, series: Dict[str, np.ndarray], driver_name: str, pct: float):
    def span(arr):
        arr = np.asarray(arr)
        return float(np.min(arr)), float(np.max(arr))
    dmin, dmax = span(series["driver"])
    print(f"\n--- {tpms} ---")
    print(f"driver: {driver_name}, driver span: {dmin:.6g} → {dmax:.6g}")
    label_map = {
        "t_mean":"mean thickness [mm]",
        "t_min": f"min P% avg [{pct:.0f}%] [mm]",
        "t_max": f"max P% avg [{pct:.0f}%] [mm]",
        "t_abs_min":"abs min [mm]",
        "t_abs_max":"abs max [mm]",
        "density":"relative density [-]",
        "offset":"offset δ (FULL) [-]"
    }
    for key in ("offset","density","t_mean","t_min","t_max","t_abs_min","t_abs_max"):
        amin, amax = span(series[key])
        print(f"{label_map[key]:>20}: {amin:.6g} → {amax:.6g}")

def plot_batch(results: Dict[str, Dict[str, np.ndarray]], driver_name: str, pct: float):
    """
    Single figure:
      - X: relative density
      - Left Y: offset δ (FULL)  — one curve per structure
      - Right Y (multiple spines): mean thickness [mm] mapped via per-structure K_tmean estimate
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = []
    structures = list(results.keys())
    for tpms in structures:
        series = results[tpms]
        rho   = series["density"]
        delta_full = series["offset"]  # FULL thickness stored
        (ln,) = ax.plot(rho, delta_full, label=tpms)
        lines.append((tpms, ln))

    ax.set_xlabel("Relative density [-]")
    ax.set_ylabel("Offset δ (FULL) [-]")
    ax.grid(True, which="both", axis="both")

    # For each structure, add a right-side axis that maps δ_full ≈ 2*δ_half -> t_mean ≈ K * δ_half
    # So t_mean ≈ (K/2) * δ_full. We estimate K from series: mean(t_mean / δ_half) = mean(2*t_mean / δ_full).
    spine_offset = 1.0
    offset_step = 0.08

    for i, (tpms, ln) in enumerate(lines):
        series = results[tpms]
        delta_full = series["offset"]
        tmean  = series["t_mean"]

        mask = delta_full > 0
        if np.any(mask):
            # K_half := mean(t_mean / δ_half) = mean(2 * t_mean / δ_full)
            K_half = float(np.mean((2.0 * tmean[mask] / delta_full[mask]).astype(np.float64)))
        else:
            K_half = 1.0

        # Relationship: t_mean ≈ (K_half/2) * δ_full
        ax_r = ax.twinx()
        ax_r.spines["right"].set_position(("axes", spine_offset + i*offset_step))
        ax_r.set_frame_on(True)
        ax_r.patch.set_visible(False)

        color = ln.get_color()
        ax_r.spines["right"].set_color(color)
        ax_r.tick_params(axis="y", colors=color)
        ax_r.yaxis.label.set_color(color)

        ylim_delta_full = ax.get_ylim()
        ax_r.set_ylim(ylim_delta_full[0]*(K_half/2.0), ylim_delta_full[1]*(K_half/2.0))
        ax_r.set_ylabel(f"Mean thickness [mm] — {tpms}")

    ax.legend(loc="best", title="Structures")
    fig.tight_layout()
    plt.show()

# ------------------------------ Orchestrators ------------------------------
def run_single():
    # validate exactly one driver
    provided = [x for x in [OFFSET, DENSITY, TMEAN, TMIN, TMAX] if x is not None]
    if len(provided) != 1:
        raise SystemExit("Single case: set exactly ONE of OFFSET/DENSITY/TMEAN/TMIN/TMAX (others must be None)")

    pre, _ = precompute_on_midsurface(TPMS_SINGLE, L=L, res=RES)

    # Compute INTERNAL δ_half based on the chosen driver
    if OFFSET is not None:
        # OFFSET is full thickness from the user → convert to half-band
        delta_half = _to_half_delta_from_offset_full(OFFSET)
    elif DENSITY is not None:
        if not (0.0 < DENSITY < 1.0):
            raise SystemExit("DENSITY must be in (0,1)")
        delta_half = invert_delta_from_target("density", float(DENSITY), pre, PCT)
    elif TMEAN is not None:
        delta_half = invert_delta_from_target("tmean", float(TMEAN), pre, PCT)
    elif TMIN is not None:
        delta_half = invert_delta_from_target("tmin", float(TMIN), pre, PCT)
    else:
        delta_half = invert_delta_from_target("tmax", float(TMAX), pre, PCT)

    m = metrics_from_delta(delta_half, pre, PCT)

    # summary (report FULL thickness)
    print("\n=== TPMS Unit Cell Metrics (single case) ===")
    print(f"TPMS:               {TPMS_SINGLE}")
    print(f"Cell size L:        {L:.6g} mm")
    print(f"Resolution:         {RES}³")
    print(f"Percent (P):        {PCT:.3g}%  (for P%-averaged min/max)")
    print("---- Metrics ----")
    print(f"Offset δ (FULL):    {m['offset']:.8f}  [-]")
    print(f"Relative density:   {m['density']*100:.4f}%")
    print(f"Mean thickness:     {m['t_mean']:.6f} mm  (area-weighted)")
    print(f"Min thickness (P%): {m['t_min']:.6f} mm  (avg of lowest {PCT}% by area)")
    print(f"Max thickness (P%): {m['t_max']:.6f} mm  (avg of top   {PCT}% by area)")
    print(f"Absolute min:       {m['t_abs_min']:.6f} mm")
    print(f"Absolute max:       {m['t_abs_max']:.6f} mm")

    if PREVIEW:
        preview_shell(pre, delta_half, title=f"{TPMS_SINGLE} | δ_full={m['offset']:.4f} (using |φ| ≤ δ_full/2)")

    if EXPORT_STL:
        out_path = Path(STL_OUT) if STL_OUT else Path(
            f"tpms_{TPMS_SINGLE}_L{L:g}_res{RES}_deltaFULL{m['offset']:.5f}.stl"
        )
        export_stl_shell(pre, delta_half, out_path)
        print(f"\nSTL written: {out_path.resolve()}")

def run_batch():
    start, stop, num = RANGE
    res_dict = batch_sweep(STRUCTURES, L=L, res=RES, driver=DRIVER,
                           start=float(start), stop=float(stop), num=int(num), pct=PCT)

    print("\n=== TPMS Batch Summary ===")
    print(f"L={L} mm, res={RES}³, P={PCT}%")
    print(f"Driver: {DRIVER}, range: {start} → {stop} (n={num})")

    for tpms, series in res_dict.items():
        print_batch_summary(tpms, series, DRIVER, PCT)

    if PLOTS:
        plot_batch(res_dict, driver_name=DRIVER, pct=PCT)

# ------------------------------ Entry point ------------------------------
if __name__ == "__main__":
    # NOTE: On Windows, ProcessPoolExecutor requires the guard above.
    if MODE == "single":
        run_single()
    elif MODE == "batch":
        run_batch()
    else:
        raise SystemExit("MODE must be 'single' or 'batch'")
