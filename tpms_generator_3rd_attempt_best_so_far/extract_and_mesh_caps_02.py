#!/usr/bin/env python3
# select_partitions_all_faces_lift_3d.py
# Python 3.10+

from typing import List, Tuple, Sequence, Iterable, Dict, Set, Optional
import numpy as np

# Compatibility shims for libs expecting NumPy < 2.0 aliases
if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "int_"):     np.int_     = np.int64
if not hasattr(np, "complex_"): np.complex_ = np.complex128

from shapely.geometry import (
    Polygon, LineString, MultiLineString, GeometryCollection, LinearRing, Point
)
from shapely.ops import unary_union, linemerge, polygonize, orient
from shapely.strtree import STRtree
from shapely.prepared import prep
from skimage import measure
from vtkmodules.vtkCommonCore import vtkMultiThreader

print("Before:", vtkMultiThreader.GetGlobalDefaultNumberOfThreads())
vtkMultiThreader.SetGlobalDefaultNumberOfThreads(0)   # 0 = let VTK decide (all cores)
# or set an explicit number:
# vtkMultiThreader.SetGlobalDefaultNumberOfThreads(16)

import matplotlib.pyplot as plt
import random
import warnings

# Optional triangulation
try:
    import mapbox_earcut as earcut
    HAVE_EARCUT = True
except Exception:
    HAVE_EARCUT = False

# Optional 3D preview
try:
    import pyvista as pv
    HAVE_PV = True
except Exception:
    HAVE_PV = False

# --- TPMS via microgen ---
from microgen import surface_functions as sf

# ====================== CONFIG ======================
TPMS_FN      = sf.schwarz_d           # microgen surface function
N_CELLS      = (3, 3, 3)           # periodic cell count in x,y,z
N_GRID       = 20                  # sampling per cell per axis (per face & iso cache)
ISOVALUE     = 0.05

FACE_NAMES   = ("x0","xL","y0","yL","z0","zL")

SNAP_DECIMALS     = 6
MIN_POLY_AREA     = 1e-10
INTERSECT_LEN_EPS = 1e-9           # length threshold for keeping a piece

# Fallback selection by point sampling (used only if no length-overlap found)
SAMPLE_PTS_PER_LINE = 21           # odd number; includes midpoint
REQUIRE_POINTS_MIN  = 1            # keep polygon if ≥ this many points fall inside

# --- Safety switch: keep only the polygon with the MOST points per filter polyline ---
ENABLE_MAXPOINT_SAFETY = False     # set True to enable the safety pass
SAFETY_PTS_PER_LINE    = SAMPLE_PTS_PER_LINE**2       # sampling points along each filter polyline
SAFETY_TIEBREAK_BY_LEN = True      # break ties using intersection length, then area

# --- Triangulation & visualization ---
ENABLE_TRIANGULATION = True        # requires mapbox_earcut; else auto-skips
SHOW_2D_PLOTS        = False       # small 2D debug per face
SHOW_3D_PREVIEW      = True        # requires PyVista

# --- 3D preview styling & reuse ---
SHOW_POLYLINE_RINGS = False         # draw lifted boundary polylines (rings)
SHOW_TRI_EDGES      = True         # show triangle edges for cap meshes
TRI_EDGE_COLOR      = "black"
TRI_FACE_OPACITY    = 1.0

# Split shells style (these are the two offset isosurfaces at SPLIT_LEVELS)
SHOW_SPLIT_ISOSURFACES = True
SPLIT_SHELL_OPACITY    = 0.35
SPLIT_SHELL_SMOOTH     = False
SPLIT_SHELL_COLORS     = ["royalblue", "orangered"]  # distinct colors for + / -

RAND_SEED  = 42

SPLIT_LEVELS  = (+0.5*ISOVALUE, -0.5*ISOVALUE)  # used for face splitting AND 3D shells
# Add 0.0 if you want: e.g. (+0.25*ISOVALUE, -0.25*ISOVALUE, 0.0)
FILTER_LEVELS = (+0.45*ISOVALUE, -0.45*ISOVALUE)
# ====================================================


# --------------------- Face domain helpers ---------------------
def _ranges(ncells: Tuple[int,int,int], n_grid: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    nx, ny, nz = ncells
    x = np.linspace(-np.pi*nx, +np.pi*nx, max(2, n_grid*nx))
    y = np.linspace(-np.pi*ny, +np.pi*ny, max(2, n_grid*ny))
    z = np.linspace(-np.pi*nz, +np.pi*nz, max(2, n_grid*nz))
    return x, y, z

def _face_info(face: str, ncells: Tuple[int,int,int]) -> Tuple[int, float, Tuple[int,int]]:
    nx, ny, nz = ncells
    if face == "x0": return 0, -np.pi*nx, (1,2)
    if face == "xL": return 0, +np.pi*nx, (1,2)
    if face == "y0": return 1, -np.pi*ny, (0,2)
    if face == "yL": return 1, +np.pi*ny, (0,2)
    if face == "z0": return 2, -np.pi*nz, (0,1)
    if face == "zL": return 2, +np.pi*nz, (0,1)
    raise ValueError("FACE_NAME must be one of: x0,xL,y0,yL,z0,zL")

def sample_face_field(tpms_fn, face: str, ncells: Tuple[int,int,int], n_grid: int):
    x, y, z = _ranges(ncells, n_grid)
    fixed_axis, fixed_val, (a,b) = _face_info(face, ncells)
    axes = [x, y, z]
    A, B = axes[a], axes[b]
    AA, BB = np.meshgrid(A, B, indexing="xy")
    shape = AA.shape
    X = np.zeros(shape); Y = np.zeros(shape); Z = np.zeros(shape)
    if fixed_axis == 0:
        X[...] = fixed_val; Y[...] = AA; Z[...] = BB
    elif fixed_axis == 1:
        X[...] = AA; Y[...] = fixed_val; Z[...] = BB
    else:
        X[...] = AA; Y[...] = BB; Z[...] = fixed_val
    G = tpms_fn(X, Y, Z)
    # (u,v) normalized to [0,1] bounds of (A,B)
    U = (AA - A.min()) / (A.max() - A.min() + 1e-30)
    V = (BB - B.min()) / (B.max() - B.min() + 1e-30)
    return U, V, G, (A.min(), A.max(), B.min(), B.max()), fixed_axis, fixed_val, (a,b)

def uv_to_plane_xyz(face: str, ncells: Tuple[int,int,int], uv: np.ndarray) -> np.ndarray:
    """
    Lift (u,v) ∈ [0,1]^2 to 3D points on the boundary plane for 'face',
    using the same metric ranges as sample_face_field.
    """
    x, y, z = _ranges(ncells, 2)  # min/max only
    fixed_axis, fixed_val, (a,b) = _face_info(face, ncells)

    axes = [x, y, z]
    Amin, Amax = axes[a].min(), axes[a].max()
    Bmin, Bmax = axes[b].min(), axes[b].max()

    A = Amin + uv[:, 0] * (Amax - Amin)
    B = Bmin + uv[:, 1] * (Bmax - Bmin)

    pts = np.zeros((len(uv), 3), dtype=float)
    if fixed_axis == 0:
        pts[:, 0] = fixed_val
        pts[:, 1] = A
        pts[:, 2] = B
    elif fixed_axis == 1:
        pts[:, 0] = A
        pts[:, 1] = fixed_val
        pts[:, 2] = B
    else:
        pts[:, 0] = A
        pts[:, 1] = B
        pts[:, 2] = fixed_val
    return pts


# --------------------- Contours ---------------------
def contours_uv_from_field(G: np.ndarray, level: float) -> List[np.ndarray]:
    cs = measure.find_contours(G, level=level)
    H, W = G.shape
    curves: List[np.ndarray] = []
    for c in cs:
        v = c[:, 0] / (H - 1)
        u = c[:, 1] / (W - 1)
        curves.append(np.column_stack([u, v]))
    return curves

def _is_lines(ls) -> bool:
    return isinstance(ls, (LineString, MultiLineString)) and (not ls.is_empty)

def _sanitize_lines(lines: Sequence) -> List[LineString | MultiLineString]:
    out: List[LineString | MultiLineString] = []
    for g in lines:
        if _is_lines(g):
            out.append(g)
    return out

def to_linestrings(curves: List[np.ndarray], snap_decimals: int = 0) -> List[LineString]:
    out: List[LineString] = []
    for arr in curves:
        if arr is None or len(arr) < 2:
            continue
        a = np.asarray(arr, dtype=float)
        if snap_decimals > 0:
            a = np.round(a, snap_decimals)
        try:
            ls = LineString(a)
            if not ls.is_empty:
                out.append(ls)
        except Exception:
            pass
    return out

def merge_as_multilines(lines: List[LineString]) -> MultiLineString:
    if not lines:
        return MultiLineString([])
    dissolved = unary_union(lines)
    if isinstance(dissolved, LineString):
        return MultiLineString([dissolved])
    if isinstance(dissolved, MultiLineString):
        merged = linemerge(dissolved)
        return merged if isinstance(merged, MultiLineString) else MultiLineString([merged])
    parts = []
    for g in getattr(dissolved, "geoms", []):
        if isinstance(g, LineString):
            parts.append(g)
        elif isinstance(g, MultiLineString):
            parts.extend(list(g.geoms))
    return MultiLineString(parts)


# --------------------- Polygonize-based partition ---------------------
def _snap_ls(ls: LineString, snap_decimals: int) -> LineString:
    xy = np.asarray(ls.coords, dtype=float)
    xy = np.round(xy, snap_decimals)
    return LineString(xy)

def _ensure_lines(obj, snap_decimals: int):
    out = []
    if isinstance(obj, LineString):
        out.append(_snap_ls(obj, snap_decimals) if snap_decimals > 0 else obj)
    elif isinstance(obj, MultiLineString):
        for ls in obj.geoms:
            out.append(_snap_ls(ls, snap_decimals) if snap_decimals > 0 else ls)
    elif isinstance(obj, LinearRing):
        ls = LineString(obj)
        out.append(_snap_ls(ls, snap_decimals) if snap_decimals > 0 else ls)
    return out

def partition_square_by_polygonize(square: Polygon, split_lines: MultiLineString,
                                   snap_decimals: int = 0) -> List[Polygon]:
    graph_lines: List[LineString] = []
    graph_lines += _ensure_lines(split_lines, snap_decimals)
    graph_lines += _ensure_lines(square.boundary, snap_decimals)
    graph = unary_union(graph_lines)
    faces = list(polygonize(graph))

    inside: List[Polygon] = []
    for f in faces:
        g = f.intersection(square)
        if g.is_empty:
            continue
        if isinstance(g, Polygon):
            if g.area > MIN_POLY_AREA:
                inside.append(g)
        elif isinstance(g, GeometryCollection):
            for h in getattr(g, "geoms", []):
                if isinstance(h, Polygon) and h.area > MIN_POLY_AREA:
                    inside.append(h)
    return inside


# --------------------- Length-based selection + safety ---------------------
def total_intersection_length(poly: Polygon, line: LineString | MultiLineString) -> float:
    inter = poly.intersection(line)
    if inter.is_empty:
        return 0.0
    t = inter.geom_type
    if t == "LineString":
        return inter.length
    if t == "MultiLineString":
        return sum(ls.length for ls in inter.geoms)
    return 0.0

def _sample_points_on_line(ls: LineString, n: int) -> Iterable[Point]:
    if n <= 1:
        yield ls.interpolate(0.5, normalized=True)
        return
    for k in range(n):
        t = k / (n - 1)
        yield ls.interpolate(t, normalized=True)

def _polygon_selected_by_points(p: Polygon, inner_lines: Sequence[LineString | MultiLineString],
                                pts_per_line: int, min_points: int) -> bool:
    pprep = prep(p)
    count = 0
    for ls in inner_lines:
        if isinstance(ls, MultiLineString):
            for seg in ls.geoms:
                for pt in _sample_points_on_line(seg, pts_per_line):
                    if pprep.contains(pt):
                        count += 1
                        if count >= min_points:
                            return True
        else:
            for pt in _sample_points_on_line(ls, pts_per_line):
                if pprep.contains(pt):
                    count += 1
                    if count >= min_points:
                        return True
    return False

def select_pieces_by_inner_lines_fast(
    pieces: List[Polygon],
    inner_lines_raw: List[LineString | MultiLineString],
    min_len: float = INTERSECT_LEN_EPS
) -> Tuple[List[Polygon], List[Polygon]]:
    pieces = [p for p in pieces if isinstance(p, Polygon) and (not p.is_empty) and p.area > 0.0]
    inner_lines = _sanitize_lines(inner_lines_raw)

    if not inner_lines or not pieces:
        return [], pieces[:]

    # --- Phase 1: fast length-overlap selection with STRtree culling
    tree = STRtree(inner_lines)
    selected_len, rejected_len = [], []
    per_poly_any_len = []

    for p in pieces:
        pprep = prep(p)
        try:
            idxs = tree.query_items(p)
            idxs = idxs.tolist() if hasattr(idxs, "tolist") else idxs
            idxs = [int(i) for i in idxs]
            cands = (inner_lines[i] for i in idxs if 0 <= i < len(inner_lines))
        except AttributeError:
            cands = tree.query(p)

        keep = False
        for ls in cands:
            if not _is_lines(ls):
                continue
            if not pprep.intersects(ls):
                continue
            if total_intersection_length(p, ls) > min_len:
                keep = True
                break

        per_poly_any_len.append(keep)
        (selected_len if keep else rejected_len).append(p)

    if any(per_poly_any_len):
        return selected_len, rejected_len

    # --- Phase 2: robust fallback — point-in-polygon sampling
    selected_pts, rejected_pts = [], []
    for p in pieces:
        keep = _polygon_selected_by_points(
            p, inner_lines, pts_per_line=SAMPLE_PTS_PER_LINE, min_points=REQUIRE_POINTS_MIN
        )
        (selected_pts if keep else rejected_pts).append(p)

    return selected_pts, rejected_pts


def _count_points_in_polygon(poly: Polygon, line: LineString | MultiLineString, n_pts: int) -> int:
    pprep = prep(poly)
    count = 0
    if isinstance(line, MultiLineString):
        for seg in line.geoms:
            for pt in _sample_points_on_line(seg, n_pts):
                if pprep.contains(pt):
                    count += 1
    else:
        for pt in _sample_points_on_line(line, n_pts):
            if pprep.contains(pt):
                count += 1
    return count

def refine_selection_by_max_points(
    selected_polys: List[Polygon],
    inner_lines: List[LineString | MultiLineString],
    n_pts: int = 101,
    tiebreak_by_len: bool = True
) -> List[Polygon]:
    if not selected_polys or not inner_lines:
        return selected_polys

    keepers: Set[int] = set()
    poly_tree = STRtree(selected_polys)
    index_map: Dict[int, Polygon] = {i: p for i, p in enumerate(selected_polys)}

    for ln in inner_lines:
        candidates = poly_tree.query(ln)
        if not candidates:
            continue
        cand_idx = [i for i, p in index_map.items() if p in candidates]

        counts = []
        for i in cand_idx:
            cnt = _count_points_in_polygon(index_map[i], ln, n_pts)
            counts.append((i, cnt))

        if not counts:
            continue

        max_cnt = max(cnt for _, cnt in counts)
        if max_cnt <= 0:
            continue

        best_idxs = [i for i, cnt in counts if cnt == max_cnt]

        if len(best_idxs) > 1 and tiebreak_by_len:
            lens = [(i, total_intersection_length(index_map[i], ln)) for i in best_idxs]
            max_len = max(L for _, L in lens)
            best_idxs = [i for i, L in lens if np.isclose(L, max_len) or L == max_len]

        if len(best_idxs) > 1:
            areas = [(i, index_map[i].area) for i in best_idxs]
            max_area = max(A for _, A in areas)
            best_idxs = [i for i, A in areas if np.isclose(A, max_area) or A == max_area]

        keepers.add(best_idxs[0])

    if not keepers:
        return selected_polys
    refined = [index_map[i] for i in sorted(keepers)]
    return refined


# --------------------- 2D→3D lifting & triangulation ---------------------
def polygon_uv_to_xyz(face: str, ncells: Tuple[int,int,int], poly: Polygon) -> Dict[str, List[np.ndarray]]:
    """
    Convert a shapely Polygon in UV to dict of 3D rings on the face plane.
    Returns {"exterior": [Nx3], "interiors": [Mx3, ...]} as numpy arrays.
    """
    def ring_to_xyz(coords) -> np.ndarray:
        pts2 = np.array(coords, dtype=float)
        # remove closing duplicate if present
        if len(pts2) > 1 and np.allclose(pts2[0], pts2[-1]):
            pts2 = pts2[:-1]
        return uv_to_plane_xyz(face, ncells, pts2)

    out = {
        "exterior": ring_to_xyz(poly.exterior.coords),
        "interiors": [ring_to_xyz(ring.coords) for ring in poly.interiors]
    }
    return out

def _clean_ring_coords(r: LinearRing | LineString) -> np.ndarray:
    xy = np.asarray(r.coords, dtype=np.float32)
    if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    # must have at least 3 unique verts
    if xy.shape[0] >= 3:
        # drop exact duplicates in sequence (rare but can happen after rounding)
        dedup = [xy[0]]
        for p in xy[1:]:
            if not np.allclose(p, dedup[-1]):
                dedup.append(p)
        xy = np.asarray(dedup, dtype=np.float32)
    return xy if xy.shape[0] >= 3 else np.empty((0, 2), dtype=np.float32)

def _earcut_buffers_from_poly_all(poly: Polygon) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build all buffer variants for different mapbox_earcut builds.
    Returns:
      V2:(n,2)float32
      ring_end:(k,)uint32  cumulative end indices for [outer, hole1, hole2, ...]
      holes_idx:(h,)uint32 offsets where each hole starts (classic earcut)
      flat:(2n,)float32   interleaved xy
    """
    # Enforce CCW exterior / CW holes (earcut convention)
    poly = orient(poly, sign=1.0)  # CCW exterior, CW holes

    ex = _clean_ring_coords(poly.exterior)
    if ex.size == 0:
        return (np.empty((0, 2), np.float32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.float32))

    holes = []
    for hr in poly.interiors:
        h = _clean_ring_coords(hr)
        if h.size:
            holes.append(h)

    parts = [ex] + holes
    V2 = np.vstack(parts).astype(np.float32, copy=False)
    V2 = np.ascontiguousarray(V2)

    # ring_end cumulative ends for outer + each hole
    ends = []
    off = 0
    for part in parts:
        off += part.shape[0]
        ends.append(off)
    ring_end = np.asarray(ends, dtype=np.uint32)
    ring_end = np.ascontiguousarray(ring_end)

    # classic holes_idx: offsets where each hole starts (skip outer)
    hole_offsets = []
    off = ex.shape[0]
    for h in holes:
        hole_offsets.append(off)
        off += h.shape[0]
    holes_idx = np.asarray(hole_offsets, dtype=np.uint32)
    holes_idx = np.ascontiguousarray(holes_idx)

    flat = np.ascontiguousarray(V2.ravel().astype(np.float32, copy=False))
    return V2, ring_end, holes_idx, flat

def triangulate_polygon_2d(poly: Polygon) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Robust earcut wrapper supporting multiple Python bindings:
      A) triangulate_float32(V2:(n,2), ring_end_indices:(k,))
      B) triangulate_float32(V2:(n,2), holes_idx:(h,))
      C) triangulate_float32(flat:(2n,), holes_idx:(h,))
    Returns (V2, F) with V2 float32 (n,2), F int32 (m,3).
    """
    if not HAVE_EARCUT:
        return None

    # one-time repair for near-degenerate polygons
    if (not poly.is_valid) or (poly.area <= 0):
        poly = poly.buffer(0.0)
        if not isinstance(poly, Polygon) or poly.area <= 0:
            return None

    V2, ring_end, holes_idx, flat = _earcut_buffers_from_poly_all(poly)
    if V2.size == 0:
        return None

    tris = None
    # Try variant A: (V2, ring_end_indices)
    try:
        tris = earcut.triangulate_float32(V2, ring_end)
    except Exception:
        tris = None

    # Try variant B: (V2, holes_idx)
    if tris is None:
        try:
            tris = earcut.triangulate_float32(V2, holes_idx)
        except Exception:
            tris = None

    # Try variant C: (flat, holes_idx)
    if tris is None:
        try:
            tris = earcut.triangulate_float32(flat, holes_idx)
        except Exception:
            tris = None

    # Last-chance: simplify slightly then rebuild buffers and retry A→B→C
    if (tris is None) or (np.asarray(tris).size == 0):
        simp = poly.buffer(0.0)  # robustify
        if isinstance(simp, Polygon) and simp.is_valid and simp.area > 0:
            V2, ring_end, holes_idx, flat = _earcut_buffers_from_poly_all(simp)
            if V2.size:
                for args in ((V2, ring_end), (V2, holes_idx), (flat, holes_idx)):
                    try:
                        tris = earcut.triangulate_float32(*args)
                        if np.asarray(tris).size:
                            break
                    except Exception:
                        continue

    if (tris is None) or (np.asarray(tris).size == 0):
        return None

    F = np.asarray(tris, dtype=np.uint32).reshape(-1, 3).astype(np.int32, copy=False)
    return V2, F

def lift_triangles_to_3d(face: str, ncells: Tuple[int,int,int], V2: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map 2D (u,v) vertices V2 to 3D on face plane; faces F unchanged.
    Returns (V3, F) suitable for PyVista/Trimesh.
    """
    V3 = uv_to_plane_xyz(face, ncells, V2)
    return V3, F


# --------------------- Split isosurface cache (build once, reuse) ---------------------
_SPLIT_ISO_CACHE: Optional[List["pv.PolyData"]] = None

def get_split_isosurfaces(tpms_fn, ncells: Tuple[int,int,int], n_grid: int, levels: Sequence[float]) -> List["pv.PolyData"]:
    """
    Build the two split shells (at 'levels') ONCE using the same metric ranges and n_grid,
    then cache & reuse for the rest of the run. If PyVista is unavailable, returns [].
    """
    global _SPLIT_ISO_CACHE
    if not HAVE_PV or not SHOW_SPLIT_ISOSURFACES:
        return []

    if _SPLIT_ISO_CACHE is not None:
        return _SPLIT_ISO_CACHE

    # Sample the field on a rectilinear grid aligned with _ranges and n_grid
    x, y, z = _ranges(ncells, n_grid)
    grid = pv.RectilinearGrid(x, y, z)

    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")
    G = tpms_fn(XX, YY, ZZ).astype(np.float32)
    grid["G"] = np.ascontiguousarray(G.ravel(order="F"))

    meshes: List[pv.PolyData] = []
    for lvl in levels:
        try:
            m = grid.contour([float(lvl)], scalars="G", preference="points")
        except Exception:
            # tiny nudge if level hits a sampling void
            eps = 1e-6 * (np.nanmax(G) - np.nanmin(G) + 1.0)
            m = grid.contour([float(lvl) + eps], scalars="G", preference="points")
        meshes.append(m)

    _SPLIT_ISO_CACHE = meshes
    return meshes


# --------------------- Per-face pipeline ---------------------
def process_face(face: str) -> Dict:
    # 1) Sample field on face
    U, V, G, ab_bounds, fixed_axis, fixed_val, ab_idx = sample_face_field(TPMS_FN, face, N_CELLS, N_GRID)

    # 2) Split lines
    split_curves = []
    for lvl in SPLIT_LEVELS:
        split_curves += contours_uv_from_field(G, lvl)
    split_lines  = to_linestrings(split_curves, snap_decimals=SNAP_DECIMALS)
    if not split_lines:
        return {"face": face, "pieces": [], "selected": [], "inner_lines": [], "tri_meshes": []}

    mls_split = merge_as_multilines(split_lines)

    # 3) Partition by polygonize
    square = Polygon([(0,0),(1,0),(1,1),(0,1)])
    pieces = partition_square_by_polygonize(square, mls_split, snap_decimals=SNAP_DECIMALS)
    pieces = [p for p in pieces if p.area > MIN_POLY_AREA]

    # 4) Inner lines (filters)
    inner_curves = []
    for lvl in FILTER_LEVELS:
        inner_curves += contours_uv_from_field(G, lvl)
    inner_lines  = to_linestrings(inner_curves, snap_decimals=SNAP_DECIMALS)
    inner_lines  = _sanitize_lines(inner_lines)

    # 5) Select by overlap length (+fallback)
    if not inner_lines:
        selected, rejected = [], pieces[:]
    else:
        selected, rejected = select_pieces_by_inner_lines_fast(pieces, inner_lines, min_len=INTERSECT_LEN_EPS)

    # 6) Safety refinement (optional)
    if ENABLE_MAXPOINT_SAFETY and selected and inner_lines:
        selected = refine_selection_by_max_points(
            selected, inner_lines, n_pts=SAFETY_PTS_PER_LINE, tiebreak_by_len=SAFETY_TIEBREAK_BY_LEN
        )

    # 7) Triangulation in UV + lift to 3D (optional)
    tri_meshes = []
    if ENABLE_TRIANGULATION and HAVE_EARCUT:
        for poly in selected:
            tri = triangulate_polygon_2d(poly)
            if tri is None:
                continue
            V2, F = tri
            V3, F3 = lift_triangles_to_3d(face, N_CELLS, V2, F)
            tri_meshes.append((V3, F3))
    elif ENABLE_TRIANGULATION and not HAVE_EARCUT:
        warnings.warn("[triangulation] mapbox_earcut not available; skipping triangulation.")

    # 8) Prepare rings lifted to 3D
    rings3d = [polygon_uv_to_xyz(face, N_CELLS, p) for p in selected]

    return {
        "face": face,
        "pieces": pieces,
        "selected": selected,
        "inner_lines": inner_lines,
        "rings3d": rings3d,       # list of dicts {"exterior": Nx3, "interiors":[...]}
        "tri_meshes": tri_meshes  # list of (V3, F3) if triangulated
    }


# --------------------- Optional debug plots ---------------------
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

def _ring_to_path(coords):
    pts = list(coords)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if not pts:
        return [], []
    verts = [(pts[0][0], pts[0][1])]
    codes = [MplPath.MOVETO]
    for x, y in pts[1:]:
        verts.append((x, y))
        codes.append(MplPath.LINETO)
    verts.append((0, 0))
    codes.append(MplPath.CLOSEPOLY)
    return verts, codes

def polygon_to_path(poly: Polygon) -> MplPath:
    verts, codes = [], []
    v, c = _ring_to_path(poly.exterior.coords)
    verts += v; codes += c
    for ring in poly.interiors:
        v, c = _ring_to_path(ring.coords)
        verts += v; codes += c
    return MplPath(verts, codes)

def add_polygon_patch(ax, poly: Polygon, facecolor, edgecolor="white", linewidth=0.8, zorder=6, alpha=0.7):
    path = polygon_to_path(poly)
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    return patch


# --------------------- Main ---------------------
if __name__ == "__main__":
    random.seed(RAND_SEED)

    results = {}
    for face in FACE_NAMES:
        res = process_face(face)
        results[face] = res
        print(f"[{face}] pieces={len(res['pieces'])} selected={len(res['selected'])} "
              f"tri_meshes={len(res['tri_meshes'])}")

        if SHOW_2D_PLOTS:
            # Quick 2D face plot
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            sq = Polygon([(0,0),(1,0),(1,1),(0,1)])
            sx, sy = zip(*sq.exterior.coords)
            ax.plot(sx, sy, color="black", linewidth=2, zorder=10)
            # inner lines
            for ls in res["inner_lines"]:
                x, y = ls.xy
                ax.plot(x, y, color="crimson", linewidth=1.4, alpha=0.9, zorder=30)
            # rejected grey (compute on the fly)
            rejected = [p for p in res["pieces"] if p not in set(res["selected"])]
            for p in rejected:
                add_polygon_patch(ax, p, facecolor=(0.85, 0.85, 0.85, 0.6),
                                  edgecolor="white", linewidth=0.7, zorder=5, alpha=0.6)
            # selected colored
            for p in res["selected"]:
                color = (random.random()*0.5+0.3, random.random()*0.5+0.3, random.random()*0.5+0.3, 0.7)
                add_polygon_patch(ax, p, facecolor=color, edgecolor="white", linewidth=0.8, zorder=6, alpha=0.7)

            ax.set_title(f"{face}: Selected={len(res['selected'])} / Total={len(res['pieces'])}")
            ax.set_xlabel("u"); ax.set_ylabel("v")
            plt.tight_layout()
            plt.show()

    if SHOW_3D_PREVIEW and HAVE_PV:
        pl = pv.Plotter()

        # (A) Split-level isosurfaces: build once (from current TPMS_FN, N_CELLS, N_GRID), reuse thereafter
        if SHOW_SPLIT_ISOSURFACES:
            iso_meshes = get_split_isosurfaces(TPMS_FN, N_CELLS, N_GRID, SPLIT_LEVELS)
            for i, m in enumerate(iso_meshes):
                if m is None or getattr(m, "n_points", 0) == 0:
                    continue
                pl.add_mesh(
                    m,
                    color="royalblue",
                    opacity=TRI_FACE_OPACITY,
                    show_edges=SHOW_TRI_EDGES,
                    edge_color=TRI_EDGE_COLOR if SHOW_TRI_EDGES else None,
                    smooth_shading=True,
                )

        # (B) Boundary polylines (optional)
        if SHOW_POLYLINE_RINGS:
            for face, res in results.items():
                for ringdict in res["rings3d"]:
                    ext = ringdict["exterior"]
                    if len(ext) >= 2:
                        pl.add_mesh(pv.Spline(ext, len(ext)), line_width=2)
                    for hole in ringdict["interiors"]:
                        if len(hole) >= 2:
                            pl.add_mesh(pv.Spline(hole, len(hole)), line_width=2)

        # (C) Triangulated caps with triangle edges
        for face, res in results.items():
            for V3, F3 in res["tri_meshes"]:
                faces_pv = np.hstack([np.full((len(F3),1), 3, dtype=np.int32), F3]).ravel()
                m = pv.PolyData(V3, faces_pv)
                pl.add_mesh(
                    m,
                    color="royalblue",
                    opacity=TRI_FACE_OPACITY,
                    show_edges=SHOW_TRI_EDGES,
                    edge_color=TRI_EDGE_COLOR if SHOW_TRI_EDGES else None,
                    smooth_shading=True,
                )

        pl.add_axes()
        pl.show()
    elif SHOW_3D_PREVIEW and not HAVE_PV:
        warnings.warn("[preview] PyVista not available; set SHOW_3D_PREVIEW=False or install pyvista.")
