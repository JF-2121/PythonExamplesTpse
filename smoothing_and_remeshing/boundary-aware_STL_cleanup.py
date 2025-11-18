#!/usr/bin/env python3
# tpms_taubin_cvt_cpt_odt_lmo_safe_plus.py
# Python 3.8+
#
# Build a TPMS midsurface (marching cubes), then smooth it using one of:
#   - 'taubin'           : non-shrinking Taubin (λ–μ) with SAFE rails (your original)
#   - 'cvt-centroid'     : manifold CVT/Lloyd analogue (area-weighted 1-ring centroids)
#   - 'cpt-fixed-point'  : CPT fixed-point (density-preserving laplacian-like iteration)
#   - 'odt-circumcenter' : manifold ODT analogue (area-weighted 1-ring circumcenters)
#
# Optional *global* LMO pass (Nealen et al.):
#   - lmo_mode='shape'  : triangle-shape optimization (fix skinny triangles)
#   - lmo_mode='smooth' : feature-preserving smoothing (outer fairness)
#
# Boundary modes:
#   - 'fixed'         : all 6 planes frozen
#   - 'planar-slide'  : plane vertices move only along boundary polylines; edges/corners pinned
#   - 'periodic-wrap' : opposite planes move with identical tangential displacement (periodic)
#
# Extras retained:
#   - Unit-cube normalization for robust step sizes
#   - Per-vertex step clamp (prevents collapse)
#   - Closed- & open-curve boundary arclength equalization (optional)
#   - Deterministic periodic pairing via rounded tangential keys
#   - Side-by-side viz and optional STL/INP export
#
# Deps: numpy, scipy, pyvista, microgen (surface_functions)
# pip install numpy scipy pyvista

import os
import sys
import numpy as np

# Compatibility shims for libs expecting NumPy < 2.0 aliases
if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "int_"):     np.int_     = np.int64
if not hasattr(np, "complex_"): np.complex_ = np.complex128

import pyvista as pv
from scipy.sparse import coo_matrix, csr_matrix, diags, identity, vstack, hstack
from scipy.sparse.linalg import lsqr
from microgen import surface_functions as sf
from pathlib import Path

# (Optional) Abaqus exporter
abaqus_script_dir = Path(r"C:\Users\nm40ralo\Nextcloud\LSM\Abaqus\python_code_abaqus")
if str(abaqus_script_dir) not in sys.path:
    sys.path.append(str(abaqus_script_dir))
try:
    from exportMeshtoAbaqus import exportMeshtoAbaqus
except Exception:
    exportMeshtoAbaqus = None  # safe if you don't export .inp

# =======================================================================
# User controls
# =======================================================================
tpms_func = sf.schwarz_p   # e.g. sf.gyroid, sf.schwarz_p, sf.schwarz_d
nx, ny, nz = 1, 1, 1
n = 35                  # marching-cubes grid density (↑ = smoother base)

# --- choose smoothing algorithm ---
# "taubin" | "cvt-centroid" | "cpt-fixed-point" | "odt-circumcenter"
smoothing_algo = "cvt-centroid"
cpt_omega = 0.5    # step size for CPT fixed-point (0.1..1.0 is typical)

# --- OPTIONAL: one-shot LMO pass after the chosen smoother ---
# None        -> skip LMO
# "shape"     -> global triangle-shape optimization
# "smooth"    -> feature-preserving smoothing
lmo_mode = None

# LMO parameters
lmo_lap_type_L = "uniform"   # 'uniform'|'cotan'|'cotan-kappa'  (L used in constraints)
lmo_pos_weight = "cdf"       # 'cdf'|'linear'|'const'
lmo_pos_scale  = 1.0         # scales positional weights
lmo_wl_min     = 0.2         # min factor for Laplacian rows at features (0..1)
lmo_tangent_plane = True     # add soft tangent-plane constraints
step_clamp_frac = 0.08       # SAFE trust region

# Laplacian operator (used by Taubin and CPT):
#   'uniform' = degree-normalized umbrella (very stable)
#   'cotan'   = cotangent Laplacian (sharper, use once stable)
laplacian_type = 'uniform'

# Taubin parameters (non-shrinking smoothing)
taubin_lambda = 0.33
taubin_mu     = -0.34
taubin_iters  = 30

# Boundary mode: 'fixed', 'planar-slide', or 'periodic-wrap'
boundary_mode = 'periodic-wrap'

# Plane detection / deterministic pairing (unit cube coordinates)
plane_eps = 1e-12          # detect plane membership
pair_round_ndigits = 12    # rounding for tangential keys (deterministic pairing)
pairing_verbose = True     # print [PAIR] stats (one-time)

# Step limiting (prevents collapse), in normalized units
max_step_frac          = 0.20  # interior vertices
max_step_frac_boundary = 0.08  # single-plane vertices (stricter)

# Boundary polyline handling
boundary_curve_tangent    = True   # project updates onto boundary *tangent*
equalize_boundary_arclen  = False  # equal-arc-length redistribution at end
print_chain_stats         = False  # toggle chain-length stats

# Visualization
show_edges_wireframe = True
show_feature_edges   = True
show_boundary_pairs  = True  # optional boundary-pair dots overlay

# Pairing / diagnostics (for stubborn cases like split_p)
enable_pair_fallback    = True     # try a tiny tol fallback for leftover singles
pair_fallback_tol       = 5e-6     # fallback tol in tangential space (try 5e-8..1e-7..1e-6)
enable_pair_diagnostics = True     # print per-axis pairing stats

# Optional STL export (smoothed mesh)
export_stl  = False
export_path = r"C:\Users\John\Desktop\Remeshed\tpms_smoothed.stl"

# ========= Abaqus INP export (via your own module) =========
export_inp           = False
inp_element_type     = 'S3R'  # 'S3', 'S3R', or 'STRI65'
inp_path             = r"C:\Users\nm40ralo\Documents\Python\stl_output\primitive.inp"
materials = []  # e.g. [{'name':'Steel','E':210e9,'nu':0.3}]
sections  = []  # e.g. [{'elset':'EALL','material':'Steel'}]
steps     = []  # e.g. [{'name':'Static','nlgeom':'NO','stepType':'Static','additionalLines':''}]
stri65_tolerance = 1e-6  # only used if inp_element_type == 'STRI65'

# =======================================================================
# Helpers
# =======================================================================

def create_grid_and_field(tpms_fn, nx, ny, nz, n):
    x = np.linspace(-np.pi*nx, +np.pi*nx, n*nx)
    y = np.linspace(-np.pi*ny, +np.pi*ny, n*ny)
    z = np.linspace(-np.pi*nz, +np.pi*nz, n*nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    F = tpms_fn(X, Y, Z)
    return x, y, z, X, Y, Z, F

def ensure_tris(poly: pv.PolyData) -> pv.PolyData:
    tri = poly.triangulate()
    tri = tri.clean(point_merging=True, tolerance=1e-12)
    return tri

def feature_edges(poly: pv.PolyData) -> pv.PolyData:
    try:
        return poly.extract_feature_edges(boundary_edges=True, feature_edges=True,
                                          manifold_edges=False, non_manifold_edges=True)
    except Exception:
        return poly.extract_feature_edges(boundary_edges=True)

def detect_plane_masks_unit(pts, eps):
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    return {
        'xmin': np.isclose(x, 0.0, atol=eps),
        'xmax': np.isclose(x, 1.0, atol=eps),
        'ymin': np.isclose(y, 0.0, atol=eps),
        'ymax': np.isclose(y, 1.0, atol=eps),
        'zmin': np.isclose(z, 0.0, atol=eps),
        'zmax': np.isclose(z, 1.0, atol=eps),
    }

def build_adjacency(nV, F):
    I = np.hstack([F[:,0], F[:,1], F[:,1], F[:,2], F[:,2], F[:,0]])
    J = np.hstack([F[:,1], F[:,0], F[:,2], F[:,1], F[:,0], F[:,2]])
    W = np.ones_like(I, dtype=np.float64)
    A = coo_matrix((W, (I, J)), shape=(nV, nV)).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return A, deg

def uniform_laplacian(V, F):
    nV = V.shape[0]
    A, deg = build_adjacency(nV, F)
    invdeg = np.zeros_like(deg)
    nz = deg > 0
    invdeg[nz] = 1.0/deg[nz]
    Dinv = diags(invdeg)
    L = Dinv @ A - identity(nV, format='csr')
    return L, deg

def cotangent_laplacian(V, F):
    I = []; J = []; W = []
    nV = V.shape[0]
    M_diag = np.zeros(nV, dtype=np.float64)

    def tri_area(v0, v1, v2):
        return 0.5*np.linalg.norm(np.cross(v1-v0, v2-v0))

    def cot(a, b):
        la = np.linalg.norm(a); lb = np.linalg.norm(b)
        if la < 1e-15 or lb < 1e-15:
            return 0.0
        cosv = np.dot(a,b) / (la*lb)
        cosv = np.clip(cosv, -1.0, 1.0)
        s = np.sqrt(max(1.0 - cosv*cosv, 1e-30))
        return cosv / s

    for f in F:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        v0, v1, v2 = V[i], V[j], V[k]
        coti = cot(v1 - v0, v2 - v0)
        cotj = cot(v2 - v1, v0 - v1)
        cotk = cot(v0 - v2, v1 - v2)
        wij = 0.5 * cotk
        wjk = 0.5 * coti
        wki = 0.5 * cotj
        I += [i, j, j, k, k, i]; J += [j, i, k, j, i, k]
        W += [-wij, -wij, -wjk, -wjk, -wki, -wki]
        area = tri_area(v0, v1, v2)
        aa = area/3.0
        M_diag[i] += aa; M_diag[j] += aa; M_diag[k] += aa

    A_off = coo_matrix((W, (I, J)), shape=(nV, nV)).tocsr()
    diag_vals = -np.array(A_off.sum(axis=1)).ravel()
    A = A_off + csr_matrix((diag_vals, (np.arange(nV), np.arange(nV))), shape=(nV, nV))
    Minv_diag = np.where(M_diag > 1e-30, 1.0/M_diag, 0.0)
    Minv = diags(Minv_diag)
    L = Minv @ A
    return L, M_diag

def local_avg_edge_length(V, F):
    nV = V.shape[0]
    A, _ = build_adjacency(nV, F)
    rows, cols = A.nonzero()
    edge_len = np.linalg.norm(V[rows] - V[cols], axis=1)
    sum_len = np.zeros(nV); cnt = np.zeros(nV)
    np.add.at(sum_len, rows, edge_len); np.add.at(cnt, rows, 1)
    np.add.at(sum_len, cols, edge_len); np.add.at(cnt, cols, 1)
    h = np.zeros(nV); nz = cnt > 0
    h[nz] = sum_len[nz] / cnt[nz]
    if np.any(h<=1e-12):
        med = np.median(h[h>0]) if np.any(h>0) else 1.0
        h[h<=1e-12] = med
    return h

# ---------- vertex normals & curvature magnitude ----------
def _vertex_normals_mc(V, F):
    nV = V.shape[0]
    N = np.zeros_like(V)
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k, (i,j,kidx) in enumerate(F):
        N[i] += fn[k]; N[j] += fn[k]; N[kidx] += fn[k]
    n = np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
    return N / n

def _mean_curvature_kappa(V, F):
    Lc, _ = cotangent_laplacian(V, F)   # Lc = M^{-1}A
    K = Lc.dot(V)
    kappa = np.linalg.norm(K, axis=1)
    return kappa

# ---------- LMO (global LS optimizer) ----------
def _cdf_weights(kappa, s=1.0):
    k = np.asarray(kappa, float)
    k = np.clip(k, np.quantile(k, 0.0), np.quantile(k, 0.999))
    idx = np.argsort(k); inv = np.empty_like(idx); inv[idx] = np.arange(len(k))
    cdf = (inv.astype(float) + 1.0) / float(len(k))
    return s * cdf

def _linear_weights(kappa, s=1.0):
    k = np.asarray(kappa, float)
    lo, hi = np.quantile(k, 0.0), np.quantile(k, 0.999)
    if hi <= lo: return s*np.ones_like(k)
    return s * (np.clip(k, lo, hi) - lo) / (hi - lo)

def _laplacian_choice(V, F, which):
    if which == 'uniform':
        L, _ = uniform_laplacian(V, F)
        return L
    elif which == 'cotan':
        L, _ = cotangent_laplacian(V, F)
        return L
    elif which == 'cotan-kappa':
        L, _ = cotangent_laplacian(V, F)
        kappa = _mean_curvature_kappa(V, F)
        Wκ = diags(np.maximum(kappa, 1e-12))
        return Wκ @ L
    else:
        raise ValueError("laplacian must be 'uniform' | 'cotan' | 'cotan-kappa'")

# ---- helper: tiny union-find for grouping tied vertices ----
class _UF:
    def __init__(self, n):
        self.p = np.arange(n)
        self.r = np.zeros(n, dtype=int)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def _axes_for_coord(coord):
    if coord == 'x': return ('y','z')
    if coord == 'y': return ('x','z')
    if coord == 'z': return ('x','y')
    raise ValueError

def _build_reduction_R_for_coord(nV, pairs_dict, coord, V_unit, plane_eps):
    # identify single-plane vs edge/corner
    planes = detect_plane_masks_unit(V_unit, plane_eps)
    plane_count = (planes['xmin'].astype(int) + planes['xmax'].astype(int) +
                   planes['ymin'].astype(int) + planes['ymax'].astype(int) +
                   planes['zmin'].astype(int) + planes['zmax'].astype(int))
    edge_or_corner = (plane_count >= 2)
    single_plane   = (plane_count == 1)

    uf = _UF(nV)
    for ax in _axes_for_coord(coord):
        for i_m, i_p in pairs_dict.get(ax, []):
            if single_plane[i_m] and single_plane[i_p]:
                uf.union(i_m, i_p)

    rep_to_col = {}
    cols = []
    for i in range(nV):
        r = uf.find(i)
        if r not in rep_to_col:
            rep_to_col[r] = len(rep_to_col)
        cols.append(rep_to_col[r])
    cols = np.asarray(cols, dtype=int)
    m = len(rep_to_col)

    data = np.ones(nV, dtype=float)
    rows = np.arange(nV, dtype=int)
    R = csr_matrix((data, (rows, cols)), shape=(nV, m))
    return R

def lmo_optimize(
    V, F,
    mode='smooth',               # 'shape' or 'smooth'
    lap_type_L='uniform',        # 'uniform'|'cotan'|'cotan-kappa'
    pos_weight='cdf',            # 'cdf'|'linear'|'const'
    pos_scale=1.0,
    lap_reduce_at_features=True,
    wl_min=0.25,
    tangent_plane=True,
    periodic_pairs=None,         # dict from build_periodic_pairs_unit_deterministic()
    periodic_weight=80.0,        # only used if periodic_enforce='soft'
    plane_eps=1e-12,
    # stabilizers:
    normal_damp=True,            # if tangent_plane=False, damp (v'-v)·n ≈ 0
    normal_damp_weight=5.0,
    step_clamp_frac=0.08,
    # NEW:
    periodic_enforce='soft',     # 'soft' | 'hard'
):
    nV = V.shape[0]
    F = F.astype(int, copy=False)

    # ----- choose L and f -----
    L = _laplacian_choice(V, F, lap_type_L)
    if mode == 'shape':
        Lcot, _ = cotangent_laplacian(V, F)
        f = Lcot.dot(V)                        # (nV,3)
    elif mode == 'smooth':
        f = np.zeros_like(V)
    else:
        raise ValueError("mode must be 'shape' or 'smooth'")

    # ----- weights -----
    kappa = _mean_curvature_kappa(V, F)
    if pos_weight == 'cdf':
        Wp_diag = _cdf_weights(kappa, s=pos_scale)
    elif pos_weight == 'linear':
        Wp_diag = _linear_weights(kappa, s=pos_scale)
    else:
        Wp_diag = pos_scale * np.ones(nV)
    Wp = diags(Wp_diag)

    if lap_reduce_at_features:
        hp = (Wp_diag / (np.max(Wp_diag) + 1e-30))
        WL_diag = wl_min + (1.0 - wl_min) * (1.0 - hp)
    else:
        WL_diag = np.ones(nV)
    WL = diags(WL_diag)

    # Base system
    A_L = WL @ L
    A_P = Wp
    A_base = vstack([A_L, A_P], format='csr')
    b_base = np.vstack([ (WL @ f), (Wp @ V) ])  # (rows, 3)

    # ----- tangent-plane or normal-damp -----
    N = _vertex_normals_mc(V, F)         # (nV,3)
    A_tp_base = None
    b_tp      = None
    if tangent_plane:
        wtan = 5.0 * pos_scale
        A_tp_base = diags(wtan * np.ones(nV))
        b_tp = (wtan * np.sum(N * V, axis=1, keepdims=True))  # (nV,1)
    elif normal_damp:
        wns = normal_damp_weight * pos_scale
        A_tp_base = diags(wns * np.ones(nV))

    # ----- optional SOFT periodic rows (not used for 'hard') -----
    AperY = None; AperZ = None; perY = None; perZ = None
    if periodic_enforce == 'soft' and periodic_pairs is not None and periodic_weight > 0.0:
        def rows_for_pairs(pairs):
            if not pairs: return None
            rows = np.zeros((len(pairs), nV))
            for r, (im, ip) in enumerate(pairs):
                rows[r, im] =  1.0
                rows[r, ip] = -1.0
            return csr_matrix(rows)

        blocksY, blocksZ = [], []
        for ax, tang in (('x',[1,2]), ('y',[0,2]), ('z',[0,1])):
            pairs = periodic_pairs.get(ax, [])
            if not pairs: continue
            M = rows_for_pairs(pairs)
            if M is None: continue
            if 1 in tang: blocksY.append(M)
            if 2 in tang: blocksZ.append(M)

        if blocksY:
            AperY = vstack(blocksY, format='csr')
            AperY = diags(periodic_weight * np.ones(AperY.shape[0])) @ AperY
            perY  = np.zeros((AperY.shape[0], 1))
        if blocksZ:
            AperZ = vstack(blocksZ, format='csr')
            AperZ = diags(periodic_weight * np.ones(AperZ.shape[0])) @ AperZ
            perZ  = np.zeros((AperZ.shape[0], 1))

    # ----- HARD periodic reduction matrices (per coord) -----
    R_x = R_y = R_z = None
    if periodic_enforce == 'hard' and periodic_pairs is not None:
        R_x = _build_reduction_R_for_coord(nV, periodic_pairs, 'x', V, plane_eps)
        R_y = _build_reduction_R_for_coord(nV, periodic_pairs, 'y', V, plane_eps)
        R_z = _build_reduction_R_for_coord(nV, periodic_pairs, 'z', V, plane_eps)

    # ----- solve per coordinate -----
    Vsol = np.zeros_like(V)
    for d, coord in enumerate(['x','y','z']):
        Ad = A_base
        bd = b_base[:, [d]].copy()

        # tangent-plane / normal-damp rows
        if A_tp_base is not None:
            nd = N[:, [d]]
            A_tp_d = diags((A_tp_base.diagonal() * nd.ravel()))
            Ad = vstack([Ad, A_tp_d], format='csr')
            if tangent_plane:
                bd = np.vstack([bd, b_tp])
            else:
                bd = np.vstack([bd, (A_tp_base.diagonal()[:,None] * nd * V[:,[d]])])

        # Periodic enforcement
        if periodic_enforce == 'soft':
            if coord == 'y' and AperY is not None:
                Ad = vstack([Ad, AperY], format='csr'); bd = np.vstack([bd, perY])
            elif coord == 'z' and AperZ is not None:
                Ad = vstack([Ad, AperZ], format='csr'); bd = np.vstack([bd, perZ])
            sol = lsqr(Ad, bd.ravel(), atol=1e-10, btol=1e-10, iter_lim=2000)
            v_d = sol[0]
        else:
            R = R_x if coord == 'x' else (R_y if coord == 'y' else R_z)
            Ad_red = Ad @ R
            sol = lsqr(Ad_red, bd.ravel(), atol=1e-10, btol=1e-10, iter_lim=2000)
            u_d = sol[0]
            v_d = (R @ u_d)

        Vsol[:, d] = v_d

    # ----- SAFE trust region clamp -----
    dV = Vsol - V
    h = local_avg_edge_length(V, F)
    cap = (step_clamp_frac * h)[:,None]
    mag = np.linalg.norm(dV, axis=1, keepdims=True) + 1e-30
    scl = np.minimum(1.0, cap / mag)
    Vprime = V + dV * scl

    # final exact plane snap
    planes = detect_plane_masks_unit(Vprime, plane_eps)
    for name, mask in planes.items():
        if   name=='xmin': Vprime[mask,0] = 0.0
        elif name=='xmax': Vprime[mask,0] = 1.0
        elif name=='ymin': Vprime[mask,1] = 0.0
        elif name=='ymax': Vprime[mask,1] = 1.0
        elif name=='zmin': Vprime[mask,2] = 0.0
        elif name=='zmax': Vprime[mask,2] = 1.0

    return Vprime

# ---------- Boundary polylines (open + closed), tangents & equal-arclen ----------

def _build_boundary_chains(V, F, plane_masks):
    nV = V.shape[0]
    E = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)

    from collections import defaultdict
    chains_by_plane = {k: [] for k in plane_masks.keys()}

    for pname, mask in plane_masks.items():
        ids = np.where(mask)[0]
        if ids.size == 0: continue
        idset = set(int(i) for i in ids)

        e_mask = np.array([(int(a) in idset and int(b) in idset) for a,b in E], dtype=bool)
        Epl = E[e_mask]
        if Epl.size == 0: continue

        adj = defaultdict(list); deg = defaultdict(int)
        for a,b in map(tuple, Epl):
            adj[a].append(b); adj[b].append(a)
            deg[a]+=1; deg[b]+=1

        visited_edge = set()
        def mark_edge(u,v):
            if u > v: u,v = v,u
            visited_edge.add((u,v))
        def is_edge_visited(u,v):
            if u > v: u,v = v,u
            return (u,v) in visited_edge

        # open chains
        endpoints = [int(i) for i in ids if deg[int(i)]==1]
        for s in endpoints:
            if all(is_edge_visited(s,t) for t in adj[s]): continue
            chain = [s]; cur = s; prev = -1
            while True:
                nbrs = [u for u in adj[cur] if u != prev]
                nxts = [u for u in nbrs if not is_edge_visited(cur,u)]
                if not nxts: break
                nxt = int(nxts[0]); mark_edge(cur,nxt)
                chain.append(nxt); prev, cur = cur, nxt
                if deg[cur] == 1: break
            if len(chain) >= 2:
                chains_by_plane[pname].append( (np.asarray(chain, dtype=int), False) )

        # closed loops
        for s in ids:
            s = int(s)
            if all(is_edge_visited(s,t) for t in adj[s]): continue
            start = s; chain = [start]; cur = start; prev = -1
            while True:
                nbrs = [u for u in adj[cur] if u != prev]
                nxt = None
                for u in nbrs:
                    if not is_edge_visited(cur,u):
                        nxt = int(u); break
                if nxt is None: break
                mark_edge(cur, nxt)
                chain.append(nxt)
                prev, cur = cur, nxt
                if cur == start: break
            if len(chain) > 2 and chain[0] == chain[-1]:
                chain = chain[:-1]
            if len(chain) >= 2:
                chains_by_plane[pname].append( (np.asarray(chain, dtype=int), True) )

    return chains_by_plane

def _polyline_tangents(V, chain_idx):
    P = V[chain_idx]
    T = np.zeros_like(P)
    if len(chain_idx)==2:
        d = P[1]-P[0]; n = np.linalg.norm(d)+1e-30
        T[0]=T[1]=d/n
    else:
        d0 = P[1]-P[0]; dN = P[-1]-P[-2]
        T[0]  = d0/(np.linalg.norm(d0)+1e-30)
        T[-1] = dN/(np.linalg.norm(dN)+1e-30)
        for i in range(1, len(P)-1):
            d = P[i+1]-P[i-1]
            n = np.linalg.norm(d)+1e-30
            T[i] = d/n
    Tout = np.zeros_like(V)
    Tout[chain_idx] = T
    return Tout

def _equalize_polyline_arclen(V, chain_idx, plane_fix, closed=False):
    P = V[chain_idx]
    if len(chain_idx) < 3: return V

    if closed:
        P2 = np.vstack([P, P[0]])
        seg = np.linalg.norm(np.diff(P2, axis=0), axis=1)
        s = np.hstack([[0.0], np.cumsum(seg)])
        L = s[-1]
        if L <= 1e-14: return V
        s_new = np.linspace(0.0, L, len(chain_idx), endpoint=False)
        out = []
        j = 0
        for sn in s_new:
            while j < len(seg) and s[j+1] < sn: j += 1
            t = 0.0 if seg[j] < 1e-30 else (sn - s[j]) / seg[j]
            out.append((1.0 - t) * P2[j] + t * P2[j+1])
        Q = np.asarray(out)
    else:
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.hstack([[0.0], np.cumsum(seg)])
        L = s[-1]
        if L <= 1e-14: return V
        s_new = np.linspace(0.0, L, len(chain_idx))
        out = []
        j = 0
        for sn in s_new:
            while j < len(seg) and s[j+1] < sn: j += 1
            if j >= len(seg): out.append(P[-1])
            else:
                t = 0.0 if seg[j] < 1e-30 else (sn - s[j]) / seg[j]
                out.append((1.0 - t) * P[j] + t * P[j+1])
        Q = np.asarray(out)

    ax, val = plane_fix
    coord = {'x':0, 'y':1, 'z':2}[ax]
    Q[:,coord] = val
    V[chain_idx] = Q
    return V

def _chain_lengths_stats(V, chains, label=""):
    if not print_chain_stats: return
    def stats(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0: return 0.0, 0.0, 0.0
        m = float(vals.mean()); s = float(vals.std()); cv = float(s / (m + 1e-30))
        return m, s, cv
    print(f"[ARCLEN] ===== {label} =====")
    any_chain = False
    for pname, chs in chains.items():
        for k, entry in enumerate(chs):
            idx = entry[0] if isinstance(entry, tuple) else entry
            P = V[idx]
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            if seg.size == 0: continue
            m, s, cv = stats(seg); any_chain = True
            print(f"[ARCLEN] plane={pname:>4} chain#{k:02d}: mean={m:.4e}  std={s:.4e}  CV={cv:.3f}")
    if not any_chain:
        print("[ARCLEN] (no boundary chains found)")

# ---------- Deterministic periodic pairing (no KD-tree, no tol) ----------

def _rounded_key(a, b, ndigits):
    return np.vstack((np.round(a, ndigits), np.round(b, ndigits))).T

def build_periodic_pairs_unit_deterministic(pts, plane_eps=1e-12, ndigits=12, verbose=True):
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    out = {'x': [], 'y': [], 'z': []}

    def pair_axis(axis):
        if axis == 'x':
            mA = np.isclose(x, 0.0, atol=plane_eps); mB = np.isclose(x, 1.0, atol=plane_eps)
            keyA = _rounded_key(y[mA], z[mA], ndigits); keyB = _rounded_key(y[mB], z[mB], ndigits)
            idxA = np.where(mA)[0]; idxB = np.where(mB)[0]
        elif axis == 'y':
            mA = np.isclose(y, 0.0, atol=plane_eps); mB = np.isclose(y, 1.0, atol=plane_eps)
            keyA = _rounded_key(x[mA], z[mA], ndigits); keyB = _rounded_key(x[mB], z[mB], ndigits)
            idxA = np.where(mA)[0]; idxB = np.where(mB)[0]
        else:
            mA = np.isclose(z, 0.0, atol=plane_eps); mB = np.isclose(z, 1.0, atol=plane_eps)
            keyA = _rounded_key(x[mA], y[mA], ndigits); keyB = _rounded_key(x[mB], y[mB], ndigits)
            idxA = np.where(mA)[0]; idxB = np.where(mB)[0]

        from collections import defaultdict
        bucket = defaultdict(list)
        for j, k in zip(idxB, map(tuple, keyB)): bucket[k].append(j)

        pairs = []; unmatched = 0
        for i, k in zip(idxA, map(tuple, keyA)):
            lst = bucket.get(k, [])
            if lst: pairs.append((i, lst.pop()))
            else:   unmatched += 1
        if verbose:
            print(f"[PAIR] axis={axis}: paired={len(pairs)} | A={len(idxA)} B={len(idxB)} | unmatched A={unmatched}")
        return pairs

    for ax in ('x','y','z'):
        out[ax] = pair_axis(ax)
    return out

# ---------- Pairing diagnostics + tolerance fallback (optional) ----------

def diagnose_pairs(Vu, axis, pairs, ndigits=12, plane_eps=1e-12, title=""):
    """
    Vu: unit-cube vertices (n,3)
    axis: 'x' | 'y' | 'z'
    pairs: list[(i_minus, i_plus)]
    Prints tangential diffs and unmatched counts per plane.
    """
    x, y, z = Vu[:,0], Vu[:,1], Vu[:,2]
    if axis == 'x':
        mA = np.isclose(x, 0.0, atol=plane_eps); mB = np.isclose(x, 1.0, atol=plane_eps)
        tA = np.c_[np.round(y[mA], ndigits), np.round(z[mA], ndigits)]
        tB = np.c_[np.round(y[mB], ndigits), np.round(z[mB], ndigits)]
        iA = np.where(mA)[0]; iB = np.where(mB)[0]
    elif axis == 'y':
        mA = np.isclose(y, 0.0, atol=plane_eps); mB = np.isclose(y, 1.0, atol=plane_eps)
        tA = np.c_[np.round(x[mA], ndigits), np.round(z[mA], ndigits)]
        tB = np.c_[np.round(x[mB], ndigits), np.round(z[mB], ndigits)]
        iA = np.where(mA)[0]; iB = np.where(mB)[0]
    else:
        mA = np.isclose(z, 0.0, atol=plane_eps); mB = np.isclose(z, 1.0, atol=plane_eps)
        tA = np.c_[np.round(x[mA], ndigits), np.round(y[mA], ndigits)]
        tB = np.c_[np.round(x[mB], ndigits), np.round(y[mB], ndigits)]
        iA = np.where(mA)[0]; iB = np.where(mB)[0]

    pairedA = np.zeros(mA.sum(), dtype=bool)
    pairedB = np.zeros(mB.sum(), dtype=bool)
    mapA = {int(v):k for k,v in enumerate(iA)}
    mapB = {int(v):k for k,v in enumerate(iB)}
    diffs = []
    for a,b in pairs:
        ka = mapA.get(int(a)); kb = mapB.get(int(b))
        if ka is not None and kb is not None:
            pairedA[ka] = True; pairedB[kb] = True
            diffs.append(np.linalg.norm(tA[ka] - tB[kb]))
    diffs = np.array(diffs) if diffs else np.array([])

    print(f"[DIAG] {title} axis={axis} A={len(iA)} B={len(iB)} paired={len(pairs)}")
    if diffs.size:
        print(f"[DIAG]  tangential | mean={diffs.mean():.2e}  max={diffs.max():.2e}  95%={np.quantile(diffs,0.95):.2e}")
    print(f"[DIAG]  unmatched A={int((~pairedA).sum())}  unmatched B={int((~pairedB).sum())}")

def build_pairs_with_fallback(Vu, plane_eps=1e-12, ndigits=12, tol=5e-8, verbose=True):
    """
    Deterministic pairing first; then rescue unmatched single-plane vertices
    via 1–1 NN in tangential space within 'tol'. Returns dict like base function.
    """
    base = build_periodic_pairs_unit_deterministic(Vu, plane_eps=plane_eps, ndigits=ndigits, verbose=verbose)
    x, y, z = Vu[:,0], Vu[:,1], Vu[:,2]
    planes = detect_plane_masks_unit(Vu, plane_eps)
    pcnt = (planes['xmin'].astype(int) + planes['xmax'].astype(int) +
            planes['ymin'].astype(int) + planes['ymax'].astype(int) +
            planes['zmin'].astype(int) + planes['zmax'].astype(int))
    single_plane = (pcnt == 1)

    def fallback_axis(axis, pairs):
        if axis == 'x':
            mA = np.isclose(x, 0.0, atol=plane_eps); mB = np.isclose(x, 1.0, atol=plane_eps)
            tangA = np.c_[y[mA], z[mA]]; tangB = np.c_[y[mB], z[mB]]
            iA = np.where(mA & single_plane)[0]; iB = np.where(mB & single_plane)[0]
        elif axis == 'y':
            mA = np.isclose(y, 0.0, atol=plane_eps); mB = np.isclose(y, 1.0, atol=plane_eps)
            tangA = np.c_[x[mA], z[mA]]; tangB = np.c_[x[mB], z[mB]]
            iA = np.where(mA & single_plane)[0]; iB = np.where(mB & single_plane)[0]
        else:
            mA = np.isclose(z, 0.0, atol=plane_eps); mB = np.isclose(z, 1.0, atol=plane_eps)
            tangA = np.c_[x[mA], y[mA]]; tangB = np.c_[x[mB], y[mB]]
            iA = np.where(mA & single_plane)[0]; iB = np.where(mB & single_plane)[0]

        usedA = set(a for (a,_) in pairs); usedB = set(b for (_,b) in pairs)
        idxA = [int(a) for a in iA if int(a) not in usedA]
        idxB = [int(b) for b in iB if int(b) not in usedB]
        if not idxA or not idxB: return pairs

        try:
            from scipy.spatial import cKDTree as KDTree
        except Exception:
            from scipy.spatial import KDTree
        mapAm = {int(v):k for k,v in enumerate(np.where(mA)[0])}
        mapBp = {int(v):k for k,v in enumerate(np.where(mB)[0])}
        Acoords = np.array([tangA[mapAm[a]] for a in idxA])
        Bcoords = np.array([tangB[mapBp[b]] for b in idxB])
        tree = KDTree(Bcoords)

        takenB = set(); added = 0
        for ii, a in enumerate(idxA):
            dist, jloc = tree.query(Acoords[ii], k=1)
            if dist <= tol and jloc not in takenB:
                b = idxB[jloc]
                pairs.append((a,b))
                takenB.add(jloc); added += 1
        if verbose and added:
            print(f"[PAIR-FB] axis={axis}: +{added} fallback matches within tol={tol:g}")
        return pairs

    out = {'x': list(base['x']), 'y': list(base['y']), 'z': list(base['z'])}
    out['x'] = fallback_axis('x', out['x'])
    out['y'] = fallback_axis('y', out['y'])
    out['z'] = fallback_axis('z', out['z'])
    return out

# ---------- Geometry primitives for CVT/ODT ----------

def tri_centroid(v0, v1, v2):
    return (v0 + v1 + v2) / 3.0

def tri_area_simple(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def tri_circumcenter(v0, v1, v2):
    a = v1 - v0; b = v2 - v0
    adot = np.dot(a, a); bdot = np.dot(b, b); axb = np.cross(a, b)
    denom = 2.0 * np.maximum(np.dot(axb, axb), 1e-30)
    c = v0 + (np.cross(axb, a) * bdot + np.cross(b, axb) * adot) / denom
    return c

def one_ring_faces(F, nV):
    ring = [[] for _ in range(nV)]
    for fi, (i, j, k) in enumerate(F):
        ring[i].append(fi); ring[j].append(fi); ring[k].append(fi)
    return [np.asarray(r, dtype=int) for r in ring]

def cvt_targets(V, F):
    nV = V.shape[0]
    faces = F.astype(int, copy=False)
    V0 = V[faces[:,0]]; V1 = V[faces[:,1]]; V2 = V[faces[:,2]]
    C  = (V0 + V1 + V2) / 3.0
    A  = 0.5 * np.linalg.norm(np.cross(V1 - V0, V2 - V0), axis=1)
    rings = one_ring_faces(faces, nV)
    T = np.empty_like(V)
    for i in range(nV):
        idx = rings[i]
        if idx.size == 0: T[i] = V[i]
        else:
            w = A[idx]; s = w.sum()
            T[i] = V[i] if s <= 1e-30 else (C[idx] * w[:,None]).sum(axis=0) / s
    return T

def odt_targets(V, F):
    nV = V.shape[0]
    faces = F.astype(int, copy=False)
    V0 = V[faces[:,0]]; V1 = V[faces[:,1]]; V2 = V[faces[:,2]]
    CC = np.empty_like(V0)
    for fi in range(faces.shape[0]):
        CC[fi] = tri_circumcenter(V0[fi], V1[fi], V2[fi])
    A  = 0.5 * np.linalg.norm(np.cross(V1 - V0, V2 - V0), axis=1)
    rings = one_ring_faces(faces, nV)
    T = np.empty_like(V)
    for i in range(nV):
        idx = rings[i]
        if idx.size == 0: T[i] = V[i]
        else:
            w = A[idx]; s = w.sum()
            T[i] = V[i] if s <= 1e-30 else (CC[idx] * w[:,None]).sum(axis=0) / s
    return T

# ---------- Taubin (λ–μ) SAFE with deterministic periodic pairs ----------

def taubin_smooth_safe(
    V, F, iters=30, lam=0.33, mu=-0.34,
    lap_type='uniform', boundary_mode='planar-slide',
    plane_eps=1e-12, pair_round_ndigits=12,
    max_step_frac=0.2, max_step_frac_boundary=0.08,
    boundary_curve_tangent=True,
    equalize_boundary_arclen=True,
    pairing_logs=True,
    iter_callback=None,
    callback_every=1,
    stop_tol=None,
    static_pairs=None,
):
    V = V.copy()
    if lap_type == 'cotan': L, _ = cotangent_laplacian(V, F)
    else:                   L, _ = uniform_laplacian(V, F)

    nV = V.shape[0]
    h  = local_avg_edge_length(V, F)
    cap_int = (max_step_frac          * h)[:,None]
    cap_bnd = (max_step_frac_boundary * h)[:,None]

    planes = detect_plane_masks_unit(V, plane_eps)
    plane_count = (planes['xmin'].astype(int) + planes['xmax'].astype(int) +
                   planes['ymin'].astype(int) + planes['ymax'].astype(int) +
                   planes['zmin'].astype(int) + planes['zmax'].astype(int))
    edge_or_corner = plane_count >= 2
    single_plane   = (plane_count == 1)
    cap = np.where(single_plane[:,None], cap_bnd, cap_int)

    chains = _build_boundary_chains(V, F, planes) if boundary_curve_tangent else None
    T = np.zeros_like(V)
    if chains:
        for _, chs in chains.items():
            if not chs: continue
            for chain_idx, _ in chs:
                T += _polyline_tangents(V, chain_idx)

    if boundary_mode == 'periodic-wrap':
        if static_pairs is not None:
            pairs = static_pairs
            if pairing_logs:
                for ax in ('x','y','z'):
                    print(f"[PAIR] axis={ax}: (static) pairs={len(pairs.get(ax, []))}")
        else:
            if enable_pair_fallback:
                pairs = build_pairs_with_fallback(
                    V, plane_eps=plane_eps, ndigits=pair_round_ndigits,
                    tol=pair_fallback_tol, verbose=pairing_logs
                )
            else:
                pairs = build_periodic_pairs_unit_deterministic(
                    V, plane_eps=plane_eps, ndigits=pair_round_ndigits, verbose=pairing_logs
                )
            if enable_pair_diagnostics and pairing_logs:
                diagnose_pairs(V, 'x', pairs.get('x', []), ndigits=pair_round_ndigits, plane_eps=plane_eps, title="taubin")
                diagnose_pairs(V, 'y', pairs.get('y', []), ndigits=pair_round_ndigits, plane_eps=plane_eps, title="taubin")
                diagnose_pairs(V, 'z', pairs.get('z', []), ndigits=pair_round_ndigits, plane_eps=plane_eps, title="taubin")
    else:
        pairs = None

    def clamp(dV):
        mag = np.linalg.norm(dV, axis=1, keepdims=True) + 1e-30
        scl = np.minimum(1.0, cap / mag)
        return dV * scl

    def project_to_plane(V_prev, V_new):
        V_out = V_new.copy()
        V_out[edge_or_corner] = V_prev[edge_or_corner]
        for mask, coord, val in [
            (planes['xmin'], 0, 0.0), (planes['xmax'], 0, 1.0),
            (planes['ymin'], 1, 0.0), (planes['ymax'], 1, 1.0),
            (planes['zmin'], 2, 0.0), (planes['zmax'], 2, 1.0),
        ]:
            sp = np.logical_and(mask, ~edge_or_corner)
            if not np.any(sp): continue
            delta = V_out[sp] - V_prev[sp]
            delta[:, coord] = 0.0
            V_out[sp] = V_prev[sp] + delta
            V_out[sp, coord] = val
        return V_out

    def project_to_tangent(V_prev, V_new):
        V_mid = project_to_plane(V_prev, V_new)
        if chains is not None:
            sp = np.logical_and(single_plane, ~edge_or_corner)
            idx = np.where(sp)[0]
            if idx.size:
                d = V_mid[idx] - V_prev[idx]
                t = T[idx]
                tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-30
                t_unit = t / tn
                d_proj = (np.sum(d*t_unit, axis=1, keepdims=True)) * t_unit
                V_mid[idx] = V_prev[idx] + d_proj
        return V_mid

    def apply_update(V_prev, dV, s):
        d = clamp(s * dV)
        V_new = V_prev + d

        if boundary_mode == 'fixed':
            V_fix = V_new.copy()
            for mask in planes.values(): V_fix[mask] = V_prev[mask]
            return V_fix

        if boundary_mode == 'planar-slide':
            return project_to_tangent(V_prev, V_new) if boundary_curve_tangent else project_to_plane(V_prev, V_new)

        if boundary_mode == 'periodic-wrap':
            V_tmp = V_prev + d
            if pairs:
                axis_map = {'x':0, 'y':1, 'z':2}
                for ax in ('x','y','z'):
                    coord = axis_map[ax]
                    tangential_idx = [1,2] if ax=='x' else ([0,2] if ax=='y' else [0,1])
                    for i_minus, i_plus in pairs.get(ax, []):
                        disp_m = V_tmp[i_minus] - V_prev[i_minus]
                        disp_p = V_tmp[i_plus ] - V_prev[i_plus ]
                        disp_m[coord] = 0.0; disp_p[coord] = 0.0
                        avg_tan = 0.5 * (disp_m + disp_p)
                        disp_m[tangential_idx] = avg_tan[tangential_idx]
                        disp_p[tangential_idx] = avg_tan[tangential_idx]
                        V_tmp[i_minus] = V_prev[i_minus] + disp_m
                        V_tmp[i_plus ] = V_prev[i_plus ] + disp_p
            V_tmp = project_to_tangent(V_prev, V_tmp) if boundary_curve_tangent else project_to_plane(V_prev, V_tmp)
            return V_tmp

        return V_new

    iters_done = 0
    last_rms   = None
    for k in range(iters):
        V_before = V.copy()
        dV = L.dot(V); V = apply_update(V, dV, lam)
        dV = L.dot(V); V = apply_update(V, dV, mu)
        diff = V - V_before
        last_rms = float(np.sqrt(np.mean(np.sum(diff*diff, axis=1))))
        iters_done = k + 1

    if equalize_boundary_arclen:
        chains2 = _build_boundary_chains(V, F, detect_plane_masks_unit(V, plane_eps))
        _chain_lengths_stats(V, chains2, "before equalize")
        for pname, chs in chains2.items():
            if not chs: continue
            ax = pname[0]
            val = 0.0 if pname.endswith('min') else 1.0
            for chain_idx, is_closed in chs:
                V = _equalize_polyline_arclen(V, chain_idx, (ax, val), closed=is_closed)
        chains3 = _build_boundary_chains(V, F, detect_plane_masks_unit(V, plane_eps))
        _chain_lengths_stats(V, chains3, "after equalize")

    for name, mask in detect_plane_masks_unit(V, plane_eps).items():
        if   name=='xmin': V[mask,0] = 0.0
        elif name=='xmax': V[mask,0] = 1.0
        elif name=='ymin': V[mask,1] = 0.0
        elif name=='ymax': V[mask,1] = 1.0
        elif name=='zmin': V[mask,2] = 0.0
        elif name=='zmax': V[mask,2] = 1.0

    return V, {'iters_done': iters_done, 'last_rms': last_rms}, pairs

# ---------- Unified manifold smoother (CVT/CPT/ODT via your SAFE rails) ----------

def smooth_manifold(
    V, F,
    method="cvt-centroid",
    iters=30,
    lap_type='uniform',
    lam=0.33, mu=-0.34,
    omega=0.5,                       # for CPT
    boundary_mode='planar-slide',
    plane_eps=1e-12, pair_round_ndigits=12,
    max_step_frac=0.2, max_step_frac_boundary=0.08,
    boundary_curve_tangent=True,
    equalize_boundary_arclen=True,
    pairing_logs=True,
):
    if method == "taubin":
        return taubin_smooth_safe(
            V, F, iters=iters, lam=lam, mu=mu,
            lap_type=laplacian_type, boundary_mode=boundary_mode,
            plane_eps=plane_eps, pair_round_ndigits=pair_round_ndigits,
            max_step_frac=max_step_frac, max_step_frac_boundary=max_step_frac_boundary,
            boundary_curve_tangent=boundary_curve_tangent,
            equalize_boundary_arclen=equalize_boundary_arclen,
            pairing_logs=pairing_logs
        )

    V = V.copy()
    if lap_type == 'cotan': L, _ = cotangent_laplacian(V, F)
    else:                   L, _ = uniform_laplacian(V, F)

    planes = detect_plane_masks_unit(V, plane_eps)
    plane_count = (planes['xmin'].astype(int) + planes['xmax'].astype(int) +
                   planes['ymin'].astype(int) + planes['ymax'].astype(int) +
                   planes['zmin'].astype(int) + planes['zmax'].astype(int))
    edge_or_corner = plane_count >= 2
    single_plane   = (plane_count == 1)

    h  = local_avg_edge_length(V, F)
    cap_int = (max_step_frac          * h)[:,None]
    cap_bnd = (max_step_frac_boundary * h)[:,None]

    pairs = None
    if boundary_mode == 'periodic-wrap':
        if enable_pair_fallback:
            pairs = build_pairs_with_fallback(
                V, plane_eps=plane_eps, ndigits=pair_round_ndigits,
                tol=pair_fallback_tol, verbose=pairing_logs
            )
        else:
            pairs = build_periodic_pairs_unit_deterministic(
                V, plane_eps=plane_eps, ndigits=pair_round_ndigits, verbose=pairing_logs
            )
        if enable_pair_diagnostics and pairing_logs:
            diagnose_pairs(V, 'x', pairs.get('x', []), ndigits=pair_round_ndigits, plane_eps=plane_eps, title=method)
            diagnose_pairs(V, 'y', pairs.get('y', []), ndigits=pair_round_ndigits, plane_eps=plane_eps, title=method)
            diagnose_pairs(V, 'z', pairs.get('z', []), ndigits=pair_round_ndigits, plane_eps=plane_eps, title=method)

    def project_to_plane(V_prev, V_new):
        V_out = V_new.copy()
        V_out[edge_or_corner] = V_prev[edge_or_corner]
        for mask, coord, val in [
            (planes['xmin'], 0, 0.0), (planes['xmax'], 0, 1.0),
            (planes['ymin'], 1, 0.0), (planes['ymax'], 1, 1.0),
            (planes['zmin'], 2, 0.0), (planes['zmax'], 2, 1.0),
        ]:
            sp = np.logical_and(mask, ~edge_or_corner)
            if np.any(sp):
                delta = V_out[sp] - V_prev[sp]
                delta[:, coord] = 0.0
                V_out[sp] = V_prev[sp] + delta
                V_out[sp, coord] = val
        return V_out

    def project_periodic_and_tangent(V_prev, V_tmp):
        V_mid = V_tmp.copy()
        if boundary_mode == 'periodic-wrap' and pairs:
            axis_map = {'x':0, 'y':1, 'z':2}
            for ax in ('x','y','z'):
                coord = axis_map[ax]
                tangential_idx = [1,2] if ax=='x' else ([0,2] if ax=='y' else [0,1])
                for i_minus, i_plus in pairs.get(ax, []):
                    dm = V_mid[i_minus] - V_prev[i_minus]
                    dp = V_mid[i_plus ] - V_prev[i_plus ]
                    dm[coord] = 0.0; dp[coord] = 0.0
                    avg_tan = 0.5 * (dm + dp)
                    dm[tangential_idx] = avg_tan[tangential_idx]
                    dp[tangential_idx] = avg_tan[tangential_idx]
                    V_mid[i_minus] = V_prev[i_minus] + dm
                    V_mid[i_plus ]  = V_prev[i_plus ]  + dp

        V_mid = project_to_plane(V_prev, V_mid)

        if boundary_curve_tangent:
            ch = _build_boundary_chains(V_prev, F, planes)
            T = np.zeros_like(V_prev)
            for _, chs in ch.items():
                if not chs: continue
                for chain_idx, _cls in chs:
                    T += _polyline_tangents(V_prev, chain_idx)
            sp = np.logical_and(single_plane, ~edge_or_corner)
            idx = np.where(sp)[0]
            if idx.size:
                dloc = V_mid[idx] - V_prev[idx]
                t = T[idx]
                tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-30
                t_unit = t / tn
                d_proj = (np.sum(dloc*t_unit, axis=1, keepdims=True)) * t_unit
                V_mid[idx] = V_prev[idx] + d_proj

        return V_mid

    def apply_delta(V_prev, dV, step_scale=1.0):
        cap = np.where(single_plane[:,None], cap_bnd, cap_int)
        mag = np.linalg.norm(dV, axis=1, keepdims=True) + 1e-30
        scl = np.minimum(1.0, (cap * step_scale) / mag)
        V_tmp = V_prev + dV * scl
        return project_periodic_and_tangent(V_prev, V_tmp)

    last_rms = None
    for k in range(iters):
        V_before = V.copy()

        if method == "cpt-fixed-point":
            dV = L.dot(V)
            V = apply_delta(V, dV, step_scale=omega)

        elif method == "cvt-centroid":
            T = cvt_targets(V, F)
            V = apply_delta(V, T - V, step_scale=1.0)

        elif method == "odt-circumcenter":
            T = odt_targets(V, F)
            V = apply_delta(V, T - V, step_scale=1.0)

        else:
            raise ValueError(f"Unknown method: {method}")

        diff = V - V_before
        last_rms = float(np.sqrt(np.mean(np.sum(diff*diff, axis=1))))

    if equalize_boundary_arclen:
        chains2 = _build_boundary_chains(V, F, detect_plane_masks_unit(V, plane_eps))
        _chain_lengths_stats(V, chains2, "before equalize")
        for pname, chs in chains2.items():
            if not chs: continue
            ax = pname[0]; val = 0.0 if pname.endswith('min') else 1.0
            for chain_idx, is_closed in chs:
                V = _equalize_polyline_arclen(V, chain_idx, (ax, val), closed=is_closed)
        chains3 = _build_boundary_chains(V, F, detect_plane_masks_unit(V, plane_eps))
        _chain_lengths_stats(V, chains3, "after equalize")

    for name, mask in detect_plane_masks_unit(V, plane_eps).items():
        if   name=='xmin': V[mask,0] = 0.0
        elif name=='xmax': V[mask,0] = 1.0
        elif name=='ymin': V[mask,1] = 0.0
        elif name=='ymax': V[mask,1] = 1.0
        elif name=='zmin': V[mask,2] = 0.0
        elif name=='zmax': V[mask,2] = 1.0

    return V, {"iters_done": iters, "last_rms": last_rms}, pairs

# ---------- Post Smoothing/Exact Stitching ----------

def enforce_exact_periodicity(Vu, pairs, plane_eps=1e-12):
    """
    Make tangential coords exactly equal across opposite faces for provided pairs.
    Keeps normals snapped to {0,1}. Operates on *unit-cube* coords.
    """
    if pairs is None:
        return Vu
    V = Vu.copy()
    axis_idx = {'x':0,'y':1,'z':2}

    for ax in ('x','y','z'):
        if not pairs.get(ax): 
            continue
        c = axis_idx[ax]
        tang = [i for i in range(3) if i != c]
        for i, j in pairs[ax]:
            m = 0.5 * (V[i, tang] + V[j, tang])
            V[i, tang] = m
            V[j, tang] = m

    # Final exact plane snap
    planes = detect_plane_masks_unit(V, plane_eps)
    for name, mask in planes.items():
        if   name == 'xmin': V[mask, 0] = 0.0
        elif name == 'xmax': V[mask, 0] = 1.0
        elif name == 'ymin': V[mask, 1] = 0.0
        elif name == 'ymax': V[mask, 1] = 1.0
        elif name == 'zmin': V[mask, 2] = 0.0
        elif name == 'zmax': V[mask, 2] = 1.0
    return V

# =======================================================================
# Main: build TPMS, smooth, optional LMO, visualize, export
# =======================================================================

if __name__ == "__main__":
    # Build scalar field (parametric domain)
    x, y, z, X, Y, Z, F = create_grid_and_field(tpms_func, nx, ny, nz, n)
    dom_min = np.array([x.min(), y.min(), z.min()], dtype=float)
    dom_max = np.array([x.max(), y.max(), z.max()], dtype=float)

    # Marching cubes → midsurface
    grid = pv.StructuredGrid()
    grid.points = np.c_[X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')]
    grid.dimensions = X.shape
    grid['F'] = F.ravel(order='F')

    midsurf_orig = grid.contour([0.0], scalars='F').clean(point_merging=True, tolerance=1e-12)
    midsurf_orig = ensure_tris(midsurf_orig)

    # Normalize to unit cube (stabilizes steps)
    V0 = midsurf_orig.points.copy()
    scale = dom_max - dom_min
    scale[scale == 0.0] = 1.0
    Vn = (V0 - dom_min) / scale  # unit cube
    faces_tri = midsurf_orig.faces.reshape(-1, 4)[:, 1:]   # (nF, 3)

    print(f"[INFO] Original mesh: points={midsurf_orig.n_points}, cells={midsurf_orig.n_cells}")

    # ========== Run local smoother (and capture its pairs) ==========
    if smoothing_algo == "taubin":
        Vn_s, info, pairs_used_smooth = taubin_smooth_safe(
            Vn, faces_tri,
            iters=taubin_iters, lam=taubin_lambda, mu=taubin_mu,
            lap_type=laplacian_type, boundary_mode=boundary_mode,
            plane_eps=plane_eps, pair_round_ndigits=pair_round_ndigits,
            max_step_frac=max_step_frac, max_step_frac_boundary=max_step_frac_boundary,
            boundary_curve_tangent=boundary_curve_tangent,
            equalize_boundary_arclen=equalize_boundary_arclen,
            pairing_logs=pairing_verbose,
            iter_callback=None, callback_every=1, stop_tol=None,
            static_pairs=None
        )
        print(f".[TAUBIN] Done: iters={info['iters_done']} last_rms={(info['last_rms'] or 0.0):.3e}")
    else:
        Vn_s, info, pairs_used_smooth = smooth_manifold(
            Vn, faces_tri,
            method=smoothing_algo,
            iters=taubin_iters,
            lap_type=laplacian_type,
            lam=taubin_lambda, mu=taubin_mu,
            omega=cpt_omega,
            boundary_mode=boundary_mode,
            plane_eps=plane_eps, pair_round_ndigits=pair_round_ndigits,
            max_step_frac=max_step_frac, max_step_frac_boundary=max_step_frac_boundary,
            boundary_curve_tangent=boundary_curve_tangent,
            equalize_boundary_arclen=equalize_boundary_arclen,
            pairing_logs=pairing_verbose
        )
        print(f".[SMOOTH] {smoothing_algo} iters={info['iters_done']} last_rms={(info['last_rms'] or 0.0):.3e}")

    # === Post-smoothing exact stitch on unit cube (preserve periodicity exactly) ===
    if boundary_mode == 'periodic-wrap':
        Vn_s = enforce_exact_periodicity(Vn_s, pairs_used_smooth, plane_eps=plane_eps)

    # ========== Optional global LMO pass ==========
    if lmo_mode in ("shape", "smooth"):
        # (LMO can internally enforce periodicity; we still stitch afterwards for exactness)
        pairs_for_lmo = None
        if boundary_mode == 'periodic-wrap':
            # You can supply the same pairs to LMO (recommended):
            pairs_for_lmo = pairs_used_smooth

        Vn_lmo = lmo_optimize(
            Vn_s, faces_tri,
            mode=lmo_mode,                  # your preferred mode
            lap_type_L=lmo_lap_type_L,
            pos_weight=lmo_pos_weight,
            pos_scale=lmo_pos_scale,
            wl_min=lmo_wl_min,
            tangent_plane=lmo_tangent_plane,             # strongly recommended; or False with normal_damp=True
            periodic_pairs=pairs_for_lmo,
            periodic_enforce="hard",        # <<< EXACT periodicity via DOF reduction
            # periodic_weight is ignored in 'hard' mode
            step_clamp_frac=step_clamp_frac,             # SAFE trust region
        )
        Vn_s = Vn_lmo
        print(f".[LMO] mode={lmo_mode}  L={lmo_lap_type_L}  pos={lmo_pos_weight}  done.")

        # Stitch again after LMO so plotting & export are numerically exact.
        if boundary_mode == 'periodic-wrap':
            Vn_s = enforce_exact_periodicity(Vn_s, pairs_used_smooth, plane_eps=plane_eps)

    # Rescale back to world coords
    Vs = Vn_s * scale + dom_min

    # Build smoothed mesh
    midsurf_smooth = pv.PolyData(Vs, midsurf_orig.faces.copy())
    midsurf_smooth = midsurf_smooth.clean(point_merging=False, tolerance=0.0)
    midsurf_smooth = midsurf_smooth.compute_normals(auto_orient_normals=True, inplace=False)

    print(f"[INFO] Smoothed mesh: points={midsurf_smooth.n_points}, cells={midsurf_smooth.n_cells}")

    # Optional STL export
    if export_stl:
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            midsurf_smooth.save(export_path)
            print(f"[EXPORT] Wrote STL: {export_path}")
        except Exception as e:
            print(f"[EXPORT] STL export failed: {e}")

    # ========= Abaqus INP export (surface triangles) =========
    if export_inp and exportMeshtoAbaqus is not None:
        try:
            tri_elems = faces_tri.astype(np.int32, copy=True)
            exportMeshtoAbaqus(
                nodes=Vs,
                elements=tri_elems,
                filename=inp_path,
                element_type=inp_element_type,
                materials=materials,
                sections=sections,
                steps=steps,
                tolerance=stri65_tolerance
            )
            print(f"✅ Exported Abaqus INP: {inp_path}")
        except Exception as e:
            print(f"[INP] Export failed: {e}")

    # === Boundary-pair highlights built from the SAME pairs used in smoothing ===
    pairs_hi = pairs_used_smooth if (boundary_mode == 'periodic-wrap') else {'x':[], 'y':[], 'z':[]}

    # Collect matched indices per axis (from smoothing-time pairs)
    matched_x = set(i for (i, j) in pairs_hi.get('x', [])) | set(j for (i, j) in pairs_hi.get('x', []))
    matched_y = set(i for (i, j) in pairs_hi.get('y', [])) | set(j for (i, j) in pairs_hi.get('y', []))
    matched_z = set(i for (i, j) in pairs_hi.get('z', [])) | set(j for (i, j) in pairs_hi.get('z', []))

    # Detect ALL boundary vertices on the final unit mesh (for leftovers display)
    planes_s = detect_plane_masks_unit(Vn_s, plane_eps)
    on_x = np.where(planes_s['xmin'] | planes_s['xmax'])[0]
    on_y = np.where(planes_s['ymin'] | planes_s['ymax'])[0]
    on_z = np.where(planes_s['zmin'] | planes_s['zmax'])[0]

    # Edge/corner mask (>= 2 planes)
    pcnt = (planes_s['xmin'].astype(int) + planes_s['xmax'].astype(int) +
            planes_s['ymin'].astype(int) + planes_s['ymax'].astype(int) +
            planes_s['zmin'].astype(int) + planes_s['zmax'].astype(int))
    is_edge_corner = (pcnt >= 2)

    # Unmatched single-plane vertices per axis (relative to the smoother's pairs)
    unmatched_x = np.setdiff1d(on_x, np.fromiter(matched_x, dtype=int) if matched_x else np.array([], dtype=int))
    unmatched_y = np.setdiff1d(on_y, np.fromiter(matched_y, dtype=int) if matched_y else np.array([], dtype=int))
    unmatched_z = np.setdiff1d(on_z, np.fromiter(matched_z, dtype=int) if matched_z else np.array([], dtype=int))
    # Drop edges/corners from "unmatched" (drawn separately)
    unmatched_x = unmatched_x[~is_edge_corner[unmatched_x]]
    unmatched_y = unmatched_y[~is_edge_corner[unmatched_y]]
    unmatched_z = unmatched_z[~is_edge_corner[unmatched_z]]

    # World-space clouds for plotting (on SMOOTHED view)
    edge_pts_world   = Vs[np.where(is_edge_corner)[0]]
    x_match_pts      = Vs[np.fromiter(matched_x, dtype=int)] if matched_x else np.empty((0, 3))
    y_match_pts      = Vs[np.fromiter(matched_y, dtype=int)] if matched_y else np.empty((0, 3))
    z_match_pts      = Vs[np.fromiter(matched_z, dtype=int)] if matched_z else np.empty((0, 3))
    x_unmatched_pts  = Vs[unmatched_x]
    y_unmatched_pts  = Vs[unmatched_y]
    z_unmatched_pts  = Vs[unmatched_z]

    # (Optional) Tangential error check against the *same* pairs:
    def tangential_err(axis, Vu, pairs):
        ax = {'x':0,'y':1,'z':2}[axis]
        tang = [i for i in range(3) if i != ax]
        diffs = [np.linalg.norm(Vu[i, tang] - Vu[j, tang]) for (i,j) in pairs.get(axis, [])]
        return (float(np.mean(diffs)) if diffs else 0.0,
                float(np.max(diffs)) if diffs else 0.0,
                len(diffs))
    if boundary_mode == 'periodic-wrap':
        print("x mean/max,count:", tangential_err('x', Vn_s, pairs_hi))
        print("y mean/max,count:", tangential_err('y', Vn_s, pairs_hi))
        print("z mean/max,count:", tangential_err('z', Vn_s, pairs_hi))

    # Visualize side-by-side
    p = pv.Plotter(shape=(1, 2), window_size=(1300, 850),
                   title=f"{smoothing_algo} + LMO={lmo_mode} ({laplacian_type}, {boundary_mode}) iters={taubin_iters}")

    # Left: original
    p.subplot(0, 0)
    p.add_text("Original", font_size=12)
    p.add_mesh(midsurf_orig, color="#c9c9c9",
               show_edges=show_edges_wireframe, edge_color="black" if show_edges_wireframe else None)
    if show_feature_edges:
        p.add_mesh(feature_edges(midsurf_orig), color="#e74c3c", line_width=3, render_lines_as_tubes=True)
    p.add_axes(); p.view_isometric()

    # Right: smoothed (+ optional LMO)
    p.subplot(0, 1)
    tag = f"Smoothed ({smoothing_algo}, LMO={lmo_mode}, {boundary_mode})"
    if boundary_mode != 'fixed':
        if boundary_curve_tangent: tag += " + tangent"
        if equalize_boundary_arclen: tag += " + eq-arc"
    p.add_text(tag, font_size=12)
    p.add_mesh(midsurf_smooth, color="#c9c9c9",
               show_edges=show_edges_wireframe, edge_color="black" if show_edges_wireframe else None)
    if show_feature_edges:
        p.add_mesh(feature_edges(midsurf_smooth), color="#27ae60", line_width=3, render_lines_as_tubes=True)

    # --- Optional boundary-pair highlighting on the SMOOTHED view (uses smoothing-time pairs) ---
    if show_boundary_pairs and boundary_mode == 'periodic-wrap':
        # Matched vertices used in periodic pairs (per axis)
        if x_match_pts.size:
            p.add_mesh(pv.PolyData(x_match_pts), render_points_as_spheres=True,
                       point_size=14, color="#e74c3c")   # X: red
        if y_match_pts.size:
            p.add_mesh(pv.PolyData(y_match_pts), render_points_as_spheres=True,
                       point_size=14, color="#27ae60")   # Y: green
        if z_match_pts.size:
            p.add_mesh(pv.PolyData(z_match_pts), render_points_as_spheres=True,
                       point_size=14, color="#3498db")   # Z: blue

        # Single-plane boundary vertices that were NOT in pairs (leftovers), per axis
        if x_unmatched_pts.size:
            p.add_mesh(pv.PolyData(x_unmatched_pts), render_points_as_spheres=True,
                       point_size=8, color="#ff9aa2")   # X unmatched: light red
        if y_unmatched_pts.size:
            p.add_mesh(pv.PolyData(y_unmatched_pts), render_points_as_spheres=True,
                       point_size=8, color="#a7e8bd")   # Y unmatched: light green
        if z_unmatched_pts.size:
            p.add_mesh(pv.PolyData(z_unmatched_pts), render_points_as_spheres=True,
                       point_size=8, color="#9ecbff")   # Z unmatched: light blue

        # Edges/corners (>= 2 planes) shown separately
        if edge_pts_world.size:
            p.add_mesh(pv.PolyData(edge_pts_world), render_points_as_spheres=True,
                       point_size=10, color="#f1c40f")  # yellow

    p.add_axes(); p.view_isometric()
    p.link_views()
    p.show()
