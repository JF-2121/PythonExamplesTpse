import numpy as np
import importlib.util
import inspect
import trimesh

try:
    import pyvista as pv  # optional fallback
    _HAVE_PV = True
except Exception:
    _HAVE_PV = False


def reducepatch_trimesh(vertices,
                        faces,
                        reduction_factor=0.5,
                        verbose=False,
                        *,
                        aggression=None,
                        preserve_topology=True):
    """
    Quadric decimation via trimesh.simplify_quadric_decimation (fast-simplification backend).
    Falls back to PyVista/VTK decimate_pro if available and the fast path fails.

    Parameters
    ----------
    vertices : (V,3) float
    faces    : (F,3) int
    reduction_factor : float|int
        <1.0 -> keep fraction (0.1 keeps 10%).
        >=1  -> target absolute face count.
    verbose : bool
    aggression : int|None
        0..10; only passed if supported by your trimesh/fast-simplification versions.
    preserve_topology : bool
        Used by the VTK fallback.

    Returns
    -------
    v_out : (V',3) float64
    f_out : (F',3) int64
    """
    # ---- validate inputs
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be an (F,3) triangle index array")
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must be a (V,3) array")

    orig_F = int(f.shape[0])
    orig_V = int(v.shape[0])

    # ---- target faces
    if reduction_factor <= 0:
        raise ValueError("reduction_factor must be > 0")

    if reduction_factor < 1.0:
        face_count = int(round(orig_F * float(reduction_factor)))
    else:
        face_count = int(round(reduction_factor))

    # clamp: keep at least 4 triangles, at most original
    face_count = int(np.clip(face_count, 4, max(4, orig_F)))

    if verbose:
        mode = "keep-fraction" if reduction_factor < 1 else "target-count"
        print(f"[reducepatch] mode={mode} | original: {orig_F} faces, {orig_V} vertices | target: {face_count}")

    # no-op if target is >= original
    if face_count >= orig_F:
        if verbose:
            print("[reducepatch] target >= original -> returning input")
        return v, f

    # ---- fast path: trimesh + fast_simplification
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)

    fast_simpl_installed = importlib.util.find_spec("fast_simplification") is not None
    accepts_aggr = "aggression" in inspect.signature(tm.simplify_quadric_decimation).parameters

    kwargs = {"face_count": face_count}
    if aggression is not None and fast_simpl_installed and accepts_aggr:
        kwargs["aggression"] = int(np.clip(int(aggression), 0, 10))
        if verbose:
            print(f"[reducepatch] using aggression={kwargs['aggression']}")
    elif aggression is not None and verbose:
        print("[reducepatch] 'aggression' not supported in this setup; ignoring")

    try:
        simplified = tm.simplify_quadric_decimation(**kwargs)
        if not isinstance(simplified, trimesh.Trimesh) or simplified.faces.size == 0:
            raise RuntimeError("empty/invalid result from quadric decimation")

        # cleanup for safe indexing
        simplified.remove_degenerate_faces()
        simplified.remove_unreferenced_vertices()

        v_out = np.asarray(simplified.vertices, dtype=np.float64)
        f_out = np.asarray(simplified.faces, dtype=np.int64)

        if verbose:
            red = 100.0 * (1.0 - f_out.shape[0] / orig_F)
            print(f"[reducepatch] reduced: {f_out.shape[0]} faces, {v_out.shape[0]} vertices "
                  f"({red:.1f}% fewer faces) | watertight={simplified.is_watertight}")

        return v_out, f_out

    except Exception as e:
        if not _HAVE_PV:
            raise RuntimeError(f"[reducepatch] trimesh decimation failed and PyVista fallback not available: {e}")
        if verbose:
            print(f"[reducepatch] trimesh/fast-simplification path failed: {e}")
            print("[reducepatch] falling back to PyVista/VTK decimate_pro…")

    # ---- fallback: PyVista / VTK decimate_pro
    # Build VTK face array: [3,i0,i1,i2, 3,i0,i1,i2, ...]
    counts = np.full((f.shape[0], 1), 3, dtype=np.int32)
    vtk_faces = np.hstack((counts, f.astype(np.int32))).ravel(order="C")
    pv_mesh = pv.PolyData(v, vtk_faces).triangulate()

    # VTK wants fraction to REMOVE
    remove_fraction = float(np.clip(1.0 - (face_count / float(orig_F)), 0.0, 0.99))
    reduced_pv = pv_mesh.decimate_pro(
        target_reduction=remove_fraction,
        preserve_topology=preserve_topology,
        feature_angle=30.0,
        splitting=False
    ).clean(triangle_strips=False)

    v_out = reduced_pv.points.astype(np.float64, copy=False)
    f_out = reduced_pv.faces.reshape(-1, 4)[:, 1:].astype(np.int64, copy=False)

    if f_out.size == 0:
        raise RuntimeError("VTK decimation produced an empty mesh.")

    if verbose:
        red = 100.0 * (1.0 - f_out.shape[0] / orig_F)
        print(f"[reducepatch] VTK reduced: {f_out.shape[0]} faces, {v_out.shape[0]} vertices "
              f"({red:.1f}% fewer faces)")

    return v_out, f_out
