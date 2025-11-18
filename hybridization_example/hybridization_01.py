import sys
import os
import numpy as np
import pyvista as pv
import trimesh

# Ensure project root is on sys.path so local packages (microgen) are importable
# when the script is executed directly (sys.path[0] is the script's folder).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from microgen import surface_functions as sf

# Auto-detect `utils` folder and add it to sys.path so `isocaps_structured`
# can be imported without editing this file. This assumes the repo layout
# where `hybridization_example/` and `utils/` are siblings.
ISO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
if ISO_PATH not in sys.path:
    sys.path.append(ISO_PATH)

from isocaps_structured import isocaps_structured
from reducepatch_trimesh import reducepatch_trimesh
from blending_functions import blending_function

# --- Parameters ---
T = 0.5
n_uc_x = 1
n_uc_y = 1
n_uc_z = 3
n_per_uc = 11*3

n_x = n_uc_x * n_per_uc
n_y = n_uc_y * n_per_uc
n_z = n_uc_z * n_per_uc

x = np.linspace(-n_uc_x * np.pi, n_uc_x * np.pi, n_x)
y = np.linspace(-n_uc_y * np.pi, n_uc_y * np.pi, n_y)
z = np.linspace(-n_uc_z * np.pi, n_uc_z * np.pi, n_z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Helper Functions ---
def blend_fields(fieldA, fieldB, weight):
    return (1 - weight) * fieldA + weight * fieldB

def pv_to_trimesh(pv_mesh):
    pv_mesh = pv_mesh.triangulate()
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    vertices = pv_mesh.points
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

# --- Generate TPMS Fields ---
field1 = sf.gyroid(X, Y, Z)
field2 = sf.schwarz_d(X, Y, Z)
field3 = sf.gyroid(X, Y, Z)
field4 = sf.schwarz_d(X, Y, Z)
field5 = sf.gyroid(X, Y, Z)
field6 = sf.schwarz_d(X, Y, Z)
field7 = sf.gyroid(X, Y, Z)
field8 = sf.schwarz_d(X, Y, Z)

# --- Compute Blending Weights ---
blendz = blending_function('linear', Z, -np.pi, np.pi)
blendx = blending_function('linear', X, -np.pi, np.pi)
blendy = blending_function('linear', Y, -np.pi, np.pi)

# --- Blend the Fields ---
hybridFieldz1 = blend_fields(field1, field2, blendz)
hybridFieldz2 = blend_fields(field3, field4, blendz)
hybridFieldz3 = blend_fields(field5, field6, blendz)
hybridFieldz4 = blend_fields(field7, field8, blendz)

hybridFieldx1 = blend_fields(hybridFieldz1, hybridFieldz2, blendx)
hybridFieldx2 = blend_fields(hybridFieldz3, hybridFieldz4, blendx)
hybridField = blend_fields(hybridFieldx1, hybridFieldx2, blendy)

# --- Offset Surfaces ---
TPMS_mid = hybridField
TPMS_in = TPMS_mid - T / 2
TPMS_out = TPMS_mid + T / 2
TPMS = TPMS_in * TPMS_out

# --- Structured Grid ---
grid = pv.StructuredGrid()
grid.points = np.c_[X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')]
grid.dimensions = X.shape

# # --- Isosurface and Isocaps for TPMS_in ---
# grid["values"] = TPMS_in.ravel(order='F')
# iso_in = grid.contour([0], scalars="values")
# caps_in = isocaps_structured(x, y, z, TPMS_in, 0, which_caps='below')
# tpms_in = iso_in.merge(caps_in)

# # --- Isosurface and Isocaps for TPMS_out ---
# grid["values"] = TPMS_out.ravel(order='F')
# iso_out = grid.contour([0], scalars="values")
# caps_out = isocaps_structured(x, y, z, TPMS_out, 0, which_caps='above')
# tpms_out = iso_out.merge(caps_out)

# # --- Convert to Trimesh and Boolean Intersection ---
# tm1 = pv_to_trimesh(tpms_in)
# tm2 = pv_to_trimesh(tpms_out)
# tm1.fix_normals()
# tm2.fix_normals()

# print(f"tm1 is volume: {tm1.is_volume}")
# print(f"tm2 is volume: {tm2.is_volume}")
# print(f"tm1 Euler number: {tm1.euler_number}")
# print(f"tm2 Euler number: {tm2.euler_number}")

# result = trimesh.boolean.intersection([tm1, tm2], engine='manifold')
# print(f"result is watertight: {result.is_watertight}")

# --- Isosurface and Isocaps for TPMS ---
grid["values"] = TPMS.ravel(order='F')
iso = grid.contour([0], scalars="values")
caps = isocaps_structured(x, y, z, TPMS, 0, which_caps='below')
tpms = iso.merge(caps)

tpms = pv_to_trimesh(tpms)
tpms.fix_normals()

result = tpms

# --- Mesh Reduction ---
v_reduced, f_reduced = reducepatch_trimesh(result.vertices, result.faces, 0.1, verbose=True)
reduced_mesh = trimesh.Trimesh(vertices=v_reduced, faces=f_reduced, process=False)

# --- Export Meshes ---
# Export the reduced mesh and the full boolean result for downstream tools
reduced_mesh.export('reduced_tpms_mesh.stl')
result.export("tpms_boolean_result.stl")

# --- Convert Boolean result to PyVista for visualization ---
result_pv = pv.PolyData(result.vertices, np.hstack(
    (np.full((result.faces.shape[0], 1), 3), result.faces)).astype(np.int32))

plotter = pv.Plotter()
plotter.add_mesh(result_pv, color='red', show_edges=False)
plotter.add_axes()
plotter.show(title="Final Intersected TPMS Shell")

