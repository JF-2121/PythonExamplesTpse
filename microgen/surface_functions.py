"""surface_functions compatibility shim

Provides a minimal set of TPMS surface functions (gyroid, schwarz_d, etc.)
by delegating to the project's `utils/tpms_unitcell_library.py` implementations.

The original `microgen.surface_functions` API used in the examples expects
functions that accept numpy arrays (X, Y, Z) and return the scalar field.
The functions below simply call the corresponding `phi_*` implementations
which are already numpy-friendly.
"""

from __future__ import annotations

import numpy as np
from utils.tpms_unitcell_library import (
    phi_gyroid,
    phi_schwarz_d,
    phi_schwarz_p,
    phi_neovius,
    phi_newovius,
)

# Expose the functions the examples call
def gyroid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return phi_gyroid(X, Y, Z)


def schwarz_d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return phi_schwarz_d(X, Y, Z)


def schwarz_p(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return phi_schwarz_p(X, Y, Z)


def neovius(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return phi_neovius(X, Y, Z)


def newovius(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return phi_newovius(X, Y, Z)

# Optional: convenience map if callers prefer string lookup
def get_surface_fn(name: str):
    name = name.lower()
    mapping = {
        'gyroid': gyroid,
        'schwarz_d': schwarz_d,
        'schwarz_p': schwarz_p,
        'neovius': neovius,
        'newovius': newovius,
    }
    return mapping[name]
