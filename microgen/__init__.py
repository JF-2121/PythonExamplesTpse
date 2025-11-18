"""Local shim package for `microgen` required by the examples.

This small shim wraps the local `utils/tpms_unitcell_library.py` TPMS functions
so the example scripts can import `from microgen import surface_functions as sf`.
"""

__all__ = ["surface_functions"]
