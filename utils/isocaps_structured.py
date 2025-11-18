import numpy as np
import pyvista as pv

def isocaps_structured(x, y, z, v, isovalue, which_caps='above',
                       which_planes=('xmin','xmax','ymin','ymax','zmin','zmax'),
                       tol=1e-8):
    """
    Recreate Octave's isocaps for a structured 3D grid with support for 3D varying isovalue.

    Parameters:
        x, y, z        : 1D numpy arrays
        v              : 3D numpy array of scalar field values
        isovalue       : float or ndarray of shape v (can vary per point)
        which_caps     : 'above' or 'below' (default 'above')
        which_planes   : subset of ('xmin','xmax','ymin','ymax','zmin','zmax')
        tol            : tolerance

    Returns:
        pv.PolyData : merged caps mesh
    """
    data_min = np.min(v)
    data_max = np.max(v)
    pad_val = data_max + 1 if which_caps.lower().startswith('b') else data_min - 1

    nx, ny, nz = v.shape
    caps = []

    def get_iso_slice(field, plane):
        if isinstance(field, np.ndarray) and field.ndim == 3:
            if plane == 'xmin': return field[0, :, :][np.newaxis, :, :]
            if plane == 'xmax': return field[-1, :, :][np.newaxis, :, :]
            if plane == 'ymin': return field[:, 0, :][:, np.newaxis, :]
            if plane == 'ymax': return field[:, -1, :][:, np.newaxis, :]
            if plane == 'zmin': return field[:, :, 0][:, :, np.newaxis]
            if plane == 'zmax': return field[:, :, -1][:, :, np.newaxis]
        return None

    def cap_on_slice(Xc, Yc, Zc, cap_data, plane, iso_slice):
        sg = pv.StructuredGrid()
        sg.points = np.c_[Xc.ravel(order='F'), Yc.ravel(order='F'), Zc.ravel(order='F')]
        sg.dimensions = Xc.shape
        sg['v'] = cap_data.ravel(order='F')

        if iso_slice is not None:
            v_shifted = cap_data - iso_slice
            sg['v_shifted'] = v_shifted.ravel(order='F')
            patch = sg.contour([0.0], scalars='v_shifted')
        else:
            patch = sg.contour([isovalue], scalars='v')

        if plane == 'xmin': patch.points[:, 0] = x[0]
        if plane == 'xmax': patch.points[:, 0] = x[-1]
        if plane == 'ymin': patch.points[:, 1] = y[0]
        if plane == 'ymax': patch.points[:, 1] = y[-1]
        if plane == 'zmin': patch.points[:, 2] = z[0]
        if plane == 'zmax': patch.points[:, 2] = z[-1]
        return patch

    for which in which_planes:
        if which.startswith('x'):
            i = 0 if which == 'xmin' else nx - 1
            cap_data = np.empty((2, ny, nz))
            cap_data[1, :, :] = v[i, :, :]
            cap_data[0, :, :] = pad_val
            Xc = np.full((2, ny, nz), x[i])
            Yc, Zc = np.meshgrid(y, z, indexing='ij')
            Yc = np.repeat(Yc[np.newaxis, :, :], 2, axis=0)
            Zc = np.repeat(Zc[np.newaxis, :, :], 2, axis=0)
            iso_slice = get_iso_slice(isovalue, which)
            caps.append(cap_on_slice(Xc, Yc, Zc, cap_data, which, iso_slice))

        elif which.startswith('y'):
            j = 0 if which == 'ymin' else ny - 1
            cap_data = np.empty((nx, 2, nz))
            cap_data[:, 1, :] = v[:, j, :]
            cap_data[:, 0, :] = pad_val
            Yc = np.full((nx, 2, nz), y[j])
            Xc, Zc = np.meshgrid(x, z, indexing='ij')
            Xc = np.repeat(Xc[:, np.newaxis, :], 2, axis=1)
            Zc = np.repeat(Zc[:, np.newaxis, :], 2, axis=1)
            iso_slice = get_iso_slice(isovalue, which)
            caps.append(cap_on_slice(Xc, Yc, Zc, cap_data, which, iso_slice))

        elif which.startswith('z'):
            k = 0 if which == 'zmin' else nz - 1
            cap_data = np.empty((nx, ny, 2))
            cap_data[:, :, 1] = v[:, :, k]
            cap_data[:, :, 0] = pad_val
            Zc = np.full((nx, ny, 2), z[k])
            Xc, Yc = np.meshgrid(x, y, indexing='ij')
            Xc = np.repeat(Xc[:, :, np.newaxis], 2, axis=2)
            Yc = np.repeat(Yc[:, :, np.newaxis], 2, axis=2)
            iso_slice = get_iso_slice(isovalue, which)
            caps.append(cap_on_slice(Xc, Yc, Zc, cap_data, which, iso_slice))

    if not caps:
        return None
    merged = caps[0]
    for patch in caps[1:]:
        merged = merged.merge(patch)
    return merged
