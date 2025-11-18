import numpy as np

def linear_blend(x, start_val, end_val):
    t = (x - start_val) / (end_val - start_val)
    return np.clip(t, 0, 1)

def smoothstep_blend(x, start_val, end_val):
    t = (x - start_val) / (end_val - start_val)
    t = np.clip(t, 0, 1)
    return 3 * t**2 - 2 * t**3

def sigmoid_blend(x, start_val, end_val, k=10):
    x_c = (start_val + end_val) / 2
    return 1 / (1 + np.exp(-k * (x - x_c)))

def sinusoidal_blend(x, start_val, end_val):
    t = (x - start_val) / (end_val - start_val)
    t = np.clip(t, 0, 1)
    return 0.5 * (1 - np.cos(np.pi * t))

def polynomial_blend(x, start_val, end_val):
    t = (x - start_val) / (end_val - start_val)
    t = np.clip(t, 0, 1)
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def blending_function(blend_type, x, start_val, end_val, *args):
    blend_type = blend_type.lower()
    if blend_type == 'linear':
        return linear_blend(x, start_val, end_val)
    elif blend_type == 'smoothstep':
        return smoothstep_blend(x, start_val, end_val)
    elif blend_type == 'sigmoid':
        k = args[0] if args else 10
        return sigmoid_blend(x, start_val, end_val, k)
    elif blend_type == 'sinusoidal':
        return sinusoidal_blend(x, start_val, end_val)
    elif blend_type == 'polynomial':
        return polynomial_blend(x, start_val, end_val)
    else:
        raise ValueError(f"Unsupported blending type: {blend_type}")
