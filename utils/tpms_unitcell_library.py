# tpms_library.py
# Python 3.8+
# TPMS implicit fields φ(u,v,w) and analytical gradients ∇φ(u,v,w).
# Variables (u,v,w) live in implicit space; (x,y,z)=S*(u,v,w), S=L/(2π).

from typing import Tuple, Dict
import numpy as np

s = np.sin
c = np.cos

# ---------- φ & ∇φ definitions ----------

def phi_gyroid(u, v, w):
    # microgen-style sign convention you pasted (note the minus):
    return s(u)*c(v) - s(v)*c(w) + s(w)*c(u)

def grad_gyroid(u, v, w):
    du =  c(u)*c(v) - s(w)*s(u)
    dv = -s(u)*s(v) - c(v)*c(w)
    dw =  s(v)*s(w) + c(w)*c(u)
    return du, dv, dw

def phi_schwarz_p(u, v, w):
    return c(u) + c(v) + c(w)

def grad_schwarz_p(u, v, w):
    return -s(u), -s(v), -s(w)

def phi_schwarz_d(u, v, w):
    return (s(u)*s(v)*s(w)
            + s(u)*c(v)*c(w)
            + c(u)*s(v)*c(w)
            + c(u)*c(v)*s(w))

def grad_schwarz_d(u, v, w):
    du = ( c(u)*s(v)*s(w)
         + c(u)*c(v)*c(w)
         - s(u)*s(v)*c(w)
         - s(u)*c(v)*s(w) )
    dv = ( s(u)*c(v)*s(w)
         - s(u)*s(v)*c(w)
         + c(u)*c(v)*c(w)
         - c(u)*s(v)*s(w) )
    dw = ( s(u)*s(v)*c(w)
         - s(u)*c(v)*s(w)
         - c(u)*s(v)*s(w)
         + c(u)*c(v)*c(w) )
    return du, dv, dw

def phi_neovius(u, v, w):
    return 3*(c(u)+c(v)+c(w)) + 4*c(u)*c(v)*c(w)

def grad_neovius(u, v, w):
    du = -3*s(u) - 4*s(u)*c(v)*c(w)
    dv = -3*s(v) - 4*c(u)*s(v)*c(w)
    dw = -3*s(w) - 4*c(u)*c(v)*s(w)
    return du, dv, dw

def phi_newovius(u, v, w):
    return 3*(c(u)+c(v)+c(w)) + 8*c(u)*c(v)*c(w)

def grad_newovius(u, v, w):
    du = -3*s(u) - 8*s(u)*c(v)*c(w)
    dv = -3*s(v) - 8*c(u)*s(v)*c(w)
    dw = -3*s(w) - 8*c(u)*c(v)*s(w)
    return du, dv, dw

def phi_schoen_iwp(u, v, w):
    return 2*(c(u)*c(v) + c(v)*c(w) + c(w)*c(u)) - (c(2*u)+c(2*v)+c(2*w))

def grad_schoen_iwp(u, v, w):
    du = -2*s(u)*(c(v)+c(w)) + 2*s(2*u)
    dv = -2*s(v)*(c(u)+c(w)) + 2*s(2*v)
    dw = -2*s(w)*(c(u)+c(v)) + 2*s(2*w)
    return du, dv, dw

def phi_schoen_frd(u, v, w):
    return 4*c(u)*c(v)*c(w) - (c(2*u)*c(2*v) + c(2*v)*c(2*w) + c(2*w)*c(2*u))

def grad_schoen_frd(u, v, w):
    du = -4*s(u)*c(v)*c(w) + 2*s(2*u)*(c(2*v)+c(2*w))
    dv = -4*c(u)*s(v)*c(w) + 2*s(2*v)*(c(2*w)+c(2*u))
    dw = -4*c(u)*c(v)*s(w) + 2*s(2*w)*(c(2*u)+c(2*v))
    return du, dv, dw

def phi_fischer_koch_s(u, v, w):
    return c(2*u)*s(v)*c(w) + c(u)*c(2*v)*s(w) + s(u)*c(v)*c(2*w)

def grad_fischer_koch_s(u, v, w):
    du = -2*s(2*u)*s(v)*c(w) - s(u)*c(2*v)*s(w) + c(u)*c(v)*c(2*w)
    dv =  c(2*u)*c(v)*c(w) - 2*c(u)*s(2*v)*s(w) - s(u)*s(v)*c(2*w)
    dw = -c(2*u)*s(v)*s(w) + c(u)*c(2*v)*c(w) - 2*s(u)*c(v)*s(2*w)
    return du, dv, dw

def phi_pmy(u, v, w):
    return 2*c(u)*c(v)*c(w) + s(2*u)*s(v) + s(u)*s(2*w) + s(2*v)*s(w)

def grad_pmy(u, v, w):
    du = -2*s(u)*c(v)*c(w) + 2*c(2*u)*s(v) + c(u)*s(2*w)
    dv = -2*c(u)*s(v)*c(w) + s(2*u)*c(v) + 2*c(2*v)*s(w)
    dw = -2*c(u)*c(v)*s(w) + 2*s(u)*c(2*w) + s(2*v)*c(w)
    return du, dv, dw

def phi_lidinoid(u, v, w):
    return (s(2*u)*c(v)*s(w) + s(2*v)*c(w)*s(u) + s(2*w)*c(u)*s(v)
            - (c(2*u)*c(2*v) + c(2*v)*c(2*w) + c(2*w)*c(2*u)) + 0.3)

def grad_lidinoid(u, v, w):
    du = (2*c(2*u)*c(v)*s(w) + s(2*v)*c(w)*c(u) - s(2*w)*s(u)*s(v)
          + 2*s(2*u)*(c(2*v)+c(2*w)))
    dv = (-s(2*u)*s(v)*s(w) + 2*c(2*v)*c(w)*s(u) + s(2*w)*c(u)*c(v)
          + 2*s(2*v)*(c(2*u)+c(2*w)))
    dw = (s(2*u)*c(v)*c(w) - s(2*v)*s(w)*s(u) + 2*c(2*w)*c(u)*s(v)
          + 2*s(2*w)*(c(2*u)+c(2*v)))
    return du, dv, dw

def phi_split_p(u, v, w):
    return (1.1*(s(2*u)*c(v)*s(w) + s(2*v)*c(w)*s(u) + s(2*w)*c(u)*s(v))
            - 0.2*(c(2*u)*c(2*v) + c(2*v)*c(2*w) + c(2*w)*c(2*u))
            - 0.4*(c(2*u)+c(2*v)+c(2*w)))

def grad_split_p(u, v, w):
    du = (1.1*(2*c(2*u)*c(v)*s(w) + s(2*v)*c(w)*c(u) - s(2*w)*s(u)*s(v))
          + 0.4*s(2*u)*(c(2*v)+c(2*w)) + 0.8*s(2*u))
    dv = (1.1*(-s(2*u)*s(v)*s(w) + 2*c(2*v)*c(w)*s(u) + s(2*w)*c(u)*c(v))
          + 0.4*s(2*v)*(c(2*w)+c(2*u)) + 0.8*s(2*v))
    dw = (1.1*(s(2*u)*c(v)*c(w) - s(2*v)*s(w)*s(u) + 2*c(2*w)*c(u)*s(v))
          + 0.4*s(2*w)*(c(2*u)+c(2*v)) + 0.8*s(2*w))
    return du, dv, dw

def phi_honeycomb(u, v, w):
    return s(u)*c(v) + s(v) + c(w)

def grad_honeycomb(u, v, w):
    du = c(u)*c(v)
    dv = -s(u)*s(v) + c(v)
    dw = -s(w)
    return du, dv, dw

def phi_honeycomb_gyroid(u, v, w):
    return s(u)*c(v) + s(v) + c(u)

def grad_honeycomb_gyroid(u, v, w):
    du = c(u)*c(v) - s(u)
    dv = -s(u)*s(v) + c(v)
    dw = 0.0
    return du, dv, dw

def phi_honeycomb_schwarz_p(u, v, w):
    return c(u) + c(v)

def grad_honeycomb_schwarz_p(u, v, w):
    return -s(u), -s(v), 0.0

def phi_honeycomb_schwarz_d(u, v, w):
    return c(u)*c(v) + s(u)*s(v) + s(u)*c(v) + c(u)*s(v)

def grad_honeycomb_schwarz_d(u, v, w):
    du = -s(u)*c(v) + c(u)*s(v) + c(u)*c(v) - s(u)*s(v)
    dv = -c(u)*s(v) + s(u)*c(v) - s(u)*s(v) + c(u)*c(v)
    dw = 0.0
    return du, dv, dw

def phi_honeycomb_schoen_iwp(u, v, w):
    return c(u)*c(v) + c(v) + c(u)

def grad_honeycomb_schoen_iwp(u, v, w):
    du = -s(u)*c(v) - s(u)
    dv = -c(u)*s(v) - s(v)
    dw = 0.0
    return du, dv, dw

# ---------- registry (public) ----------

TPMS_FUNCS: Dict[str, Tuple] = {
    "gyroid":               (phi_gyroid,               grad_gyroid),
    "schwarz_p":            (phi_schwarz_p,            grad_schwarz_p),
    "primitive":            (phi_schwarz_p,            grad_schwarz_p),     # alias
    "schwarz_d":            (phi_schwarz_d,            grad_schwarz_d),
    "diamond":              (phi_schwarz_d,            grad_schwarz_d),     # alias
    "neovius":              (phi_neovius,              grad_neovius),
    "newovius":             (phi_newovius,             grad_newovius),
    "schoen_iwp":           (phi_schoen_iwp,           grad_schoen_iwp),
    "siwp":                 (phi_schoen_iwp,           grad_schoen_iwp),    # alias
    "iwp":                  (phi_schoen_iwp,           grad_schoen_iwp),    # alias
    "schoen_frd":           (phi_schoen_frd,           grad_schoen_frd),
    "fischer_koch_s":       (phi_fischer_koch_s,       grad_fischer_koch_s),
    "pmy":                  (phi_pmy,                  grad_pmy),
    "lidinoid":             (phi_lidinoid,             grad_lidinoid),
    "split_p":              (phi_split_p,              grad_split_p),
    "splitp":               (phi_split_p,              grad_split_p),       # alias
    "honeycomb":            (phi_honeycomb,            grad_honeycomb),
    "honeycomb_gyroid":     (phi_honeycomb_gyroid,     grad_honeycomb_gyroid),
    "honeycomb_schwarz_p":  (phi_honeycomb_schwarz_p,  grad_honeycomb_schwarz_p),
    "honeycomb_primitive":  (phi_honeycomb_schwarz_p,  grad_honeycomb_schwarz_p),   # alias
    "honeycomb_diamond":    (phi_honeycomb_schwarz_d,  grad_honeycomb_schwarz_d),   # alias
    "honeycomb_schwarz_d":  (phi_honeycomb_schwarz_d,  grad_honeycomb_schwarz_d),
    "honeycomb_schoen_iwp": (phi_honeycomb_schoen_iwp, grad_honeycomb_schoen_iwp),
    "honeycomb_siwp":       (phi_honeycomb_schoen_iwp, grad_honeycomb_schoen_iwp),  # alias
    "honeycomb_iwp":        (phi_honeycomb_schoen_iwp, grad_honeycomb_schoen_iwp),  # alias
}
