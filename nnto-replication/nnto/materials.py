"""Material models: design variables -> constitutive entries and derivatives.

Every model exposes

    C, dC = model.evaluate(theta)       theta (ne, nvar)
                                        C  (ne, 4)        [C11, C12, C22, C66]
                                        dC (ne, 4, nvar)

``NNMaterial`` wraps a surrogate network and makes every questionable step of
the original implementation an explicit, switchable option so that the
published behaviour can be reproduced *and* each fix can be ablated:

    predict_cast   float32 -> inputs of the forward prediction are rounded to
                   single precision (what Keras ``Model.predict`` does in the
                   original Run_* scripts);  None -> double precision.
    clip           'original'   clip C to [Cmin, Cmax] but keep the raw NN
                                derivative (gradient inconsistent with the
                                objective wherever the clip is active);
                   'consistent' clip C and zero the derivative where clipped
                                (exact derivative of the clipped model);
                   'smooth'     C^1 soft floor/ceiling with exact derivative;
                   'none'       no clipping.
    derivative     'NNJ' | 'CFD' | 'CFD-f32' (as published) | 'CSM'
                   | 'AD-fwd' | 'AD-rev' | callable(net, x) -> Jacobian
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.interpolate import RectBivariateSpline

from . import derivatives as D
from .fea import C0_VEC_ORIGINAL, C0_VEC_EXACT
from .networks import Network

__all__ = ["SimpMaterial", "NNMaterial", "VoxelTableMaterial",
           "SplineMaterial", "DERIVATIVE_METHODS"]


def _cfd_as_published(net, x):
    return D.cfd(net, x, h=1e-6, input_dtype=np.float32, exact_step=False)


DERIVATIVE_METHODS = {
    "NNJ": D.nnj,
    "CFD": lambda net, x: D.cfd(net, x, h=1e-6),
    "CFD-f32": _cfd_as_published,
    "CSM": lambda net, x: D.csm(net, x, h=1e-12),
    "AD-fwd": D.ad_forward,
    "AD-rev": D.ad_reverse,
}


def _smooth_clip(c, lo, hi, width):
    """C^1 soft clip to [lo, hi]: softplus floor and ceiling of given width.
    Returns the clipped value and d(clipped)/dc."""
    a = (c - lo) / width
    b = (hi - c) / width
    sp = lambda t: np.logaddexp(0.0, t)
    sg = lambda t: 0.5 * (1.0 + np.tanh(0.5 * t))
    y = lo + width * sp(a)                      # floor
    dy = sg(a)
    bb = (hi - y) / width
    out = hi - width * sp(bb)                   # ceiling
    dout = sg(bb) * dy
    return out, dout


@dataclass
class SimpMaterial:
    """Solid isotropic material with penalisation, eq. (14)-(15)."""
    p: float = 3.0
    Emin: float = 1e-9
    Emax: float = 1.0
    C0: np.ndarray = None
    clip: bool = True          # the original also clips the (never-active) SIMP law

    nvar = 1

    def __post_init__(self):
        if self.C0 is None:
            self.C0 = C0_VEC_ORIGINAL

    def modulus(self, theta):
        t = np.asarray(theta, dtype=float).reshape(-1)
        E = self.Emin + (self.Emax - self.Emin) * t ** self.p
        dE = self.p * (self.Emax - self.Emin) * t ** (self.p - 1)      # NB: positive
        return E, dE

    def evaluate(self, theta):
        E, dE = self.modulus(theta)
        C = E[:, None] * self.C0
        if self.clip:
            C = np.minimum(np.maximum(C, self.Emin * self.C0), self.Emax * self.C0)
        return C, (dE[:, None] * self.C0)[:, :, None]


@dataclass
class NNMaterial:
    net: Network
    derivative: Union[str, Callable] = "NNJ"
    predict_cast: Optional[type] = np.float32
    clip: str = "original"
    C0: np.ndarray = None
    Emin: float = 1e-9
    smooth_width: float = 1e-3
    scale_by_C0: Optional[bool] = None       # True: net predicts E (DBTO)

    def __post_init__(self):
        if self.C0 is None:
            self.C0 = C0_VEC_ORIGINAL
        if self.scale_by_C0 is None:
            self.scale_by_C0 = (self.net.ydim == 1)
        self.nvar = self.net.xdim
        self._dfun = (DERIVATIVE_METHODS[self.derivative]
                      if isinstance(self.derivative, str) else self.derivative)

    def raw(self, theta):
        """Un-clipped prediction and derivative, (ne, 4), (ne, 4, nvar)."""
        x = np.asarray(theta, dtype=float).reshape(-1, self.nvar)
        xin = x.astype(self.predict_cast).astype(float) if self.predict_cast else x
        y = self.net.forward(xin)
        J = self._dfun(self.net, x)                      # (ne, ydim, nvar)
        if self.scale_by_C0:
            return y * self.C0, J[:, [0], :] * self.C0[None, :, None]
        return y, J

    def evaluate(self, theta):
        C, dC = self.raw(theta)
        lo, hi = self.Emin * self.C0, self.C0
        if self.clip == "none":
            return C, dC
        if self.clip == "smooth":
            Cc, g = _smooth_clip(C, lo, hi, self.smooth_width * self.C0)
            return Cc, dC * g[:, :, None]
        Cc = np.minimum(np.maximum(C, lo), hi)
        if self.clip == "original":
            return Cc, dC
        if self.clip == "consistent":
            active = (C > lo) & (C < hi)
            return Cc, dC * active[:, :, None]
        raise ValueError(f"unknown clip mode {self.clip!r}")


@dataclass
class VoxelTableMaterial:
    """The piecewise-constant voxel model that generated the paper's HBTO
    training data (nearest lattice value).  No derivative (returns zeros)."""
    grid: np.ndarray
    C: np.ndarray
    nvar = 2

    def evaluate(self, theta):
        x = np.asarray(theta, dtype=float).reshape(-1, 2)
        step = self.grid[1] - self.grid[0]
        # an element is void when its centre is strictly inside the hole
        idx = np.clip(np.ceil((x - step / 2) / step - 1e-12).astype(int), 0, self.grid.size - 1)
        return self.C[idx[:, 0], idx[:, 1]], np.zeros((x.shape[0], 4, 2))


class SplineMaterial:
    """Smooth interpolant of the mesh-converged homogenised tensor.

    Bicubic splines through the ground-truth grid; values *and* derivatives
    are evaluated from the spline.  Outside the tabulated range the inputs are
    clamped (constant extrapolation, zero derivative)."""
    nvar = 2

    def __init__(self, grid, C, dC=None):
        self.grid = np.asarray(grid)
        self.lo, self.hi = self.grid[0], self.grid[-1]
        self.splines = [RectBivariateSpline(self.grid, self.grid, C[:, :, k], kx=3, ky=3)
                        for k in range(4)]

    def evaluate(self, theta):
        x = np.asarray(theta, dtype=float).reshape(-1, 2)
        xc = np.clip(x, self.lo, self.hi)
        inside = (x >= self.lo) & (x <= self.hi)
        C = np.stack([s.ev(xc[:, 0], xc[:, 1]) for s in self.splines], axis=1)
        d1 = np.stack([s.ev(xc[:, 0], xc[:, 1], dx=1) for s in self.splines], axis=1)
        d2 = np.stack([s.ev(xc[:, 0], xc[:, 1], dy=1) for s in self.splines], axis=1)
        dC = np.stack([d1 * inside[:, [0]], d2 * inside[:, [1]]], axis=2)
        return C, dC
