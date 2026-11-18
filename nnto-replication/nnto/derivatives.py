"""Derivatives of a network's outputs with respect to its inputs.

All functions take a :class:`nnto.networks.Network` and a batch ``x`` of shape
``(N, xdim)`` and return a batch Jacobian ``(N, ydim, xdim)`` (or a Hessian
``(N, ydim, xdim, xdim)``).  They are fully vectorised over the batch.

First order
    nnj            analytic chain rule (paper Sec. 2.1), forward accumulation
    cfd            central finite differences (paper Sec. 2.2)
    cfd_ridders    adaptive finite differences with an error estimate
    csm            complex-step method (paper Sec. 2.3)
    ad_forward     forward-mode AD (dual numbers; xdim sweeps)
    ad_reverse     reverse-mode AD (tape; ydim sweeps)

Second order
    hessian_nnj            analytic second-order chain rule
    hessian_ad             hyper-dual numbers (exact)
    hessian_cfd            second differences of function values
    hessian_cfd_of_nnj     central differences of the analytic Jacobian
    hessian_csm_of_nnj     complex step applied to the analytic Jacobian (exact)
    hessian_csm_values     complex-step formula using function values only
                           (has subtractive cancellation -- shown for contrast)
    hvp                    Hessian-vector product, forward-over-reverse AD
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import autodiff as ad
from .networks import Network

__all__ = [
    "nnj", "cfd", "cfd_ridders", "csm", "ad_forward", "ad_reverse",
    "optimal_cfd_step", "hessian_nnj", "hessian_ad", "hessian_cfd",
    "hessian_cfd_of_nnj", "hessian_csm_of_nnj", "hessian_csm_values", "hvp",
    "METHODS",
]


def _as_batch(x) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x


# --------------------------------------------------------------------------- #
# First order
# --------------------------------------------------------------------------- #
def nnj(net: Network, x) -> np.ndarray:
    """Analytic neural-network Jacobian."""
    return net.jacobian_nnj(_as_batch(x))


def optimal_cfd_step(x, dtype=np.float64, scale: float = 1.0) -> np.ndarray:
    """Near-optimal central-difference step,  h ~ eps**(1/3) * max(|x|, scale).

    Balances the truncation error  h**2 |f'''| / 6  against the round-off
    error  eps |f| / h  (the constants are problem dependent; this is the
    standard rule of thumb).
    """
    eps = np.finfo(dtype).eps
    return eps ** (1.0 / 3.0) * np.maximum(np.abs(np.asarray(x, dtype=float)), scale)


def cfd(net: Network, x, h=1e-6, input_dtype=None,
        exact_step: bool = True) -> np.ndarray:
    """Central finite differences, eq. (7) of the paper.

    Parameters
    ----------
    h : float or array broadcastable to ``x``
        Step size (the original code uses 1e-6).
    input_dtype : dtype, optional
        If given, the perturbed inputs are rounded to this dtype before the
        network is evaluated.  ``input_dtype=np.float32, exact_step=False``
        reproduces the published implementation, in which Keras'
        ``Model.predict`` casts the inputs to float32 (audit finding A1).
    exact_step : bool
        Divide by the step that was *actually* taken,
        ``(x+h) - (x-h)`` evaluated after rounding, instead of the nominal
        ``2h``.  This removes the input-rounding error to first order and is
        the standard remedy; ``False`` reproduces the original behaviour.
    """
    x = _as_batch(x)
    n, xdim = x.shape
    hh = np.broadcast_to(np.asarray(h, dtype=float), x.shape)
    cols = []
    for j in range(xdim):
        xp = np.array(x, dtype=float, copy=True)
        xm = np.array(x, dtype=float, copy=True)
        xp[:, j] += hh[:, j]
        xm[:, j] -= hh[:, j]
        if input_dtype is not None:
            xp = xp.astype(input_dtype).astype(x.dtype)
            xm = xm.astype(input_dtype).astype(x.dtype)
        else:
            xp = xp.astype(x.dtype)
            xm = xm.astype(x.dtype)
        dy = np.asarray(net.forward(xp), dtype=float) - np.asarray(net.forward(xm), dtype=float)
        if exact_step:
            den = (xp[:, j].astype(float) - xm[:, j].astype(float))[:, None]
        else:
            den = 2.0 * hh[:, j][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            # a step below the resolution of the input dtype gives x+h == x-h
            # and hence nan: the derivative cannot be formed at that precision
            cols.append(dy / den)
    return np.stack(cols, axis=-1)


def cfd_ridders(net: Network, x, h0: float = 0.05, ntab: int = 10,
                con: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive finite differences (Ridders' extrapolation to h -> 0).

    A Neville tableau of central differences with geometrically decreasing
    steps ``h0 / con**i`` is built.  For every entry of the Jacobian the
    extrapolant with the smallest internal error estimate is returned together
    with that estimate, so no step size has to be tuned by hand and the
    accuracy actually reached is reported.

    Limitation (see tests): a symmetric stencil cannot see a kink that lies
    inside it.  For non-smooth networks (ReLU) the result near a kink is a
    blend of the two one-sided slopes and the error estimate does not flag it.
    """
    x = np.asarray(_as_batch(x), dtype=float)
    n, xdim = x.shape
    ydim = net.ydim
    J = np.zeros((n, ydim, xdim))
    E = np.full((n, ydim, xdim), np.inf)
    con2 = con * con
    for j in range(xdim):
        a = np.zeros((ntab, ntab, n, ydim))

        def cd(step):
            xp = x.copy(); xm = x.copy()
            xp[:, j] += step; xm[:, j] -= step
            return (net.forward(xp) - net.forward(xm)) / ((xp[:, j] - xm[:, j])[:, None])

        hh = h0
        a[0, 0] = cd(hh)
        best = a[0, 0].copy()
        err = np.full((n, ydim), np.inf)
        for i in range(1, ntab):
            hh /= con
            a[0, i] = cd(hh)
            fac = con2
            for m in range(1, i + 1):
                a[m, i] = (a[m - 1, i] * fac - a[m - 1, i - 1]) / (fac - 1.0)
                fac *= con2
                errt = np.maximum(np.abs(a[m, i] - a[m - 1, i]),
                                  np.abs(a[m, i] - a[m - 1, i - 1]))
                upd = errt <= err
                best = np.where(upd, a[m, i], best)
                err = np.where(upd, errt, err)
        J[:, :, j] = best
        E[:, :, j] = err
    return J, E


def csm(net: Network, x, h: float = 1e-20) -> np.ndarray:
    """Complex-step method, eq. (8) of the paper:  Im f(x + i h e_j) / h.

    There is no subtraction, so ``h`` can be made so small that the O(h**2)
    truncation error vanishes below round-off (the original code uses 1e-12).
    Only valid if every operation in the forward pass is the analytic
    continuation of its real counterpart -- see ``Network.smooth`` and the
    tests for what happens with ``abs``/``max`` style operations.
    """
    x = _as_batch(x)
    n, xdim = x.shape
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128
    cols = []
    for j in range(xdim):
        xc = x.astype(ctype)
        xc[:, j] += 1j * h
        cols.append(np.imag(net.forward(xc)) / h)
    return np.stack(cols, axis=-1)


def ad_forward(net: Network, x) -> np.ndarray:
    """Forward-mode AD (one JVP per input dimension)."""
    return ad.jacobian_forward(net.forward, _as_batch(x))


def ad_reverse(net: Network, x) -> np.ndarray:
    """Reverse-mode AD (one VJP per output dimension)."""
    return ad.jacobian_reverse(net.forward, _as_batch(x))


METHODS = {
    "NNJ": nnj,
    "CFD": cfd,
    "CSM": csm,
    "AD-fwd": ad_forward,
    "AD-rev": ad_reverse,
}


# --------------------------------------------------------------------------- #
# Second order
# --------------------------------------------------------------------------- #
def hessian_nnj(net: Network, x) -> np.ndarray:
    return net.hessian_nnj(_as_batch(x))


def hessian_ad(net: Network, x) -> np.ndarray:
    return ad.hessian_forward(net.forward, _as_batch(x))


def hessian_cfd(net: Network, x, h: float = 1e-4) -> np.ndarray:
    """Second differences of *function values* (O(h**2), round-off eps/h**2)."""
    x = np.asarray(_as_batch(x), dtype=float)
    n, xdim = x.shape
    f0 = net.forward(x)
    H = np.zeros((n, net.ydim, xdim, xdim))
    for i in range(xdim):
        ei = np.zeros(xdim); ei[i] = h
        H[:, :, i, i] = (net.forward(x + ei) - 2 * f0 + net.forward(x - ei)) / h ** 2
        for j in range(i + 1, xdim):
            ej = np.zeros(xdim); ej[j] = h
            m = (net.forward(x + ei + ej) - net.forward(x + ei - ej)
                 - net.forward(x - ei + ej) + net.forward(x - ei - ej)) / (4 * h * h)
            H[:, :, i, j] = m
            H[:, :, j, i] = m
    return H


def hessian_cfd_of_nnj(net: Network, x, h: float = 1e-6) -> np.ndarray:
    """Central differences of the analytic Jacobian (round-off eps/h)."""
    x = np.asarray(_as_batch(x), dtype=float)
    n, xdim = x.shape
    H = np.zeros((n, net.ydim, xdim, xdim))
    for j in range(xdim):
        e = np.zeros(xdim); e[j] = h
        H[:, :, :, j] = (net.jacobian_nnj(x + e) - net.jacobian_nnj(x - e)) / (2 * h)
    return 0.5 * (H + np.swapaxes(H, 2, 3))


def hessian_csm_of_nnj(net: Network, x, h: float = 1e-20) -> np.ndarray:
    """Complex step through the analytic Jacobian: exact to round-off."""
    x = np.asarray(_as_batch(x), dtype=float)
    n, xdim = x.shape
    H = np.zeros((n, net.ydim, xdim, xdim))
    for j in range(xdim):
        xc = x.astype(complex)
        xc[:, j] += 1j * h
        H[:, :, :, j] = np.imag(net.jacobian_nnj(xc)) / h
    return 0.5 * (H + np.swapaxes(H, 2, 3))


def hessian_csm_values(net: Network, x, h: float = 1e-4) -> np.ndarray:
    """Second derivative from function values with complex steps.

    Diagonal:  f'' ~ 2 (f(x) - Re f(x + i h)) / h**2 .
    Mixed:     d2f/dxi dxj ~ Im( f(x + i h ei + h ej) - f(x + i h ei - h ej) ) / (2 h**2)

    Unlike the first-order complex step these formulas *do* subtract nearly
    equal numbers, so the step cannot be made arbitrarily small.
    """
    x = np.asarray(_as_batch(x), dtype=float)
    n, xdim = x.shape
    f0 = net.forward(x)
    H = np.zeros((n, net.ydim, xdim, xdim))
    for i in range(xdim):
        xc = x.astype(complex)
        xc[:, i] += 1j * h
        H[:, :, i, i] = 2.0 * (f0 - np.real(net.forward(xc))) / h ** 2
        for j in range(i + 1, xdim):
            xp = xc.copy(); xm = xc.copy()
            xp[:, j] += h; xm[:, j] -= h
            m = np.imag(net.forward(xp) - net.forward(xm)) / (2 * h * h)
            H[:, :, i, j] = m
            H[:, :, j, i] = m
    return H


def hvp(net: Network, x, v, k: int = 0) -> np.ndarray:
    """Hessian-vector product  H_k(x) v  (forward-over-reverse AD)."""
    return ad.hvp_forward_over_reverse(net.forward, _as_batch(x), _as_batch(v), k)
