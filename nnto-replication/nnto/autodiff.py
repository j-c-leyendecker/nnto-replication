"""A small operator-overloading automatic-differentiation engine for NumPy.

Two modes are provided, both exact to working precision:

``Dual``  forward mode.  ``Dual(value, tangent)`` propagates one directional
          derivative alongside the value.  Components may themselves be
          ``Dual`` (hyper-dual numbers -> exact second derivatives) or ``Var``
          (forward-over-reverse -> Hessian-vector products).

``Var``   reverse mode.  Operations are recorded on a tape; ``backward``
          propagates one adjoint seed from an output to every input.

The engine exists so that "AD" in this package is an implementation that is
*independent* of the hand-derived chain rule in :mod:`nnto.derivatives`
(the paper's "NNJ"); agreement between the two is therefore a real check.
It is deliberately tiny: only the operations needed by the networks in
:mod:`nnto.networks` are implemented.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import special as _sp

from . import functional as F

_SQRT2 = np.sqrt(2.0)
_TWO_OVER_SQRT_PI = 2.0 / np.sqrt(np.pi)


# =========================================================================== #
# Forward mode
# =========================================================================== #
class Dual:
    """value + tangent * eps,  eps**2 = 0."""

    __slots__ = ("v", "t")
    __array_priority__ = 1000.0
    __array_ufunc__ = None      # make ndarray binary operators defer to us

    def __init__(self, value, tangent):
        self.v = value
        self.t = tangent

    # -- helpers
    @staticmethod
    def _split(other):
        if isinstance(other, Dual):
            return other.v, other.t
        return other, None

    @property
    def shape(self):
        return self.v.shape

    # -- arithmetic
    def __add__(self, o):
        ov, ot = self._split(o)
        return Dual(self.v + ov, self.t if ot is None else self.t + ot)

    __radd__ = __add__

    def __sub__(self, o):
        ov, ot = self._split(o)
        return Dual(self.v - ov, self.t if ot is None else self.t - ot)

    def __rsub__(self, o):
        return Dual(o - self.v, -1.0 * self.t)

    def __neg__(self):
        return Dual(-1.0 * self.v, -1.0 * self.t)

    def __mul__(self, o):
        ov, ot = self._split(o)
        if ot is None:
            return Dual(self.v * ov, self.t * ov)
        return Dual(self.v * ov, self.t * ov + self.v * ot)

    __rmul__ = __mul__

    def __truediv__(self, o):
        ov, ot = self._split(o)
        if ot is None:
            return Dual(self.v / ov, self.t / ov)
        q = self.v / ov
        return Dual(q, (self.t - q * ot) / ov)

    def __rtruediv__(self, o):
        q = o / self.v
        return Dual(q, -1.0 * q * self.t / self.v)

    def __matmul__(self, o):
        ov, ot = self._split(o)
        if ot is None:
            return Dual(self.v @ ov, self.t @ ov)
        return Dual(self.v @ ov, self.t @ ov + self.v @ ot)

    def __rmatmul__(self, o):
        return Dual(o @ self.v, o @ self.t)

    def __pow__(self, p):
        if not isinstance(p, (int, float)):
            raise TypeError("only constant real exponents are supported")
        return Dual(self.v ** p, p * self.v ** (p - 1) * self.t)

    def sum(self, axis=None):
        return Dual(self.v.sum(axis=axis), self.t.sum(axis=axis))

    # -- elementary functions (written with F.* so that nesting works)
    def exp(self):
        e = F.exp(self.v)
        return Dual(e, e * self.t)

    def log(self):
        return Dual(F.log(self.v), self.t / self.v)

    def sin(self):
        return Dual(F.sin(self.v), F.cos(self.v) * self.t)

    def cos(self):
        return Dual(F.cos(self.v), -1.0 * F.sin(self.v) * self.t)

    def tanh(self):
        th = F.tanh(self.v)
        return Dual(th, (1.0 - th * th) * self.t)

    def sigmoid(self):
        s = F.sigmoid(self.v)
        return Dual(s, s * (1.0 - s) * self.t)

    def softplus(self):
        return Dual(F.softplus(self.v), F.sigmoid(self.v) * self.t)

    def erf(self):
        return Dual(F.erf(self.v),
                    _TWO_OVER_SQRT_PI * F.exp(-1.0 * self.v * self.v) * self.t)

    def relu(self):
        return Dual(F.relu(self.v), F.step(self.v) * self.t)

    def step(self):
        s = F.step(self.v)
        return Dual(s, 0.0 * self.t)


# =========================================================================== #
# Reverse mode
# =========================================================================== #
def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``g`` over the axes that NumPy broadcasting introduced."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class Tape:
    def __init__(self):
        self.nodes: List["Var"] = []


class Var:
    """Node of a reverse-mode computational graph (values are ndarrays)."""

    __slots__ = ("v", "tape", "parents", "grad")
    __array_priority__ = 2000.0
    __array_ufunc__ = None      # make ndarray binary operators defer to us

    def __init__(self, value, tape: Optional[Tape] = None, parents=()):
        self.v = np.asarray(value)
        self.tape = tape if tape is not None else Tape()
        self.parents: Sequence[Tuple["Var", Callable]] = parents
        self.grad = None
        self.tape.nodes.append(self)

    @property
    def shape(self):
        return self.v.shape

    def _new(self, value, parents):
        return Var(value, self.tape, parents)

    @staticmethod
    def _val(o):
        return o.v if isinstance(o, Var) else o

    # -- arithmetic
    def __add__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        ov = self._val(o)
        out = self.v + ov
        par = [(self, lambda g, s=self.v.shape: _unbroadcast(g, s))]
        if isinstance(o, Var):
            par.append((o, lambda g, s=o.v.shape: _unbroadcast(g, s)))
        return self._new(out, par)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        ov = self._val(o)
        out = self.v - ov
        par = [(self, lambda g, s=self.v.shape: _unbroadcast(g, s))]
        if isinstance(o, Var):
            par.append((o, lambda g, s=o.v.shape: _unbroadcast(-g, s)))
        return self._new(out, par)

    def __rsub__(self, o):
        return self._new(o - self.v,
                         [(self, lambda g, s=self.v.shape: _unbroadcast(-g, s))])

    def __neg__(self):
        return self._new(-self.v, [(self, lambda g: -g)])

    def __mul__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        ov = self._val(o)
        out = self.v * ov
        par = [(self, lambda g, s=self.v.shape, c=ov: _unbroadcast(g * c, s))]
        if isinstance(o, Var):
            par.append((o, lambda g, s=o.v.shape, c=self.v: _unbroadcast(g * c, s)))
        return self._new(out, par)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        ov = self._val(o)
        out = self.v / ov
        par = [(self, lambda g, s=self.v.shape, c=ov: _unbroadcast(g / c, s))]
        if isinstance(o, Var):
            par.append((o, lambda g, s=o.v.shape, q=out, c=ov:
                        _unbroadcast(-g * q / c, s)))
        return self._new(out, par)

    def __rtruediv__(self, o):
        out = o / self.v
        return self._new(out, [(self, lambda g, s=self.v.shape, q=out, c=self.v:
                                _unbroadcast(-g * q / c, s))])

    def __pow__(self, p):
        if not isinstance(p, (int, float)):
            raise TypeError("only constant real exponents are supported")
        return self._new(self.v ** p,
                         [(self, lambda g, c=p * self.v ** (p - 1): g * c)])

    def __matmul__(self, o):
        ov = self._val(o)
        out = self.v @ ov
        par = [(self, lambda g, c=ov: g @ np.swapaxes(c, -1, -2))]
        if isinstance(o, Var):
            def _back(g, a=self.v, s=o.v.shape):
                r = np.swapaxes(a, -1, -2) @ g
                return _unbroadcast(r, s)
            par.append((o, _back))
        return self._new(out, par)

    def __rmatmul__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        o = np.asarray(o)
        out = o @ self.v

        def _back(g, a=o, s=self.v.shape):
            return _unbroadcast(np.swapaxes(a, -1, -2) @ g, s)
        return self._new(out, [(self, _back)])

    def sum(self, axis=None):
        out = self.v.sum(axis=axis)
        shape = self.v.shape

        def _back(g, axis = axis, shape = shape):
            g = np.asarray(g)
            if axis is not None:
                g = np.expand_dims(g, axis)
            return np.broadcast_to(g, shape).copy()
        return self._new(out, [(self, _back)])

    # -- elementary functions
    def _unary(self, value, dval):
        return self._new(value, [(self, lambda g, d=dval: g * d)])

    def exp(self):
        e = np.exp(self.v)
        return self._unary(e, e)

    def log(self):
        return self._unary(np.log(self.v), 1.0 / self.v)

    def sin(self):
        return self._unary(np.sin(self.v), np.cos(self.v))

    def cos(self):
        return self._unary(np.cos(self.v), -np.sin(self.v))

    def tanh(self):
        th = np.tanh(self.v)
        return self._unary(th, 1.0 - th * th)

    def sigmoid(self):
        s = F.sigmoid(self.v)
        return self._unary(s, s * (1.0 - s))

    def softplus(self):
        return self._unary(F.softplus(self.v), F.sigmoid(self.v))

    def erf(self):
        return self._unary(_sp.erf(self.v),
                           _TWO_OVER_SQRT_PI * np.exp(-self.v * self.v))

    def relu(self):
        return self._unary(F.relu(self.v), F.step(self.v))

    def step(self):
        return self._unary(F.step(self.v), np.zeros_like(self.v))

    # -- reverse sweep
    def backward(self, seed=None) -> None:
        """Accumulate d<seed, self>/d(node) into ``node.grad`` for all nodes."""
        nodes = self.tape.nodes
        for n in nodes:
            n.grad = None
        self.grad = np.ones_like(self.v) if seed is None else np.asarray(seed)
        # the tape is already a topological order (parents precede children)
        for n in reversed(nodes):
            if n.grad is None:
                continue
            for parent, fn in n.parents:
                g = fn(n.grad)
                parent.grad = g if parent.grad is None else parent.grad + g


def _concat_var(a: "Var", b: "Var") -> "Var":
    """Concatenate two tape nodes along the last axis."""
    na = a.v.shape[-1]
    out = np.concatenate([a.v, b.v], axis=-1)
    return Var(out, a.tape, [(a, lambda g, n=na: g[..., :n]),
                             (b, lambda g, n=na: g[..., n:])])


def _take_col(a: "Var", k: int) -> "Var":
    """Column ``k`` of a (N, m) tape node, shape (N,)."""
    shape = a.v.shape

    def _back(g, k=k, shape=shape):
        out = np.zeros(shape, dtype=np.result_type(g, float))
        out[:, k] = g
        return out
    return Var(a.v[:, k], a.tape, [(a, _back)])


def _stack_var(cols) -> "Var":
    """Stack (N,) tape nodes (or arrays) into a (N, m) node."""
    tape = next(c.tape for c in cols if isinstance(c, Var))
    vals = [c.v if isinstance(c, Var) else np.asarray(c) for c in cols]
    out = np.stack(vals, axis=1)
    parents = [(c, (lambda g, k=k: g[:, k])) for k, c in enumerate(cols) if isinstance(c, Var)]
    return Var(out, tape, parents)


# =========================================================================== #
# Drivers for batched functions  f: (N, xdim) -> (N, ydim), rows independent
# =========================================================================== #
def jacobian_forward(f: Callable, x: np.ndarray) -> np.ndarray:
    """Batch Jacobian (N, ydim, xdim) with ``xdim`` forward sweeps (JVPs)."""
    x = np.asarray(x)
    n, xdim = x.shape
    cols = []
    for j in range(xdim):
        t = np.zeros_like(x)
        t[:, j] = 1.0
        cols.append(f(Dual(x, t)).t)
    return np.stack(cols, axis=-1)


def jacobian_reverse(f: Callable, x: np.ndarray) -> np.ndarray:
    """Batch Jacobian (N, ydim, xdim) with ``ydim`` reverse sweeps (VJPs).

    Because the rows of the batch are independent, seeding output column k for
    *all* samples at once returns row k of every sample's Jacobian.  This is
    the O(N) alternative to forming the (N, ydim, N, xdim) cross-batch
    Jacobian and summing it, which is what the original code does.
    """
    x = np.asarray(x)
    xin = Var(x)
    y = f(xin)
    n, ydim = y.v.shape
    rows = []
    for k in range(ydim):
        seed = np.zeros_like(y.v)
        seed[:, k] = 1.0
        y.backward(seed)
        rows.append(xin.grad.copy())
    return np.stack(rows, axis=1)


def jvp(f: Callable, x: np.ndarray, v: np.ndarray):
    """(f(x), J(x) v) with one forward sweep."""
    out = f(Dual(np.asarray(x), np.asarray(v)))
    return out.v, out.t


def vjp(f: Callable, x: np.ndarray, w: np.ndarray):
    """(f(x), w^T J(x)) with one reverse sweep."""
    xin = Var(np.asarray(x))
    y = f(xin)
    y.backward(np.asarray(w))
    return y.v, xin.grad


def hessian_forward(f: Callable, x: np.ndarray) -> np.ndarray:
    """Batch Hessian (N, ydim, xdim, xdim) with hyper-dual numbers.

    Uses xdim*(xdim+1)/2 nested forward sweeps; exact to working precision
    (no step size, no subtractive cancellation).
    """
    x = np.asarray(x)
    n, xdim = x.shape
    H = None
    zero = np.zeros_like(x)
    for i in range(xdim):
        ei = zero.copy()
        ei[:, i] = 1.0
        for j in range(i, xdim):
            ej = zero.copy()
            ej[:, j] = 1.0
            xx = Dual(Dual(x, ej), Dual(ei, zero))
            d2 = f(xx).t.t
            if H is None:
                H = np.zeros((n, d2.shape[1], xdim, xdim), dtype=d2.dtype)
            H[:, :, i, j] = d2
            H[:, :, j, i] = d2
    return H


def hvp_forward_over_reverse(f: Callable, x: np.ndarray, v: np.ndarray,
                             k: int = 0) -> np.ndarray:
    """Hessian-vector product  H_k(x) v  for output component ``k``.

    One forward-over-reverse sweep per sample batch: the tangent of the output
    is itself a tape node, so differentiating it in reverse gives d(Jv)_k/dx.
    """
    x = np.asarray(x)
    tape = Tape()
    xin = Var(x, tape)
    out = f(Dual(xin, np.asarray(v)))
    jv = out.t
    if not isinstance(jv, Var):
        return np.zeros_like(x)
    seed = np.zeros_like(jv.v)
    seed[:, k] = 1.0
    jv.backward(seed)
    return xin.grad if xin.grad is not None else np.zeros_like(x)
