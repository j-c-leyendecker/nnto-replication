"""Feed-forward surrogate networks with method-agnostic forward passes.

A :class:`Network` is a sequence of blocks.  ``Network.forward`` is written
only with operators and :mod:`nnto.functional`, so the *same* code evaluates

* real float32/float64 arrays             (prediction, finite differences),
* complex arrays                          (complex-step method),
* :class:`~nnto.autodiff.Dual` numbers    (forward-mode AD, hyper-dual),
* :class:`~nnto.autodiff.Var` nodes       (reverse-mode AD).

In addition every block implements the hand-derived chain rule used by the
paper's "neural network Jacobian" (NNJ), here vectorised over the batch and
extended to second order (``push_hess``), and to blocks other than dense
layers (residual blocks, Fourier features).

Conventions
-----------
Keras convention for dense layers: ``y = act(x @ W + b)`` with ``W`` of shape
``(n_in, n_out)``.  Jacobians are returned in numerator layout
``J[n, k, j] = d y_k / d x_j`` (shape ``(N, ydim, xdim)``); Hessians as
``H[n, k, i, j] = d^2 y_k / d x_i d x_j``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import functional as F


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
@dataclass
class Affine:
    """Element-wise affine map ``y = x * gain + offset`` (no trainable use).

    Input standardisation ``(x - mean) / std`` is ``gain = 1/std``,
    ``offset = -mean/std``.
    """
    gain: np.ndarray
    offset: np.ndarray

    def forward(self, x):
        return x * self.gain + self.offset

    def push_jac(self, x, J):
        return x * self.gain + self.offset, J * self.gain[None, :, None]

    def push_hess(self, x, J, H):
        return (x * self.gain + self.offset, J * self.gain[None, :, None],
                H * self.gain[None, :, None, None])

    def params(self):
        return []

    def astype(self, dtype):
        return Affine(self.gain.astype(dtype), self.offset.astype(dtype))


@dataclass
class Dense:
    """``y = act(x @ W + b)``."""
    W: np.ndarray
    b: np.ndarray
    activation: str = "linear"

    def forward(self, x):
        f = F.get_activation(self.activation)[0]
        return f(x @ self.W + self.b)

    def push_jac(self, x, J):
        f, df, _, _ = F.get_activation(self.activation)
        z = x @ self.W + self.b
        # d z_o / d x0_j = sum_i W[i, o] J[n, i, j]
        Jz = np.einsum("nij,io->noj", J, self.W)
        return f(z), df(z)[:, :, None] * Jz

    def push_hess(self, x, J, H):
        f, df, ddf, _ = F.get_activation(self.activation)
        z = x @ self.W + self.b
        Jz = np.einsum("nij,io->noj", J, self.W)
        Hz = np.einsum("nijk,io->nojk", H, self.W)
        d1 = df(z)
        d2 = ddf(z)
        Ha = (d1[:, :, None, None] * Hz
              + d2[:, :, None, None] * Jz[:, :, :, None] * Jz[:, :, None, :])
        return f(z), d1[:, :, None] * Jz, Ha

    def params(self):
        return [self.W, self.b]

    def astype(self, dtype):
        return Dense(self.W.astype(dtype), self.b.astype(dtype), self.activation)


@dataclass
class Residual:
    """``y = x + g(x)`` with ``g`` a stack of blocks (same in/out width)."""
    body: List[object]

    def forward(self, x):
        h = x
        for blk in self.body:
            h = blk.forward(h)
        return x + h

    def push_jac(self, x, J):
        h, Jh = x, J
        for blk in self.body:
            h, Jh = blk.push_jac(h, Jh)
        return x + h, J + Jh

    def push_hess(self, x, J, H):
        h, Jh, Hh = x, J, H
        for blk in self.body:
            h, Jh, Hh = blk.push_hess(h, Jh, Hh)
        return x + h, J + Jh, H + Hh

    def params(self):
        out = []
        for blk in self.body:
            out += blk.params()
        return out

    def astype(self, dtype):
        return Residual([b.astype(dtype) for b in self.body])


@dataclass
class FourierFeatures:
    """``y = [sin(x @ B), cos(x @ B)]`` with a fixed projection ``B``."""
    B: np.ndarray

    def forward(self, x):
        p = x @ self.B
        s, c = F.sin(p), F.cos(p)
        return _concat_last(s, c)

    def push_jac(self, x, J):
        p = x @ self.B
        s, c = np.sin(p), np.cos(p)
        Jp = np.einsum("nij,io->noj", J, self.B)
        return (np.concatenate([s, c], axis=1),
                np.concatenate([c[:, :, None] * Jp, -s[:, :, None] * Jp], axis=1))

    def push_hess(self, x, J, H):
        p = x @ self.B
        s, c = np.sin(p), np.cos(p)
        Jp = np.einsum("nij,io->noj", J, self.B)
        Hp = np.einsum("nijk,io->nojk", H, self.B)
        JJ = Jp[:, :, :, None] * Jp[:, :, None, :]
        Hs = c[:, :, None, None] * Hp - s[:, :, None, None] * JJ
        Hc = -s[:, :, None, None] * Hp - c[:, :, None, None] * JJ
        return (np.concatenate([s, c], axis=1),
                np.concatenate([c[:, :, None] * Jp, -s[:, :, None] * Jp], axis=1),
                np.concatenate([Hs, Hc], axis=1))

    def params(self):
        return []

    def astype(self, dtype):
        return FourierFeatures(self.B.astype(dtype))


@dataclass
class SoftFloor:
    """``y = lo + softplus(x)`` element-wise: outputs bounded below by ``lo``.

    Used as the last block of a density surrogate so that the predicted
    modulus can never be negative -- no clipping (and hence no
    objective/gradient mismatch) is needed in the optimisation."""
    lo: np.ndarray

    def forward(self, x):
        return F.softplus(x) + self.lo

    def push_jac(self, x, J):
        return F.softplus(x) + self.lo, F.sigmoid(x)[:, :, None] * J

    def push_hess(self, x, J, H):
        s = F.sigmoid(x)
        ds = s * (1.0 - s)
        return (F.softplus(x) + self.lo, s[:, :, None] * J,
                s[:, :, None, None] * H + ds[:, :, None, None] * J[:, :, :, None] * J[:, :, None, :])

    def params(self):
        return []

    def astype(self, dtype):
        return SoftFloor(self.lo.astype(dtype))


@dataclass
class CholeskyHead:
    """Map 4 raw outputs to an admissible orthotropic plane tensor.

    With  l11 = softplus(a), l21 = b, l22 = softplus(c):

        C11 = l11^2,  C12 = l11 l21,  C22 = l21^2 + l22^2,  C66 = softplus(d)

    so [[C11, C12], [C12, C22]] = L L^T is positive definite and C66 > 0 for
    every input, by construction.  ``scale`` (4,) rescales the outputs."""
    scale: np.ndarray

    def _parts(self, x):
        a, b, c, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return F.softplus(a), b, F.softplus(c), F.softplus(d)

    def forward(self, x):
        from .autodiff import Dual, Var
        if isinstance(x, (Dual, Var)):
            cols = [_col(x, k) for k in range(4)]
            l11, l21, l22, c66 = F.softplus(cols[0]), cols[1], F.softplus(cols[2]), F.softplus(cols[3])
            outs = [l11 * l11 * self.scale[0], l11 * l21 * self.scale[1],
                    (l21 * l21 + l22 * l22) * self.scale[2], c66 * self.scale[3]]
            return _stack_cols(outs)
        l11, l21, l22, c66 = self._parts(x)
        return np.stack([l11 * l11, l11 * l21, l21 * l21 + l22 * l22, c66], axis=1) * self.scale

    def _local(self, x):
        """Value, Jacobian d out/d raw (N,4,4) and Hessian (N,4,4,4)."""
        a, b, c, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        l11, l22, c66 = F.softplus(a), F.softplus(c), F.softplus(d)
        s1, s2, s3 = F.sigmoid(a), F.sigmoid(c), F.sigmoid(d)
        t1, t2, t3 = s1 * (1 - s1), s2 * (1 - s2), s3 * (1 - s3)
        n = x.shape[0]
        y = np.stack([l11 * l11, l11 * b, b * b + l22 * l22, c66], axis=1)
        G = np.zeros((n, 4, 4), dtype=y.dtype)
        G[:, 0, 0] = 2 * l11 * s1
        G[:, 1, 0] = s1 * b
        G[:, 1, 1] = l11
        G[:, 2, 1] = 2 * b
        G[:, 2, 2] = 2 * l22 * s2
        G[:, 3, 3] = s3
        Hh = np.zeros((n, 4, 4, 4), dtype=y.dtype)
        Hh[:, 0, 0, 0] = 2 * (s1 * s1 + l11 * t1)
        Hh[:, 1, 0, 0] = t1 * b
        Hh[:, 1, 0, 1] = Hh[:, 1, 1, 0] = s1
        Hh[:, 2, 1, 1] = 2.0
        Hh[:, 2, 2, 2] = 2 * (s2 * s2 + l22 * t2)
        Hh[:, 3, 3, 3] = t3
        sc = self.scale
        return y * sc, G * sc[None, :, None], Hh * sc[None, :, None, None]

    def push_jac(self, x, J):
        y, G, _ = self._local(x)
        return y, np.einsum("nki,nij->nkj", G, J)

    def push_hess(self, x, J, H):
        y, G, Hh = self._local(x)
        Jy = np.einsum("nki,nij->nkj", G, J)
        Hy = (np.einsum("nki,nijl->nkjl", G, H)
              + np.einsum("nkim,nij,nml->nkjl", Hh, J, J))
        return y, Jy, Hy

    def params(self):
        return []

    def astype(self, dtype):
        return CholeskyHead(self.scale.astype(dtype))


def _col(x, k):
    """Column k of a (N, m) Dual/Var/array as shape (N,)."""
    from .autodiff import Dual, Var, _take_col
    if isinstance(x, Dual):
        return Dual(_col(x.v, k), _col(x.t, k))
    if isinstance(x, Var):
        return _take_col(x, k)
    return x[:, k]


def _stack_cols(cols):
    from .autodiff import Dual, Var, _stack_var
    c0 = cols[0]
    if isinstance(c0, Dual):
        return Dual(_stack_cols([c.v for c in cols]), _stack_cols([c.t for c in cols]))
    if isinstance(c0, Var):
        return _stack_var(cols)
    return np.stack(cols, axis=1)


@dataclass
class ChebyKAN:
    """Kolmogorov-Arnold layer with Chebyshev-polynomial edge functions.

    Every input-output pair (i, o) carries its own learnable 1-D function

        phi_io(x_i) = sum_k C[k][i, o] T_k(tanh(x_i)),   k = 0..K,

    and the layer output is  y_o = sum_i phi_io(x_i)  (the "ChebyKAN" variant
    of Kolmogorov-Arnold networks, which replaces B-splines by Chebyshev
    polynomials of a tanh-squashed input).  ``C`` is a list of K+1 matrices
    of shape (n_in, n_out).  The map is smooth (analytic), so the analytic
    chain rule, complex step and AD all apply; mixed second derivatives of one
    layer vanish because the layer is a sum of univariate functions.
    """
    C: List[np.ndarray]

    @property
    def degree(self):
        return len(self.C) - 1

    def forward(self, x):
        t = F.tanh(x)
        n_in = self.C[0].shape[0]
        ones = np.ones((x.shape[0], n_in))
        y = ones @ self.C[0]
        if self.degree >= 1:
            Tkm1, Tk = ones, t
            y = y + Tk @ self.C[1]
            for k in range(2, self.degree + 1):
                Tkp1 = 2.0 * t * Tk - Tkm1
                y = y + Tkp1 @ self.C[k]
                Tkm1, Tk = Tk, Tkp1
        return y

    def _basis(self, x):
        """T_k(t), dT_k/dt and d2T_k/dt2 for t = tanh(x); each (K+1, N, n_in)."""
        t = np.tanh(x)
        K = self.degree
        T = np.empty((K + 1,) + x.shape, dtype=np.result_type(x, float))
        dT = np.zeros_like(T); d2T = np.zeros_like(T)
        T[0] = 1.0
        if K >= 1:
            T[1] = t; dT[1] = 1.0
        for k in range(1, K):
            T[k + 1] = 2 * t * T[k] - T[k - 1]
            dT[k + 1] = 2 * T[k] + 2 * t * dT[k] - dT[k - 1]
            d2T[k + 1] = 4 * dT[k] + 2 * t * d2T[k] - d2T[k - 1]
        return t, T, dT, d2T

    def _local(self, x, second=False):
        t, T, dT, d2T = self._basis(x)
        Cs = np.stack(self.C)                                    # (K+1, n_in, n_out)
        y = np.einsum("kni,kio->no", T, Cs)
        tp = 1.0 - t * t                                         # dt/dx
        G = np.einsum("kni,kio->nio", dT, Cs) * tp[:, :, None]   # d phi_io / d x_i
        if not second:
            return y, G
        tpp = -2.0 * t * tp                                      # d2t/dx2
        G2 = (np.einsum("kni,kio->nio", d2T, Cs) * (tp * tp)[:, :, None]
              + np.einsum("kni,kio->nio", dT, Cs) * tpp[:, :, None])
        return y, G, G2

    def push_jac(self, x, J):
        y, G = self._local(x)
        return y, np.einsum("nio,nij->noj", G, J)

    def push_hess(self, x, J, H):
        y, G, G2 = self._local(x, second=True)
        Jy = np.einsum("nio,nij->noj", G, J)
        Hy = (np.einsum("nio,nijl->nojl", G, H)
              + np.einsum("nio,nij,nil->nojl", G2, J, J))
        return y, Jy, Hy

    def params(self):
        return list(self.C)

    def astype(self, dtype):
        return ChebyKAN([c.astype(dtype) for c in self.C])


def _concat_last(a, b):
    """Concatenate along the feature axis for arrays, Duals and Vars."""
    from .autodiff import Dual, Var, _concat_var
    if isinstance(a, Dual):
        return Dual(_concat_last(a.v, b.v), _concat_last(a.t, b.t))
    if isinstance(a, Var):
        return _concat_var(a, b)
    return np.concatenate([a, b], axis=-1)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
@dataclass
class Network:
    blocks: List[object]
    xdim: int
    ydim: int
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------- evaluation
    def forward(self, x):
        h = x
        for blk in self.blocks:
            h = blk.forward(h)
        return h

    __call__ = forward

    def predict(self, x):
        return self.forward(np.atleast_2d(np.asarray(x)))

    # ------------------------------------------------ analytic chain rule
    def jacobian_nnj(self, x, return_value: bool = False):
        """Analytic Jacobian by forward accumulation of the chain rule."""
        x = np.atleast_2d(np.asarray(x))
        n = x.shape[0]
        J = np.broadcast_to(np.eye(self.xdim, dtype=x.dtype),
                            (n, self.xdim, self.xdim))
        h = x
        for blk in self.blocks:
            h, J = blk.push_jac(h, J)
        return (h, J) if return_value else J

    def hessian_nnj(self, x, return_all: bool = False):
        """Analytic Hessian by forward accumulation (second-order chain rule)."""
        x = np.atleast_2d(np.asarray(x))
        n = x.shape[0]
        J = np.broadcast_to(np.eye(self.xdim, dtype=x.dtype),
                            (n, self.xdim, self.xdim))
        H = np.zeros((n, self.xdim, self.xdim, self.xdim), dtype=x.dtype)
        h = x
        for blk in self.blocks:
            h, J, H = blk.push_hess(h, J, H)
        return (h, J, H) if return_all else H

    # ------------------------------------------------------------ utilities
    def params(self) -> List[np.ndarray]:
        out = []
        for blk in self.blocks:
            out += blk.params()
        return out

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params()))

    def astype(self, dtype) -> "Network":
        return Network([b.astype(dtype) for b in self.blocks], self.xdim,
                       self.ydim, dict(self.meta))

    @property
    def smooth(self) -> bool:
        def _ok(blk):
            if isinstance(blk, Dense):
                return F.get_activation(blk.activation)[3]
            if isinstance(blk, Residual):
                return all(_ok(b) for b in blk.body)
            return True
        return all(_ok(b) for b in self.blocks)


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def mlp(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
        activations: Sequence[str], in_mean=None, in_std=None,
        out_gain=None, out_offset=None, meta: Optional[dict] = None) -> Network:
    """Plain multilayer perceptron with optional input/output affine maps."""
    xdim = weights[0].shape[0]
    ydim = weights[-1].shape[1]
    blocks: List[object] = []
    if in_mean is not None:
        in_mean = np.asarray(in_mean, dtype=float).reshape(-1)
        in_std = np.asarray(in_std, dtype=float).reshape(-1)
        blocks.append(Affine(1.0 / in_std, -in_mean / in_std))
    for W, b, a in zip(weights, biases, activations):
        blocks.append(Dense(np.asarray(W), np.asarray(b).reshape(-1), a))
    if out_gain is not None:
        blocks.append(Affine(np.asarray(out_gain, dtype=float).reshape(-1),
                             np.asarray(out_offset, dtype=float).reshape(-1)))
    return Network(blocks, xdim, ydim, meta or {})


def load_keras_mlp(saved_model_dir: str, hidden_activation: str = "sigmoid",
                   output_activation: str = "linear") -> Network:
    """Load one of the original repository's SavedModel MLPs without TensorFlow.

    Layer layout written by Train_DBTO_NN.py / Train_HBTO_NN.py:

        Normalization -> Dense(sigmoid) x NL -> Dense(linear) -> Normalization

    NOTE (audit finding A7): the last layer is created with ``invert=False``,
    i.e. it applies ``(z - mean_y) / std_y`` to the network output although the
    source comment and the paper call it a *de*-normalisation layer.  We
    reproduce exactly what the saved model computes.
    """
    import os
    from .tf_checkpoint import load_variables

    v = load_variables(os.path.join(saved_model_dir, "variables"))
    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    layers = {}
    for name, arr in v.items():
        if not name.startswith("layer_with_weights-"):
            continue
        idx, attr = name[len("layer_with_weights-"):-len(suffix)].split("/", 1)
        layers.setdefault(int(idx), {})[attr] = arr
    order = sorted(layers)
    first, last = layers[order[0]], layers[order[-1]]
    if "mean" not in first or "mean" not in last:
        raise ValueError("unexpected layer layout (no Normalization layers)")
    dense = [layers[i] for i in order[1:-1]]
    eps = 1e-7                                    # keras.backend.epsilon()
    in_std = np.maximum(np.sqrt(first["variance"]), eps)
    out_std = np.maximum(np.sqrt(last["variance"]), eps)
    acts = [hidden_activation] * (len(dense) - 1) + [output_activation]
    net = mlp([d["kernel"] for d in dense], [d["bias"] for d in dense], acts,
              in_mean=first["mean"], in_std=in_std,
              out_gain=1.0 / out_std, out_offset=-last["mean"] / out_std,
              meta={"source": saved_model_dir,
                    "in_mean": first["mean"].copy(), "in_std": in_std.copy(),
                    "out_mean": last["mean"].copy(), "out_std": out_std.copy(),
                    "norm_count": int(first["count"]),
                    "adam_iterations": int(v["optimizer/_iterations" + suffix])})
    return net


# --------------------------------------------------------------------------- #
# Serialisation (plain .npz, no pickling)
# --------------------------------------------------------------------------- #
def _flatten_blocks(blocks, prefix, store, spec):
    for i, blk in enumerate(blocks):
        key = f"{prefix}{i}"
        if isinstance(blk, Affine):
            spec.append((key, "Affine", ""))
            store[key + "_gain"], store[key + "_offset"] = blk.gain, blk.offset
        elif isinstance(blk, Dense):
            spec.append((key, "Dense", blk.activation))
            store[key + "_W"], store[key + "_b"] = blk.W, blk.b
        elif isinstance(blk, FourierFeatures):
            spec.append((key, "FourierFeatures", ""))
            store[key + "_B"] = blk.B
        elif isinstance(blk, SoftFloor):
            spec.append((key, "SoftFloor", ""))
            store[key + "_lo"] = blk.lo
        elif isinstance(blk, CholeskyHead):
            spec.append((key, "CholeskyHead", ""))
            store[key + "_scale"] = blk.scale
        elif isinstance(blk, ChebyKAN):
            spec.append((key, "ChebyKAN", str(len(blk.C))))
            for k, c in enumerate(blk.C):
                store[f"{key}_C{k}"] = c
        elif isinstance(blk, Residual):
            spec.append((key, "Residual", str(len(blk.body))))
            _flatten_blocks(blk.body, key + "r", store, spec)
        else:
            raise TypeError(type(blk))


def save_network(path: str, net: Network) -> None:
    store, spec = {}, []
    _flatten_blocks(net.blocks, "b", store, spec)
    store["__spec__"] = np.array(["|".join(s) for s in spec])
    store["__dims__"] = np.array([net.xdim, net.ydim])
    np.savez(path, **store)


def load_network(path: str) -> Network:
    z = np.load(path, allow_pickle=False)
    spec = [s.split("|") for s in z["__spec__"]]
    pos = 0

    def build(n, prefix):
        nonlocal pos
        out = []
        for _ in range(n):
            key, kind, arg = spec[pos]
            pos += 1
            if kind == "Affine":
                out.append(Affine(z[key + "_gain"], z[key + "_offset"]))
            elif kind == "Dense":
                out.append(Dense(z[key + "_W"], z[key + "_b"], arg))
            elif kind == "FourierFeatures":
                out.append(FourierFeatures(z[key + "_B"]))
            elif kind == "SoftFloor":
                out.append(SoftFloor(z[key + "_lo"]))
            elif kind == "CholeskyHead":
                out.append(CholeskyHead(z[key + "_scale"]))
            elif kind == "ChebyKAN":
                out.append(ChebyKAN([z[f"{key}_C{k}"] for k in range(int(arg))]))
            elif kind == "Residual":
                out.append(Residual(build(int(arg), key + "r")))
        return out

    top = sum(1 for k, _, _ in spec if "r" not in k[1:])
    blocks = build(top, "b")
    xdim, ydim = (int(a) for a in z["__dims__"])
    return Network(blocks, xdim, ydim, {"source": path})
