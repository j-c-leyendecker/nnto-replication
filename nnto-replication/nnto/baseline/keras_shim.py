"""NumPy stand-in for the TensorFlow/Keras API surface used by the original
``Run_DBTO_NN.py`` / ``Run_HBTO_NN.py`` scripts.

Only the calls those scripts make are provided.  The semantics that matter:

``Model.predict(x)``
    Keras' data adapter converts floating NumPy inputs to
    ``keras.backend.floatx()`` *before* they reach the first layer
    (``keras/engine/data_adapter.py::_process_tensorlike`` in Keras 2.12).
    The original Run_* scripts never call ``set_floatx('float64')``, so inputs
    are rounded to float32 while the float64 layers then compute in double
    precision.  ``ShimModel.floatx`` controls this ('float32' = as published).

``layer(x)``  (direct ``__call__``)
    No floatx cast: the input is converted with its own dtype and then cast to
    the layer dtype (float64), so these paths are double precision.

``tf.GradientTape().jacobian(y, x)``
    Returns the full cross-batch Jacobian of shape ``y.shape + x.shape``,
    which the scripts then reduce with ``tf.reduce_sum(..., axis=[2])``.
    The shim computes it with :mod:`nnto.autodiff` (reverse mode, one sweep
    per output entry) so that both the values and the O(N^2) memory footprint
    of the original AD path are reproduced.
"""
from __future__ import annotations

import types

import numpy as np

from .. import functional as F
from ..autodiff import Var, Tape
from ..networks import Network, Affine, Dense


class _Tensor:
    """Eager-tensor stand-in exposing ``.numpy()``."""

    def __init__(self, value):
        self.value = value

    def numpy(self):
        v = self.value
        return v.v if isinstance(v, Var) else np.asarray(v)

    @property
    def shape(self):
        v = self.value
        return v.v.shape if isinstance(v, Var) else np.shape(v)


def _unwrap(x):
    if isinstance(x, (_Tensor, _Variable)):
        return x.value
    return np.asarray(x)


class _Variable(_Tensor):
    pass


class _Activation:
    def __init__(self, name):
        self.__name__ = name


class _NormLayer:
    def __init__(self, mean, variance, dtype=np.float64):
        self.mean = _Tensor(np.asarray(mean, dtype=dtype).reshape(1, -1))
        self.variance = _Tensor(np.asarray(variance, dtype=dtype).reshape(1, -1))
        self._std = np.maximum(np.sqrt(self.variance.value), 1e-7)
        self.output_shape = (None, self.mean.value.shape[1])

    def __call__(self, x):
        x = _unwrap(x)
        if not isinstance(x, Var):
            x = np.asarray(x, dtype=np.float64)
        return _Tensor((x - self.mean.value) / self._std)


class _DenseLayer:
    def __init__(self, W, b, activation):
        self.kernel = _Tensor(np.asarray(W))
        self.bias = _Tensor(np.asarray(b))
        self.weights = [self.kernel, self.bias]
        self.activation = _Activation(activation)
        self.output_shape = (None, W.shape[1])

    def __call__(self, x):
        x = _unwrap(x)
        f = F.get_activation(self.activation.__name__)[0]
        return _Tensor(f(x @ self.kernel.value + self.bias.value))


class ShimModel:
    """Sequential( Normalization, Dense..., Dense, Normalization )."""

    def __init__(self, net: Network, floatx: str = "float32"):
        blocks = net.blocks
        if not (isinstance(blocks[0], Affine) and isinstance(blocks[-1], Affine)
                and all(isinstance(b, Dense) for b in blocks[1:-1])):
            raise ValueError("ShimModel needs the original layer layout")
        m = net.meta
        self.layers = ([_NormLayer(m["in_mean"], m["in_std"] ** 2)]
                       + [_DenseLayer(b.W, b.b, b.activation) for b in blocks[1:-1]]
                       + [_NormLayer(m["out_mean"], m["out_std"] ** 2)])
        self.output_shape = (None, net.ydim)
        self.input_shape = (None, net.xdim)
        self.floatx = floatx
        self.n_predict_calls = 0

    def predict(self, x, verbose=0):
        self.n_predict_calls += 1
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.floating):
            x = x.astype(self.floatx)             # <- Keras data-adapter cast
        if x.ndim == 1:
            x = x.reshape(-1, 1)                  # Keras expands rank-1 input
        h = x.astype(np.float64)                  # layers are float64
        for layer in self.layers:
            h = layer(h).value
        return np.asarray(h)


class _GradientTape:
    def __init__(self, persistent=False):
        self.persistent = persistent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def jacobian(self, y, x):
        """Full Jacobian with shape y.shape + x.shape (as TensorFlow returns)."""
        yv, xv = y.value, x.value
        if not isinstance(yv, Var):
            raise RuntimeError("output was not computed from a tf.Variable")
        out = np.zeros(yv.v.shape + xv.v.shape)
        seed = np.zeros_like(yv.v)
        for idx in np.ndindex(*yv.v.shape):
            seed[idx] = 1.0
            yv.backward(seed)
            out[idx] = xv.grad
            seed[idx] = 0.0
        return _Tensor(out)


def _variable(value, dtype=None):
    return _Variable(Var(np.asarray(value, dtype=np.float64), Tape()))


def _reduce_sum(t, axis=None):
    v = _unwrap(t)
    if isinstance(axis, (list, tuple)):
        axis = tuple(axis)
    return _Tensor(np.sum(v, axis=axis))


def make_tf_module():
    """Return an object usable as the scripts' global ``tf``."""
    tf = types.SimpleNamespace()
    tf.float64 = np.float64
    tf.float32 = np.float32
    tf.Variable = _variable
    tf.GradientTape = _GradientTape
    tf.reduce_sum = _reduce_sum
    return tf
