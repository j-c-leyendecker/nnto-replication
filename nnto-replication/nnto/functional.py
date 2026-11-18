"""Type-generic elementary functions.

Every function here accepts

* real or complex ``numpy`` arrays (complex inputs are what the complex-step
  method needs, so nothing in this file may silently drop an imaginary part),
* :class:`nnto.autodiff.Dual` numbers (forward-mode AD, nestable), and
* :class:`nnto.autodiff.Var` nodes (reverse-mode AD),

by dispatching to a method of the same name when the argument is not a plain
array.  Network forward passes written with these functions therefore serve
*all* derivative methods from a single definition.

Numerical notes
---------------
``sigmoid`` and ``softplus`` are evaluated in overflow-free form.  The original
repository uses ``exp(z) / (1 + exp(z))**2`` for the sigmoid derivative, which
returns ``nan`` for z > ~709.8 (``inf/inf``) -- see tests/test_activations.py.
"""
from __future__ import annotations

import numpy as np
from scipy import special as _sp

_SQRT2 = np.sqrt(2.0)
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def _is_array(x) -> bool:
    return isinstance(x, (np.ndarray, np.generic, float, int, complex))


# ----------------------------------------------------------------- primitives
def exp(x):
    return np.exp(x) if _is_array(x) else x.exp()


def log(x):
    return np.log(x) if _is_array(x) else x.log()


def sin(x):
    return np.sin(x) if _is_array(x) else x.sin()


def cos(x):
    return np.cos(x) if _is_array(x) else x.cos()


def tanh(x):
    return np.tanh(x) if _is_array(x) else x.tanh()


def erf(x):
    return _sp.erf(x) if _is_array(x) else x.erf()


def sigmoid(x):
    """Logistic function 1/(1+exp(-x)), overflow-free, complex-safe."""
    if not _is_array(x):
        return x.sigmoid()
    x = np.asarray(x)
    if np.iscomplexobj(x):
        out = np.empty_like(x)
        pos = x.real >= 0
        e = np.exp(-x[pos])
        out[pos] = 1.0 / (1.0 + e)
        e = np.exp(x[~pos])
        out[~pos] = e / (1.0 + e)
        return out
    if x.dtype == np.float32:
        return _sp.expit(x).astype(np.float32)
    return _sp.expit(x)


def softplus(x):
    """log(1+exp(x)), overflow-free, complex-safe."""
    if not _is_array(x):
        return x.softplus()
    x = np.asarray(x)
    if np.iscomplexobj(x):
        out = np.empty_like(x)
        pos = x.real >= 0
        out[pos] = x[pos] + np.log1p(np.exp(-x[pos]))
        out[~pos] = np.log1p(np.exp(x[~pos]))
        return out
    return np.logaddexp(0.0, x).astype(x.dtype, copy=False)


def relu(x):
    """max(x, 0).  For complex input the branch is chosen on the real part and
    the imaginary part is carried through, which is what makes the complex
    step return the correct one-sided derivative away from the kink."""
    if not _is_array(x):
        return x.relu()
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.where(x.real > 0, x, 0.0 * x)
    return np.maximum(x, 0)


def step(x):
    """Heaviside step of the real part (derivative of relu, 0 at the kink)."""
    if not _is_array(x):
        return x.step()
    x = np.asarray(x)
    return (x.real > 0).astype(x.real.dtype)


# ------------------------------------------------- activations (f, f', f'')
def _linear(z):
    return z


def _d_linear(z):
    return 1.0 + 0.0 * z


def _dd_linear(z):
    return 0.0 * z


def _d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def _dd_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def _d_tanh(z):
    t = tanh(z)
    return 1.0 - t * t


def _dd_tanh(z):
    t = tanh(z)
    return -2.0 * t * (1.0 - t * t)


def _d_relu(z):
    return step(z)


def _dd_relu(z):
    return 0.0 * z


def _dd_softplus(z):
    return _d_sigmoid(z)


def _swish(z):
    return z * sigmoid(z)


def _d_swish(z):
    s = sigmoid(z)
    return s + z * s * (1.0 - s)


def _dd_swish(z):
    s = sigmoid(z)
    ds = s * (1.0 - s)
    return 2.0 * ds + z * ds * (1.0 - 2.0 * s)


def _gelu(z):
    return 0.5 * z * (1.0 + erf(z / _SQRT2))


def _gauss_pdf(z):
    return exp(-0.5 * z * z) * (_SQRT_2_OVER_PI / 2.0)


def _d_gelu(z):
    return 0.5 * (1.0 + erf(z / _SQRT2)) + z * _gauss_pdf(z)


def _dd_gelu(z):
    return (2.0 - z * z) * _gauss_pdf(z)


def _d_sin(z):
    return cos(z)


def _dd_sin(z):
    return -1.0 * sin(z)


ACTIVATIONS = {
    #  name      : (f, f', f'', smooth?)
    "linear": (_linear, _d_linear, _dd_linear, True),
    "sigmoid": (sigmoid, _d_sigmoid, _dd_sigmoid, True),
    "tanh": (tanh, _d_tanh, _dd_tanh, True),
    "relu": (relu, _d_relu, _dd_relu, False),
    "softplus": (softplus, sigmoid, _dd_softplus, True),
    "swish": (_swish, _d_swish, _dd_swish, True),
    "gelu": (_gelu, _d_gelu, _dd_gelu, True),
    "sin": (sin, _d_sin, _dd_sin, True),
}


def get_activation(name: str):
    try:
        return ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown activation {name!r}; available: {sorted(ACTIVATIONS)}") from exc
