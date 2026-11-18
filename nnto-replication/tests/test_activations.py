"""Activation functions: derivatives, complex safety, overflow behaviour."""
import numpy as np
import pytest

from nnto import functional as F


@pytest.mark.parametrize("name", sorted(F.ACTIVATIONS))
def test_first_and_second_derivatives(name):
    f, df, ddf, smooth = F.get_activation(name)
    z = np.linspace(-4.0, 4.0, 161) + 0.0123          # avoid the ReLU kink
    h = 1e-20
    d_cs = np.imag(f(z + 1j * h)) / h                 # complex step: exact
    assert np.allclose(df(z), d_cs, rtol=1e-12, atol=1e-13)
    dd_cs = np.imag(df(z + 1j * h)) / h
    assert np.allclose(ddf(z), dd_cs, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("name", ["sigmoid", "softplus", "tanh", "swish"])
def test_no_overflow_at_extreme_arguments(name):
    f, df, ddf, _ = F.get_activation(name)
    z = np.array([-800.0, -710.0, 710.0, 800.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for g in (f, df, ddf):
            assert np.all(np.isfinite(g(z)))
            assert np.all(np.isfinite(g(z.astype(complex) + 1e-20j)))


def test_original_sigmoid_derivative_formula_overflows():
    """Audit finding A9: ``exp(z) / (1 + exp(z))**2`` (original dact_fn) is
    nan for z > ~709.8, while the true value is 0."""
    z = np.array([800.0])
    with np.errstate(over="ignore", invalid="ignore"):
        original = np.exp(z) / ((1 + np.exp(z)) ** 2)
    assert np.isnan(original[0])
    assert F.get_activation("sigmoid")[1](z)[0] == 0.0


def test_original_relu_derivative_formula_is_nan_at_zero():
    """Audit finding A10: ``max(z, 0) / z`` (original dact_fn) is 0/0 at z = 0."""
    z = np.array([0.0, -0.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        original = np.maximum(z, 0) / z
    assert np.all(np.isnan(original))
    assert np.all(F.get_activation("relu")[1](z) == 0.0)


def test_relu_keeps_imaginary_part():
    z = np.array([0.3 + 1e-20j, -0.3 + 1e-20j])
    out = F.relu(z)
    assert out[0] == z[0] and out[1] == 0


def test_complex_step_fails_for_modulus_based_activation():
    """np.abs returns the complex modulus, which is not the analytic
    continuation of |x|: the complex step then yields a wrong derivative
    without any warning.  (softsign(x) = x / (1 + |x|), derivative 1/(1+|x|)^2)"""
    x = np.array([-0.7, 0.4])
    h = 1e-20
    naive = lambda z: z / (1 + np.abs(z))
    safe = lambda z: z / (1 + np.where(z.real >= 0, z, -z))
    exact = 1 / (1 + np.abs(x)) ** 2
    d_naive = np.imag(naive(x + 1j * h)) / h
    d_safe = np.imag(safe(x + 1j * h)) / h
    assert np.allclose(d_safe, exact, rtol=1e-14)
    assert not np.allclose(d_naive, exact, rtol=1e-2)       # off by 1/(1+|x|) vs 1/(1+|x|)^2
    assert np.allclose(d_naive, 1 / (1 + np.abs(x)), rtol=1e-14)
