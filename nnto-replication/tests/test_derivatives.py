"""Finite differences, complex step and the single-precision artefact."""
import numpy as np
import pytest

from nnto import derivatives as D
from nnto.networks import Network, Dense
from conftest import random_networks


def test_cfd_error_is_u_shaped_in_step_size(rng):
    """Truncation (h^2) dominates for large h, round-off (eps/h) for small h."""
    net = random_networks(rng)["mlp-sigmoid-1x64"]
    x = rng.uniform(0, 1, size=(200, 2))
    J = D.nnj(net, x)
    err = {h: np.abs(D.cfd(net, x, h) - J).max() for h in (1e-1, 1e-3, 1e-5, 1e-9, 1e-12)}
    assert err[1e-1] > err[1e-3] > err[1e-5]                 # truncation branch
    assert err[1e-12] > err[1e-9] > err[1e-5]                # round-off branch
    assert err[1e-5] < 1e-8


def test_cfd_truncation_is_second_order(rng):
    net = random_networks(rng)["mlp-tanh-6x32"]
    x = rng.uniform(-1, 1, size=(50, 3))
    J = D.nnj(net, x)
    e1 = np.abs(D.cfd(net, x, 4e-2) - J).max()
    e2 = np.abs(D.cfd(net, x, 2e-2) - J).max()
    assert 3.3 < e1 / e2 < 4.7                                # ratio ~ 2^2


def test_optimal_step_rule_beats_fixed_step(rng):
    net = random_networks(rng)["mlp-sigmoid-1x64"]
    x = rng.uniform(0, 1, size=(200, 2))
    J = D.nnj(net, x)
    e_opt = np.abs(D.cfd(net, x, D.optimal_cfd_step(x)) - J).max()
    e_1e6 = np.abs(D.cfd(net, x, 1e-6) - J).max()
    assert e_opt < e_1e6


def test_float32_input_quantisation_reproduces_published_cfd_error():
    """Audit finding A1.  Keras' ``Model.predict`` rounds NumPy inputs to
    float32 (the Run_* scripts never call set_floatx('float64')).  At
    theta = 0.5 and theta = sqrt(0.5) the perturbed inputs theta +- 1e-6 are
    rounded so that the step actually taken is 1.01327896e-6: the published
    CFD sensitivities are 1.33 % too large, uniformly."""
    for x0 in (0.5, np.sqrt(0.5)):
        xp, xm = np.float32(x0 + 1e-6), np.float32(x0 - 1e-6)
        ratio = (float(xp) - float(xm)) / 2e-6
        assert abs(ratio - 1.01327896) < 1e-8
    # same thing through the derivative routine, on a linear "network"
    net = Network([Dense(np.array([[2.0]]), np.array([0.3]), "linear")], 1, 1)
    x = np.array([[0.5]])
    assert np.isclose(D.cfd(net, x, 1e-6, np.float32, exact_step=False)[0, 0, 0], 2.0 * 1.01327896, rtol=1e-8)
    # dividing by the step actually taken removes the error
    assert np.isclose(D.cfd(net, x, 1e-6, np.float32, exact_step=True)[0, 0, 0], 2.0, rtol=1e-12)
    # and in double precision there is nothing to fix
    assert np.isclose(D.cfd(net, x, 1e-6)[0, 0, 0], 2.0, rtol=1e-9)


def test_float32_cfd_error_values(rng):
    """On [0.5, 1) the float32 spacing is 2**-24, so the nominal step 2e-6 is
    33.55 ulps and the step actually taken is 33 or 34 ulps: the published
    CFD derivative is either 1.65 % too small or 1.33 % too large, depending
    on theta.  The error bound ulp32(theta) / (2h) shrinks for smaller theta."""
    net = Network([Dense(np.array([[1.0]]), np.array([0.0]), "linear")], 1, 1)
    x = rng.uniform(0.5, 1.0, size=(20000, 1))
    d = D.cfd(net, x, 1e-6, np.float32, exact_step=False)[:, 0, 0]
    vals = np.unique(np.round(d, 8))
    assert np.allclose(vals, [33 * 2 ** -24 / 2e-6, 34 * 2 ** -24 / 2e-6], atol=1e-8)
    assert np.allclose(vals, [0.98347664, 1.01327896], atol=1e-8)
    x = rng.uniform(0.01, 0.02, size=(20000, 1))
    d = D.cfd(net, x, 1e-6, np.float32, exact_step=False)[:, 0, 0]
    assert np.abs(d - 1).max() < 2 ** -29 / 2e-6 + 1e-9        # < 0.1 %


def test_csm_is_step_size_independent(rng):
    net = random_networks(rng)["fourier-gelu-softplus"]
    x = rng.uniform(-1, 1, size=(30, 3))
    J = D.nnj(net, x)
    for h in (1e-8, 1e-12, 1e-20, 1e-100):
        assert np.allclose(D.csm(net, x, h), J, rtol=1e-12, atol=1e-13)


def test_csm_in_single_precision_needs_small_step(rng):
    """With complex64 the usual h = 1e-12 still works (no subtraction),
    and the result carries float32 round-off (~1e-6 relative)."""
    net = random_networks(rng)["mlp-sigmoid-1x64"]
    x = rng.uniform(0, 1, size=(100, 2))
    J = D.nnj(net, x)
    J32 = D.csm(net.astype(np.float32), x.astype(np.float32), 1e-12)
    rel = np.abs(J32 - J).max() / np.abs(J).max()
    assert 1e-9 < rel < 1e-4


def test_relu_network_methods(rng):
    """Away from kinks all methods agree.  Next to a kink the analytic chain
    rule, AD and the complex step return the one-sided slope, while finite
    differences (fixed-step or adaptive) blend the two slopes -- silently."""
    net = Network([Dense(np.array([[1.0, -1.0]]), np.array([-0.3, 0.6]), "relu"),
                   Dense(np.array([[2.0], [3.0]]), np.array([0.0]), "linear")], 1, 1)
    # y = 2 relu(x - 0.3) + 3 relu(0.6 - x);  y' = -3 (x<0.3), -1 (0.3<x<0.6), 2 (x>0.6)
    x = np.array([[0.1], [0.45], [0.9]])
    for J in (D.nnj(net, x), D.ad_forward(net, x), D.ad_reverse(net, x), D.csm(net, x), D.cfd(net, x, 1e-6)):
        assert np.allclose(J[:, 0, 0], [-3.0, -1.0, 2.0], atol=1e-8)
    xk = np.array([[0.3 + 2e-7]])                              # 2e-7 to the right of the kink
    for J in (D.nnj(net, xk), D.ad_forward(net, xk), D.ad_reverse(net, xk), D.csm(net, xk)):
        assert np.isclose(J[0, 0, 0], -1.0)
    assert abs(D.cfd(net, xk, 1e-6)[0, 0, 0] + 1.0) > 0.5      # FD mixes both branches
    Jr, err = D.cfd_ridders(net, xk)
    assert abs(Jr[0, 0, 0] + 2.0) < 1e-3                       # average of -3 and -1 ...
    assert err[0, 0, 0] < 1e-4                                 # ... reported with confidence
    # second derivative of a ReLU network is identically zero wherever it exists
    assert np.all(D.hessian_nnj(net, x) == 0.0)


def test_ridders_error_estimate_is_honest(rng):
    net = random_networks(rng)["mlp-tanh-6x32"]
    x = rng.uniform(-1, 1, size=(60, 3))
    J = D.nnj(net, x)
    Jr, err = D.cfd_ridders(net, x)
    actual = np.abs(Jr - J)
    assert actual.max() < 1e-9
    assert np.mean(actual <= 10 * err + 1e-12) > 0.99


def test_second_order_from_values_has_cancellation(rng):
    net = random_networks(rng)["mlp-sigmoid-1x64"]
    x = rng.uniform(0, 1, size=(50, 2))
    H = D.hessian_nnj(net, x)
    scale = np.abs(H).max()
    good = np.abs(D.hessian_cfd(net, x, 1e-4) - H).max() / scale
    bad = np.abs(D.hessian_cfd(net, x, 1e-7) - H).max() / scale
    assert good < 1e-5 and bad > 1e-4
    # the complex step does NOT rescue second derivatives taken from values
    bad_cs = np.abs(D.hessian_csm_values(net, x, 1e-7) - H).max() / scale
    assert bad_cs > 1e-4
    # ... but applied to the analytic Jacobian it is exact
    assert np.abs(D.hessian_csm_of_nnj(net, x) - H).max() / scale < 1e-12
