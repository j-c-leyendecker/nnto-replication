"""The AD engine agrees with the analytic chain rule and with complex step."""
import numpy as np
import pytest

from nnto import autodiff as ad
from nnto import functional as F
from nnto import derivatives as D
from conftest import random_networks


def test_dual_elementary_rules(rng):
    x = rng.uniform(0.2, 1.5, size=(7, 3))
    t = rng.normal(size=x.shape)
    f = lambda z: F.sigmoid(z) * F.tanh(z) + F.exp(-1.0 * z) / (1.0 + z * z) + F.sin(z) ** 3 - F.log(z)
    out = f(ad.Dual(x, t))
    h = 1e-20
    ref = np.imag(f(x + 1j * h * t)) / h
    assert np.allclose(out.t, ref, rtol=1e-13, atol=1e-14)


def test_reverse_matches_forward_on_scalar_function(rng):
    x = rng.uniform(0.2, 1.5, size=(5, 4))
    W = rng.normal(size=(4, 6))
    f = lambda z: (F.softplus(z @ W) * F.cos(z @ W)).sum(axis=1)

    xin = ad.Var(x)
    y = f(xin)
    y.backward(np.ones_like(y.v))
    g_rev = xin.grad
    g_fwd = np.stack([f(ad.Dual(x, np.eye(4)[j][None, :] * np.ones_like(x))).t for j in range(4)], axis=1)
    assert np.allclose(g_rev, g_fwd, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("name", ["mlp-sigmoid-1x64", "mlp-tanh-6x32", "resnet-swish",
                                  "fourier-gelu-softplus", "siren"])
def test_jacobians_agree_across_methods(name, rng):
    net = random_networks(rng)[name]
    x = rng.uniform(-1, 1, size=(40, net.xdim))
    J = D.nnj(net, x)
    assert J.shape == (40, net.ydim, net.xdim)
    for other in (D.ad_forward(net, x), D.ad_reverse(net, x), D.csm(net, x)):
        assert np.allclose(other, J, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("name", ["mlp-sigmoid-1x64", "mlp-tanh-6x32", "resnet-swish",
                                  "fourier-gelu-softplus", "siren"])
def test_hessians_agree_across_methods(name, rng):
    net = random_networks(rng)[name]
    x = rng.uniform(-1, 1, size=(15, net.xdim))
    H = D.hessian_nnj(net, x)
    assert H.shape == (15, net.ydim, net.xdim, net.xdim)
    assert np.allclose(H, np.swapaxes(H, 2, 3))                      # symmetry
    assert np.allclose(D.hessian_ad(net, x), H, rtol=1e-10, atol=1e-11)
    assert np.allclose(D.hessian_csm_of_nnj(net, x), H, rtol=1e-10, atol=1e-11)


def test_hvp_forward_over_reverse(rng):
    net = random_networks(rng)["resnet-swish"]
    x = rng.uniform(-1, 1, size=(9, net.xdim))
    v = rng.normal(size=x.shape)
    H = D.hessian_nnj(net, x)
    for k in range(net.ydim):
        assert np.allclose(D.hvp(net, x, v, k), np.einsum("nij,nj->ni", H[:, k], v),
                           rtol=1e-10, atol=1e-11)


def test_jvp_vjp_duality(rng):
    """<w, J v> computed by one forward sweep equals <J^T w, v> from one reverse sweep."""
    net = random_networks(rng)["mlp-tanh-6x32"]
    x = rng.uniform(-1, 1, size=(11, net.xdim))
    v = rng.normal(size=x.shape)
    w = rng.normal(size=(11, net.ydim))
    _, jv = ad.jvp(net.forward, x, v)
    _, wj = ad.vjp(net.forward, x, w)
    assert np.allclose(np.sum(w * jv, axis=1), np.sum(wj * v, axis=1), rtol=1e-12)
