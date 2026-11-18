"""Newer surrogate architectures: Chebyshev KAN and SIREN initialisation.

All derivative methods must agree on them exactly as on the paper's MLP, and
their Sobolev-loss gradients must pass a finite-difference check."""
import numpy as np
import pytest

from nnto import derivatives as D
from nnto.networks import Affine, ChebyKAN, Network, load_network, save_network
from nnto.training import _loss_and_grad, init_mlp


def _kan(rng):
    return Network([Affine(np.array([2.0, 1.5]), np.array([-1.0, 0.3])),
                    ChebyKAN([rng.normal(0, 0.3, (2, 8)) for _ in range(6)]),
                    ChebyKAN([rng.normal(0, 0.2, (8, 3)) for _ in range(6)])], 2, 3)


def test_kan_first_and_second_derivatives(rng):
    net = _kan(rng)
    x = rng.uniform(0, 1, (25, 2))
    J, H = D.nnj(net, x), D.hessian_nnj(net, x)
    scale = np.abs(J).max()
    for other in (D.csm(net, x), D.ad_forward(net, x), D.ad_reverse(net, x)):
        assert np.abs(other - J).max() < 1e-12 * scale
    assert np.abs(D.hessian_ad(net, x) - H).max() < 1e-11 * np.abs(H).max()
    assert np.allclose(H, np.swapaxes(H, 2, 3))


def test_kan_single_layer_has_no_mixed_partials(rng):
    net = Network([ChebyKAN([rng.normal(size=(2, 1)) for _ in range(5)])], 2, 1)
    H = D.hessian_nnj(net, rng.uniform(-1, 1, (10, 2)))
    assert np.allclose(H[:, 0, 0, 1], 0.0) and np.allclose(H[:, 0, 1, 0], 0.0)


@pytest.mark.parametrize("kw", [dict(hidden=(8,), kan_degree=5),
                                dict(hidden=(8,), kan_degree=5, head="cholesky"),
                                dict(hidden=(16, 16), activation="sin", siren_omega0=5.0)])
def test_sobolev_loss_gradient_and_roundtrip(kw, rng, tmp_path):
    X = rng.uniform(0, 1, (40, 2))
    Y = np.stack([np.sin(3 * X[:, 0]) * X[:, 1], X[:, 0] ** 2 + X[:, 1],
                  np.cos(X[:, 1]) + 1, X[:, 0] * X[:, 1] + 0.5], 1)
    G = rng.normal(size=(40, 4, 2))
    net = init_mlp(2, 4, X=X, Y=Y, seed=1, **kw)
    shapes = [p.shape for p in net.params()]
    flat = np.concatenate([p.ravel() for p in net.params()])
    wy, wg = np.ones(4) / 40, np.ones((4, 2)) / 40
    l, g = _loss_and_grad(net, shapes, flat, X, Y, G, wy, wg, 0.7)
    d = rng.normal(size=flat.size)
    h = 1e-6
    fd = (_loss_and_grad(net, shapes, flat + h * d, X, Y, G, wy, wg, 0.7)[0]
          - _loss_and_grad(net, shapes, flat - h * d, X, Y, G, wy, wg, 0.7)[0]) / (2 * h)
    assert abs(fd - g @ d) < 1e-7 * abs(fd)
    path = str(tmp_path / "net.npz")
    save_network(path, net)
    assert np.array_equal(load_network(path).forward(X), net.forward(X))
