"""Shared fixtures.

Tests that compare against the original repository need its location.  Set
the environment variable NNTO_UPSTREAM to the folder that contains
``Run_DBTO_NN.py`` (i.e. ``.../nn-derivative-to-main/NN_derivatives_TO_v3_final``)
or put a clone of https://github.com/edrl-purdue/nn-derivative-to (or its
unzipped download, ``nn-derivative-to-main``) next to this package.  Those tests
are skipped when the folder cannot be found.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

_CANDIDATES = [
    os.environ.get("NNTO_UPSTREAM", ""),
    os.path.join(ROOT, "nn-derivative-to-main", "NN_derivatives_TO_v3_final"),
    os.path.join(ROOT, "..", "nn-derivative-to-main", "NN_derivatives_TO_v3_final"),
    os.path.join(ROOT, "nn-derivative-to", "NN_derivatives_TO_v3_final"),          # git clone
    os.path.join(ROOT, "..", "nn-derivative-to", "NN_derivatives_TO_v3_final"),
    os.path.join(ROOT, "..", "upstream", "nn-derivative-to-main", "NN_derivatives_TO_v3_final"),
]


def _find_upstream():
    for c in _CANDIDATES:
        if c and os.path.isfile(os.path.join(c, "Run_DBTO_NN.py")):
            return os.path.abspath(c)
    return None


@pytest.fixture(scope="session")
def upstream():
    path = _find_upstream()
    if path is None:
        pytest.skip("original repository not found (set NNTO_UPSTREAM)")
    return path


@pytest.fixture(scope="session")
def paper_results(upstream):
    return os.path.join(upstream, "JMD Journal Paper Results")


@pytest.fixture(scope="session")
def dbto_net(upstream):
    from nnto.networks import load_keras_mlp
    return load_keras_mlp(os.path.join(upstream, "NN_model_DBTO_1e+04"))


@pytest.fixture(scope="session")
def hbto_net(upstream):
    from nnto.networks import load_keras_mlp
    return load_keras_mlp(os.path.join(upstream, "NN_model_HBTO_1e+04"))


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)


def random_networks(rng):
    """A zoo of architectures used by several test modules."""
    from nnto.networks import Network, Dense, Residual, FourierFeatures, Affine

    def dense(i, o, act):
        return Dense(rng.normal(size=(i, o)) / np.sqrt(i), 0.1 * rng.normal(size=o), act)

    nets = {
        "mlp-sigmoid-1x64": Network([Affine(np.array([2.0, 3.0]), np.array([-1.0, 0.5])),
                                     dense(2, 64, "sigmoid"), dense(64, 4, "linear"),
                                     Affine(np.array([1.5, 2.0, 0.5, 3.0]), np.array([0.1, -0.2, 0.3, 0.0]))], 2, 4),
        "mlp-tanh-6x32": Network([dense(3, 32, "tanh")] + [dense(32, 32, "tanh") for _ in range(5)]
                                 + [dense(32, 4, "linear")], 3, 4),
        "resnet-swish": Network([dense(3, 24, "swish")]
                                + [Residual([dense(24, 24, "swish"), dense(24, 24, "linear")]) for _ in range(3)]
                                + [dense(24, 2, "linear")], 3, 2),
        "fourier-gelu-softplus": Network([FourierFeatures(2.0 * rng.normal(size=(3, 12))),
                                          dense(24, 24, "gelu"), dense(24, 24, "softplus"),
                                          dense(24, 3, "linear")], 3, 3),
        "siren": Network([dense(2, 24, "sin"), dense(24, 24, "sin"), dense(24, 1, "linear")], 2, 1),
    }
    return nets
