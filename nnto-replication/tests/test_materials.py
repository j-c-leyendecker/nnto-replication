"""Material models and the clip / gradient consistency issue."""
import numpy as np
import pytest

from nnto.materials import SimpMaterial, NNMaterial, _smooth_clip


def test_simp_derivative_sign_and_value():
    """Paper eq. (15) prints dE/dtheta = -p (Emax - Emin) theta^(p-1); the
    derivative of eq. (14) is +p (Emax - Emin) theta^(p-1)."""
    mat = SimpMaterial(p=3.0)
    t = np.linspace(0.05, 1.0, 20)
    E, dE = mat.modulus(t)
    h = 1e-6
    fd = (mat.modulus(t + h)[0] - mat.modulus(t - h)[0]) / (2 * h)
    assert np.allclose(dE, fd, rtol=1e-8)
    assert np.all(dE > 0)
    printed_eq15 = -3.0 * (1.0 - 1e-9) * t ** 2
    assert np.allclose(printed_eq15, -fd, rtol=1e-8)          # wrong sign


def test_smooth_clip_derivative():
    c = np.linspace(-0.01, 1.01, 2001)[:, None]
    lo, hi, w = np.array([1e-9]), np.array([1.0]), np.array([1e-3])
    y, dy = _smooth_clip(c, lo, hi, w)
    h = 1e-7
    fd = (_smooth_clip(c + h, lo, hi, w)[0] - _smooth_clip(c - h, lo, hi, w)[0]) / (2 * h)
    assert np.allclose(dy, fd, atol=1e-6)
    assert y.min() > 0 and y.max() < 1.0 + 1e-12
    mid = (c[:, 0] > 0.02) & (c[:, 0] < 0.98)
    assert np.allclose(y[mid], c[mid], atol=1e-8)


def test_published_dbto_surrogate_predicts_negative_modulus(dbto_net):
    """Audit finding A3: the trained surrogate is negative on [0, 0.081]."""
    t = np.linspace(0, 1, 100001).reshape(-1, 1)
    E = dbto_net.predict(t)[:, 0]
    assert np.isclose(E[0], -3.7508e-4, rtol=1e-3)
    neg = t[E < 1e-9, 0]
    assert neg.min() == 0.0 and abs(neg.max() - 0.0811) < 1e-3
    dE = dbto_net.jacobian_nnj(t)[:, 0, 0]
    wrong_sign = t[dE < 0, 0]
    assert abs(wrong_sign.max() - 0.0289) < 1e-3


@pytest.mark.parametrize("mode,consistent", [("original", False), ("consistent", True), ("smooth", True)])
def test_clip_gradient_consistency(dbto_net, mode, consistent):
    """In the clipped region the original code returns a non-zero derivative
    for a model that is locally constant."""
    mat = NNMaterial(dbto_net, derivative="NNJ", predict_cast=None, clip=mode)
    t = np.array([[0.02], [0.05], [0.07], [0.3], [0.8]])
    C, dC = mat.evaluate(t)
    h = 1e-6
    fd = (mat.evaluate(t + h)[0] - mat.evaluate(t - h)[0]) / (2 * h)
    ok = np.allclose(dC[:, :, 0], fd, rtol=1e-5, atol=1e-9)
    assert ok is consistent
    if mode == "original":
        assert np.all(fd[:3] == 0.0) and np.all(np.abs(dC[:3, 0, 0]) > 1e-3)
