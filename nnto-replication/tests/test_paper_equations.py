"""Independent checks of the paper's printed equations and worked example."""
import numpy as np

from nnto.networks import mlp
from nnto import derivatives as D
from nnto import functional as F

# Appendix B, eqs. (B13)-(B14)
W1 = np.array([[1.735, -0.850, 1.404]])
W2 = np.array([[-2.257, 3.355, 2.304], [1.418, -0.558, -1.067], [3.993, -4.024, -3.978]])
W3 = np.array([[4.323], [-1.276], [-2.209]])
B1 = np.array([2.424, -0.203, -2.053])
B2 = np.array([0.520, -1.258, -0.828])
B3 = np.array([0.588])


def test_appendix_b_final_value():
    net = mlp([W1, W2, W3], [B1, B2, B3], ["sigmoid", "sigmoid", "linear"])
    x = np.array([[0.812]])
    d = D.nnj(net, x)[0, 0, 0]
    assert abs(d - 1.623) < 2e-3                         # eq. (B18c)
    assert abs((2 * 0.812 - d) / (2 * 0.812)) < 2e-3     # "0.062 % error" up to rounding
    for other in (D.csm(net, x), D.ad_forward(net, x), D.ad_reverse(net, x), D.cfd(net, x)):
        assert np.isclose(other[0, 0, 0], d, rtol=1e-8)


def test_appendix_b_intermediate_vectors_have_sign_typos():
    """The printed intermediate vectors of eqs. (B15b), (B16c) and (B18b)
    contain sign errors; the final result is unaffected."""
    x = 0.812
    z1 = x * W1[0] + B1
    assert np.allclose(z1, [3.833, -0.893, -0.913], atol=1e-3)        # (B15b) prints +0.913
    y1 = F.sigmoid(z1)
    assert np.allclose(y1, [0.979, 0.290, 0.286], atol=1e-3)          # (B16c) prints -0.290
    d1 = W1[0] * y1 * (1 - y1)
    assert np.allclose(d1, [0.036, -0.175, 0.287], atol=1e-3)         # (B15d) correct
    z2 = y1 @ W2 + B2
    assert np.allclose(z2, [-0.134, 0.711, -0.022], atol=4e-3)        # (B16d) correct
    y2 = F.sigmoid(z2)
    d2 = W2 * (y2 * (1 - y2))[None, :]
    row = d1 @ d2
    assert np.allclose(row, [0.203, -0.207, -0.218], atol=1.5e-3)     # (B18b) prints +0.218
    assert abs(row @ W3[:, 0] - 1.623) < 2e-3
    assert abs(np.array([0.203, -0.207, 0.218]) @ W3[:, 0] - 1.623) > 0.9   # as printed


def test_eq4_needs_the_row_vector_convention():
    """Eq. (1) writes y_n = F(W_n y_{n-1} + B_n) but eqs. (3)-(6) multiply the
    layer Jacobians in the order that is only valid for y_n = F(y_{n-1} W_n + B_n)
    (the convention of Appendix B and of the code).  With the column
    convention of eq. (1) the factors are not even conformable."""
    rng = np.random.default_rng(0)
    Wa, Wb = rng.normal(size=(2, 5)), rng.normal(size=(5, 3))       # row convention shapes
    ba, bb = rng.normal(size=5), rng.normal(size=3)
    x = rng.normal(size=(1, 2))
    net = mlp([Wa, Wb], [ba, bb], ["sigmoid", "linear"])
    za = x @ Wa + ba
    J_eq6 = (Wa * F.get_activation("sigmoid")[1](za)) @ Wb          # dy1/dx . dy2/dy1
    assert np.allclose(J_eq6.T, D.nnj(net, x)[0])
    # column convention: W_1 is (5, 2), W_2 is (3, 5); eq. (6) asks for W_1-term @ W_2-term
    try:
        (Wa.T * 1.0) @ Wb.T
        conformable = True
    except ValueError:
        conformable = False
    assert not conformable
