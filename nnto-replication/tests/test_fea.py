"""Finite-element kernel and compliance sensitivities."""
import numpy as np
import pytest

from nnto.fea import mbb_half, cantilever, ke_basis, C0_VEC_EXACT, C0_VEC_ORIGINAL
from nnto.materials import SimpMaterial
from nnto.topopt import ComplianceProblem


def test_element_matrix_properties():
    A = ke_basis()
    KE = np.einsum("k,kab->ab", C0_VEC_EXACT, A)
    assert np.allclose(KE, KE.T)
    w = np.linalg.eigvalsh(KE)
    assert np.sum(np.abs(w) < 1e-12) == 3                 # three rigid-body modes
    assert w.min() > -1e-12
    # constant-strain patch: u = (x, 0) -> energy = C11 * area
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    u = np.zeros(8); u[0::2] = xy[:, 0]
    assert np.isclose(u @ KE @ u, C0_VEC_EXACT[0])
    u = np.zeros(8); u[0::2] = xy[:, 1]; u[1::2] = xy[:, 0]      # pure shear, gamma = 2
    assert np.isclose(u @ KE @ u, 4 * C0_VEC_EXACT[3])


def test_hard_coded_constants_are_rounded():
    """Audit finding A13: the scripts use 4-digit constants for the solid tensor."""
    rel = np.abs(C0_VEC_ORIGINAL - C0_VEC_EXACT) / C0_VEC_EXACT
    assert 1e-5 < rel.max() < 1e-4


def test_boundary_conditions_code_vs_figure():
    """Audit finding P4: the code solves the half MBB beam (initial SIMP
    compliance 356.48, as in the authors' log); the cantilever drawn in
    Fig. 4(a) / described in Sec. 4.2 would give 330.31."""
    mat = SimpMaterial()
    c = {}
    for name, mesh in (("mbb", mbb_half()), ("cant", cantilever())):
        prob = ComplianceProblem(mesh, mat, filter="none")
        c[name] = prob.compliance(prob.x0())
    assert abs(c["mbb"] - 356.4817) < 1e-3
    assert abs(c["cant"] - 330.3107) < 1e-3


@pytest.mark.parametrize("filt", ["none", "density"])
def test_gradient_passes_taylor_test(filt, rng):
    """Eq. (11) with the correct sign:  d phi / d theta = - u^T (dK/dtheta) u."""
    mesh = mbb_half(24, 12, rmin=1.8)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter=filt, volume_jac_mode="exact")
    x = rng.uniform(0.2, 0.9, prob.n)
    d = rng.normal(size=prob.n)
    g = prob.gradient(x) @ d
    errs = []
    for h in (1e-3, 1e-4, 1e-5):
        fd = (prob.compliance(x + h * d) - prob.compliance(x - h * d)) / (2 * h)
        errs.append(abs(fd - g) / abs(g))
    assert errs[-1] < 1e-7
    assert prob.gradient(x).max() < 0          # more material never increases compliance


def test_paper_eq11_sign_is_wrong(rng):
    """Paper eq. (11) reads  d phi/d theta = + u^T dK u.  The finite-difference
    derivative of the compliance is the *negative* of that."""
    mesh = mbb_half(16, 8, rmin=1.5)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter="none")
    x = rng.uniform(0.3, 0.9, prob.n)
    e = 37
    h = 1e-4
    xp, xm = x.copy(), x.copy(); xp[e] += h; xm[e] -= h
    fd = (prob.compliance(xp) - prob.compliance(xm)) / (2 * h)
    ev = prob._evaluate(x)
    C, dC = prob.material.evaluate(x.reshape(-1, 1))
    U, q = mesh.solve(C)
    uT_dK_u = float(np.sum(dC[e, :, 0] * q[0, e]))            # + u^T dK u  (paper eq. 11)
    assert uT_dK_u > 0 and fd < 0
    assert np.isclose(fd, -uT_dK_u, rtol=1e-5)


def test_sensitivity_filter_is_not_a_gradient(rng):
    """Audit finding A5: the heuristic sensitivity filter returns a vector
    that is not the derivative of the objective (by construction).  Feeding
    it to a trust-region SQP method breaks the model/merit consistency the
    method relies on."""
    mesh = mbb_half(24, 12, rmin=2.7)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter="sensitivity")
    x = rng.uniform(0.05, 1.0, prob.n)
    d = rng.normal(size=prob.n)
    h = 1e-5
    fd = (prob.compliance(x + h * d) - prob.compliance(x - h * d)) / (2 * h)
    assert abs(prob.gradient(x) @ d - fd) / abs(fd) > 0.05
    assert abs(prob.raw_gradient(x) @ d - fd) / abs(fd) < 1e-6
