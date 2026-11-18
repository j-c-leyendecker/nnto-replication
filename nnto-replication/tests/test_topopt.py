"""Problem definition: constraint Jacobian, caching, optimiser smoke tests."""
import numpy as np
import pytest

from nnto.fea import mbb_half
from nnto.materials import SimpMaterial
from nnto.topopt import ComplianceProblem, run_oc, run_trust_constr


def _fd_jac(fun, x, h=1e-6):
    g = np.zeros(x.size)
    for i in range(x.size):
        e = np.zeros(x.size); e[i] = h
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


def test_dbto_volume_jacobian_original_is_off_by_ne(rng):
    """Audit finding A2: Run_DBTO_*.py::dnlcon returns ones although the
    constraint is mean(x) - Vt, i.e. the Jacobian is 1/ne."""
    mesh = mbb_half(12, 6, rmin=1.5)
    x = rng.uniform(0.1, 0.9, mesh.ne)
    orig = ComplianceProblem(mesh, SimpMaterial(), volume_jac_mode="original")
    exact = ComplianceProblem(mesh, SimpMaterial(), volume_jac_mode="exact")
    fd = _fd_jac(orig.volume, x)
    assert np.allclose(exact.volume_jac(x), fd, rtol=1e-6)
    assert np.allclose(orig.volume_jac(x), mesh.ne * fd, rtol=1e-6)


def test_hbto_volume_jacobian_is_exact(rng):
    mesh = mbb_half(10, 5, rmin=1.5)
    prob = ComplianceProblem(mesh, SimpMaterial(), kind="hole", volume_jac_mode="original")
    x = rng.uniform(0.1, 0.9, 2 * mesh.ne)
    assert np.allclose(prob.volume_jac(x), _fd_jac(prob.volume, x), rtol=1e-6)
    assert np.isclose(prob.volume(prob.x0()), 0.0, atol=1e-12)


def test_density_filter_volume_is_consistent(rng):
    mesh = mbb_half(12, 6, rmin=2.0)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter="density", volume_jac_mode="exact")
    x = rng.uniform(0.1, 0.9, mesh.ne)
    assert np.allclose(prob.volume_jac(x), _fd_jac(prob.volume, x), rtol=1e-6)


def test_cache_avoids_second_solve():
    mesh = mbb_half(12, 6)
    prob = ComplianceProblem(mesh, SimpMaterial())
    x = prob.x0()
    prob.compliance(x); prob.gradient(x); prob.compliance(x)
    assert prob.n_eval == 1


def test_oc_reduces_compliance_and_keeps_volume():
    mesh = mbb_half(30, 10, rmin=1.5)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter="sensitivity")
    r = run_oc(prob, maxiter=60)
    assert r["fun"] < 0.6 * prob.compliance(prob.x0())
    assert abs(prob.volume(r["x"])) < 2e-3


def test_trust_constr_runs_and_respects_bounds():
    mesh = mbb_half(16, 8, rmin=1.5)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter="density", volume_jac_mode="exact")
    r = run_trust_constr(prob, maxiter=25, sparse_jacobian=True)
    assert r["fun"] < prob.compliance(prob.x0())
    assert r["x"].min() >= 0 and r["x"].max() <= 1
    assert abs(prob.volume(r["x"])) < 1e-6


def test_mma_reduces_compliance_and_satisfies_volume():
    from nnto.topopt import run_mma
    mesh = mbb_half(30, 10, rmin=1.5)
    prob = ComplianceProblem(mesh, SimpMaterial(), filter="density", volume_jac_mode="exact")
    r = run_mma(prob, maxiter=300, tol=1e-3)
    assert r["fun"] < 0.6 * prob.compliance(prob.x0())
    assert prob.volume(r["x"]) < 1e-6                       # inequality v - Vt <= 0
    assert r["x"].min() >= 0 and r["x"].max() <= 1


def test_mma_hole_parametrisation_respects_bounds():
    from nnto.topopt import run_mma
    mesh = mbb_half(12, 6, rmin=1.5)
    prob = ComplianceProblem(mesh, SimpMaterial(), kind="hole", filter="density", volume_jac_mode="exact")
    # SIMP stands in for a hole-size model here: only the optimiser mechanics are tested
    prob.material = type("M", (), {"evaluate": staticmethod(
        lambda t: (np.repeat((1 - t[:, :1] * t[:, 1:]) ** 3 * 1.0, 4, axis=1) * [1.3462, 0.5769, 1.3462, 0.3846],
                   np.stack([np.repeat(-3 * (1 - t[:, :1] * t[:, 1:]) ** 2 * t[:, 1:], 4, axis=1) * [1.3462, 0.5769, 1.3462, 0.3846],
                             np.repeat(-3 * (1 - t[:, :1] * t[:, 1:]) ** 2 * t[:, :1], 4, axis=1) * [1.3462, 0.5769, 1.3462, 0.3846]], axis=2)))})()
    r = run_mma(prob, maxiter=200, tol=1e-3, lb=0.01, ub=0.99)
    assert r["x"].min() >= 0.01 - 1e-12 and r["x"].max() <= 0.99 + 1e-12
    assert r["fun"] < prob.compliance(prob.x0())
