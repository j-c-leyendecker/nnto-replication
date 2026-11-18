"""Tests for nnto.scoring and the fixes of 2026-11-18 (FIXES_2026-11-18.md)."""
import os

import numpy as np
import pytest

from nnto.fea import C0_VEC_EXACT, mbb_half
from nnto.materials import NNMaterial
from nnto.scoring import (SmallHolesSolid, compare_vectors, error_summary, hole_volume_fraction,
                          project_volume, sensitivity_bases)


# ----------------------------------------------------------------- volume
def test_project_volume_hits_target(rng):
    P = rng.uniform(0.05, 0.95, size=(1800, 2))
    for target in (0.3, 0.5, 0.7):
        Q, s = project_volume(P, target, lb=0.01, ub=0.99)
        assert abs(hole_volume_fraction(Q) - target) < 1e-12
        assert np.all((Q >= 0.01) & (Q <= 0.99))
        # one common factor for both fields wherever no bound is active
        inside = (s * P > 0.01) & (s * P < 0.99)
        assert np.allclose(Q[inside], s * P[inside])


def test_project_volume_leaves_feasible_design_unchanged():
    P = np.full((100, 2), np.sqrt(0.5))
    Q, s = project_volume(P, 0.5)
    assert s == 1.0 and np.array_equal(Q, P)


def test_project_volume_direction():
    # too little material (holes too large) -> holes shrink, and vice versa
    P = np.full((10, 2), 0.72)                      # solid fraction 1 - 0.5184 = 0.4816 < 0.5
    _, s = project_volume(P, 0.5)
    assert s < 1
    P = np.full((10, 2), 0.70)                      # 1 - 0.49 = 0.51 > 0.5
    _, s = project_volume(P, 0.5)
    assert s > 1


# ------------------------------------------------------------ error summary
def test_error_summary_does_not_let_signs_cancel():
    e = error_summary([0.027, 0.0002, -0.079])     # the improved network's three runs (%)
    assert e["mean_signed"] == pytest.approx(-0.0173, abs=1e-4)
    assert e["mean_abs"] == pytest.approx(0.0354, abs=1e-4)
    assert e["max_abs"] == pytest.approx(0.079)
    assert e["mean_abs"] >= abs(e["mean_signed"])


def test_compare_vectors():
    b = np.array([1.0, 2.0, 2.0])
    r = compare_vectors(1.1 * b, b)
    assert r["rel_l2"] == pytest.approx(0.1)
    assert r["cosine"] == pytest.approx(1.0)
    assert r["mse"] == pytest.approx(np.mean((0.1 * b) ** 2))


# ------------------------------------------------- small-hole convention
class _Slit:
    """Toy reference on [0.01, 0.99]: clamps like SplineMaterial and is much
    softer than solid when one side is at the lower end (a thin slit)."""
    lo, hi = 0.01, 0.99

    def evaluate(self, theta):
        x = np.clip(np.asarray(theta, float).reshape(-1, 2), self.lo, self.hi)
        f = (1 - x[:, :1]) * (1 - x[:, 1:]) * (1 - 0.4 * (x.min(axis=1, keepdims=True) <= self.lo))
        return f * C0_VEC_EXACT, np.zeros((x.shape[0], 4, 2))


def test_small_holes_solid_convention():
    base = _Slit()
    m = SmallHolesSolid(base, C0_VEC_EXACT)
    th = np.array([[0.005, 0.5],     # side below the range: slit under clamping, solid here
                   [0.01, 0.5],      # at the range: unchanged
                   [0.3, 0.4]])      # interior: unchanged
    C, dC = m.evaluate(th)
    Cb, _ = base.evaluate(th)
    assert np.allclose(C[0], C0_VEC_EXACT) and np.all(dC[0] == 0)
    assert np.all(Cb[0] < C0_VEC_EXACT)
    assert np.allclose(C[1:], Cb[1:])
    assert m.mask(th).tolist() == [True, False, False]


# ------------------------- the paper's Fig. 11 MSE is on FILTERED sensitivities
def test_stored_dc0_is_the_filtered_sensitivity(hbto_net, paper_results):
    """The authors save dc0 = dfeafun(x0), which returns the sensitivity AFTER
    filterfun.  So the paper's 1.07e-4 (Fig. 11) is a filtered-basis MSE and a
    network-error MSE must be put on the same basis before taking a ratio."""
    mesh = mbb_half()
    x0 = np.full(2 * mesh.ne, np.sqrt(0.5))
    nnj = sensitivity_bases(mesh, NNMaterial(hbto_net, "NNJ", predict_cast=np.float32, clip="original"), x0)
    cfd = sensitivity_bases(mesh, NNMaterial(hbto_net, "CFD-f32", predict_cast=np.float32, clip="original"), x0)
    a = np.load(os.path.join(paper_results, "HBTO_NN_result_dy1", "HBTO_NN_dy1_it1225final.npz"))["dc0"].ravel()
    c = np.load(os.path.join(paper_results, "HBTO_NN_result_dy2", "HBTO_NN_dy2_it1105final.npz"))["dc0"].ravel()
    assert np.linalg.norm(a - nnj["filtered"]) / np.linalg.norm(a) < 1e-5
    assert np.linalg.norm(a - nnj["raw"]) / np.linalg.norm(a) > 0.5
    # the published CFD-vs-NNJ MSE is reproduced on the filtered basis only
    assert compare_vectors(cfd["filtered"], nnj["filtered"])["mse"] == pytest.approx(1.07e-4, rel=0.01)
    assert compare_vectors(c, a)["mse"] == pytest.approx(1.07e-4, rel=0.01)
    assert compare_vectors(cfd["raw"], nnj["raw"])["mse"] > 2 * 1.07e-4
