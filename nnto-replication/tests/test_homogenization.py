"""Numerical homogenization: reproduction of the published data set and the
body-fitted ground truth for derivatives."""
import os

import numpy as np
import pytest
from scipy.io import loadmat

from nnto.homogenization import (voxel_CH, fitted_CH, voxel_effective_theta,
                                 plane_strain_C, _homogenize_rectilinear)


def test_voxel_model_reproduces_published_dataset(upstream):
    m = loadmat(os.path.join(upstream, "HBTO_CH_2D_N10000_100x100.mat"), squeeze_me=True)
    P, C = m["SP_para"], m["SP_C"][:, [0, 1, 2, 5]]
    for i in (0, 17, 4242, 9999):
        assert np.allclose(voxel_CH(P[i, 0], P[i, 1]), C[i], atol=1e-11)


def test_published_targets_are_piecewise_constant(upstream):
    """Audit finding A6: 10 000 samples, 2 455 distinct targets -- the hole
    snaps to the 100 x 100 grid, so theta is effectively rounded to 0.02."""
    m = loadmat(os.path.join(upstream, "HBTO_CH_2D_N10000_100x100.mat"), squeeze_me=True)
    P, C = m["SP_para"][:, :2], m["SP_C"][:, [0, 1, 2, 5]]
    assert np.unique(np.round(C, 10), axis=0).shape[0] == 2455
    q = np.round(voxel_effective_theta(P) / 0.02).astype(int)
    key = q[:, 0] * 1000 + q[:, 1]
    for k in np.unique(key)[::25]:
        assert np.ptp(C[key == k], axis=0).max() == 0.0


def test_fitted_mesh_equals_voxel_mesh_when_they_coincide():
    cv = voxel_CH(0.4, 0.4, nel=50)                       # hole 20 x 20 voxels, walls 15
    cf = fitted_CH(0.4, 0.4, ns=15, nh=20, grading=1.0, void_scale=1e-9, derivatives=False)
    assert np.allclose(cv, cf, atol=1e-12)


@pytest.mark.parametrize("t1,t2", [(0.3, 0.7), (0.7071, 0.7071), (0.9, 0.2), (0.05, 0.5)])
def test_analytic_parameter_derivative(t1, t2):
    C, dC = fitted_CH(t1, t2, ns=8, nh=16)
    h = 1e-6
    fd1 = (fitted_CH(t1 + h, t2, ns=8, nh=16, derivatives=False)
           - fitted_CH(t1 - h, t2, ns=8, nh=16, derivatives=False)) / (2 * h)
    fd2 = (fitted_CH(t1, t2 + h, ns=8, nh=16, derivatives=False)
           - fitted_CH(t1, t2 - h, ns=8, nh=16, derivatives=False)) / (2 * h)
    assert np.allclose(dC[:, 0], fd1, atol=5e-8)
    assert np.allclose(dC[:, 1], fd2, atol=5e-8)


def test_symmetry_and_physical_bounds():
    C, dC = fitted_CH(0.35, 0.8, ns=8, nh=16)
    Cs, dCs = fitted_CH(0.8, 0.35, ns=8, nh=16)
    assert np.allclose(C[[2, 1, 0, 3]], Cs, atol=1e-12)           # x <-> y
    assert np.allclose(dC[[2, 1, 0, 3]][:, ::-1], dCs, atol=1e-10)
    C0 = plane_strain_C()
    M = np.array([[C[0], C[1], 0], [C[1], C[2], 0], [0, 0, C[3]]])
    voigt = (1 - 0.35 * 0.8) * C0
    assert np.linalg.eigvalsh(M).min() > 0                        # positive definite
    assert np.linalg.eigvalsh(voigt - M).min() > 0                # below the Voigt bound
    assert np.all(dC[[0, 2, 3]] < 0)                              # bigger hole -> softer


def test_mesh_convergence_is_second_order_with_grading():
    seq = [fitted_CH(0.5, 0.6, ns=n, nh=2 * n, grading=2.5)[0] for n in (8, 16, 32)]
    rate = np.log2(np.abs(seq[1] - seq[0]) / np.abs(seq[2] - seq[1]))
    assert np.all(rate > 1.6)
    seq = [fitted_CH(0.5, 0.6, ns=n, nh=2 * n, grading=1.0)[0] for n in (8, 16, 32)]
    rate_u = np.log2(np.abs(seq[1] - seq[0]) / np.abs(seq[2] - seq[1]))
    assert np.all(rate_u < 1.4)                # corner singularity limits uniform meshes


def test_solid_cell_returns_base_material():
    C0 = plane_strain_C()
    h = np.full(6, 1 / 6)
    CH, _ = _homogenize_rectilinear(h, h, np.ones((6, 6)), C0)
    assert np.allclose(CH, C0, atol=1e-12)
