"""Homogenisation-based runs with a consistent formulation, scored by the
mesh-converged ("true") material model.

    improved   Sobolev-trained, positive-definite-by-construction surrogate
               (results/models/hbto_sobolev_cholesky.npz), analytic Jacobian,
               double precision, no clipping, linear filter on both hole
               fields with the exact chain rule, exact constraint Jacobian.
    truth      the same formulation driven by the converged model itself
               (bicubic spline of results/fitted_grid.npz): what a perfect
               surrogate would deliver.
    published-consistent
               the authors' network inside the consistent formulation
               (isolates the surrogate from the formulation).

Hole sizes are bounded to [0.01, 0.99], the range covered by the reference.
The published designs (authors' .npz files) are scored in
exp_surrogate_truth.py with the same true model.

usage: python exp_hbto_runs.py <improved|truth|published-consistent> [maxiter]
Output: results/hbto_run_<name>.json / .npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SplineMaterial  # noqa: E402
from nnto.networks import load_keras_mlp, load_network  # noqa: E402
from nnto.topopt import ComplianceProblem, run_trust_constr  # noqa: E402


def main(name, maxiter=400):
    mesh = mbb_half()
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    truth = SplineMaterial(g["grid"], g["C"])
    if name == "improved":
        net = load_network(os.path.join(RESULTS, "models", "hbto_sobolev_cholesky.npz"))
        mat = NNMaterial(net, "NNJ", predict_cast=None, clip="none")
    elif name == "truth":
        mat = truth
    elif name == "published-consistent":
        net = load_keras_mlp(os.path.join(find_upstream(), "NN_model_HBTO_1e+04"))
        mat = NNMaterial(net, "NNJ", predict_cast=None, clip="consistent")
    else:
        raise SystemExit(__doc__)
    prob = ComplianceProblem(mesh, mat, kind="hole", filter="density", volume_jac_mode="exact")
    t0 = time.time()
    r = run_trust_constr(prob, tol=1e-3, maxiter=maxiter, sparse_jacobian=True, lb=0.01, ub=0.99)
    x = r["x"]
    X = x.reshape(-1, 2, order="F")
    P = np.stack([mesh.H @ X[:, j] / mesh.Hs for j in range(2)], axis=1)
    c_true = ComplianceProblem(mesh, truth, kind="hole", filter="none").compliance(P.flatten("F"))
    out = dict(name=name, compliance_model=r["fun"], compliance_true=c_true,
               model_error_pct=100 * (c_true / r["fun"] - 1), nit=r["nit"], message=r["message"],
               optimality=r["optimality"], volume_fraction=float(1 - np.mean(P[:, 0] * P[:, 1])),
               seconds=time.time() - t0, maxiter=maxiter,
               history_fun=r["history"]["fun"][::10])
    save_json(f"hbto_run_{name}.json", out)
    np.savez(os.path.join(RESULTS, f"hbto_run_{name}.npz"), x=x, P=P)
    print({k: v for k, v in out.items() if k != "history_fun"})

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
