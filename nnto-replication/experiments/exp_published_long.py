"""How far does the published HBTO network get with more MMA iterations?
(added 2026-11-18, see FIXES_2026-11-18.md)

With the 3,000-iteration cap of exp_newer_models_to.py the published network
never converges. This script runs the same formulation (MMA, consistent
density filter, exact volume Jacobian, hole sizes in [0.01, 0.99]) for many
more iterations and scores the design every ``every`` iterations with the
mesh-converged reference, so the comparison with the newer networks can be
made at more than one iteration budget.

usage:  python exp_published_long.py [maxiter] [every]
Output: results/published_long.json
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SplineMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.scoring import hole_volume_fraction, project_volume  # noqa: E402
from nnto.topopt import ComplianceProblem, run_mma  # noqa: E402


def main(maxiter=30000, every=1000):
    mesh = mbb_half()
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    truth = SplineMaterial(g["grid"], g["C"])
    net = load_keras_mlp(os.path.join(find_upstream(), "NN_model_HBTO_1e+04"))
    mat = NNMaterial(net, "NNJ", predict_cast=None, clip="consistent")      # as in exp_newer_models_to.py
    prob = ComplianceProblem(mesh, mat, kind="hole", filter="density", volume_jac_mode="exact")
    snaps, count, model_compliance = [], [0], prob.compliance

    def recording(x):                     # run_mma evaluates the objective once per iteration
        c = model_compliance(x)
        count[0] += 1
        if count[0] % every == 0:
            snaps.append((count[0], np.array(x, copy=True), c))
        return c

    prob.compliance = recording
    t0 = time.time()
    r = run_mma(prob, maxiter=maxiter, tol=1e-3, move=0.2, lb=0.01, ub=0.99)
    if not snaps or snaps[-1][0] != r["nit"]:
        snaps.append((r["nit"], r["x"].copy(), r["fun"]))
    scorer = ComplianceProblem(mesh, truth, kind="hole", filter="none")
    rows = []
    for it, x, c_model in snaps:
        X = x.reshape(-1, 2, order="F")
        P = np.stack([mesh.H @ X[:, j] / mesh.Hs for j in range(2)], axis=1)
        Pp, _ = project_volume(P, prob.volfrac, lb=0.01, ub=0.99)
        c_true = scorer.compliance(P.flatten("F"))
        rows.append(dict(iteration=int(it), compliance_model=float(c_model), compliance_true=c_true,
                         model_error_pct=100 * (c_model / c_true - 1), volume_fraction=hole_volume_fraction(P),
                         compliance_true_at_target_volume=scorer.compliance(Pp.flatten("F"))))
        print(f"it {it:6d}: true {c_true:.3f} (at target volume {rows[-1]['compliance_true_at_target_volume']:.3f}) "
              f"model {c_model:.3f} ({rows[-1]['model_error_pct']:+.2f} %) volume {rows[-1]['volume_fraction']:.5f}", flush=True)
    out = dict(label="published", maxiter=maxiter, converged=r["converged"], nit=r["nit"],
               seconds=time.time() - t0, trajectory=rows,
               last_change=float(r["history"]["change"][-1]))
    save_json("published_long.json", out)
    print(f"converged {r['converged']} after {r['nit']} iterations, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
