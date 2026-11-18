"""Is the ranking of derivative methods in the paper distinguishable from noise?

The paper compares ONE optimisation run per derivative method and recommends
NNJ and AD over CSM because their runs ended closest to each other, although
NNJ, CSM and AD return the same sensitivities to working precision.

Here every published configuration is re-run, and the NNJ / CFD / SIMP runs
are repeated with a relative perturbation of the gradient of size 1e-10 (the
size of the differences between the authors' own NNJ, CSM and AD initial
sensitivities) and 1e-14 (a few ulps).  If runs that differ only by such a
perturbation end as far apart as runs that use different derivative methods,
the single-run ranking carries no information.

Everything else is exactly as published: trained network, float32 prediction
cast, clipping, sensitivity filter, constraint Jacobian, trust-constr with
tol = 1e-3 and at most 1000 iterations.  (Constraint Jacobians are passed in
sparse form, which changes only the linear algebra -- see exp_efficiency.py.)

Output: results/dbto_ensemble.npz, results/dbto_ensemble.json
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS, mean_ci  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SimpMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.topopt import ComplianceProblem, run_trust_constr  # noqa: E402

UP = find_upstream()


def run_one(job):
    label, method, noise, seed = job
    mesh = mbb_half()
    if method == "SIMP":
        mat = SimpMaterial()
    else:
        net = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))
        mat = NNMaterial(net, derivative=method)           # published defaults
    prob = ComplianceProblem(mesh, mat, kind="density", filter="sensitivity",
                             volume_jac_mode="original", grad_noise=noise, noise_seed=seed)
    r = run_trust_constr(prob, tol=1e-3, maxiter=1000, sparse_jacobian=True)
    x = r["x"]
    true = ComplianceProblem(mesh, SimpMaterial(), filter="none")
    return dict(label=label, method=method, noise=noise, seed=seed, x=x,
                comp=r["fun"], comp_true=true.compliance(x), nit=r["nit"],
                status=r["status"], message=r["message"], time=r["time"],
                optimality=r["optimality"], hist_fun=r["history"]["fun"],
                grey=float(np.mean(4 * x * (1 - x))))


def main(n_seeds=10, workers=2):
    jobs = [(m, m, 0.0, 0) for m in ("SIMP", "NNJ", "CFD-f32", "CSM", "AD-rev", "CFD")]
    for eps in (1e-10, 1e-14):
        jobs += [(f"NNJ+{eps:g}", "NNJ", eps, s) for s in range(1, n_seeds + 1)]
    jobs += [(f"CFD-f32+1e-10", "CFD-f32", 1e-10, s) for s in range(1, n_seeds + 1)]
    jobs += [(f"SIMP+1e-10", "SIMP", 1e-10, s) for s in range(1, n_seeds + 1)]
    t0 = time.time()
    out = []
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs)):
            out.append(r)
            print(f"[{k + 1}/{len(jobs)}] {r['label']:16s} seed {r['seed']:2d}: comp={r['comp']:.3f} "
                  f"true={r['comp_true']:.3f} nit={r['nit']} ({r['message'][:28]}) {time.time() - t0:.0f}s",
                  flush=True)
            np.savez(os.path.join(RESULTS, "dbto_ensemble.npz"),
                     labels=np.array([o["label"] for o in out]),
                     seeds=np.array([o["seed"] for o in out]),
                     X=np.array([o["x"] for o in out]),
                     comp=np.array([o["comp"] for o in out]),
                     comp_true=np.array([o["comp_true"] for o in out]),
                     nit=np.array([o["nit"] for o in out]),
                     status=np.array([o["status"] for o in out]),
                     grey=np.array([o["grey"] for o in out]))
    summary = {}
    for lab in sorted(set(o["label"] for o in out)):
        sel = [o for o in out if o["label"] == lab]
        c = [o["comp"] for o in sel]; ct = [o["comp_true"] for o in sel]; it = [o["nit"] for o in sel]
        X = np.array([o["x"] for o in sel])
        pair = [float(np.mean((X[i] - X[j]) ** 2)) for i in range(len(sel)) for j in range(i + 1, len(sel))]
        summary[lab] = dict(n=len(sel), comp=mean_ci(c), comp_min=min(c), comp_max=max(c),
                            comp_true=mean_ci(ct), nit=mean_ci(it), nit_min=min(it), nit_max=max(it),
                            theta_mse_pairwise_mean=float(np.mean(pair)) if pair else None,
                            theta_mse_pairwise_max=float(np.max(pair)) if pair else None,
                            messages=sorted(set(o["message"] for o in sel)))
    save_json("dbto_ensemble.json", summary)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
