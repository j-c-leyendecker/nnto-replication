"""Does the float64 patch improve the density-based optimisation results?

Configurations (all with the published network, sensitivity filter, constraint
Jacobian, trust-constr tol = 1e-3, maxiter = 1000):

    CFD published   predict() rounds inputs to float32, CFD divides by 2h
    CFD patched     set_floatx('float64'): everything in double precision
    NNJ patched     exact derivative of the surrogate (reference)
    NNJ published   exact derivative, but objective evaluated with float32 inputs

Part A: one unperturbed run per configuration, every iterate recorded, to see
        how long each run follows the exact-derivative run and how large the
        derivative error is along the path.
Part B: 5 repeats of CFD published / CFD patched / NNJ patched in which the
        gradient is multiplied by (1 + 1e-10 * N(0,1)) -- the size of the
        round-off differences between the authors' own NNJ, CSM and AD
        sensitivities -- to measure the run-to-run spread that any comparison
        of final designs has to beat.

Every final design is also scored with the exact SIMP law ("true compliance").

Output: results/patch_check.npz, results/patch_check.json
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS, mean_ci  # noqa: E402

from nnto import derivatives as D  # noqa: E402
from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SimpMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.topopt import ComplianceProblem, run_trust_constr  # noqa: E402

UP = find_upstream()


def cfd_published(net, x):
    return D.cfd(net, x, 1e-6, input_dtype=np.float32, exact_step=False)


def cfd_patched(net, x):
    return D.cfd(net, x, 1e-6, input_dtype=None, exact_step=False)


CONFIGS = {
    "CFD published": dict(derivative=cfd_published, predict_cast=np.float32),
    "CFD patched": dict(derivative=cfd_patched, predict_cast=None),
    "NNJ patched": dict(derivative="NNJ", predict_cast=None),
    "NNJ published": dict(derivative="NNJ", predict_cast=np.float32),
}


def material(name, net):
    return NNMaterial(net, **CONFIGS[name])


def run_one(job):
    name, noise, seed, record = job
    net = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))
    mesh = mbb_half()
    prob = ComplianceProblem(mesh, material(name, net), kind="density", filter="sensitivity",
                             volume_jac_mode="original", grad_noise=noise, noise_seed=seed)
    r = run_trust_constr(prob, tol=1e-3, maxiter=1000, sparse_jacobian=True,
                         record_designs=list(range(1, 1001)) if record else None)
    x = r["x"]
    true = ComplianceProblem(mesh, SimpMaterial(), filter="none").compliance(x)
    out = dict(name=name, noise=noise, seed=seed, x=x, comp=r["fun"], comp_true=true,
               nit=r["nit"], message=r["message"], optimality=r["optimality"],
               time=r["time"], hist=r["history"]["fun"])
    if record:
        its = sorted(r["designs"])
        out["iterates"] = np.array([r["designs"][k] for k in its])
        # derivative error of this run's method along its own path (every 10th iterate)
        exact = NNMaterial(net, derivative="NNJ", predict_cast=None, clip="none")
        mine = NNMaterial(net, derivative=CONFIGS[name]["derivative"], predict_cast=None, clip="none")
        errs = []
        for k in range(0, len(its), 10):
            th = out["iterates"][k].reshape(-1, 1)
            dE_exact = exact.raw(th)[1][:, 0, 0]
            dE_mine = mine.raw(th)[1][:, 0, 0]
            errs.append(float(np.max(np.abs(dE_mine - dE_exact)) / np.max(np.abs(dE_exact))))
        out["deriv_err_path"] = np.array(errs)
    return out


def main(n_rep=5, workers=2):
    jobs = [(n, 0.0, 0, True) for n in CONFIGS]
    jobs += [(n, 1e-10, s, False) for s in range(1, n_rep + 1)
             for n in ("CFD published", "CFD patched", "NNJ patched")]
    t0 = time.time()
    res = []
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs)):
            res.append(r)
            print(f"[{k + 1}/{len(jobs)}] {r['name']:14s} seed {r['seed']}: comp {r['comp']:.3f} "
                  f"true {r['comp_true']:.3f} nit {r['nit']} ({r['message'][:5]}) {time.time() - t0:.0f}s",
                  flush=True)
            _save(res)
    print("done in %.0f s" % (time.time() - t0))


def _save(res):
    np.savez(os.path.join(RESULTS, "patch_check.npz"),
             names=np.array([r["name"] for r in res]), seeds=np.array([r["seed"] for r in res]),
             X=np.array([r["x"] for r in res]), comp=np.array([r["comp"] for r in res]),
             comp_true=np.array([r["comp_true"] for r in res]), nit=np.array([r["nit"] for r in res]),
             **{f"iterates_{r['name'].replace(' ', '_')}": r["iterates"] for r in res if "iterates" in r},
             **{f"derr_{r['name'].replace(' ', '_')}": r["deriv_err_path"] for r in res if "iterates" in r},
             **{f"hist_{r['name'].replace(' ', '_')}_{r['seed']}": r["hist"] for r in res})
    save_json("patch_check.json", [{k: v for k, v in r.items()
                                    if k not in ("x", "iterates", "hist", "deriv_err_path")} for r in res])


if __name__ == "__main__":
    main()
