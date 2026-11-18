"""Second batch of density-based runs (single process).

(a) more perturbation seeds for the published NNJ configuration and for the
    exact SIMP model (continuation of exp_dbto_ensemble.py);
(b) ablation: the published configuration with one correction added at a
    time, ending with a consistent formulation and a Sobolev-trained,
    positive-by-construction surrogate.

Every final design is re-evaluated with the exact SIMP law on its physical
density field ("true compliance"), so that runs with different surrogates and
filters can be compared on the same footing.

Output: results/dbto_part2.json, results/dbto_part2.npz
"""
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SimpMaterial  # noqa: E402
from nnto.networks import load_keras_mlp, save_network  # noqa: E402
from nnto.topopt import ComplianceProblem, run_trust_constr  # noqa: E402
from nnto.training import init_mlp, train  # noqa: E402

UP = find_upstream()
mesh = mbb_half()
pub = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))


def sobolev_positive_net(seed=0):
    X = qmc.LatinHypercube(1, seed=seed).random(7000)
    Y = 1e-9 + (1 - 1e-9) * X ** 3
    G = (3 * (1 - 1e-9) * X ** 2).reshape(-1, 1, 1)
    net = init_mlp(1, 1, (64,), "sigmoid", X, Y, seed=seed, head="softfloor")
    train(net, X, Y, G, lam=1.0, maxiter=3000)
    return net


def true_compliance(x, filt):
    P = mesh.H @ x / mesh.Hs if filt == "density" else x
    return ComplianceProblem(mesh, SimpMaterial(), filter="none").compliance(P), P


def run(label, material, filt, vj, noise=0.0, seed=0):
    prob = ComplianceProblem(mesh, material, kind="density", filter=filt, volume_jac_mode=vj,
                             grad_noise=noise, noise_seed=seed)
    r = run_trust_constr(prob, tol=1e-3, maxiter=1000, sparse_jacobian=True)
    ct, P = true_compliance(r["x"], filt)
    # consistency of the supplied gradient with the objective at the final design
    d = np.random.default_rng(1).normal(size=prob.n)
    chk = ComplianceProblem(mesh, material, kind="density", filter=filt, volume_jac_mode=vj)
    h = 1e-6
    fd = (chk.compliance(np.clip(r["x"] + h * d, 0, 1)) - chk.compliance(np.clip(r["x"] - h * d, 0, 1))) / (2 * h)
    return dict(label=label, noise=noise, seed=seed, comp=r["fun"], comp_true=ct, nit=r["nit"],
                message=r["message"], optimality=r["optimality"], time=r["time"], x=r["x"],
                grey=float(np.mean(4 * P * (1 - P))), volume=float(P.mean()),
                frac_below_clip=float(np.mean(P < 0.0811)))


def main():
    t0 = time.time()
    jobs = []
    for s in (3, 4, 5, 6):
        jobs.append(("NNJ+1e-10", lambda: NNMaterial(pub, "NNJ"), "sensitivity", "original", 1e-10, s))
    for s in (1, 2, 3, 4, 5):
        jobs.append(("SIMP+1e-10", lambda: SimpMaterial(), "sensitivity", "original", 1e-10, s))
    sob = sobolev_positive_net()
    os.makedirs(os.path.join(RESULTS, "models"), exist_ok=True)
    save_network(os.path.join(RESULTS, "models", "dbto_sobolev_positive.npz"), sob)
    t = np.linspace(0, 1, 2001).reshape(-1, 1)
    info = dict(sobolev_net_fn_rmse=float(np.sqrt(np.mean((sob.forward(t)[:, 0] - (1e-9 + (1 - 1e-9) * t[:, 0] ** 3)) ** 2))),
                sobolev_net_d_rmse=float(np.sqrt(np.mean((sob.jacobian_nnj(t)[:, 0, 0] - 3 * (1 - 1e-9) * t[:, 0] ** 2) ** 2))),
                sobolev_net_min_E=float(sob.forward(t).min()), sobolev_net_min_dE=float(sob.jacobian_nnj(t).min()))
    print("Sobolev positive surrogate:", info, flush=True)
    abl = [
        ("A0 published", lambda: NNMaterial(pub, "NNJ"), "sensitivity", "original"),
        ("A1 +float64 prediction", lambda: NNMaterial(pub, "NNJ", predict_cast=None), "sensitivity", "original"),
        ("A2 +exact volume Jacobian", lambda: NNMaterial(pub, "NNJ", predict_cast=None), "sensitivity", "exact"),
        ("A3 +consistent clip", lambda: NNMaterial(pub, "NNJ", predict_cast=None, clip="consistent"), "sensitivity", "exact"),
        ("A4 +density filter", lambda: NNMaterial(pub, "NNJ", predict_cast=None, clip="consistent"), "density", "exact"),
        ("A5 +Sobolev positive surrogate", lambda: NNMaterial(sob, "NNJ", predict_cast=None, clip="none"), "density", "exact"),
        ("A6 exact SIMP, consistent", lambda: SimpMaterial(), "density", "exact"),
    ]
    for lab, mat, f, vj in abl[1:]:
        jobs.append((lab, mat, f, vj, 0.0, 0))
    for lab, mat, f, vj in (abl[5], abl[6]):
        for s in (1, 2):
            jobs.append((lab + " +1e-10", mat, f, vj, 1e-10, s))
    out = []
    for k, (lab, mat, f, vj, noise, seed) in enumerate(jobs):
        r = run(lab, mat(), f, vj, noise, seed)
        out.append(r)
        print(f"[{k + 1}/{len(jobs)}] {lab:34s} seed {seed}: comp={r['comp']:.3f} true={r['comp_true']:.3f} nit={r['nit']} "
              f"grey={r['grey']:.3f} ({r['message'][:26]}) {time.time() - t0:.0f}s", flush=True)
        np.savez(os.path.join(RESULTS, "dbto_part2.npz"), labels=np.array([o["label"] for o in out]),
                 seeds=np.array([o["seed"] for o in out]), X=np.array([o["x"] for o in out]),
                 comp=np.array([o["comp"] for o in out]), comp_true=np.array([o["comp_true"] for o in out]),
                 nit=np.array([o["nit"] for o in out]), grey=np.array([o["grey"] for o in out]))
        save_json("dbto_part2.json", dict(info=info, runs=[{k2: v for k2, v in o.items() if k2 != "x"} for o in out]))


if __name__ == "__main__":
    main()
