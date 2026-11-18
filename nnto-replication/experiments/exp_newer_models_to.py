"""Microstructure (HBTO) optimisation with the newer surrogate models.

Same formulation for every model (so that differences come from the model):
consistent density-type filter on both hole fields with the exact chain rule,
exact volume-constraint Jacobian, double precision, hole sizes in
[0.01, 0.99] (the range of the reference), MMA (move 0.2) until the largest
design change is below 1e-3 or 3000 iterations.  Unconstrained networks are
clipped to [1e-9 C0, C0] with the derivative zeroed where clipped
(consistent); Cholesky-head networks need no clipping.

Every final design is scored with the mesh-converged reference model
(bicubic spline of results/fitted_grid.npz; values within 6e-5 relative of
the converged FE results at independent off-grid points).  The reference
itself is also optimised: that is what a perfect surrogate would give.

Fixes of 2026-11-18 (see FIXES_2026-11-18.md): each run also records its
volume-constraint violation and the true compliance after scaling both hole
fields to the exact volume target (runs stopped by the iteration cap can end
slightly short of it); the summary reports the mean absolute and largest
model error, not only the signed mean, whose terms can cancel.

usage:  python exp_newer_models_to.py [--workers=N] [--out=NAME] [--seeds=N] [--maxiter=N]
                                      "<config 1>" "<config 2>" ...
        a config is "published", "reference", or "<architecture> | <regime>"
        as in results/newer_models.json; one run per trained seed.
Output: results/newer_models_to.json, results/newer_models_to.npz
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS, mean_ci, model_ref, resolve_model  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SplineMaterial  # noqa: E402
from nnto.networks import CholeskyHead, load_keras_mlp, load_network  # noqa: E402
from nnto.scoring import error_summary, hole_volume_fraction, project_volume  # noqa: E402
from nnto.topopt import ComplianceProblem, run_mma  # noqa: E402

UP = find_upstream()
MAXITER = 3000


def reference():
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    return SplineMaterial(g["grid"], g["C"])


def run_one(job):
    label, path, seed, noise, maxiter = job if len(job) == 5 else (*job, MAXITER)
    mesh = mbb_half()
    truth = reference()
    if label == "reference":
        mat = truth
    else:
        net = load_keras_mlp(resolve_model(path)) if label == "published" else load_network(resolve_model(path))
        chol = any(isinstance(b, CholeskyHead) for b in net.blocks)
        mat = NNMaterial(net, "NNJ", predict_cast=None, clip="none" if chol else "consistent")
    prob = ComplianceProblem(mesh, mat, kind="hole", filter="density", volume_jac_mode="exact",
                             grad_noise=noise, noise_seed=seed)
    r = run_mma(prob, maxiter=maxiter, tol=1e-3, move=0.2, lb=0.01, ub=0.99)
    X = r["x"].reshape(-1, 2, order="F")
    P = np.stack([mesh.H @ X[:, j] / mesh.Hs for j in range(2)], axis=1)
    scorer = ComplianceProblem(mesh, truth, kind="hole", filter="none")
    c_true = scorer.compliance(P.flatten("F"))
    vf = hole_volume_fraction(P)
    Pp, s = project_volume(P, prob.volfrac, lb=0.01, ub=0.99)
    c_true_v = scorer.compliance(Pp.flatten("F"))
    return dict(label=label, seed=seed, noise=noise, path=model_ref(path), compliance_model=r["fun"], compliance_true=c_true,
                model_error_pct=100 * (r["fun"] / c_true - 1), nit=r["nit"], converged=r["converged"],
                volume_fraction=vf, volume_violation=vf - prob.volfrac,
                compliance_true_at_target_volume=c_true_v, volume_projection_scale=s, maxiter=maxiter,
                seconds=r["time"], x=r["x"])


def main(configs, workers=2, out_name="newer_models_to", n_seeds=None, maxiter=MAXITER):
    meta_path = os.path.join(RESULTS, "newer_models.json")
    if not os.path.exists(meta_path):                      # training still running
        meta_path = os.path.join(RESULTS, "newer_models_runs.json")
    runs_meta = json.load(open(meta_path))["runs"] if os.path.exists(meta_path) else []
    jobs = []
    for c in configs:
        # "published*N" / "reference*N": the run plus N repeats whose gradient
        # is perturbed by a relative 1e-10 (run-to-run noise floor)
        base, _, rep = c.partition("*")
        nrep = int(rep) if rep else 0
        if base in ("published", "reference"):
            path = os.path.join(UP, "NN_model_HBTO_1e+04") if base == "published" else ""
            jobs.append((base, path, 0, 0.0, maxiter))
            jobs += [(base, path, s, 1e-10, maxiter) for s in range(1, nrep + 1)]
        else:
            arch, regime = [s.strip() for s in c.split("|")]
            sel = sorted([r for r in runs_meta if r["arch"] == arch and r["regime"] == regime], key=lambda r: r["seed"])
            if n_seeds is not None:
                sel = sel[:n_seeds]
            if not sel:
                raise SystemExit(f"no trained models for {c!r}")
            jobs += [(c, r["path"], r["seed"], 0.0, maxiter) for r in sel]
    t0 = time.time()
    out = []
    # resume: keep runs that finished before an interruption
    prev_json = os.path.join(RESULTS, out_name + ".json")
    prev_npz = os.path.join(RESULTS, out_name + ".npz")
    if os.path.exists(prev_json) and os.path.exists(prev_npz):
        meta = json.load(open(prev_json)); z = np.load(prev_npz)
        for m, x in zip(meta, z["X"]):
            m["x"] = x
            out.append(m)
        done = {(m["label"], m["seed"], m.get("noise", 0.0)) for m in out}
        jobs = [j for j in jobs if (j[0], j[2], j[3]) not in done]
        print(f"resuming: {len(out)} runs already done, {len(jobs)} to go", flush=True)
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs)):
            out.append(r)
            print(f"[{k + 1}/{len(jobs)}] {r['label']:58s} s{r['seed']}: true {r['compliance_true']:.3f} "
                  f"model {r['compliance_model']:.3f} ({r['model_error_pct']:+.2f} %) nit {r['nit']} "
                  f"conv {r['converged']} {r['seconds']:.0f}s [{time.time() - t0:.0f}s]", flush=True)
            np.savez(os.path.join(RESULTS, out_name + ".npz"), labels=np.array([o["label"] for o in out]),
                     seeds=np.array([o["seed"] for o in out]), X=np.array([o["x"] for o in out]))
            save_json(out_name + ".json", [{k2: v for k2, v in o.items() if k2 != "x"} for o in out])
    print("summary (true compliance, mean ± 95% CI over runs; model error = surrogate's own compliance vs truth):")
    for c in dict.fromkeys(o["label"] for o in out):
        sel = [o for o in out if o["label"] == c]
        v = [o["compliance_true"] for o in sel]
        es = error_summary([o["model_error_pct"] for o in sel])
        m, sd, hw = mean_ci(v) if len(v) > 1 else (v[0], float("nan"), float("nan"))
        line = (f"  {c:58s} n={len(v)} true {m:.3f} ± {hw:.3f}  model error: mean {es['mean_signed']:+.2f} %, "
                f"mean |.| {es['mean_abs']:.2f} %, max |.| {es['max_abs']:.2f} %  "
                f"volume {min(o['volume_fraction'] for o in sel):.5f}-{max(o['volume_fraction'] for o in sel):.5f}")
        if all("compliance_true_at_target_volume" in o for o in sel):
            vp = [o["compliance_true_at_target_volume"] for o in sel]
            line += f"  true at target volume {np.mean(vp):.3f}"
        print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    workers, out_name, n_seeds, maxiter = 2, "newer_models_to", None, MAXITER
    while args and args[0].startswith("--"):
        k, v = args.pop(0)[2:].split("=")
        if k == "workers":
            workers = int(v)
        elif k == "out":
            out_name = v
        elif k == "seeds":
            n_seeds = int(v)
        elif k == "maxiter":
            maxiter = int(v)
    main(args, workers=workers, out_name=out_name, n_seeds=n_seeds, maxiter=maxiter)
