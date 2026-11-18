"""Newer surrogate models for the homogenised stiffness of the rectangular-hole cell.

Architectures (hyper-parameters fixed in advance, not tuned on the test set):
    MLP 1x64 sigmoid       the paper's network
    Deep MLP 3x64 tanh
    ResNet 4x64 swish      three residual blocks after the first layer
    Fourier + 2x64 tanh    16 random Fourier features (Tancik et al. 2020), sigma = 2
    SIREN 3x64             sine activations, SIREN initialisation, omega0 = 5
    KAN 2-16-16-4          Chebyshev Kolmogorov-Arnold network, degree 6

Data regimes:
    paper data      the authors' 10,000 voxel samples (values only; first 7,000
                    rows = their training split)
    converged data  the 51 x 51 mesh-converged grid (results/fitted_grid.npz):
                    values only, or values + exact derivative labels (Sobolev)
    + Cholesky      Sobolev training with an output head that makes the
                    predicted tensor positive definite by construction

Test set: 300 off-grid Latin-hypercube points with mesh-converged values and
derivatives (results/offgrid_truth.npz), never used for training.
Admissibility is checked on a 197 x 197 grid over [0.01, 0.99]^2.

usage:  python exp_newer_models.py [n_seeds] [workers]
Output: results/newer_models.json, results/models/new_*.npz
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS, mean_ci, model_ref, resolve_model  # noqa: E402

from nnto.networks import load_keras_mlp, save_network, load_network  # noqa: E402
from nnto.training import init_mlp, train  # noqa: E402

UP = find_upstream()
MODELS = os.path.join(RESULTS, "models")
os.makedirs(MODELS, exist_ok=True)

ARCHS = {
    "MLP 1x64 sigmoid (paper)": dict(hidden=(64,), activation="sigmoid"),
    "Deep MLP 3x64 tanh": dict(hidden=(64, 64, 64), activation="tanh"),
    "ResNet 4x64 swish": dict(hidden=(64, 64, 64, 64), activation="swish", residual=True),
    "Fourier + 2x64 tanh": dict(hidden=(64, 64), activation="tanh", fourier=16, fourier_scale=2.0),
    "SIREN 3x64": dict(hidden=(64, 64, 64), activation="sin", siren_omega0=5.0),
    "KAN 2-16-16-4": dict(hidden=(16, 16), kan_degree=6),
}


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


def load_data():
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    T1, T2 = np.meshgrid(g["grid"], g["grid"], indexing="ij")
    grid = dict(X=np.stack([T1.ravel(), T2.ravel()], 1), Y=g["C"].reshape(-1, 4), G=g["dC"].reshape(-1, 4, 2))
    m = loadmat(os.path.join(UP, "HBTO_CH_2D_N10000_100x100.mat"), squeeze_me=True)
    paper = dict(X=m["SP_para"][:7000, :2], Y=m["SP_C"][:7000][:, [0, 1, 2, 5]])
    t = np.load(os.path.join(RESULTS, "offgrid_truth.npz"))
    test = dict(X=t["P"], Y=t["C"], G=t["dC"])
    return grid, paper, test


def rel_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def score(net, test):
    X, Y, G = test["X"], test["Y"], test["G"]
    inner = np.all((X >= 0.1) & (X <= 0.9), axis=1)
    Yp = net.forward(X); Jp = net.jacobian_nnj(X)
    f = np.linspace(0.01, 0.99, 197); F1, F2 = np.meshgrid(f, f, indexing="ij")
    C = net.forward(np.stack([F1.ravel(), F2.ravel()], 1))
    not_pd = (C[:, 0] <= 0) | (C[:, 2] <= 0) | (C[:, 3] <= 0) | (C[:, 0] * C[:, 2] - C[:, 1] ** 2 <= 0)
    return dict(fn=rel_rmse(Yp, Y), d=rel_rmse(Jp, G),
                fn_interior=rel_rmse(Yp[inner], Y[inner]), d_interior=rel_rmse(Jp[inner], G[inner]),
                frac_not_pd=float(np.mean(not_pd)), n_params=int(net.n_params()))


def run_job(job):
    arch, regime, seed = job
    grid, paper, test = load_data()
    kw = dict(ARCHS[arch])
    if regime == "paper data":
        D = paper
        net = init_mlp(2, 4, X=D["X"], Y=D["Y"], seed=seed, **kw)
        t0 = time.time(); r = train(net, D["X"], D["Y"], maxiter=2000)
    else:
        D = grid
        if regime == "converged + Sobolev + Cholesky":
            kw["head"] = "cholesky"
        net = init_mlp(2, 4, X=D["X"], Y=D["Y"], seed=seed, **kw)
        lam = 0.0 if regime == "converged, values only" else 1.0
        t0 = time.time(); r = train(net, D["X"], D["Y"], D["G"], lam=lam, maxiter=2000)
    out = score(net, test)
    out.update(arch=arch, regime=regime, seed=seed, seconds=time.time() - t0, n_iter=int(r.n_iter))
    path = os.path.join(MODELS, f"new_{slug(arch)}__{slug(regime)}__s{seed}.npz")
    save_network(path, net)
    out["path"] = model_ref(path)
    return out


class Ensemble:
    """Average of several networks (values and Jacobians)."""

    def __init__(self, nets):
        self.nets = nets
        self.xdim, self.ydim = nets[0].xdim, nets[0].ydim

    def forward(self, x):
        return np.mean([n.forward(x) for n in self.nets], axis=0)

    __call__ = forward

    def jacobian_nnj(self, x):
        return np.mean([n.jacobian_nnj(x) for n in self.nets], axis=0)

    def n_params(self):
        return sum(n.n_params() for n in self.nets)


def main(n_seeds=3, workers=2):
    grid, paper, test = load_data()
    regimes = ["paper data", "converged, values only", "converged + Sobolev"]
    jobs = [(a, r, s) for r in regimes for a in ARCHS for s in range(n_seeds)]
    jobs += [(a, "converged + Sobolev + Cholesky", s) for a in ("Deep MLP 3x64 tanh", "KAN 2-16-16-4")
             for s in range(n_seeds)]
    t0 = time.time()
    runs = []
    # resume: keep runs that finished before an interruption
    prev = os.path.join(RESULTS, "newer_models_runs.json")
    if os.path.exists(prev):
        import json
        runs = [r for r in json.load(open(prev))["runs"] if os.path.exists(resolve_model(r["path"]))]
        done = {(r["arch"], r["regime"], r["seed"]) for r in runs}
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(runs)} runs already done, {len(jobs)} to go", flush=True)
    pub = load_keras_mlp(os.path.join(UP, "NN_model_HBTO_1e+04"))
    published = score(pub, test)
    print("published model:", {k: round(v, 4) for k, v in published.items()}, flush=True)
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            runs.append(r)
            print(f"[{len(runs)}/{len(runs) - k - 1 + len(jobs)}] {r['arch']:26s} | {r['regime']:31s} | s{r['seed']} fn {r['fn']:.4f} d {r['d']:.4f} "
                  f"(interior {r['fn_interior']:.4f} / {r['d_interior']:.4f}) notPD {r['frac_not_pd']:.3f} "
                  f"{r['seconds']:.0f}s  [{time.time() - t0:.0f}s]", flush=True)
            save_json("newer_models_runs.json", dict(published=published, runs=runs))
    summary = {}
    for (a, rg) in sorted(set((r["arch"], r["regime"]) for r in runs)):
        sel = [r for r in runs if r["arch"] == a and r["regime"] == rg]
        agg = {m: dict(zip(("mean", "sd", "ci95"), mean_ci([r[m] for r in sel])))
               for m in ("fn", "d", "fn_interior", "d_interior", "frac_not_pd", "seconds")}
        ens = Ensemble([load_network(resolve_model(r["path"])) for r in sel])
        agg["ensemble"] = score(ens, test)
        agg["n_params"] = sel[0]["n_params"]
        agg["n"] = len(sel)
        summary[f"{a} | {rg}"] = agg
    save_json("newer_models.json", dict(published=published, summary=summary, runs=runs))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
