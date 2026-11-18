"""Do the paper's conclusions survive other architectures, training regimes
and data-set sizes?  (SIMP target, where exact derivatives are known.)

For every network: accuracy of the function, first and second derivative
against the exact SIMP law; physical admissibility (E >= Emin, E' >= 0); and
the agreement between derivative methods on the *trained* network.

Training: full-batch L-BFGS on Latin-hypercube samples (the paper used Adam,
batch 32, 5000 epochs; L-BFGS reaches a lower loss in seconds and is
deterministic for a given seed).  'Sobolev' adds the exact derivative labels
E'(theta) to the loss with unit relative weight.

Output: results/architectures.json
"""
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, mean_ci  # noqa: E402

from nnto import derivatives as D  # noqa: E402
from nnto.training import init_mlp, train  # noqa: E402

simp = lambda t: 1e-9 + (1 - 1e-9) * t ** 3
dsimp = lambda t: 3 * (1 - 1e-9) * t ** 2
d2simp = lambda t: 6 * (1 - 1e-9) * t

ARCHS = {
    "1x64 sigmoid (paper)": dict(hidden=(64,), activation="sigmoid"),
    "1x64 relu": dict(hidden=(64,), activation="relu"),
    "3x32 tanh": dict(hidden=(32, 32, 32), activation="tanh"),
    "3x32 swish residual": dict(hidden=(32, 32, 32), activation="swish", residual=True),
    "fourier(8) + 2x32 tanh": dict(hidden=(32, 32), activation="tanh", fourier=8, fourier_scale=3.0),
    "1x64 sigmoid + positive head": dict(hidden=(64,), activation="sigmoid", head="softfloor"),
}


def evaluate(net):
    t = np.linspace(0, 1, 4001).reshape(-1, 1)
    E = net.forward(t)[:, 0]; J = D.nnj(net, t); dE = J[:, 0, 0]; d2E = D.hessian_nnj(net, t)[:, 0, 0, 0]
    sc = np.abs(J).max()
    r = dict(fn_rmse=float(np.sqrt(np.mean((E - simp(t[:, 0])) ** 2))),
             d1_rmse=float(np.sqrt(np.mean((dE - dsimp(t[:, 0])) ** 2))),
             d2_rmse=float(np.sqrt(np.mean((d2E - d2simp(t[:, 0])) ** 2))),
             frac_E_below_Emin=float(np.mean(E < 1e-9)), frac_dE_negative=float(np.mean(dE < 0)),
             min_E=float(E.min()),
             csm_vs_nnj=float(np.abs(D.csm(net, t) - J).max() / sc),
             adrev_vs_nnj=float(np.abs(D.ad_reverse(net, t) - J).max() / sc),
             cfd_vs_nnj=float(np.abs(D.cfd(net, t, 1e-6) - J).max() / sc),
             cfd_frac_bad=float(np.mean(np.abs(D.cfd(net, t, 1e-6) - J)[:, 0, 0] / sc > 1e-6)))
    return r


def fit(arch, n_train, sobolev, seed, maxiter=1500):
    X = qmc.LatinHypercube(1, seed=seed).random(n_train)
    Y = simp(X); G = dsimp(X).reshape(-1, 1, 1)
    kw = dict(ARCHS[arch])
    net = init_mlp(1, 1, X=X, Y=Y, seed=seed, **kw)
    t0 = time.time()
    train(net, X, Y, G, lam=1.0 if sobolev else 0.0, maxiter=maxiter)
    r = evaluate(net)
    r["train_seconds"] = time.time() - t0
    return r


def summarise(runs):
    keys = runs[0].keys()
    return {k: dict(zip(("mean", "sd", "ci95"), mean_ci([r[k] for r in runs]))) for k in keys}


def main(seeds=(0, 1, 2)):
    out = {"architectures": {}, "dataset_size": {}}
    for arch in ARCHS:
        for sob in (False, True):
            runs = [fit(arch, 2000, sob, s) for s in seeds]
            tag = f"{arch} | {'Sobolev' if sob else 'values only'}"
            out["architectures"][tag] = summarise(runs)
            m = out["architectures"][tag]
            print(f"{tag:48s} fn {m['fn_rmse']['mean']:.1e} d1 {m['d1_rmse']['mean']:.1e} d2 {m['d2_rmse']['mean']:.1e} "
                  f"E<Emin {m['frac_E_below_Emin']['mean']:.3f} dE<0 {m['frac_dE_negative']['mean']:.3f} "
                  f"CSM {m['csm_vs_nnj']['mean']:.0e} AD {m['adrev_vs_nnj']['mean']:.0e} CFD {m['cfd_vs_nnj']['mean']:.0e} "
                  f"({m['train_seconds']['mean']:.0f}s)", flush=True)
    for n in (70, 700, 7000):
        for sob in (False, True):
            runs = [fit("1x64 sigmoid (paper)", n, sob, s) for s in seeds]
            tag = f"N_train={n} | {'Sobolev' if sob else 'values only'}"
            out["dataset_size"][tag] = summarise(runs)
            m = out["dataset_size"][tag]
            print(f"{tag:48s} fn {m['fn_rmse']['mean']:.1e} d1 {m['d1_rmse']['mean']:.1e} d2 {m['d2_rmse']['mean']:.1e} "
                  f"E<Emin {m['frac_E_below_Emin']['mean']:.3f} dE<0 {m['frac_dE_negative']['mean']:.3f}", flush=True)
    save_json("architectures.json", out)


if __name__ == "__main__":
    main()
