"""Better HBTO surrogates, judged against the converged ground truth.

(1) Deep ensemble on the *published* data (same architecture as the paper,
    different seeds): does my training reproduce the published accuracy, and
    does the ensemble spread indicate where the surrogate is wrong?
(2) Surrogates trained on the mesh-converged data (exact values and exact
    derivative labels at 70 % of the 51 x 51 reference nodes):
        values only          (paper's architecture)
        Sobolev              (paper's architecture + derivative labels)
        Sobolev + Cholesky   (2 x 64 tanh, admissible tensor by construction)
    The remaining 30 % of the nodes are the test set.

Output: results/hbto_surrogates.json and results/models/*.npz
"""
import os
import sys
import time

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS, mean_ci  # noqa: E402

from nnto.networks import load_keras_mlp, save_network  # noqa: E402
from nnto.training import init_mlp, train, predict_ensemble  # noqa: E402

UP = find_upstream()
MODELS = os.path.join(RESULTS, "models")
os.makedirs(MODELS, exist_ok=True)
COMP = ["C11", "C12", "C22", "C66"]


def rel_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def score(net, pts, C, dC, mask=None):
    Cn = net.forward(pts); Jn = net.jacobian_nnj(pts)
    if mask is None:
        mask = np.ones(pts.shape[0], bool)
    r = dict(fn_rel_rmse=rel_rmse(Cn[mask], C[mask]), d_rel_rmse=rel_rmse(Jn[mask], dC[mask]),
             fn_mse=float(np.mean((Cn[mask] - C[mask]) ** 2)), d_mse=float(np.mean((Jn[mask] - dC[mask]) ** 2)))
    for k, c in enumerate(COMP):
        r[f"d_dth1_rel_rmse_{c}"] = rel_rmse(Jn[mask][:, k, 0], dC[mask][:, k, 0])
    det = Cn[:, 0] * Cn[:, 2] - Cn[:, 1] ** 2
    r["frac_not_pd"] = float(np.mean((det <= 0) | (Cn[:, 0] <= 0) | (Cn[:, 3] <= 0)))
    r["frac_wrong_sign"] = float(np.mean((Jn[:, 0, 1] > 0) | (Jn[:, 2, 0] > 0)))
    return r


def main():
    out = {}
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    grid, Ct, dCt = g["grid"], g["C"], g["dC"]
    T1, T2 = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([T1.ravel(), T2.ravel()], 1); C = Ct.reshape(-1, 4); dC = dCt.reshape(-1, 4, 2)
    interior = np.all((pts >= 0.1) & (pts <= 0.9), axis=1)

    pub = load_keras_mlp(os.path.join(UP, "NN_model_HBTO_1e+04"))
    out["published"] = {"all": score(pub, pts, C, dC), "interior": score(pub, pts, C, dC, interior)}

    # ---------------- (2) surrogates trained on the converged data
    rng = np.random.default_rng(0)
    perm = rng.permutation(pts.shape[0]); ntr = int(0.7 * perm.size)
    tr, te = perm[:ntr], perm[ntr:]
    test_mask = np.zeros(pts.shape[0], bool); test_mask[te] = True
    configs = {
        "Sobolev + Cholesky (2x64 tanh)": dict(hidden=(64, 64), activation="tanh", head="cholesky", lam=1.0, seeds=2),
        "values only (1x64 sigmoid)": dict(hidden=(64,), activation="sigmoid", head="denorm", lam=0.0, seeds=3),
        "Sobolev (1x64 sigmoid)": dict(hidden=(64,), activation="sigmoid", head="denorm", lam=1.0, seeds=3),
        "values only (2x64 tanh)": dict(hidden=(64, 64), activation="tanh", head="denorm", lam=0.0, seeds=2),
        "Sobolev (2x64 tanh)": dict(hidden=(64, 64), activation="tanh", head="denorm", lam=1.0, seeds=2),
    }
    out["trained_on_converged_data"] = {}
    for name, cfg in configs.items():
        runs = []
        for s in range(cfg["seeds"]):
            net = init_mlp(2, 4, cfg["hidden"], cfg["activation"], pts[tr], C[tr], seed=s, head=cfg["head"])
            t0 = time.time(); train(net, pts[tr], C[tr], dC[tr], lam=cfg["lam"], maxiter=2500)
            r = {"test": score(net, pts, C, dC, test_mask), "test_interior": score(net, pts, C, dC, test_mask & interior),
                 "seconds": time.time() - t0}
            runs.append(r)
            if s == 0:
                save_network(os.path.join(MODELS, "hbto_" + name.split(" (")[0].replace(" + ", "_").replace(" ", "_").lower() + ".npz"), net)
        agg = {}
        for part in ("test", "test_interior"):
            agg[part] = {k: dict(zip(("mean", "sd", "ci95"), mean_ci([r[part][k] for r in runs]))) for k in runs[0][part]}
        agg["seconds"] = float(np.mean([r["seconds"] for r in runs]))
        out["trained_on_converged_data"][name] = agg
        print(f"{name:34s} test: fn rel-rmse {agg['test']['fn_rel_rmse']['mean']:.4f}  d rel-rmse {agg['test']['d_rel_rmse']['mean']:.4f} "
              f"| interior: fn {agg['test_interior']['fn_rel_rmse']['mean']:.4f} d {agg['test_interior']['d_rel_rmse']['mean']:.4f} "
              f"| not PD {agg['test']['frac_not_pd']['mean']:.4f}  ({agg['seconds']:.0f}s)", flush=True)
        save_json("hbto_surrogates.json", out)
    # ---------------- (1) deep ensemble on the published data
    m = loadmat(os.path.join(UP, "HBTO_CH_2D_N10000_100x100.mat"), squeeze_me=True)
    P, Cd = m["SP_para"][:, :2], m["SP_C"][:, [0, 1, 2, 5]]
    Xtr, Ytr, Xte, Yte = P[:7000], Cd[:7000], P[8500:], Cd[8500:]
    ens, test_mse = [], []
    for s in range(5):
        net = init_mlp(2, 4, (64,), "sigmoid", Xtr, Ytr, seed=s)
        t0 = time.time(); train(net, Xtr, Ytr, maxiter=3000)
        test_mse.append(float(np.mean((net.forward(Xte) - Yte) ** 2)))
        ens.append(net)
        print(f"ensemble member {s}: test MSE vs published labels {test_mse[-1]:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    mu, sd, Jmu, Jsd = predict_ensemble(ens, pts)
    err_f = np.abs(mu - C); err_d = np.abs(Jmu - dC)
    out["ensemble_on_published_data"] = dict(
        test_mse_members=test_mse, test_mse_published_model=float(np.mean((pub.forward(Xte) - Yte) ** 2)),
        spearman_std_vs_error_function=float(spearmanr(sd.ravel(), err_f.ravel()).statistic),
        spearman_std_vs_error_derivative=float(spearmanr(Jsd.ravel(), err_d.ravel()).statistic),
        mean_derivative_std_interior=float(Jsd[interior].mean()), mean_derivative_std_edge=float(Jsd[~interior].mean()),
        mean_derivative_err_interior=float(err_d[interior].mean()), mean_derivative_err_edge=float(err_d[~interior].mean()),
        coverage_2sigma_function=float(np.mean(err_f <= 2 * sd)), coverage_2sigma_derivative=float(np.mean(err_d <= 2 * Jsd)),
        ensemble_mean_vs_truth={"all": dict(fn_rel_rmse=rel_rmse(mu, C), d_rel_rmse=rel_rmse(Jmu, dC)),
                                "interior": dict(fn_rel_rmse=rel_rmse(mu[interior], C[interior]),
                                                 d_rel_rmse=rel_rmse(Jmu[interior], dC[interior]))})

    save_json("hbto_surrogates.json", out)
    p = out["published"]
    print("published model vs truth: all fn %.4f d %.4f | interior fn %.4f d %.4f | not PD %.4f"
          % (p["all"]["fn_rel_rmse"], p["all"]["d_rel_rmse"], p["interior"]["fn_rel_rmse"], p["interior"]["d_rel_rmse"], p["all"]["frac_not_pd"]))
    e = out["ensemble_on_published_data"]
    print("ensemble:", {k: v for k, v in e.items() if k != "ensemble_mean_vs_truth"})


if __name__ == "__main__":
    main()
