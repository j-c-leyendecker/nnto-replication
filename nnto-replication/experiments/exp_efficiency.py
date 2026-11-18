"""Computational cost of the derivative methods and of the optimisation loop.

1. Derivatives of the published networks for all 1800 elements:
   the original routine (``dcfun``, run verbatim through the NumPy stand-in
   for Keras) versus the vectorised implementations of this package.
   NOTE: the stand-in has no TensorFlow call overhead, so the "original"
   column measures the algorithmic structure only; the authors' own
   TensorFlow timings (paper Table 2) are quoted for reference.
2. Memory needed by the original AD path (full cross-batch Jacobian) versus
   a batch Jacobian.
3. One objective + gradient evaluation: original functions versus vectorised.
4. trust-constr iteration time with dense (as published) and sparse
   constraint Jacobians, for three mesh sizes.
5. Gradient of a scalar function of the network outputs versus input
   dimension: reverse-mode needs one sweep, the other methods need O(xdim).

Output: results/efficiency.json
"""
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json  # noqa: E402

from nnto import autodiff as ad  # noqa: E402
from nnto import derivatives as D  # noqa: E402
from nnto.baseline.keras_shim import ShimModel  # noqa: E402
from nnto.baseline.original import load_original_namespace, setup_mbb  # noqa: E402
from nnto.fea import Mesh, mbb_half  # noqa: E402
from nnto.materials import NNMaterial  # noqa: E402
from nnto.networks import Dense, Network, load_keras_mlp  # noqa: E402
from nnto.topopt import ComplianceProblem, run_trust_constr  # noqa: E402

UP = find_upstream()
PAPER_TABLE2 = {"DBTO": {"NNJ": 2.741, "CFD": 0.342, "CSM": 0.088, "AD": 2.568},
                "HBTO": {"NNJ": 2.988, "CFD": 0.682, "CSM": 0.180, "AD": 8.553}}


def best_of(f, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    out = {"paper_table2_seconds": PAPER_TABLE2}
    rng = np.random.default_rng(0)

    # ---- 1. derivative routines, N = 1800
    out["derivatives_N1800"] = {}
    for name, script in (("DBTO", "Run_DBTO_NN.py"), ("HBTO", "Run_HBTO_NN.py")):
        net = load_keras_mlp(os.path.join(UP, f"NN_model_{name}_1e+04"))
        ns = load_original_namespace(os.path.join(UP, script))
        model, _ = setup_mbb(ns, hbto=(name == "HBTO"))
        shim = ShimModel(net)
        x = rng.uniform(0.05, 0.95, size=(1800, net.xdim))
        res = {}
        for der, lab in ((1, "NNJ"), (2, "CFD"), (3, "CSM"), (4, "AD")):
            model.der_type = der
            res[f"original {lab}"] = best_of(lambda: _orig_dcfun(ns, model, x, shim),
                                             n=2 if der == 4 else 3)
        for lab, f in (("NNJ", lambda: D.nnj(net, x)), ("CFD", lambda: D.cfd(net, x, 1e-6)),
                       ("CSM", lambda: D.csm(net, x)), ("AD-fwd", lambda: D.ad_forward(net, x)),
                       ("AD-rev", lambda: D.ad_reverse(net, x))):
            res[f"vectorised {lab}"] = best_of(f, n=20)
        out["derivatives_N1800"][name] = res
        ydim, xdim = net.ydim, net.xdim
        out.setdefault("ad_memory_bytes", {})[name] = dict(
            cross_batch_jacobian=1800 * ydim * 1800 * xdim * 8, batch_jacobian=1800 * ydim * xdim * 8)

    # ---- 3. objective + gradient evaluation
    net = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))
    ns = load_original_namespace(os.path.join(UP, "Run_DBTO_NN.py"))
    model, x0 = setup_mbb(ns); model.nn_model = ShimModel(net); model.der_type = 1
    x = rng.uniform(0.2, 1.0, 1800)
    t_orig = best_of(lambda: (ns["feafun"](x, model), ns["dfeafun"](x, model)), n=2)
    prob = ComplianceProblem(mbb_half(), NNMaterial(net, "NNJ"))

    def new():
        prob._cache_x = None
        prob.compliance(x); prob.gradient(x)
    out["objective_plus_gradient_seconds"] = dict(original=t_orig, vectorised=best_of(new, n=10))

    # ---- 4. optimiser iteration time, dense vs sparse constraint Jacobians
    out["trust_constr_seconds_per_iteration"] = {}
    for nelx, nely, its_dense in ((30, 15, 8), (40, 20, 6), (60, 30, 3)):
        mesh = Mesh(nelx, nely, mbb_half(nelx, nely).fixed, mbb_half(nelx, nely).F, 2.7)
        row = {}
        for mode, its in (("dense", its_dense), ("sparse", 20)):
            p = ComplianceProblem(mesh, NNMaterial(net, "NNJ"))
            t = time.perf_counter()
            r = run_trust_constr(p, maxiter=its, sparse_jacobian=(True if mode == "sparse" else None))
            row[mode] = (time.perf_counter() - t) / r["nit"]
        row["speedup"] = row["dense"] / row["sparse"]
        out["trust_constr_seconds_per_iteration"][f"{nelx}x{nely}"] = row
        print(f"trust-constr {nelx}x{nely}: dense {row['dense']:.3f} s/it, sparse {row['sparse']:.3f} s/it, x{row['speedup']:.0f}", flush=True)

    # ---- 5. gradient of a scalar objective vs input dimension
    out["scalar_gradient_vs_input_dim"] = {}
    for xdim in (1, 4, 16, 64, 256):
        W1 = rng.normal(size=(xdim, 64)) / np.sqrt(xdim); W2 = rng.normal(size=(64, 64)) / 8; W3 = rng.normal(size=(64, 4)) / 8
        netd = Network([Dense(W1, np.zeros(64), "tanh"), Dense(W2, np.zeros(64), "tanh"), Dense(W3, np.zeros(4), "linear")], xdim, 4)
        xb = rng.normal(size=(256, xdim)); w = rng.normal(size=(256, 4))
        row = {"reverse (1 VJP)": best_of(lambda: ad.vjp(netd.forward, xb, w), n=5),
               "forward (xdim JVPs)": best_of(lambda: [ad.jvp(netd.forward, xb, np.eye(xdim)[j][None].repeat(256, 0)) for j in range(xdim)], n=2),
               "NNJ (full Jacobian)": best_of(lambda: D.nnj(netd, xb), n=3),
               "CFD (2 xdim evals)": best_of(lambda: D.cfd(netd, xb, 1e-6), n=2),
               "CSM (xdim evals)": best_of(lambda: D.csm(netd, xb), n=2)}
        out["scalar_gradient_vs_input_dim"][str(xdim)] = row
    save_json("efficiency.json", out)
    for name, res in out["derivatives_N1800"].items():
        print(name, {k: f"{v * 1e3:.2f} ms" for k, v in res.items()})
    print("AD memory:", out["ad_memory_bytes"])
    print("objective+gradient:", out["objective_plus_gradient_seconds"])
    for k, v in out["scalar_gradient_vs_input_dim"].items():
        print("xdim", k, {a: f"{b * 1e3:.1f} ms" for a, b in v.items()})


def _orig_dcfun(ns, model, x, shim):
    """The original module-level ``dcfun(x_test, nn_model)`` (it reads the
    derivative type from the global ``model`` class)."""
    return ns["dcfun"](x, shim)


if __name__ == "__main__":
    main()
