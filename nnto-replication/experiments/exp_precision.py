"""Derivative accuracy versus step size and floating-point precision.

Reference: the analytic Jacobian (NNJ) in float64, which agrees with forward-
and reverse-mode AD and with the complex step to ~1e-15 (tests/).

Reported for the two published networks:
  * CFD error vs step size h in four settings
        f64        everything in double precision
        f32-input  inputs rounded to float32, network in float64, nominal step
                   (= the published implementation; Keras predict cast)
        f32-fixed  same, but divided by the step actually taken
        f32-all    network weights and arithmetic in float32
  * CSM error vs step size in complex128 and complex64
  * AD / NNJ error when the whole network runs in float32
  * the uniform sensitivity scaling the float32 cast produces at the paper's
    initial designs and the MSE values it implies (Figs. 8 and 11).

Output: results/precision.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json  # noqa: E402

from nnto import derivatives as D  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402

UP = find_upstream()
RES = os.path.join(UP, "JMD Journal Paper Results")


def relerr(J, Jref):
    scale = np.abs(Jref).max(axis=0, keepdims=True)          # per (output, input) entry
    e = np.abs(J - Jref) / scale
    return float(e.max()), float(np.sqrt(np.mean(e ** 2)))


def main():
    out = {}
    rng = np.random.default_rng(0)
    for name in ("DBTO", "HBTO"):
        net = load_keras_mlp(os.path.join(UP, f"NN_model_{name}_1e+04"))
        net32 = net.astype(np.float32)
        x = rng.uniform(0.0, 1.0, size=(4000, net.xdim))
        x32 = x.astype(np.float32)
        J = D.nnj(net, x)
        J_at32 = D.nnj(net, x32.astype(float))                 # reference at the rounded points
        res = {"cfd": {}, "csm": {}}
        for e in range(-12, 0):
            h = 10.0 ** e
            res["cfd"][f"1e{e}"] = {
                "f64": relerr(D.cfd(net, x, h), J),
                "f32-input": relerr(D.cfd(net, x, h, np.float32, exact_step=False), J),
                "f32-fixed": relerr(D.cfd(net, x, h, np.float32, exact_step=True), J),
                "f32-all": relerr(D.cfd(net32, x32, h, exact_step=True), J_at32),
            }
        for e in (-4, -8, -12, -20, -30, -100, -200):
            h = 10.0 ** e
            res["csm"][f"1e{e}"] = {"c128": relerr(D.csm(net, x, h), J)}
            if e >= -30:
                res["csm"][f"1e{e}"]["c64"] = relerr(D.csm(net32, x32, h), J_at32)
        res["ad_float32"] = {"NNJ": relerr(D.nnj(net32, x32), J_at32),
                             "AD-fwd": relerr(D.ad_forward(net32, x32), J_at32),
                             "AD-rev": relerr(D.ad_reverse(net32, x32), J_at32)}
        res["agreement_f64"] = {m: relerr(f(net, x), J) for m, f in
                                [("AD-fwd", D.ad_forward), ("AD-rev", D.ad_reverse),
                                 ("CSM h=1e-12", lambda n, v: D.csm(n, v, 1e-12))]}
        hopt = D.optimal_cfd_step(x)
        res["cfd_optimal_step"] = relerr(D.cfd(net, x, hopt), J)
        Jr, Er = D.cfd_ridders(net, x)
        res["cfd_ridders"] = relerr(Jr, J)
        # error of the published CFD as a function of theta (binade structure), 1-D only
        if net.xdim == 1:
            t = np.linspace(0.001, 0.999, 1997).reshape(-1, 1)
            ratio = D.cfd(net, t, 1e-6, np.float32, False)[:, 0, 0] / D.nnj(net, t)[:, 0, 0]
            bins = [(0.0, 0.03125), (0.03125, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0)]
            res["published_cfd_error_by_binade"] = {
                f"[{a},{b})": [float(np.min(ratio[(t[:, 0] >= a) & (t[:, 0] < b)] - 1)),
                               float(np.max(ratio[(t[:, 0] >= a) & (t[:, 0] < b)] - 1))] for a, b in bins}
        out[name] = res

    # --- the numbers printed in the paper's Figs. 8 and 11
    fig = {}
    for x0, tag, d1, d2 in [(0.5, "DBTO", "DBTO_NN_result_dy1/DBTO_NN_dy1.npz", "DBTO_NN_result_dy2/DBTO_NN_dy2.npz"),
                            (np.sqrt(0.5), "HBTO", "HBTO_NN_result_dy1/HBTO_NN_dy1_it1225final.npz",
                             "HBTO_NN_result_dy2/HBTO_NN_dy2_it1105final.npz")]:
        ratio = (float(np.float32(x0 + 1e-6)) - float(np.float32(x0 - 1e-6))) / 2e-6
        a = np.load(os.path.join(RES, d1))["dc0"]; b = np.load(os.path.join(RES, d2))["dc0"]
        fig[tag] = {"predicted_step_ratio": ratio,
                    "stored_ratio_mean": float(np.mean(b / a)), "stored_ratio_std": float(np.std(b / a)),
                    "stored_mse_cfd_vs_nnj": float(np.mean((a - b) ** 2)),
                    "mse_predicted_from_ratio": float((ratio - 1) ** 2 * np.mean(a ** 2))}
    out["paper_figures"] = fig
    save_json("precision.json", out)

    # ------------------------------------------------------------- console
    for name in ("DBTO", "HBTO"):
        print(f"\n{name}: CFD max relative error vs step (reference NNJ, float64)")
        print("   h        f64        f32-input(published)  f32-fixed   f32-all")
        for k, v in out[name]["cfd"].items():
            print(f"   {k:6s}  {v['f64'][0]:.2e}   {v['f32-input'][0]:.2e}            {v['f32-fixed'][0]:.2e}   {v['f32-all'][0]:.2e}")
        print("   CSM:", {k: {kk: f"{vv[0]:.1e}" for kk, vv in v.items()} for k, v in out[name]["csm"].items()})
        print("   float32 network: ", {k: f"{v[0]:.1e}" for k, v in out[name]["ad_float32"].items()})
        print("   float64 agreement:", {k: f"{v[0]:.1e}" for k, v in out[name]["agreement_f64"].items()})
        print("   CFD optimal step: %.1e   Ridders: %.1e" % (out[name]["cfd_optimal_step"][0], out[name]["cfd_ridders"][0]))
    print("\npublished CFD error by float32 binade (DBTO):", out["DBTO"]["published_cfd_error_by_binade"])
    print("paper figures:", fig)


if __name__ == "__main__":
    main()
