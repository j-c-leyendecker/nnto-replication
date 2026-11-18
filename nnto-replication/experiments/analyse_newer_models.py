"""Tables for the newer-model study (results/newer_models.json and the
optimisation runs in results/to_baselines.json, results/newer_models_to.json)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, mean_ci  # noqa: E402

from nnto.scoring import error_summary  # noqa: E402

ORDER = ["MLP 1x64 sigmoid (paper)", "Deep MLP 3x64 tanh", "ResNet 4x64 swish",
         "Fourier + 2x64 tanh", "SIREN 3x64", "KAN 2-16-16-4"]
REGIMES = ["paper data", "converged, values only", "converged + Sobolev", "converged + Sobolev + Cholesky"]


def accuracy():
    d = json.load(open(os.path.join(RESULTS, "newer_models.json")))
    pub = d["published"]
    print("Surrogate accuracy on 300 off-grid points (relative RMSE; mean of 3 seeds; ensemble of the 3 in brackets)")
    print(f"{'model':28s} {'training':32s} {'values':>8s} {'derivs, full':>14s} {'derivs, interior':>17s} {'not PD':>7s} {'params':>7s} {'fit s':>6s}")
    print(f"{'published model':28s} {'paper data (as published)':32s} {pub['fn']:8.4f} {pub['d']:14.4f} {pub['d_interior']:17.4f} {pub['frac_not_pd']:7.3f} {pub['n_params']:7d}")
    rows = []
    for rg in REGIMES:
        for a in ORDER:
            k = f"{a} | {rg}"
            if k not in d["summary"]:
                continue
            s = d["summary"][k]
            e = s["ensemble"]
            print(f"{a:28s} {rg:32s} {s['fn']['mean']:8.4f} {s['d']['mean']:8.4f} ({e['d']:.3f}) "
                  f"{s['d_interior']['mean']:8.4f} ({e['d_interior']:.4f}) {s['frac_not_pd']['mean']:7.3f} {s['n_params']:7d} {s['seconds']['mean']:6.0f}")
            rows.append((a, rg, s))
    return d, rows


def optimisation():
    runs = []
    for name in ("to_baselines.json", "newer_models_to.json", "newer_models_to2.json", "newer_models_to3.json", "newer_models_to4.json"):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            runs += json.load(open(p))
    if not runs:
        return
    print("\nMicrostructure optimisation (MMA, consistent formulation); compliance scored with the converged reference")
    print("error = the surrogate's own compliance vs the truth: signed mean, and mean / max absolute value "
          "(signed errors of opposite sign cancel; fix 2026-11-18)")
    print(f"{'surrogate':62s} {'n':>2s} {'true compliance':>22s} {'surrogate says':>15s} {'error':>8s} {'|error|':>8s} "
          f"{'max|e|':>7s} {'iters':>6s} {'volume (min)':>12s}")
    for lab in dict.fromkeys(r["label"] for r in runs):
        sel = [r for r in runs if r["label"] == lab and r.get("noise", 0.0) == 0.0]
        rep = [r for r in runs if r["label"] == lab]
        v = [r["compliance_true"] for r in sel]
        m, sd, hw = mean_ci(v) if len(v) > 1 else (v[0], float("nan"), float("nan"))
        ci = f"± {hw:.3f}" if len(v) > 1 else "      "
        es = error_summary([r["model_error_pct"] for r in sel])
        print(f"{lab:62s} {len(sel):2d} {m:12.3f} {ci:>9s} {np.mean([r['compliance_model'] for r in sel]):15.3f} "
              f"{es['mean_signed']:+7.2f}% {es['mean_abs']:7.2f}% {es['max_abs']:6.2f}% {np.mean([r['nit'] for r in sel]):6.0f} "
              f"{min(r['volume_fraction'] for r in rep):12.5f}"
              + (f"   perturbed repeats: {', '.join(f'{r['compliance_true']:.3f}' for r in rep if r.get('noise', 0) > 0)}"
                 if len(rep) > len(sel) else ""))
    p = os.path.join(RESULTS, "final_designs_rescored.json")
    if os.path.exists(p):
        print("\nAt equal material (both hole fields scaled to the exact volume target), from final_designs_rescored.json:")
        for lab, s in json.load(open(p))["mma_runs"]["by_label"].items():
            print(f"  {lab:60s} true {s['true']['mean']:.3f} -> at target volume {s['true_at_target_volume']['mean']:.3f}")


if __name__ == "__main__":
    if os.path.exists(os.path.join(RESULTS, "newer_models.json")):
        accuracy()
    optimisation()
