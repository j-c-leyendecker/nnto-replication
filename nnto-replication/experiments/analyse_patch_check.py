"""Summarise results/patch_check.npz (see exp_patch_check.py)."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, mean_ci, save_json  # noqa: E402

z = np.load(os.path.join(RESULTS, "patch_check.npz"))
names, seeds, X, comp, ct, nit = z["names"], z["seeds"], z["X"], z["comp"], z["comp_true"], z["nit"]
runs = json.load(open(os.path.join(RESULTS, "patch_check.json")))
msg = {(r["name"], r["seed"]): r["message"][:6] for r in runs}
out = {"per_config": {}, "divergence": {}, "derivative_error_along_path": {}}

print("Final designs (seed 0 unperturbed + seeds 1-5 with 1e-10 gradient perturbation)")
print(f"{'config':15s} {'n':>2s} {'true compliance mean ± 95% CI':>32s} {'min':>7s} {'max':>7s} {'iterations':>14s} {'gtol stops':>10s}")
for cfg in ("CFD published", "CFD patched", "NNJ patched", "NNJ published"):
    m = names == cfg
    c = ct[m]; it = nit[m]
    if c.size == 0:      # robustness fix 2026-11-18: an interrupted run leaves configs out
        print(f"{cfg:15s}  0   (no runs in patch_check.npz)")
        continue
    g = sum(msg[(cfg, int(s))].startswith("`gtol") for s in seeds[m])
    mc, sd, hw = mean_ci(c) if c.size > 1 else (c[0], np.nan, np.nan)
    mi, sdi, hwi = mean_ci(it) if it.size > 1 else (it[0], np.nan, np.nan)
    low = int(np.sum(c < 81.4))
    out["per_config"][cfg] = dict(n=int(c.size), true_comp=c.tolist(), mean=mc, sd=sd, ci95=hw,
                                  nit=it.tolist(), nit_mean=mi, nit_ci95=hwi, gtol_stops=g,
                                  n_low_cluster=low)
    print(f"{cfg:15s} {c.size:2d} {mc:12.3f} ± {hw:6.3f} (sd {sd:5.3f})   {c.min():7.3f} {c.max():7.3f} {mi:7.1f} ± {hwi:5.1f} {g:5d}/{c.size}"
          f"   low cluster (<81.4): {low}/{c.size}")

for a, b in (("CFD published", "CFD patched"), ("CFD published", "NNJ patched"), ("CFD patched", "NNJ patched")):
    ca, cb = ct[names == a], ct[names == b]
    if ca.size < 2 or cb.size < 2:
        continue
    t = stats.ttest_ind(ca, cb, equal_var=False)
    u = stats.mannwhitneyu(ca, cb)
    out.setdefault("tests", {})[f"{a} vs {b}"] = dict(welch_p=float(t.pvalue), mannwhitney_p=float(u.pvalue),
                                                       mean_diff=float(ca.mean() - cb.mean()))
    print(f"{a} vs {b}: mean difference {ca.mean() - cb.mean():+.3f}, Welch p = {t.pvalue:.2f}, Mann-Whitney p = {u.pvalue:.2f}")

# ---------- how long does each run follow the exact-derivative (NNJ patched) run?
have = lambda cfg, pre: f"{pre}_{cfg.replace(' ', '_')}" in z.files
ref = z["iterates_NNJ_patched"] if "iterates_NNJ_patched" in z.files else None
print("\nDistance of the iterates from the exact-derivative run (max |theta - theta_NNJ|):")
print(f"{'iteration':>9s} " + " ".join(f"{c:>15s}" for c in ("CFD patched", "CFD published", "NNJ published")))
rows = [1, 2, 5, 10, 20, 30, 40, 60, 100, 150, 200]
for cfg in ("CFD patched", "CFD published", "NNJ published"):
    if ref is None or not have(cfg, "iterates"):
        out["divergence"][cfg] = dict(max_abs_diff=[], first_iteration_above_1em3=None, first_iteration_above_1em6=None)
        continue
    it = z[f"iterates_{cfg.replace(' ', '_')}"]
    n = min(len(it), len(ref))
    d = np.abs(it[:n] - ref[:n]).max(axis=1)
    first = int(np.argmax(d > 1e-3)) + 1 if np.any(d > 1e-3) else None
    out["divergence"][cfg] = dict(max_abs_diff=d.tolist(), first_iteration_above_1em3=first,
                                  first_iteration_above_1em6=int(np.argmax(d > 1e-6)) + 1 if np.any(d > 1e-6) else None)
for r in rows:
    vals = []
    for cfg in ("CFD patched", "CFD published", "NNJ published"):
        d = out["divergence"][cfg]["max_abs_diff"]
        vals.append(f"{d[r - 1]:15.2e}" if r - 1 < len(d) else f"{'-':>15s}")
    print(f"{r:9d} " + " ".join(vals))
for cfg in ("CFD patched", "CFD published", "NNJ published"):
    print(f"  {cfg}: first iteration where the design differs by > 1e-6: {out['divergence'][cfg]["first_iteration_above_1em6"]}, by > 1e-3: {out['divergence'][cfg]["first_iteration_above_1em3"]}")

print("\nMax relative error of the network derivative along each run's own path (every 10th iterate):")
for cfg in ("CFD published", "CFD patched", "NNJ patched"):
    if not have(cfg, "derr"):
        print(f"  {cfg:14s} (no runs)")
        continue
    e = z[f"derr_{cfg.replace(' ', '_')}"]
    out["derivative_error_along_path"][cfg] = dict(median=float(np.median(e)), max=float(e.max()), min=float(e.min()))
    print(f"  {cfg:14s} median {np.median(e):.2e}  range [{e.min():.2e}, {e.max():.2e}]")
save_json("patch_check_summary.json", out)
