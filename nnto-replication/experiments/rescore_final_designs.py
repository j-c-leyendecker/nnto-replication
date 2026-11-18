"""Re-score every saved final HBTO design and produce the final comparison
numbers (added 2026-11-18, see FIXES_2026-11-18.md).

Nothing is re-optimised: the designs are the ones already saved in results/
(to_baselines.npz, newer_models_to*.npz) and in the authors' repository.

For the authors' four published designs:
    reported compliance, the network's recomputed value, the voxel model,
    the converged reference (sides below 0.01 clamped to a 1 % slit), and the
    converged reference with sides below 0.01 counted as solid.

For every MMA run:
    true compliance (recomputed and checked against the stored value), the
    volume fraction and its violation, and the true compliance after scaling
    both hole fields to the exact volume target.

Summaries report the mean absolute model error as well as the signed mean.

usage:  python rescore_final_designs.py
Output: results/final_designs_rescored.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, find_upstream, mean_ci, save_json  # noqa: E402

from nnto.fea import C0_VEC_EXACT, mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SplineMaterial, VoxelTableMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.scoring import SmallHolesSolid, error_summary, hole_volume_fraction, project_volume  # noqa: E402
from nnto.topopt import ComplianceProblem  # noqa: E402

UP = find_upstream()
RES = os.path.join(UP, "JMD Journal Paper Results")
RUN_FILES = ["to_baselines", "newer_models_to", "newer_models_to2", "newer_models_to3", "newer_models_to4"]
IMPROVED = "Deep MLP 3x64 tanh | converged + Sobolev"
VOLFRAC = 0.5


def stat(v):
    v = [float(a) for a in v]
    m, sd, hw = mean_ci(v) if len(v) > 1 else (v[0], float("nan"), float("nan"))
    return dict(mean=m, sd=sd, ci95=hw, min=min(v), max=max(v), n=len(v), values=v)


def welch(a, b):
    """Welch test and 95 % CI of mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    d = a.mean() - b.mean()
    hw = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return dict(diff=float(d), ci95=[float(d - hw), float(d + hw)], df=float(df),
                p=float(stats.ttest_ind(a, b, equal_var=False).pvalue))


def main():
    mesh = mbb_half()
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    v = np.load(os.path.join(RESULTS, "voxel_lattice.npz"))
    spl = SplineMaterial(g["grid"], g["C"])
    vox = VoxelTableMaterial(v["grid"], v["C"])
    solid = SmallHolesSolid(spl, C0_VEC_EXACT)
    score = lambda mat, x: ComplianceProblem(mesh, mat, kind="hole", filter="none").compliance(x)
    out = {}

    # --- the authors' published designs
    net = load_keras_mlp(os.path.join(UP, "NN_model_HBTO_1e+04"))
    fin = {"NNJ": (1, "it1225final"), "CFD": (2, "it1105final"), "CSM": (3, "it1371final"), "AD": (4, "it1342final")}
    auth = {}
    for name, (d, tag) in fin.items():
        z = np.load(os.path.join(RES, f"HBTO_NN_result_dy{d}", f"HBTO_NN_dy{d}_{tag}.npz"))
        x = np.concatenate([z["xopt1"].flatten("F"), z["xopt2"].flatten("F")])
        X = x.reshape(-1, 2, order="F")
        r = dict(reported=float(z["comp"]), network_recomputed=score(NNMaterial(net, "NNJ"), x),
                 voxel=score(vox, x), converged_slit=score(spl, x), converged_solid=score(solid, x),
                 iterations=int(z["niter"]), volume_fraction=hole_volume_fraction(X),
                 frac_cells_side_below_0p01=float(np.mean(solid.mask(X))), min_theta=float(X.min()))
        for k in ("voxel", "converged_slit", "converged_solid"):
            r[f"{k}_vs_reported_pct"] = 100 * (r[k] / r["reported"] - 1)
            r[f"network_error_vs_{k}_pct"] = 100 * (r["reported"] / r[k] - 1)
        auth[name] = r
    out["authors_designs"] = auth

    # --- every MMA run saved in results/
    runs = []
    for f in RUN_FILES:
        pj, pz = os.path.join(RESULTS, f + ".json"), os.path.join(RESULTS, f + ".npz")
        if not (os.path.exists(pj) and os.path.exists(pz)):
            continue
        meta, z = json.load(open(pj)), np.load(pz)
        for m, lab, seed, x in zip(meta, z["labels"], z["seeds"], z["X"]):
            assert m["label"] == str(lab) and m["seed"] == int(seed), (f, m["label"], lab)
            Xd = x.reshape(-1, 2, order="F")
            P = np.stack([mesh.H @ Xd[:, j] / mesh.Hs for j in range(2)], axis=1)
            c_true = score(spl, P.flatten("F"))
            Pp, s = project_volume(P, VOLFRAC, lb=0.01, ub=0.99)
            vf = hole_volume_fraction(P)
            runs.append(dict(file=f, label=m["label"], seed=m["seed"], noise=m.get("noise", 0.0),
                             true_stored=m["compliance_true"], true=c_true,
                             recompute_rel_diff=abs(c_true / m["compliance_true"] - 1),
                             surrogate=m["compliance_model"], model_error_pct=m["model_error_pct"],
                             nit=m["nit"], converged=m["converged"], volume_fraction=vf,
                             volume_violation=vf - VOLFRAC, projection_scale=s,
                             true_at_target_volume=score(spl, Pp.flatten("F")),
                             volume_after_projection=hole_volume_fraction(Pp)))
    by = {}
    for lab in dict.fromkeys(r["label"] for r in runs):
        sel = [r for r in runs if r["label"] == lab]
        by[lab] = dict(n=len(sel), true=stat([r["true"] for r in sel]),
                       true_at_target_volume=stat([r["true_at_target_volume"] for r in sel]),
                       model_error=error_summary([r["model_error_pct"] for r in sel]),
                       iterations=[r["nit"] for r in sel], converged=sum(r["converged"] for r in sel),
                       volume_fraction=[min(r["volume_fraction"] for r in sel), max(r["volume_fraction"] for r in sel)])
    out["mma_runs"] = dict(runs=runs, by_label=by,
                           max_recompute_rel_diff=max(r["recompute_rel_diff"] for r in runs))

    # --- headline comparisons
    ref = by["reference"]["true"]["mean"]
    pub = [r for r in runs if r["label"] == "published"]
    imp = [r for r in runs if r["label"] == IMPROVED]
    head = {"reference_optimum": ref}
    for key in ("true", "true_at_target_volume"):
        a, b = [r[key] for r in pub], [r[key] for r in imp]
        on_target = min(pub, key=lambda r: abs(r["volume_violation"]))[key]
        head[key] = dict(published=stat(a), improved=stat(b), welch_published_minus_improved=welch(a, b),
                         pct_lower_vs_published_mean=100 * (1 - np.mean(b) / np.mean(a)),
                         pct_lower_vs_published_on_target_run=100 * (1 - np.mean(b) / on_target),
                         improved_vs_reference_pct=[100 * (c / ref - 1) for c in b])
    best = {k: min(auth[n][k] for n in auth) for k in ("voxel", "converged_slit", "converged_solid")}
    head["vs_authors_best_design"] = {k: dict(authors_best=best[k],
                                              pct_lower=100 * (1 - head["true"]["improved"]["mean"] / best[k]))
                                      for k in best}
    head["vs_authors_designs_range_pct_lower"] = {
        k: [100 * (1 - head["true"]["improved"]["mean"] / max(auth[n][k] for n in auth)),
            100 * (1 - head["true"]["improved"]["mean"] / min(auth[n][k] for n in auth))] for k in best}
    e_imp = error_summary([r["model_error_pct"] for r in imp])
    e_auth = {k: auth["NNJ"][f"network_error_vs_{k}_pct"] for k in best}
    head["model_error"] = dict(authors_NNJ_design=e_auth, improved=e_imp,
                               ratio_vs_max_abs={k: abs(e) / e_imp["max_abs"] for k, e in e_auth.items()},
                               ratio_vs_mean_abs={k: abs(e) / e_imp["mean_abs"] for k, e in e_auth.items()})
    head["gap_closed_pct"] = {lab: 100 * (by["published"]["true"]["mean"] - s["true"]["mean"])
                              / (by["published"]["true"]["mean"] - ref) for lab, s in by.items()}
    out["headline"] = head
    save_json("final_designs_rescored.json", out)

    # --- console
    print("Authors' published HBTO designs (compliance; % vs the value reported in the paper)")
    for n, r in auth.items():
        print(f"  {n:4s} reported {r['reported']:.2f} | voxel {r['voxel']:.2f} ({r['voxel_vs_reported_pct']:+.1f}%) | "
              f"converged, sides<0.01 as slits {r['converged_slit']:.2f} ({r['converged_slit_vs_reported_pct']:+.1f}%) | "
              f"converged, sides<0.01 solid {r['converged_solid']:.2f} ({r['converged_solid_vs_reported_pct']:+.1f}%) | "
              f"cells with a side<0.01 {100 * r['frac_cells_side_below_0p01']:.1f}% | volume {r['volume_fraction']:.5f}")
    print(f"\nMMA runs: {len(runs)} re-scored; largest difference to the stored true compliance {out['mma_runs']['max_recompute_rel_diff']:.1e}")
    print(f"{'surrogate':58s} {'n':>2s} {'true':>16s} {'at target vol.':>15s} {'err':>7s} {'|err|':>6s} {'max':>6s} {'conv':>5s} {'volume':>17s}")
    for lab, s in by.items():
        print(f"{lab:58s} {s['n']:2d} {s['true']['mean']:8.3f} ± {s['true']['ci95']:.3f} {s['true_at_target_volume']['mean']:15.3f} "
              f"{s['model_error']['mean_signed']:+6.2f}% {s['model_error']['mean_abs']:5.2f}% {s['model_error']['max_abs']:5.2f}% "
              f"{s['converged']:2d}/{s['n']:<2d} {s['volume_fraction'][0]:.5f}-{s['volume_fraction'][1]:.5f}")
    print("\nHeadline (improved = deep 3x64 tanh MLP, converged data + derivative labels):")
    for key in ("true", "true_at_target_volume"):
        h = head[key]; w = h["welch_published_minus_improved"]
        print(f"  {key:22s}: published {h['published']['mean']:.3f} vs improved {h['improved']['mean']:.3f}; "
              f"{h['pct_lower_vs_published_mean']:.2f}% lower ({h['pct_lower_vs_published_on_target_run']:.2f}% vs the on-target run); "
              f"gap {w['diff']:.3f}, 95% CI {w['ci95'][0]:.2f}-{w['ci95'][1]:.2f}, Welch p {w['p']:.1e}; "
              f"vs reference {', '.join(f'{c:+.3f}%' for c in h['improved_vs_reference_pct'])}")
    for k, d in head["vs_authors_best_design"].items():
        lo, hi = head["vs_authors_designs_range_pct_lower"][k]
        print(f"  vs authors' best design ({k}): {d['authors_best']:.3f} -> {d['pct_lower']:.2f}% lower (all four designs: {lo:.2f}-{hi:.2f}%)")
    me = head["model_error"]
    print(f"  model error: authors' NNJ design {', '.join(f'{k} {e:+.2f}%' for k, e in me['authors_NNJ_design'].items())}; "
          f"improved mean {me['improved']['mean_signed']:+.3f}%, mean |.| {me['improved']['mean_abs']:.3f}%, max |.| {me['improved']['max_abs']:.3f}%; "
          f"ratio {min(me['ratio_vs_max_abs'].values()):.0f}-{max(me['ratio_vs_mean_abs'].values()):.0f}x")
    print("  share of the published -> reference gap closed:", {k: round(v, 1) for k, v in head["gap_closed_pct"].items()})


if __name__ == "__main__":
    main()
