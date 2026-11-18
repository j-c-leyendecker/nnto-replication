"""Figures for the report (PNG, written to reports/figures/)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, RESULTS, ROOT  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import SimpMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.topopt import ComplianceProblem  # noqa: E402

# NNTO_FIGURES redirects the figures, e.g. for a verification rerun (added 2026-11-18)
FIG = os.path.abspath(os.environ.get("NNTO_FIGURES", os.path.join(ROOT, "reports", "figures")))
os.makedirs(FIG, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def fig_precision():
    d = load("precision.json")["DBTO"]["cfd"]
    hs = np.array([float(k) for k in d])
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=160)
    series = [("f64", "Double precision", BLUE), ("f32-input", "As published: inputs rounded to float32", ORANGE),
              ("f32-fixed", "float32 inputs, divided by the step actually taken", AQUA)]
    for key, lab, col in series:
        y = np.array([d[k][key][0] if d[k][key][0] is not None else np.nan for k in d], dtype=float)
        ok = np.isfinite(y)
        ax.loglog(hs[ok], y[ok], color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=lab,
                  solid_capstyle="round")
    ax.axvline(1e-6, color=INK2, lw=0.8)
    ax.text(1.25e-6, 3e2, "step used in the paper (h = 1e-6)", color=INK2, fontsize=8.5, va="center")
    # labels computed from the data (the f64 label was a stale hard-coded "2.8e-10" until 18 Nov 2026)
    ax.annotate(f"{100 * d['1e-6']['f32-input'][0]:.2f} %", (1e-6, d["1e-6"]["f32-input"][0]), xytext=(8, 6),
                textcoords="offset points", fontsize=9, color=INK)
    ax.annotate(f"{d['1e-6']['f64'][0]:.1e}", (1e-6, d["1e-6"]["f64"][0]), xytext=(8, -12),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("finite-difference step h"); ax.set_ylabel("max relative error of dE/dθ")
    ax.set_title("Central differences on the published DBTO network")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=9, handlelength=2.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_cfd_precision.png")); plt.close(fig)


def fig_hbto_error():
    up = find_upstream()
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    grid, dC = g["grid"], g["dC"]
    net = load_keras_mlp(os.path.join(up, "NN_model_HBTO_1e+04"))
    T1, T2 = np.meshgrid(grid, grid, indexing="ij"); pts = np.stack([T1.ravel(), T2.ravel()], 1)
    J = net.jacobian_nnj(pts).reshape(51, 51, 4, 2)
    rel = np.linalg.norm((J - dC).reshape(51, 51, -1), axis=2) / np.linalg.norm(dC.reshape(51, 51, -1), axis=2)
    res = os.path.join(up, "JMD Journal Paper Results")
    cells = []
    for d, tag in ((1, "it1225final"), (2, "it1105final"), (3, "it1371final"), (4, "it1342final")):
        z = np.load(os.path.join(res, f"HBTO_NN_result_dy{d}", f"HBTO_NN_dy{d}_{tag}.npz"))
        cells.append(np.stack([z["xopt1"].ravel(), z["xopt2"].ravel()], 1))
    cells = np.concatenate(cells)
    Hc, _, _ = np.histogram2d(cells[:, 0], cells[:, 1], bins=25, range=[[0, 1], [0, 1]])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=160)
    im = axs[0].pcolormesh(grid, grid, np.minimum(rel, 2.0).T, cmap=SEQ, vmin=0, vmax=2.0, shading="nearest")
    axs[0].set_title("Relative error of the surrogate's Jacobian")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03); cb.set_label("‖J_NN − J_true‖ / ‖J_true‖  (capped at 2)", color=INK2)
    im2 = axs[1].pcolormesh(np.linspace(0, 1, 26), np.linspace(0, 1, 26), 100 * Hc.T / cells.shape[0], cmap=SEQ, shading="flat")
    axs[1].set_title("Where the published final designs sit")
    cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.03); cb2.set_label("% of cells (four designs, 7 200 cells)", color=INK2)
    for ax in axs:
        ax.set_xlabel("hole width θ1"); ax.set_ylabel("hole height θ2"); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_hbto_derivative_error.png")); plt.close(fig)


def fig_dbto_runs():
    up = find_upstream(); res = os.path.join(up, "JMD Journal Paper Results")
    mesh = mbb_half(); true = ComplianceProblem(mesh, SimpMaterial(), filter="none")
    paper = {"Exact SIMP": "DBTO_SIMP_result/DBTO_SIMP.npz", "NNJ": "DBTO_NN_result_dy1/DBTO_NN_dy1.npz",
             "CFD (as published)": "DBTO_NN_result_dy2/DBTO_NN_dy2.npz", "CSM": "DBTO_NN_result_dy3/DBTO_NN_dy3.npz",
             "AD": "DBTO_NN_result_dy4/DBTO_NN_dy4.npz"}
    paper_true = {k: true.compliance(np.load(os.path.join(res, v))["xopt"].flatten("F")) for k, v in paper.items()}
    # the shipped results kept the first 8 runs of an interrupted ensemble in dbto_ensemble_part1.npz;
    # a complete rerun writes dbto_ensemble.npz (fallback added 2026-11-18)
    p1 = os.path.join(RESULTS, "dbto_ensemble_part1.npz")
    z1 = np.load(p1 if os.path.exists(p1) else os.path.join(RESULTS, "dbto_ensemble.npz"))
    z2 = np.load(os.path.join(RESULTS, "dbto_part2.npz"))
    lab = np.concatenate([z1["labels"], z2["labels"]]); ct = np.concatenate([z1["comp_true"], z2["comp_true"]])
    mine = {"Exact SIMP": "SIMP", "NNJ": "NNJ", "CFD (as published)": "CFD-f32", "CSM": "CSM", "AD": "AD-rev",
            "CFD (double precision)": "CFD"}
    rows = ["Exact SIMP", "NNJ", "CSM", "AD", "CFD (as published)", "CFD (double precision)",
            "NNJ, gradient perturbed by 1e-10", "Exact SIMP, gradient perturbed by 1e-10"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r in mine:
            ax.plot(ct[lab == mine[r]], [y] * int(np.sum(lab == mine[r])), "o", color=BLUE, ms=9, mec=SURFACE, mew=2,
                    label="this study, one run" if i == 0 else None, zorder=3)
        if r in paper_true:
            ax.plot([paper_true[r]], [y], "s", color=ORANGE, ms=8.5, mec=SURFACE, mew=2,
                    label="authors' saved design" if i == 0 else None, zorder=3)
        key = {"NNJ, gradient perturbed by 1e-10": "NNJ+1e-10", "Exact SIMP, gradient perturbed by 1e-10": "SIMP+1e-10"}.get(r)
        if key:
            v = ct[lab == key]
            ax.plot(v, [y] * v.size, "o", color=AQUA, ms=9, mec=SURFACE, mew=2,
                    label="this study, repeated with perturbed gradient" if key == "NNJ+1e-10" else None, zorder=3)
            ax.text(v.max() + 0.06, y, f"n = {v.size}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows[::-1]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("compliance of the final design, re-evaluated with the exact SIMP law")
    ax.set_title("Density-based MBB beam: where single runs end")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3, fontsize=8.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_dbto_runs.png")); plt.close(fig)


def fig_hbto_reported_vs_true():
    # Since 2026-11-18: if results/final_designs_rescored.json exists, the orange bar is the
    # conservative value (hole sides below 0.01 counted as solid, as in the authors' voxel
    # data) and the whisker reaches the value with such sides scored as thin slits.
    p = os.path.join(RESULTS, "final_designs_rescored.json")
    if os.path.exists(p):
        d = json.load(open(p))["authors_designs"]
        names = list(d)
        rep = [d[n]["reported"] for n in names]
        tru = [d[n]["converged_solid"] for n in names]; hi = [d[n]["converged_slit"] for n in names]
        lab = "same design, mesh-converged model (bar: holes < 1 % solid; whisker: as thin slits)"
    else:
        d = load("surrogate_truth.json")["HBTO"]["final_designs"]
        names = list(d)
        rep = [d[n]["compliance_reported"] for n in names]; tru = [d[n]["compliance_converged_truth"] for n in names]
        hi = None
        lab = "same design, mesh-converged material model"
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=160)
    x = np.arange(len(names)); w = 0.3
    b1 = ax.bar(x - w / 2 - 0.02, rep, w, color=BLUE, label="reported (evaluated by the surrogate)")
    b2 = ax.bar(x + w / 2 + 0.02, tru, w, color=ORANGE, label=lab)
    if hi is not None:
        for b, h in zip(b2, hi):
            xc = b.get_x() + b.get_width() / 2
            ax.plot([xc, xc], [b.get_height(), h], color=INK, lw=1.2)
            ax.plot([xc - 0.05, xc + 0.05], [h, h], color=INK, lw=1.2)
    for k, bars in enumerate((b1, b2)):
        for j, b in enumerate(bars):
            top = b.get_height() if (k == 0 or hi is None) else hi[j]
            txt = f"{b.get_height():.1f}" if (k == 0 or hi is None) else f"{b.get_height():.1f}-{hi[j]:.1f}"
            ax.text(b.get_x() + b.get_width() / 2, top + 1.0, txt, ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("compliance"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    ax.set_title("Homogenization-based designs: reported vs true compliance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_hbto_reported_vs_true.png")); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["precision", "hbto_error", "dbto_runs", "hbto_reported"]
    for w in which:
        {"precision": fig_precision, "hbto_error": fig_hbto_error, "dbto_runs": fig_dbto_runs,
         "hbto_reported": fig_hbto_reported_vs_true}[w]()
        print("wrote", w)
