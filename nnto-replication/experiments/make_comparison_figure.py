"""Older vs newer microstructure (HBTO) results, side by side.

Panels: the authors' saved NNJ design; the published network re-optimised
with the same optimiser and formulation as the newer model; the newer model
(deep 3x64 tanh network trained with exact derivative labels, seed 0 -- the
seed fixed in advance, not the best one); the reference model's own optimum.
Each cell is drawn as the unit cell with its rectangular hole (width theta1,
height theta2), anti-aliased by exact pixel coverage.

Also prints the starting-design sensitivity error of both networks.
Output: reports/figures/fig5_older_vs_newer.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, RESULTS, ROOT, resolve_model  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SplineMaterial  # noqa: E402
from nnto.networks import load_keras_mlp, load_network  # noqa: E402
from nnto.topopt import ComplianceProblem  # noqa: E402

UP = find_upstream()
# NNTO_FIGURES redirects the figures, e.g. for a verification rerun (added 2026-11-18)
FIG = os.path.abspath(os.environ.get("NNTO_FIGURES", os.path.join(ROOT, "reports", "figures")))
os.makedirs(FIG, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SOLID = "#33322f"


def cell_image(t1, t2, k=14):
    """(nely*k, nelx*k) image, 1 = solid, 0 = void, from hole-size fields (nely, nelx)."""
    nely, nelx = t1.shape
    edges = np.arange(k + 1, dtype=float)

    def overlap(w):
        a = (k - w * k) / 2.0
        b = (k + w * k) / 2.0
        lo = np.maximum(edges[:-1][None, :], a.reshape(-1, 1))
        hi = np.minimum(edges[1:][None, :], b.reshape(-1, 1))
        return np.clip(hi - lo, 0.0, 1.0)                      # (ncell, k)

    ox = overlap(t1.ravel()); oy = overlap(t2.ravel())
    hole = oy[:, :, None] * ox[:, None, :]                    # (ncell, k, k)
    img = 1.0 - hole
    return img.reshape(nely, nelx, k, k).transpose(0, 2, 1, 3).reshape(nely * k, nelx * k)


def physical(x, mesh):
    X = np.asarray(x).reshape(-1, 2, order="F")
    P = np.stack([mesh.H @ X[:, j] / mesh.Hs for j in range(2)], axis=1)
    return (P[:, 0].reshape(mesh.nely, mesh.nelx, order="F"),
            P[:, 1].reshape(mesh.nely, mesh.nelx, order="F"))


def pick(npz, label, seed):
    z = np.load(os.path.join(RESULTS, npz))
    i = [j for j, (l, s) in enumerate(zip(z["labels"], z["seeds"])) if l == label and s == seed][0]
    return z["X"][i]


def main():
    import json
    mesh = mbb_half()
    runs = {}
    for f in ("to_baselines", "newer_models_to"):
        for r in json.load(open(os.path.join(RESULTS, f + ".json"))):
            runs[(r["label"], r["seed"])] = r
    newer_label = "Deep MLP 3x64 tanh | converged + Sobolev"
    a = np.load(os.path.join(UP, "JMD Journal Paper Results", "HBTO_NN_result_dy1", "HBTO_NN_dy1_it1225final.npz"))
    st = json.load(open(os.path.join(RESULTS, "surrogate_truth.json")))["HBTO"]["final_designs"]["NNJ"]
    panels = [
        ("Older, as published", "authors' design (paper's network, their optimizer)",
         (a["xopt1"], a["xopt2"]), st["compliance_reported"], st["compliance_converged_truth"]),
        ("Older network, same optimizer as the newer one", "paper's network, MMA, exact gradients",
         physical(pick("to_baselines.npz", "published", 0), mesh),
         runs[("published", 0)]["compliance_model"], runs[("published", 0)]["compliance_true"]),
        ("Newer", "deep network + derivative labels (seed 0)",
         physical(pick("newer_models_to.npz", newer_label, 0), mesh),
         runs[(newer_label, 0)]["compliance_model"], runs[(newer_label, 0)]["compliance_true"]),
        ("Reference optimum", "optimized with the reference model itself",
         physical(pick("to_baselines.npz", "reference", 0), mesh), None, runs[("reference", 0)]["compliance_true"]),
    ]
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
                         "font.size": 10, "text.color": INK})
    fig = plt.figure(figsize=(12.4, 9.6), dpi=140)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.08)
    for i, (title, sub, (t1, t2), pred, true) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(cell_image(np.asarray(t1), np.asarray(t2)), cmap=matplotlib.colors.ListedColormap(
            [plt.cm.colors.to_rgba(c) for c in np.array([SURFACE, SOLID])]) if False else
            matplotlib.colors.LinearSegmentedColormap.from_list("bw", [SURFACE, SOLID]),
            vmin=0, vmax=1, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=20)
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=9, color=INK2)
        line = f"true compliance {true:.2f}" + (f"   ·   network predicted {pred:.2f}" if pred is not None else "")
        ax.text(0, -0.09, line, transform=ax.transAxes, fontsize=9.5, color=INK, va="top")
    # dot plot: predicted vs true compliance, joined by a line (the gap is the error)
    ax = fig.add_subplot(gs[2, :])
    names = ["Older, as published", "Older network, same optimizer", "Newer"]
    pred = [p[3] for p in panels[:3]]; true = [p[4] for p in panels[:3]]
    ref = panels[3][4]
    ys = np.arange(3)[::-1]
    ax.axvline(ref, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_facecolor(SURFACE)
    BOX = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
    ax.text(ref + 0.12, 2.62, f"reference optimum {ref:.2f}", fontsize=8.5, color=INK2, ha="left", va="bottom", bbox=BOX)
    for y, pv, tv in zip(ys, pred, true):
        ax.plot([pv, tv], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=2)
        ax.scatter([pv], [y], s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([tv], [y], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
        gap = 100 * (pv / tv - 1)
        gtxt = (f"{gap:+.2f}" if abs(gap) < 1 else f"{gap:+.1f}").replace("-", "\u2212")
        if abs(tv - pv) > 0.6:
            ax.text(pv, y + 0.22, f"{pv:.2f}", ha="center", fontsize=9, color=INK, bbox=BOX)
            ax.text(tv, y + 0.22, f"{tv:.2f}", ha="center", fontsize=9, color=INK, bbox=BOX)
        else:
            ax.text(tv + 0.25, y + 0.22, f"predicted {pv:.2f} \u00b7 true {tv:.2f} (dots overlap)", ha="left", fontsize=9, color=INK, bbox=BOX)
        ax.text(86.6, y, f"network off by {gtxt} %", fontsize=9, color=INK, va="center", ha="right")
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9.5); ax.tick_params(axis="y", length=0)
    ax.set_xlim(74.5, 86.8); ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("compliance of the final design (lower is better)", color=INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("What the network predicted vs the true compliance of its final design", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, markersize=8, label="network's prediction"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, markersize=8, label="true compliance (mesh-converged reference)")],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, fontsize=9)
    fig.savefig(os.path.join(FIG, "fig5_older_vs_newer.png"), bbox_inches="tight")
    print("wrote", os.path.join(FIG, "fig5_older_vs_newer.png"))

    # starting-design sensitivity error of both networks against the reference
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz"))
    truth = SplineMaterial(g["grid"], g["C"])
    x0 = np.full(2 * mesh.ne, np.sqrt(0.5))
    g_true = ComplianceProblem(mesh, truth, kind="hole", filter="none").raw_gradient(x0)
    nets = {"older (published)": load_keras_mlp(os.path.join(UP, "NN_model_HBTO_1e+04")),
            "newer (deep + derivative labels, s0)": load_network(resolve_model(runs[(newer_label, 0)]["path"]))}
    for name, net in nets.items():
        p = ComplianceProblem(mesh, NNMaterial(net, "NNJ", predict_cast=None, clip="none"), kind="hole", filter="none")
        gn = p.raw_gradient(x0)
        print(f"{name}: starting-design compliance {p.compliance(x0):.3f} (true {ComplianceProblem(mesh, truth, kind='hole', filter='none').compliance(x0):.3f}); "
              f"sensitivity error (relative L2) {np.linalg.norm(gn - g_true) / np.linalg.norm(g_true):.4f}")


if __name__ == "__main__":
    main()
