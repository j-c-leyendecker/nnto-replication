"""How accurate are the published surrogates and their derivatives?

DBTO: exact SIMP law (the paper's own ground truth).
HBTO: two independent references that the paper did not have
      (a) the voxel model that generated the training data (piecewise constant)
      (b) the mesh-converged homogenised tensor and its exact derivatives
          (results/fitted_grid.npz, see experiments/gt_fitted_grid.py).

Also re-evaluates the authors' *final HBTO designs* with both references to
see how much of the reported compliance is real.

Fixes of 2026-11-18 (see FIXES_2026-11-18.md):
  * the start-design sensitivity error of the network is now compared with the
    paper's CFD-vs-NNJ MSE on the *same* basis.  The paper's Figs. 8 and 11
    use filtered sensitivities (the authors' ``dc0`` is saved after
    ``filterfun``); the earlier "1,160x" divided a raw-gradient MSE by that
    filtered MSE.  Both bases are reported in ``sensitivity_bases``.
  * the authors' designs have hole sides below 0.01, outside the converged
    reference's range, which clamps them to a 1 % slit.  They are now also
    scored with such holes counted as absent (solid), as in the authors' voxel
    data.

Output: results/surrogate_truth.json
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json, RESULTS  # noqa: E402

from nnto.fea import C0_VEC_EXACT, mbb_half  # noqa: E402
from nnto.materials import NNMaterial, SplineMaterial, VoxelTableMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.scoring import SmallHolesSolid, compare_vectors, sensitivity_bases  # noqa: E402
from nnto.topopt import ComplianceProblem  # noqa: E402

UP = find_upstream()
RES = os.path.join(UP, "JMD Journal Paper Results")
COMP = ["C11", "C12", "C22", "C66"]


def stats(err, ref):
    return dict(mse=float(np.mean(err ** 2)), rmse=float(np.sqrt(np.mean(err ** 2))),
                max_abs=float(np.abs(err).max()),
                rel_rmse=float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(ref ** 2))))


def dbto():
    net = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))
    t = np.linspace(0, 1, 200001).reshape(-1, 1)
    E = net.predict(t)[:, 0]; dE = net.jacobian_nnj(t)[:, 0, 0]; d2E = net.hessian_nnj(t)[:, 0, 0, 0]
    Et = 1e-9 + (1 - 1e-9) * t[:, 0] ** 3; dEt = 3 * (1 - 1e-9) * t[:, 0] ** 2; d2Et = 6 * (1 - 1e-9) * t[:, 0]
    neg = t[E < 1e-9, 0]; wrong = t[dE < 0, 0]
    return dict(function=stats(E - Et, Et), first_derivative=stats(dE - dEt, dEt),
                second_derivative=stats(d2E - d2Et, d2Et),
                E_at_0=float(E[0]), E_at_1=float(E[-1]), dE_at_0=float(dE[0]),
                clipped_below_Emin=[float(neg.min()), float(neg.max())],
                fraction_clipped=float(np.mean(E < 1e-9)),
                wrong_sign_derivative=[float(wrong.min()), float(wrong.max())],
                derivative_in_clipped_region=[float(dE[E < 1e-9].min()), float(dE[E < 1e-9].max())],
                above_Emax=bool(np.any(E > 1.0)))


def hbto():
    out = {}
    net = load_keras_mlp(os.path.join(UP, "NN_model_HBTO_1e+04"))
    g = np.load(os.path.join(RESULTS, "fitted_grid.npz")); v = np.load(os.path.join(RESULTS, "voxel_lattice.npz"))
    grid, Ct, dCt = g["grid"], g["C"], g["dC"]
    out["truth_quality"] = dict(C_err_max=float(g["C_err"].max()), C_err_median=float(np.median(g["C_err"])),
                                dC_err_max=float(g["dC_err"].max()), dC_err_median=float(np.median(g["dC_err"])),
                                levels=[int(a) for a in g["levels"]])
    spl = SplineMaterial(grid, Ct); vox = VoxelTableMaterial(v["grid"], v["C"])
    # spline derivative vs exact derivative at the grid nodes (interpolation error of the reference)
    T1, T2 = np.meshgrid(grid, grid, indexing="ij"); pts = np.stack([T1.ravel(), T2.ravel()], 1)
    _, dS = spl.evaluate(pts)
    out["truth_quality"]["spline_derivative_vs_exact_at_nodes_rms"] = float(np.sqrt(np.mean((dS - dCt.reshape(-1, 4, 2)) ** 2)))

    # --- published data set
    m = loadmat(os.path.join(UP, "HBTO_CH_2D_N10000_100x100.mat"), squeeze_me=True)
    P, Cd = m["SP_para"][:, :2], m["SP_C"][:, [0, 1, 2, 5]]
    test = np.arange(P.shape[0]) >= 8500
    out["dataset"] = dict(n=int(P.shape[0]), unique_targets=int(np.unique(np.round(Cd, 10), axis=0).shape[0]),
                          test_mse_nn_vs_data=float(np.mean((net.predict(P[test]) - Cd[test]) ** 2)),
                          paper_table1_mse=1.47e-3)
    ins = np.all((P >= grid[0]) & (P <= grid[-1]), axis=1)
    Cs, _ = spl.evaluate(P[ins])
    out["dataset"]["mse_labels_vs_converged_truth"] = float(np.mean((Cd[ins] - Cs) ** 2))
    out["dataset"]["mse_nn_vs_labels_inside"] = float(np.mean((net.predict(P[ins]) - Cd[ins]) ** 2))
    out["dataset"]["mse_nn_vs_converged_truth_inside"] = float(np.mean((net.predict(P[ins]) - Cs) ** 2))
    edge = ~ins
    out["dataset"]["mse_nn_vs_labels_edge_band"] = float(np.mean((net.predict(P[edge]) - Cd[edge]) ** 2))
    out["dataset"]["n_edge_band"] = int(edge.sum())

    # --- surrogate vs converged truth on the reference grid (exact derivative labels)
    Cn = net.predict(pts).reshape(51, 51, 4); Jn = net.jacobian_nnj(pts).reshape(51, 51, 4, 2)
    regions = {"all [0.01,0.99]^2": np.ones((51, 51), bool),
               "interior [0.1,0.9]^2": (T1 >= 0.1) & (T1 <= 0.9) & (T2 >= 0.1) & (T2 <= 0.9),
               "edge (any theta < 0.1 or > 0.9)": ~((T1 >= 0.1) & (T1 <= 0.9) & (T2 >= 0.1) & (T2 <= 0.9))}
    out["vs_truth"] = {}
    for rn, mask in regions.items():
        r = {"function": {}, "d_dtheta1": {}, "d_dtheta2": {}}
        for k, c in enumerate(COMP):
            r["function"][c] = stats(Cn[mask][:, k] - Ct[mask][:, k], Ct[mask][:, k])
            r["d_dtheta1"][c] = stats(Jn[mask][:, k, 0] - dCt[mask][:, k, 0], dCt[mask][:, k, 0])
            r["d_dtheta2"][c] = stats(Jn[mask][:, k, 1] - dCt[mask][:, k, 1], dCt[mask][:, k, 1])
        r["function_all"] = stats(Cn[mask] - Ct[mask], Ct[mask])
        r["derivative_all"] = stats(Jn[mask] - dCt[mask], dCt[mask])
        # sign errors: stiffness must not increase with hole size
        r["frac_positive_dC11_dth2_or_dC22_dth1"] = float(np.mean((Jn[mask][:, 0, 1] > 0) | (Jn[mask][:, 2, 0] > 0)))
        out["vs_truth"][rn] = r

    # --- admissibility of the surrogate on a fine grid over [0,1]^2
    f = np.linspace(0, 1, 401); F1, F2 = np.meshgrid(f, f, indexing="ij"); fp = np.stack([F1.ravel(), F2.ravel()], 1)
    Cf = net.predict(fp)
    C0 = np.array([1.3462, 0.5769, 1.3462, 0.3846])
    Cc = np.minimum(np.maximum(Cf, 1e-9 * C0), C0)
    out["admissibility"] = dict(
        frac_any_component_clipped=float(np.mean(np.any((Cf < 1e-9 * C0) | (Cf > C0), axis=1))),
        frac_negative_raw_component=float(np.mean(np.any(Cf < 0, axis=1))),
        frac_not_positive_definite_raw=float(np.mean((Cf[:, 0] <= 0) | (Cf[:, 2] <= 0) | (Cf[:, 3] <= 0)
                                                     | (Cf[:, 0] * Cf[:, 2] - Cf[:, 1] ** 2 <= 0))),
        frac_not_positive_definite_after_clip=float(np.mean(Cc[:, 0] * Cc[:, 2] - Cc[:, 1] ** 2 <= 0)),
        min_raw=[float(a) for a in Cf.min(axis=0)], max_raw=[float(a) for a in Cf.max(axis=0)])

    # --- initial design: sensitivity error against the truth
    mesh = mbb_half()
    x0 = np.full(2 * mesh.ne, np.sqrt(0.5))
    nn = ComplianceProblem(mesh, NNMaterial(net, "NNJ", predict_cast=None, clip="none"), kind="hole", filter="none")
    tr = ComplianceProblem(mesh, spl, kind="hole", filter="none")
    g_nn, g_tr = nn.raw_gradient(x0), tr.raw_gradient(x0)
    out["initial_design"] = dict(
        C_truth=[float(a) for a in spl.evaluate(x0[:1].reshape(1, 1).repeat(2, 1))[0][0]],
        C_nn=[float(a) for a in net.predict(np.array([[x0[0], x0[0]]]))[0]],
        C_voxel=[float(a) for a in vox.evaluate(np.array([[x0[0], x0[0]]]))[0][0]],
        dC_dth1_truth=[float(a) for a in spl.evaluate(np.array([[x0[0], x0[0]]]))[1][0, :, 0]],
        dC_dth1_nn=[float(a) for a in net.jacobian_nnj(np.array([[x0[0], x0[0]]]))[0, :, 0]],
        compliance_nn=nn.compliance(x0), compliance_truth=tr.compliance(x0),
        sens_mse_nn_vs_truth=float(np.mean((g_nn - g_tr) ** 2)),
        sens_rel_l2=float(np.linalg.norm(g_nn - g_tr) / np.linalg.norm(g_tr)),
        sens_cosine=float(g_nn @ g_tr / np.linalg.norm(g_nn) / np.linalg.norm(g_tr)),
        sens_mse_cfd_f32_vs_nnj_paper=1.07e-4,
        note="sens_* above are on RAW gradients; the paper's 1.07e-4 is on FILTERED ones. "
             "Use sensitivity_bases for like-for-like ratios.")

    # --- network error vs derivative-method difference on matching bases (fix 2026-11-18)
    nn64 = NNMaterial(net, "NNJ", predict_cast=None, clip="none")
    pub_nnj = NNMaterial(net, "NNJ", predict_cast=np.float32, clip="original")      # as published
    pub_cfd = NNMaterial(net, "CFD-f32", predict_cast=np.float32, clip="original")  # as published
    S_nn, S_tr = sensitivity_bases(mesh, nn64, x0), sensitivity_bases(mesh, spl, x0)
    S_nnj, S_cfd = sensitivity_bases(mesh, pub_nnj, x0), sensitivity_bases(mesh, pub_cfd, x0)
    bases = {}
    for b in ("raw", "filtered"):
        net_err = compare_vectors(S_nn[b], S_tr[b])
        meth = compare_vectors(S_cfd[b], S_nnj[b])
        bases[b] = dict(network_vs_truth=net_err, cfd_vs_nnj_as_published=meth,
                        mse_ratio=net_err["mse"] / meth["mse"],
                        rel_l2_ratio=net_err["rel_l2"] / meth["rel_l2"])
    a = np.load(os.path.join(RES, "HBTO_NN_result_dy1", "HBTO_NN_dy1_it1225final.npz"))["dc0"].ravel()
    c = np.load(os.path.join(RES, "HBTO_NN_result_dy2", "HBTO_NN_dy2_it1105final.npz"))["dc0"].ravel()
    bases["authors_stored_dc0"] = dict(
        rel_diff_to_recomputed_filtered=float(np.linalg.norm(a - S_nnj["filtered"]) / np.linalg.norm(a)),
        rel_diff_to_recomputed_raw=float(np.linalg.norm(a - S_nnj["raw"]) / np.linalg.norm(a)),
        mse_cfd_vs_nnj=float(np.mean((c - a) ** 2)), paper_printed_mse=1.07e-4)
    out["sensitivity_bases"] = bases

    # --- the authors' final designs, re-evaluated
    solid_small = SmallHolesSolid(spl, C0_VEC_EXACT)
    fin = {"NNJ": (1, "it1225final"), "CFD": (2, "it1105final"), "CSM": (3, "it1371final"), "AD": (4, "it1342final")}
    out["final_designs"] = {}
    for name, (d, tag) in fin.items():
        z = np.load(os.path.join(RES, f"HBTO_NN_result_dy{d}", f"HBTO_NN_dy{d}_{tag}.npz"))
        x = np.concatenate([z["xopt1"].flatten("F"), z["xopt2"].flatten("F")])
        X = x.reshape(-1, 2, order="F")
        c_nn = ComplianceProblem(mesh, NNMaterial(net, "NNJ"), kind="hole", filter="none").compliance(x)
        c_vox = ComplianceProblem(mesh, vox, kind="hole", filter="none").compliance(x)
        c_spl = ComplianceProblem(mesh, spl, kind="hole", filter="none").compliance(x)
        c_sol = ComplianceProblem(mesh, solid_small, kind="hole", filter="none").compliance(x)
        Craw = net.predict(X); Cc = np.minimum(np.maximum(Craw, 1e-9 * C0), C0)
        out["final_designs"][name] = dict(
            compliance_reported=float(z["comp"]), compliance_nn_recomputed=c_nn,
            compliance_voxel_truth=c_vox, compliance_converged_truth=c_spl,
            compliance_converged_truth_small_holes_solid=c_sol,
            underestimate_vs_voxel_pct=100 * (c_vox / c_nn - 1), underestimate_vs_converged_pct=100 * (c_spl / c_nn - 1),
            underestimate_vs_converged_small_holes_solid_pct=100 * (c_sol / c_nn - 1),
            frac_cells_side_below_reference_range=float(np.mean(solid_small.mask(X))),
            min_theta=float(X.min()),
            volume_fraction=float(1 - np.mean(X[:, 0] * X[:, 1])),
            frac_cells_clipped=float(np.mean(np.any((Craw < 1e-9 * C0) | (Craw > C0), axis=1))),
            frac_cells_not_pd_after_clip=float(np.mean(Cc[:, 0] * Cc[:, 2] - Cc[:, 1] ** 2 <= 0)),
            frac_slit_cells=float(np.mean((X.min(axis=1) < 0.02) & (X.max(axis=1) > 0.1))),
            frac_near_bounds=float(np.mean((X < 0.02) | (X > 0.98))))
    return out


def main():
    out = {"DBTO": dbto(), "HBTO": hbto()}
    save_json("surrogate_truth.json", out)
    d = out["DBTO"]
    print("DBTO  fn rmse %.2e | d rmse %.2e (rel %.2e) | d2 rmse %.2e (rel %.2e) | clipped on [0, %.4f] (%.1f%% of range) | wrong-sign derivative on [0, %.4f]"
          % (d["function"]["rmse"], d["first_derivative"]["rmse"], d["first_derivative"]["rel_rmse"], d["second_derivative"]["rmse"],
             d["second_derivative"]["rel_rmse"], d["clipped_below_Emin"][1], 100 * d["fraction_clipped"], d["wrong_sign_derivative"][1]))
    h = out["HBTO"]
    print("HBTO truth quality:", h["truth_quality"])
    print("HBTO dataset:", h["dataset"])
    for rn, r in h["vs_truth"].items():
        print(f"  {rn}: fn rel-rmse {r['function_all']['rel_rmse']:.3f} | derivative rel-rmse {r['derivative_all']['rel_rmse']:.3f} | wrong-sign frac {r['frac_positive_dC11_dth2_or_dC22_dth1']:.3f}")
        print("      d/dth1 rel-rmse by component:", {c: round(r['d_dtheta1'][c]['rel_rmse'], 3) for c in COMP})
    print("  admissibility:", h["admissibility"])
    print("  initial design:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in h["initial_design"].items()})
    print("  start-design sensitivities, network error vs derivative-method difference on the SAME basis:")
    for b in ("raw", "filtered"):
        s = h["sensitivity_bases"][b]
        print(f"    {b:8s}: network vs truth MSE {s['network_vs_truth']['mse']:.4e} (rel L2 {s['network_vs_truth']['rel_l2']:.4f}) | "
              f"CFD vs NNJ as published MSE {s['cfd_vs_nnj_as_published']['mse']:.4e} (rel L2 {s['cfd_vs_nnj_as_published']['rel_l2']:.4f}) | "
              f"ratio {s['mse_ratio']:.0f}x in MSE, {s['rel_l2_ratio']:.1f}x in relative L2")
    st = h["sensitivity_bases"]["authors_stored_dc0"]
    print(f"    authors' stored dc0: differs from the recomputed filtered NNJ sensitivity by {st['rel_diff_to_recomputed_filtered']:.1e}, "
          f"from the raw one by {st['rel_diff_to_recomputed_raw']:.2f}; stored CFD-vs-NNJ MSE {st['mse_cfd_vs_nnj']:.4e} (paper {st['paper_printed_mse']:.2e})")
    for n, r in h["final_designs"].items():
        print(f"  final {n}: reported {r['compliance_reported']:.2f} | voxel truth {r['compliance_voxel_truth']:.2f} ({r['underestimate_vs_voxel_pct']:+.1f}%) | converged truth {r['compliance_converged_truth']:.2f} ({r['underestimate_vs_converged_pct']:+.1f}%) | "
              f"converged, sides < 0.01 solid {r['compliance_converged_truth_small_holes_solid']:.2f} ({r['underestimate_vs_converged_small_holes_solid_pct']:+.1f}%) | "
              f"cells with a side < 0.01 {100*r['frac_cells_side_below_reference_range']:.1f}% | clipped cells {100*r['frac_cells_clipped']:.1f}% | not PD {100*r['frac_cells_not_pd_after_clip']:.1f}% | slit cells {100*r['frac_slit_cells']:.1f}%")


if __name__ == "__main__":
    main()
