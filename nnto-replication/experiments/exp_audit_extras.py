"""Small audit computations on the authors' saved designs.

* share of design variables in the clipped region of the DBTO surrogate;
* how far the filtered "sensitivity" handed to the optimiser is from the
  actual gradient at the saved final designs (cosine similarity);
* whether the theta-MSE values printed in Fig. 11 can be recovered from the
  saved arrays (with and without the 7.5 % thresholding used for plotting).

Output: results/audit_extras.json
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json  # noqa: E402

from nnto.fea import mbb_half  # noqa: E402
from nnto.materials import NNMaterial  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402
from nnto.topopt import ComplianceProblem  # noqa: E402

UP = find_upstream(); RES = os.path.join(UP, "JMD Journal Paper Results")
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    out = {"DBTO": {}, "HBTO": {}}
    mesh = mbb_half()
    net = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))
    for d, n in ((1, "NNJ"), (2, "CFD"), (3, "CSM"), (4, "AD")):
        z = np.load(os.path.join(RES, f"DBTO_NN_result_dy{d}", f"DBTO_NN_dy{d}.npz"))
        x = z["xopt"].flatten("F")
        prob = ComplianceProblem(mesh, NNMaterial(net, "NNJ"))
        g_f, g_r = prob.gradient(x), prob.raw_gradient(x)
        cons = ComplianceProblem(mesh, NNMaterial(net, "NNJ", clip="consistent"), filter="none")
        g_c = cons.raw_gradient(x)
        rng = np.random.default_rng(0); dvec = rng.normal(size=x.size); h = 1e-6
        fd = (cons.compliance(np.clip(x + h * dvec, 0, 1)) - cons.compliance(np.clip(x - h * dvec, 0, 1))) / (2 * h)
        out["DBTO"][n] = dict(frac_in_clipped_region=float(np.mean(x < 0.0811)),
                              frac_wrong_sign_region=float(np.mean(x < 0.0289)),
                              cos_filtered_vs_raw=cos(g_f, g_r),
                              cos_unmasked_vs_true_gradient=cos(g_r, g_c),
                              rel_norm_spurious_part=float(np.linalg.norm(g_r - g_c) / np.linalg.norm(g_c)),
                              directional_fd_vs_true=float(abs(fd - g_c @ dvec) / abs(fd)),
                              directional_fd_vs_supplied=float(abs(fd - g_f @ dvec) / abs(fd)))
    net = load_keras_mlp(os.path.join(UP, "NN_model_HBTO_1e+04"))
    fin = {"NNJ": (1, "it1225final"), "CFD": (2, "it1105final"), "CSM": (3, "it1371final"), "AD": (4, "it1342final")}
    X = {}
    for n, (d, tag) in fin.items():
        z = np.load(os.path.join(RES, f"HBTO_NN_result_dy{d}", f"HBTO_NN_dy{d}_{tag}.npz"))
        x = np.concatenate([z["xopt1"].flatten("F"), z["xopt2"].flatten("F")]); X[n] = x
        prob = ComplianceProblem(mesh, NNMaterial(net, "NNJ"), kind="hole")
        g_f, g_r = prob.gradient(x), prob.raw_gradient(x)
        out["HBTO"][n] = dict(cos_filtered_vs_raw=cos(g_f, g_r),
                              max_abs_filtered_over_max_abs_raw=float(np.abs(g_f).max() / np.abs(g_r).max()),
                              frac_vars_below_0p001=float(np.mean(x < 1e-3)))
    thr = lambda v: np.where(v < 0.075, 0.0, np.where(v > 0.925, 1.0, v))
    printed = {("CFD", "NNJ"): 4.09e-2, ("CSM", "NNJ"): 3.69e-3, ("AD", "NNJ"): 3.23e-3,
               ("CSM", "CFD"): 4.12e-2, ("AD", "CFD"): 3.78e-2, ("AD", "CSM"): 3.64e-3}
    out["fig11_theta_mse"] = {}
    for (b, a), p in printed.items():
        out["fig11_theta_mse"][f"{b} vs {a}"] = dict(printed=p, from_saved=float(np.mean((X[a] - X[b]) ** 2)),
                                                    from_saved_thresholded=float(np.mean((thr(X[a]) - thr(X[b])) ** 2)))
    save_json("audit_extras.json", out)
    for k, v in out.items():
        print(k)
        for kk, vv in v.items():
            print("  ", kk, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in vv.items()})


if __name__ == "__main__":
    main()
