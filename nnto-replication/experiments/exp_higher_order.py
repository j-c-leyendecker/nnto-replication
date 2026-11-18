"""Second derivatives: which methods stay accurate, and is the surrogate's
curvature usable?

Part 1 (method accuracy).  Reference = analytic second-order chain rule in
float64 (identical to hyper-dual AD to ~1e-13).  Compared: second differences
of values, complex-step formulas on values, central differences / complex step
applied to the analytic Jacobian, and the same in single precision.

Part 2 (surrogate fidelity).  For the published DBTO network the exact SIMP
derivatives are known (E' = 3 theta^2, E'' = 6 theta): the error is reported
per derivative order.

Output: results/higher_order.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import find_upstream, save_json  # noqa: E402

from nnto import derivatives as D  # noqa: E402
from nnto.networks import load_keras_mlp  # noqa: E402

UP = find_upstream()


def rel(H, Href):
    s = np.abs(Href).max()
    return float(np.abs(H - Href).max() / s)


def main():
    out = {}
    rng = np.random.default_rng(0)
    for name in ("DBTO", "HBTO"):
        net = load_keras_mlp(os.path.join(UP, f"NN_model_{name}_1e+04"))
        x = rng.uniform(0.02, 0.98, size=(1000, net.xdim))
        H = D.hessian_nnj(net, x)
        r = {"hyper_dual_AD": rel(D.hessian_ad(net, x), H),
             "complex_step_of_NNJ": rel(D.hessian_csm_of_nnj(net, x), H),
             "cfd_of_values": {}, "csm_of_values": {}, "cfd_of_NNJ": {}}
        for e in range(-8, -1):
            h = 10.0 ** e
            r["cfd_of_values"][f"1e{e}"] = rel(D.hessian_cfd(net, x, h), H)
            r["csm_of_values"][f"1e{e}"] = rel(D.hessian_csm_values(net, x, h), H)
            r["cfd_of_NNJ"][f"1e{e}"] = rel(D.hessian_cfd_of_nnj(net, x, h), H)
        net32 = net.astype(np.float32); x32 = x.astype(np.float32)
        H32ref = D.hessian_nnj(net, x32.astype(float))
        r["float32_analytic"] = rel(D.hessian_nnj(net32, x32), H32ref)
        r["float32_hyper_dual"] = rel(D.hessian_ad(net32, x32), H32ref)
        v = rng.normal(size=x.shape)
        r["hvp_forward_over_reverse"] = float(np.abs(D.hvp(net, x, v, 0) - np.einsum("nij,nj->ni", H[:, 0], v)).max()
                                              / np.abs(H[:, 0]).max())
        out[name] = r
    # surrogate fidelity per derivative order (DBTO, exact SIMP reference)
    net = load_keras_mlp(os.path.join(UP, "NN_model_DBTO_1e+04"))
    t = np.linspace(0, 1, 20001).reshape(-1, 1)
    ref = [1e-9 + (1 - 1e-9) * t[:, 0] ** 3, 3 * (1 - 1e-9) * t[:, 0] ** 2, 6 * (1 - 1e-9) * t[:, 0]]
    got = [net.predict(t)[:, 0], net.jacobian_nnj(t)[:, 0, 0], net.hessian_nnj(t)[:, 0, 0, 0]]
    out["DBTO_surrogate_vs_SIMP"] = {
        f"order_{k}": dict(rmse=float(np.sqrt(np.mean((g - r) ** 2))),
                           rel_rmse=float(np.sqrt(np.mean((g - r) ** 2)) / np.sqrt(np.mean(r ** 2))),
                           max_abs=float(np.abs(g - r).max()))
        for k, (g, r) in enumerate(zip(got, ref))}
    save_json("higher_order.json", out)
    for name in ("DBTO", "HBTO"):
        r = out[name]
        print(f"{name}: hyper-dual {r['hyper_dual_AD']:.1e} | CS-of-NNJ {r['complex_step_of_NNJ']:.1e} | HVP {r['hvp_forward_over_reverse']:.1e} | f32 analytic {r['float32_analytic']:.1e}")
        print("   h      CFD(values)  CSM(values)  CFD(NNJ)")
        for k in r["cfd_of_values"]:
            print(f"   {k:5s}  {r['cfd_of_values'][k]:.1e}      {r['csm_of_values'][k]:.1e}      {r['cfd_of_NNJ'][k]:.1e}")
    print("DBTO surrogate vs SIMP:", out["DBTO_surrogate_vs_SIMP"])


if __name__ == "__main__":
    main()
