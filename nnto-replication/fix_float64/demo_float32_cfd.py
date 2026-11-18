"""Shows, using only NumPy and the authors' own saved results, that the
central-difference (CFD) sensitivities in Najmon & Tovar (2024) are scaled by
the float32 rounding of the inputs to Keras' Model.predict.

usage: python demo_float32_cfd.py "<path to 'JMD Journal Paper Results'>"
"""
import os
import sys

import numpy as np

res = sys.argv[1]
h = 1e-6


def float32_step_ratio(x0):
    """Step actually taken by (f(x+h) - f(x-h)) / (2h) once x +- h are rounded
    to float32, divided by the nominal step 2h."""
    return (float(np.float32(x0 + h)) - float(np.float32(x0 - h))) / (2 * h)


cases = [("DBTO", 0.5, "DBTO_NN_result_dy1/DBTO_NN_dy1.npz", "DBTO_NN_result_dy2/DBTO_NN_dy2.npz", 1.11e-3),
         ("HBTO", np.sqrt(0.5), "HBTO_NN_result_dy1/HBTO_NN_dy1_it1225final.npz",
          "HBTO_NN_result_dy2/HBTO_NN_dy2_it1105final.npz", 1.07e-4)]
for name, x0, f_nnj, f_cfd, printed in cases:
    nnj = np.load(os.path.join(res, f_nnj))["dc0"]
    cfd = np.load(os.path.join(res, f_cfd))["dc0"]
    r = cfd / nnj
    pred = float32_step_ratio(x0)
    print(f"{name}: initial design theta = {x0:.6f}")
    print(f"  predicted CFD/NNJ ratio from float32 rounding : {pred:.10f}")
    print(f"  ratio in the authors' saved sensitivities     : {r.mean():.10f}  (std {r.std():.1e}, {r.size} values)")
    print(f"  MSE(CFD, NNJ) printed in the paper            : {printed:.3g}")
    print(f"  MSE recomputed from the saved arrays          : {np.mean((cfd - nnj) ** 2):.6e}")
    print(f"  MSE predicted from the rounding alone         : {(pred - 1) ** 2 * np.mean(nnj ** 2):.6e}")

print("\nError of the published CFD derivative versus theta (pure rounding effect):")
for lo, hi in [(0.5, 1.0), (0.25, 0.5), (0.125, 0.25), (0.0625, 0.125)]:
    t = np.linspace(lo, hi, 20001)[:-1]
    e = np.array([float32_step_ratio(v) for v in t]) - 1
    print(f"  theta in [{lo:<6}, {hi:<5}): {100 * e.min():+.2f} % to {100 * e.max():+.2f} %")
