"""Independent test set for the HBTO surrogates and for the spline 'truth'.

300 Latin-hypercube points in [0.01, 0.99]^2 that are NOT on the 51 x 51
reference grid.  Each is homogenised on body-fitted, corner-graded meshes at
two levels (ns = 16 and 32) and Richardson-extrapolated, exactly as in
gt_fitted_grid.py, giving C and its exact derivatives dC/dtheta.

Output: results/offgrid_truth.npz
"""
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnto.homogenization import fitted_CH  # noqa: E402

# honours NNTO_RESULTS like the other scripts (fix 2026-11-18)
OUT = os.path.abspath(os.environ.get("NNTO_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")))


def main(n=300, seed=7, levels=(16, 32), beta=2.5):
    P = 0.01 + 0.98 * qmc.LatinHypercube(2, seed=seed).random(n)
    C = np.zeros((n, 4)); dC = np.zeros((n, 4, 2)); Ce = np.zeros((n, 4)); dCe = np.zeros((n, 4, 2))
    t0 = time.time()
    for i, (a, b) in enumerate(P):
        c1, d1 = fitted_CH(a, b, ns=levels[0], nh=2 * levels[0], grading=beta)
        c2, d2 = fitted_CH(a, b, ns=levels[1], nh=2 * levels[1], grading=beta)
        C[i] = c2 + (c2 - c1) / 3.0; dC[i] = d2 + (d2 - d1) / 3.0
        Ce[i] = np.abs(C[i] - c2); dCe[i] = np.abs(dC[i] - d2)
        if i % 25 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "offgrid_truth.npz"), P=P, C=C, dC=dC, C_err=Ce, dC_err=dCe)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
