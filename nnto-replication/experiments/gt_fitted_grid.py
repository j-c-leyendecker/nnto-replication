"""Ground truth (b): mesh-converged homogenised tensor *and its derivatives*.

Body-fitted, corner-graded meshes at two refinement levels (ns = 16 and 32
elements across each wall, twice that across the hole) are combined by
Richardson extrapolation with the observed second-order rate.  The difference
between the fine level and the extrapolant is stored as a discretisation-error
estimate for every entry.

Output: results/fitted_grid.npz
    grid (51,)                 0.01, 0.02, 0.04, ..., 0.98, 0.99
    C   (51,51,4)              [C11, C12, C22, C66]
    dC  (51,51,4,2)            d/dtheta1, d/dtheta2
    C_err, dC_err              |fine - extrapolated|
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnto.homogenization import fitted_CH  # noqa: E402

# honours NNTO_RESULTS like the other scripts (fix 2026-11-18)
OUT = os.path.abspath(os.environ.get("NNTO_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")))
SWAP = [2, 1, 0, 3]


def main(levels=(16, 32), beta=2.5):
    grid = np.concatenate([[0.01], np.arange(1, 50) * 0.02, [0.99]])
    n = grid.size
    C = np.zeros((n, n, 4)); dC = np.zeros((n, n, 4, 2))
    Ce = np.zeros_like(C); dCe = np.zeros_like(dC)
    t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            c1, d1 = fitted_CH(grid[i], grid[j], ns=levels[0], nh=2 * levels[0], grading=beta)
            c2, d2 = fitted_CH(grid[i], grid[j], ns=levels[1], nh=2 * levels[1], grading=beta)
            ce = c2 + (c2 - c1) / 3.0
            de = d2 + (d2 - d1) / 3.0
            C[i, j], dC[i, j] = ce, de
            Ce[i, j], dCe[i, j] = np.abs(ce - c2), np.abs(de - d2)
            # mirror: C(t2,t1) = swap(C(t1,t2)); d/dt1 <-> d/dt2
            C[j, i] = ce[SWAP]; Ce[j, i] = Ce[i, j][SWAP]
            dC[j, i] = de[SWAP][:, ::-1]; dCe[j, i] = dCe[i, j][SWAP][:, ::-1]
        if i % 5 == 0:
            print(f"row {i}/{n}  {time.time() - t0:.0f}s", flush=True)
            np.savez(os.path.join(OUT, "fitted_grid_partial.npz"), grid=grid, C=C, dC=dC, C_err=Ce, dC_err=dCe, rows_done=i + 1)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "fitted_grid.npz"), grid=grid, C=C, dC=dC,
             C_err=Ce, dC_err=dCe, levels=np.array(levels), beta=beta)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
