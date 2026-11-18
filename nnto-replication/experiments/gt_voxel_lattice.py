"""Ground truth (a): the voxel model that generated the paper's training set.

Evaluates the 100x100 voxel homogenisation on the full lattice of hole sizes
it can represent (theta = 0, 0.02, ..., 1.0 in each direction; 51 x 51).  The
map theta -> C^H used for the published data is piecewise constant on this
lattice, so this table *is* the data-generating function.

Output: results/voxel_lattice.npz  (theta grid, C[51,51,4])
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnto.homogenization import voxel_CH, plane_strain_C  # noqa: E402

# honours NNTO_RESULTS like the other scripts (fix 2026-11-18)
OUT = os.path.abspath(os.environ.get("NNTO_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")))


def main(nel=100):
    grid = np.arange(0, nel // 2 + 1) * (2.0 / nel)        # 0, 0.02, ..., 1
    n = grid.size
    C = np.zeros((n, n, 4))
    t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            # representative theta strictly inside the lattice cell
            t1 = grid[i] - 1e-9 if i > 0 else 0.0
            t2 = grid[j] - 1e-9 if j > 0 else 0.0
            c = voxel_CH(t1, t2, nel=nel) if (i > 0 and j > 0) else \
                np.array(plane_strain_C()[[0, 0, 1, 2], [0, 1, 1, 2]])
            C[i, j] = c
            C[j, i] = c[[2, 1, 0, 3]]                      # x <-> y symmetry
        if i % 5 == 0:
            print(f"row {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "voxel_lattice.npz"), grid=grid, C=C, nel=nel)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
