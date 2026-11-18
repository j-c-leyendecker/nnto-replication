"""Vectorised 2-D plane finite-element analysis on a structured Q4 grid.

Same mesh, numbering, boundary conditions and element as the original
``Run_*`` scripts (which follow the 88-line MATLAB code), but:

* the element stiffness is assembled from four constant basis matrices,
  ``KE = C11*A11 + C12*A12 + C22*A22 + C66*A66`` (the element is linear in the
  constitutive entries), instead of calling a 64-statement Python function
  once per element per evaluation;
* the compliance sensitivity re-uses the element strain energies
  ``q[e, k] = u_e^T A_k u_e`` so no second pass over the elements is needed.

``tests/test_fea.py`` checks the basis matrices against the original
``kefun_C`` and the assembled solution against the original ``feafun``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .homogenization import _unit_square_integrals, plane_strain_C

__all__ = ["Mesh", "ke_basis", "mbb_half", "cantilever", "C0_VEC_ORIGINAL",
           "C0_VEC_EXACT"]

# Constitutive entries [C11, C12, C22, C66] of the solid (E = 1, nu = 0.3,
# plane strain).  The original scripts hard-code values rounded to 4 digits.
C0_VEC_ORIGINAL = np.array([1.3462, 0.5769, 1.3462, 0.3846])
_C = plane_strain_C(1.0, 0.3)
C0_VEC_EXACT = np.array([_C[0, 0], _C[0, 1], _C[1, 1], _C[2, 2]])


def ke_basis() -> np.ndarray:
    """Basis matrices A (4, 8, 8) with KE = sum_k C_k A_k for a unit square
    Q4 element, dof order [u1 v1 u2 v2 u3 v3 u4 v4], nodes counter-clockwise
    from the lower-left corner."""
    A = np.zeros((4, 8, 8))
    for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 1), (2, 2)]):
        C = np.zeros((3, 3))
        C[i, j] = C[j, i] = 1.0
        Kxx, Kyy, Kxy, _, _ = _unit_square_integrals(C)
        A[k] = Kxx + Kyy + Kxy
    return A


@dataclass
class Mesh:
    nelx: int
    nely: int
    fixed: np.ndarray
    F: np.ndarray                              # (ndof, n_loadcases)
    rmin: float = 2.7
    A: np.ndarray = field(default_factory=ke_basis)

    def __post_init__(self):
        nelx, nely = self.nelx, self.nely
        self.ne = nelx * nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        # element dof table, identical to the original feaprep()
        nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape(nely + 1, nelx + 1, order="F") + 1
        edofVec = (2 * nodenrs[:-1, :-1] + 1).reshape(-1, 1, order="F")
        off = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
        self.edof = (edofVec + off - 1).astype(np.int64)                 # (ne, 8)
        self.iK = np.repeat(self.edof, 8, axis=1).ravel()
        self.jK = np.tile(self.edof, (1, 8)).ravel()
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        if self.F.ndim == 1:
            self.F = self.F.reshape(-1, 1)
        # assembly pattern on the free dofs, built once
        fmap = -np.ones(self.ndof, dtype=np.int64)
        fmap[self.free] = np.arange(self.free.size)
        ii, jj = fmap[self.iK], fmap[self.jK]
        self._keep = (ii >= 0) & (jj >= 0)
        nf = self.free.size
        pat = sparse.coo_matrix((np.ones(self._keep.sum()), (ii[self._keep], jj[self._keep])),
                                shape=(nf, nf)).tocsc()
        pat.sum_duplicates()
        self._csc_indices, self._csc_indptr = pat.indices, pat.indptr
        # map every kept COO entry to its slot in the CSC data array
        order = np.lexsort((ii[self._keep], jj[self._keep]))
        key = jj[self._keep][order] * nf + ii[self._keep][order]
        uniq, inv = np.unique(key, return_inverse=True)
        slot = np.empty(order.size, dtype=np.int64)
        slot[order] = inv
        self._slot, self._nnz, self._nf = slot, uniq.size, nf
        self._Avec = self.A.reshape(4, 64)
        self.H, self.Hs = self._filter(self.rmin)

    # ------------------------------------------------------------- filter
    def _filter(self, rmin: float):
        """Linear-hat filter weights, identical to the original filterprep()."""
        nelx, nely = self.nelx, self.nely
        r = ceil(rmin) - 1
        ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        ex, ey = ex.ravel(), ey.ravel()
        rows, cols, vals = [], [], []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                w = rmin - np.sqrt(dx * dx + dy * dy)
                if w <= 0:
                    continue
                x2, y2 = ex + dx, ey + dy
                ok = (x2 >= 0) & (x2 < nelx) & (y2 >= 0) & (y2 < nely)
                rows.append((ex * nely + ey)[ok])
                cols.append((x2 * nely + y2)[ok])
                vals.append(np.full(ok.sum(), w))
        H = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.ne, self.ne)).tocsr()
        Hs = np.asarray(H.sum(axis=1)).ravel()
        return H, Hs

    # -------------------------------------------------------------- solve
    def solve(self, C: np.ndarray):
        """Assemble and solve for constitutive entries ``C`` (ne, 4).

        Returns (U (ndof, nload), q (nload, ne, 4)) where
        ``q[l, e, k] = u_e^T A_k u_e`` so that the compliance of load case l is
        ``sum(C * q[l])`` and  d c_l / d C[e, k] = -q[l, e, k].
        """
        sK = (C @ self._Avec).ravel()                         # (ne*64,)
        data = np.bincount(self._slot, weights=sK[self._keep], minlength=self._nnz)
        K = sparse.csc_matrix((data, self._csc_indices, self._csc_indptr),
                              shape=(self._nf, self._nf))
        U = np.zeros((self.ndof, self.F.shape[1]))
        # K is symmetric positive definite for admissible C: use SuperLU's
        # symmetric mode (about 2x faster than the default ordering).  If the
        # material model hands us an indefinite tensor the factorisation still
        # goes through (it is an LU, not a Cholesky), as in the original.
        lu = splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options=dict(SymmetricMode=True))
        U[self.free] = lu.solve(self.F[self.free])
        ue = U[self.edof]                                      # (ne, 8, nload)
        q = np.einsum("eal,ekal->lek", ue, np.einsum("kab,ebl->ekal", self.A, ue))
        return U, q


def _dense_force(ndof, dof, value):
    F = np.zeros((ndof, 1))
    F[dof, 0] = value
    return F


def mbb_half(nelx=60, nely=30, rmin=2.7) -> Mesh:
    """Half MBB beam exactly as implemented in the original scripts:
    unit downward load on the top-left node, horizontal symmetry rollers on
    the left edge, vertical roller at the bottom-right node."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    fixed = np.concatenate([np.arange(0, 2 * (nely + 1), 2), [ndof - 1]])
    return Mesh(nelx, nely, fixed, _dense_force(ndof, 1, -1.0), rmin)


def cantilever(nelx=60, nely=30, rmin=2.7) -> Mesh:
    """Cantilever as *drawn* in Fig. 4(a) of the paper: left edge clamped
    (u = 0), unit downward load at the bottom-right corner."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    fixed = np.arange(0, 2 * (nely + 1))
    return Mesh(nelx, nely, fixed, _dense_force(ndof, ndof - 1, -1.0), rmin)
