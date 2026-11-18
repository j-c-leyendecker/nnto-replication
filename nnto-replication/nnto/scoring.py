"""Scoring helpers for final designs and sensitivity comparisons (added 2026-11-18).

They make four choices explicit that earlier versions of the experiment
scripts left implicit:

``sensitivity_bases`` / ``compare_vectors``
    The paper's Fig. 8 and Fig. 11 sensitivity MSEs are computed on *filtered*
    sensitivities: the authors' ``dc0`` is the output of ``dfeafun``, i.e. it
    is saved after ``filterfun``.  Any ratio against those MSEs must use the
    same basis.  (The earlier "1,160x" figure divided a raw-gradient MSE by the
    paper's filtered one.)

``SmallHolesSolid``
    The converged reference is tabulated on [0.01, 0.99] and clamps inputs
    outside that range, so a hole side below 0.01 is scored as a 1 % slit,
    which is much softer than solid.  The authors' voxel data (and their
    plotting, which snaps sides within 7.5 % of a bound) count such a hole as
    absent.  Designs that stay inside [0.01, 0.99] are not affected.

``project_volume``
    Runs stopped by an iteration cap can end slightly off the volume target.
    Scaling both hole fields by one factor restores the target exactly, so
    that designs can also be compared at equal material.

``error_summary``
    Signed errors of opposite sign cancel in a mean.  Report the mean absolute
    and the largest absolute error as well.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SmallHolesSolid", "hole_volume_fraction", "project_volume",
           "error_summary", "compare_vectors", "sensitivity_bases"]


class SmallHolesSolid:
    """Wrap a reference material: cells whose smaller hole side is below ``lo``
    (default: the lower end of the reference's tabulated range) get the solid
    tensor ``C_solid`` and a zero derivative."""
    nvar = 2

    def __init__(self, base, C_solid, lo=None):
        self.base = base
        self.C_solid = np.asarray(C_solid, dtype=float)
        self.lo = float(base.lo if lo is None else lo)

    def mask(self, theta):
        x = np.asarray(theta, dtype=float).reshape(-1, 2)
        return x.min(axis=1) < self.lo

    def evaluate(self, theta):
        x = np.asarray(theta, dtype=float).reshape(-1, 2)
        C, dC = self.base.evaluate(x)
        m = self.mask(x)
        C = np.array(C, dtype=float, copy=True)
        dC = np.array(dC, dtype=float, copy=True)
        C[m] = self.C_solid
        dC[m] = 0.0
        return C, dC


def hole_volume_fraction(P) -> float:
    """Solid fraction 1 - mean(theta1 * theta2) of a hole design, P (ne, 2)."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    return float(1.0 - np.mean(P[:, 0] * P[:, 1]))


def project_volume(P, volfrac, lb=0.0, ub=1.0, tol=1e-14, maxit=200):
    """Scale both hole fields by one factor s (then clip to [lb, ub]) so that
    the solid fraction equals ``volfrac``.  Returns (P_projected, s).

    A design that already meets the target (to ``tol``) is returned unchanged
    with s = 1."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    target = 1.0 - volfrac                       # required mean(theta1 * theta2)

    def f(s):
        Q = np.clip(s * P, lb, ub)
        return float(np.mean(Q[:, 0] * Q[:, 1]) - target)

    f1 = f(1.0)
    if abs(f1) <= tol:
        return P.copy(), 1.0
    if f1 > 0:                                   # too little material: shrink holes
        lo_s, hi_s = 0.5, 1.0
        while f(lo_s) > 0 and lo_s > 1e-6:
            lo_s *= 0.5
    else:                                        # too much material: grow holes
        lo_s, hi_s = 1.0, 2.0
        while f(hi_s) < 0 and hi_s < 1e6:
            hi_s *= 2.0
    for _ in range(maxit):
        mid = 0.5 * (lo_s + hi_s)
        if f(mid) > 0:
            hi_s = mid
        else:
            lo_s = mid
        if hi_s - lo_s <= 1e-15 * max(1.0, hi_s):
            break
    s = 0.5 * (lo_s + hi_s)
    return np.clip(s * P, lb, ub), float(s)


def error_summary(errors) -> dict:
    """Signed mean, mean absolute and largest absolute value of ``errors``."""
    e = np.asarray(errors, dtype=float).ravel()
    return dict(mean_signed=float(e.mean()), mean_abs=float(np.abs(e).mean()),
                max_abs=float(np.abs(e).max()), n=int(e.size))


def compare_vectors(a, b) -> dict:
    """MSE, relative L2 error (w.r.t. ``b``) and cosine between two vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    d = a - b
    return dict(mse=float(np.mean(d ** 2)),
                rel_l2=float(np.linalg.norm(d) / np.linalg.norm(b)),
                cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def sensitivity_bases(mesh, material, x, kind="hole") -> dict:
    """Compliance sensitivities at ``x`` on two bases:

    raw       d(compliance)/d(physical variable), no filter
    filtered  after the authors' heuristic sensitivity filter (``filterfun``
              with ft = 1), which is what the paper's Figs. 8 and 11 compare
    """
    from .topopt import ComplianceProblem
    raw = ComplianceProblem(mesh, material, kind=kind, filter="none").raw_gradient(x)
    filt = ComplianceProblem(mesh, material, kind=kind, filter="sensitivity").gradient(x)
    return dict(raw=np.asarray(raw, dtype=float), filtered=np.asarray(filt, dtype=float))
