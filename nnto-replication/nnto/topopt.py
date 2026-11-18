"""Compliance topology optimisation with pluggable material models.

``ComplianceProblem`` evaluates the objective, its gradient and the volume
constraint for density-based (one variable per element) or
homogenisation-based (hole width and height per element) parametrisations.

Options named ``*_original`` reproduce the published scripts bit-for-bit in
structure; the defaults of the *improved* configuration are documented in
``improved_options()``.  Every deviation from the original is a named option
so that its effect can be isolated (see experiments/ablation_*.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import optimize, sparse

from .fea import Mesh

__all__ = ["ComplianceProblem", "run_trust_constr", "run_oc", "run_mma",
           "original_options", "improved_options"]


def original_options() -> dict:
    """Settings that reproduce Run_DBTO_NN.py / Run_HBTO_NN.py."""
    return dict(filter="sensitivity", volume_jac_mode="original")


def improved_options() -> dict:
    """Consistent formulation: the gradient handed to the optimiser is the
    derivative of the objective that is evaluated."""
    return dict(filter="density", volume_jac_mode="exact")


@dataclass
class ComplianceProblem:
    mesh: Mesh
    material: object
    volfrac: float = 0.5
    kind: str = "density"            # 'density' (DBTO) or 'hole' (HBTO)
    filter: str = "sensitivity"      # 'sensitivity' | 'density' | 'none'
    volume_jac_mode: str = "original"   # 'original' | 'exact'
    grad_noise: float = 0.0          # relative round-off-level perturbation
    noise_seed: int = 0

    def __post_init__(self):
        self.ne = self.mesh.ne
        self.nvar = 1 if self.kind == "density" else 2
        self.n = self.ne * self.nvar
        self._cache_x = None
        self._cache = None
        self.n_eval = 0
        self._rng = np.random.default_rng(self.noise_seed)

    # ------------------------------------------------------------ helpers
    def x0(self) -> np.ndarray:
        if self.kind == "density":
            return np.full(self.ne, self.volfrac)
        return np.full(2 * self.ne, np.sqrt(1.0 - self.volfrac))

    def _split(self, x):
        return np.asarray(x, dtype=float).reshape(self.ne, self.nvar, order="F")

    def _phys(self, X):
        """Physical fields seen by the material model."""
        if self.filter == "density":
            return np.stack([self.mesh.H @ X[:, j] / self.mesh.Hs
                             for j in range(self.nvar)], axis=1)
        return X

    def _evaluate(self, x):
        xb = np.asarray(x, dtype=float)
        if self._cache_x is not None and np.array_equal(xb, self._cache_x):
            return self._cache
        X = self._split(xb)
        P = self._phys(X)
        C, dC = self.material.evaluate(P)
        U, q = self.mesh.solve(C)
        comp = float(np.sum(C[None] * q))                   # sum over load cases
        qs = q.sum(axis=0)                                  # (ne, 4)
        dc = -np.einsum("ek,ekj->ej", qs, dC)               # (ne, nvar)  eq. (11), with the sign
        self.n_eval += 1
        self._cache_x = xb.copy()
        self._cache = dict(comp=comp, dc_raw=dc, X=X, P=P, C=C, U=U)
        return self._cache

    # ---------------------------------------------------------- objective
    def compliance(self, x) -> float:
        return self._evaluate(x)["comp"]

    def raw_gradient(self, x) -> np.ndarray:
        """d(compliance)/d(physical variable), before any filter."""
        return self._evaluate(x)["dc_raw"].flatten("F")

    def gradient(self, x) -> np.ndarray:
        """Gradient passed to the optimiser."""
        ev = self._evaluate(x)
        dc, X = ev["dc_raw"], ev["X"]
        m = self.mesh
        if self.filter == "sensitivity":
            # Sigmund's heuristic filter, as in the original filterfun()
            g = np.stack([m.H @ (X[:, j] * dc[:, j]) / m.Hs / np.maximum(1e-3, X[:, j])
                          for j in range(self.nvar)], axis=1)
        elif self.filter == "density":
            # exact chain rule through the linear density filter
            g = np.stack([m.H.T @ (dc[:, j] / m.Hs) for j in range(self.nvar)], axis=1)
        elif self.filter == "none":
            g = dc
        else:
            raise ValueError(self.filter)
        g = g.flatten("F")
        if self.grad_noise:
            g = g * (1.0 + self.grad_noise * self._rng.standard_normal(g.size))
        return g

    # --------------------------------------------------------- constraint
    def volume(self, x) -> float:
        """Volume-fraction constraint value  v(theta) - Vt  (eqs. 12-13).

        NOTE: like the original, the constraint is evaluated on the design
        variables themselves; with ``filter='density'`` it is evaluated on
        the filtered (physical) field so that it is consistent with the
        stiffness model."""
        X = self._split(x)
        P = self._phys(X) if self.filter == "density" else X
        if self.kind == "density":
            return float(P.sum() / self.ne - self.volfrac)
        return float((self.ne - np.sum(P[:, 0] * P[:, 1])) / self.ne - self.volfrac)

    def volume_jac(self, x) -> np.ndarray:
        X = self._split(x)
        m = self.mesh
        if self.kind == "density":
            if self.volume_jac_mode == "original":
                return np.ones(self.ne)               # original dnlcon(): missing 1/ne
            g = np.full(self.ne, 1.0 / self.ne)
            if self.filter == "density":
                g = m.H.T @ (g / m.Hs)
            return g
        P = self._phys(X) if self.filter == "density" else X
        g = np.stack([-P[:, 1] / self.ne, -P[:, 0] / self.ne], axis=1)
        if self.filter == "density":
            g = np.stack([m.H.T @ (g[:, j] / m.Hs) for j in range(2)], axis=1)
        return g.flatten("F")


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
def run_trust_constr(prob: ComplianceProblem, x0=None, tol: float = 1e-3,
                     maxiter: int = 1000, sparse_jacobian: Optional[bool] = None,
                     lb: float = 0.0, ub: float = 1.0, verbose: int = 0,
                     record_designs: Optional[List[int]] = None,
                     hess=None) -> dict:
    """SciPy ``trust-constr`` exactly as called by the original scripts
    (bounds [0, 1] kept feasible, equality volume constraint, ``tol`` applied
    to xtol/gtol/barrier_tol, BFGS Hessian approximation).

    ``sparse_jacobian=True`` only changes the linear algebra used for the
    constraint projections (sparse LU instead of dense QR); the algorithm is
    otherwise identical.
    """
    x0 = prob.x0() if x0 is None else np.asarray(x0, dtype=float)
    hist = dict(nit=[], fun=[], tr_radius=[], optimality=[], constr_violation=[],
                nfev=[], barrier=[], time=[])
    designs: Dict[int, np.ndarray] = {}
    t0 = time.time()

    def cb(xk, state):
        hist["nit"].append(state.nit)
        hist["fun"].append(state.fun)
        hist["tr_radius"].append(state.tr_radius)
        hist["optimality"].append(state.optimality)
        hist["constr_violation"].append(state.constr_violation)
        hist["nfev"].append(state.nfev)
        hist["barrier"].append(state.barrier_parameter)
        hist["time"].append(time.time() - t0)
        if record_designs and state.nit in record_designs:
            designs[state.nit] = xk.copy()
        return False

    if sparse_jacobian:
        jac = lambda x: sparse.csr_matrix(prob.volume_jac(x).reshape(1, -1))
    else:
        jac = prob.volume_jac
    kw = {}
    if hess is not None:
        kw["hess"] = hess
    bds = optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)
    con = optimize.NonlinearConstraint(prob.volume, lb=0.0, ub=0.0, jac=jac,
                                       keep_feasible=True)
    opts = {"maxiter": maxiter, "verbose": verbose}
    if sparse_jacobian is not None:
        opts["sparse_jacobian"] = sparse_jacobian
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize.minimize(prob.compliance, x0, method="trust-constr",
                                jac=prob.gradient, bounds=bds, constraints=con,
                                tol=tol, options=opts, callback=cb, **kw)
    return dict(x=res.x, fun=float(res.fun), nit=int(res.nit), nfev=int(res.nfev),
                status=int(res.status), message=str(res.message),
                optimality=float(res.optimality),
                constr_violation=float(res.constr_violation),
                time=time.time() - t0, history={k: np.array(v) for k, v in hist.items()},
                designs=designs)


def run_oc(prob: ComplianceProblem, x0=None, maxiter: int = 500, tol: float = 1e-3,
           move: float = 0.2, record: bool = True) -> dict:
    """Optimality-criteria update of the 88-line code (density problems).

    Stops when the largest design change is below ``tol`` -- which is the
    criterion the paper *states* ("until the maximum change in theta is less
    than 1e-3")."""
    if prob.kind != "density":
        raise ValueError("run_oc supports density parametrisations only")
    x = prob.x0() if x0 is None else np.asarray(x0, dtype=float).copy()
    hist = dict(fun=[], change=[])
    t0 = time.time()
    m = prob.mesh
    for it in range(1, maxiter + 1):
        c = prob.compliance(x)
        g = prob.gradient(x)
        dv = np.ones(prob.ne)
        if prob.filter == "density":
            dv = m.H.T @ (dv / m.Hs)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lm = 0.5 * (l1 + l2)
            xn = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(
                x + move, x * np.sqrt(np.maximum(-g, 0.0) / dv / lm)))))
            if prob.volume(xn) > 0:
                l1 = lm
            else:
                l2 = lm
        change = float(np.max(np.abs(xn - x)))
        x = xn
        hist["fun"].append(c)
        hist["change"].append(change)
        if change < tol:
            break
    return dict(x=x, fun=float(prob.compliance(x)), nit=it, time=time.time() - t0,
                history={k: np.array(v) for k, v in hist.items()})


def run_mma(prob: ComplianceProblem, x0=None, maxiter: int = 500, tol: float = 1e-3,
            move: float = 0.2, lb: float = 0.0, ub: float = 1.0,
            asyinit: float = 0.5, asyincr: float = 1.2, asydecr: float = 0.7) -> dict:
    """Method of Moving Asymptotes (Svanberg 1987; 2007 update of the
    approximation coefficients) for one inequality constraint
    v(theta) - Vt <= 0.

    With a single constraint the convex MMA subproblem is solved exactly
    through its one-dimensional dual: for a given multiplier the minimiser is
    closed-form per variable, and the multiplier is found by bisection on the
    (monotone) approximated constraint.  Stops when the largest design change
    is below ``tol`` -- the criterion the paper states -- or after
    ``maxiter`` iterations.
    """
    n = prob.n
    x = prob.x0() if x0 is None else np.asarray(x0, dtype=float).copy()
    xmin = np.full(n, float(lb)); xmax = np.full(n, float(ub))
    x = np.clip(x, xmin, xmax)
    span = np.maximum(xmax - xmin, 1e-5)
    raa0 = 1e-5
    xold1 = x.copy(); xold2 = x.copy()
    low = x - asyinit * span; upp = x + asyinit * span
    hist = dict(fun=[], change=[], volume=[], time=[])
    t0 = time.time()
    it = 0
    for it in range(1, maxiter + 1):
        f0 = prob.compliance(x); g0 = prob.gradient(x)
        f1 = prob.volume(x); g1 = np.asarray(prob.volume_jac(x)).ravel()
        # --- asymptotes
        if it > 2:
            zz = (x - xold1) * (xold1 - xold2)
            fac = np.where(zz > 0, asyincr, np.where(zz < 0, asydecr, 1.0))
            low = x - fac * (xold1 - low)
            upp = x + fac * (upp - xold1)
            low = np.clip(low, x - 10 * span, x - 0.01 * span)
            upp = np.clip(upp, x + 0.01 * span, x + 10 * span)
        alpha = np.maximum.reduce([xmin, low + 0.1 * (x - low), x - move * span])
        beta = np.minimum.reduce([xmax, upp - 0.1 * (upp - x), x + move * span])
        ux2 = (upp - x) ** 2; xl2 = (x - low) ** 2
        def pq(g):
            gp, gm = np.maximum(g, 0), np.maximum(-g, 0)
            return (ux2 * (1.001 * gp + 0.001 * gm + raa0 / span),
                    xl2 * (0.001 * gp + 1.001 * gm + raa0 / span))
        p0, q0 = pq(g0); p1, q1 = pq(g1)
        b = np.sum(p1 / (upp - x) + q1 / (x - low)) - f1      # approx: sum(p1/(U-x)+q1/(x-L)) - b <= 0

        def xstar(lam):
            P = p0 + lam * p1; Q = q0 + lam * q1
            sp, sq = np.sqrt(P), np.sqrt(Q)
            return np.clip((upp * sq + low * sp) / (sp + sq), alpha, beta)

        def gapprox(xx):
            return np.sum(p1 / (upp - xx) + q1 / (xx - low)) - b

        lam_lo, lam_hi = 0.0, 1.0
        if gapprox(xstar(0.0)) <= 0:
            xn = xstar(0.0)
        else:
            while gapprox(xstar(lam_hi)) > 0 and lam_hi < 1e12:
                lam_hi *= 10.0
            for _ in range(100):
                lam = 0.5 * (lam_lo + lam_hi)
                if gapprox(xstar(lam)) > 0:
                    lam_lo = lam
                else:
                    lam_hi = lam
                if lam_hi - lam_lo <= 1e-12 * max(1.0, lam_hi):
                    break
            xn = xstar(lam_hi)
        change = float(np.max(np.abs(xn - x)))
        xold2, xold1, x = xold1, x.copy(), xn
        hist["fun"].append(f0); hist["change"].append(change); hist["volume"].append(f1)
        hist["time"].append(time.time() - t0)
        if change < tol:
            break
    return dict(x=x, fun=float(prob.compliance(x)), nit=it, time=time.time() - t0,
                volume=float(prob.volume(x)), converged=bool(hist["change"][-1] < tol),
                history={k: np.array(v) for k, v in hist.items()})
