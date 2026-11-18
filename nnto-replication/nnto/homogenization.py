"""Numerical homogenization of the square cell with a centred rectangular hole.

Two discretisations of the same unit cell are provided.

``voxel_CH``
    The discretisation used to build the paper's training set
    (``HBTO_CH_2D_N10000_100x100.mat``): a uniform ``nel x nel`` grid whose
    elements are solid or void according to the position of their centres.
    The resulting map ``theta -> C^H`` is *piecewise constant* (the hole snaps
    to the grid), so it has no useful derivative.

``fitted_CH``
    A rectilinear mesh whose grid lines follow the hole boundary, so that the
    element sizes -- and therefore ``C^H`` -- depend smoothly on ``theta``.
    The exact derivative of the discrete ``C^H`` with respect to the hole
    dimensions is returned as well.  Because ``C^H_ij`` is a stationary value
    of the cell energy, no additional linear solves are needed for it
    (envelope theorem / self-adjointness):

        dC^H_ij/dp = 1/|Y| * [ sum_e dA_e C_ij - chi_i^T dF_j - dF_i^T chi_j
                               + chi_i^T dK chi_j ]

    with the cell solutions ``chi`` held fixed.  Mesh refinement of this
    quantity is the independent ground truth for the HBTO sensitivities that
    the paper states is unavailable.

Notation: ``theta1`` = hole width / cell width, ``theta2`` = hole height /
cell height (as in ``create_Micro`` of the original Run_HBTO_NN.py); Voigt
order ``[xx, yy, xy]`` with engineering shear strain; plane strain.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

__all__ = ["plane_strain_C", "voxel_CH", "fitted_CH", "fitted_CH_converged",
           "voxel_effective_theta"]


def plane_strain_C(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, mu]])


# --------------------------------------------------------------------------- #
# Q4 element integrals on the unit square (2x2 Gauss, exact for rectangles)
# --------------------------------------------------------------------------- #
def _unit_square_integrals(C: np.ndarray):
    g = 0.5 + np.array([-1.0, 1.0]) / (2.0 * np.sqrt(3.0))
    Kxx = np.zeros((8, 8))
    Kyy = np.zeros((8, 8))
    Kxy = np.zeros((8, 8))
    Fx = np.zeros((8, 3))
    Fy = np.zeros((8, 3))
    for xi in g:
        for eta in g:
            dNx = np.array([-(1 - eta), (1 - eta), eta, -eta])
            dNy = np.array([-(1 - xi), -xi, xi, (1 - xi)])
            Gx = np.zeros((3, 8))
            Gy = np.zeros((3, 8))
            Gx[0, 0::2] = dNx          # eps_xx = du/dx
            Gx[2, 1::2] = dNx          # gamma_xy gets dv/dx
            Gy[1, 1::2] = dNy          # eps_yy = dv/dy
            Gy[2, 0::2] = dNy          # gamma_xy gets du/dy
            w = 0.25
            Kxx += w * Gx.T @ C @ Gx
            Kyy += w * Gy.T @ C @ Gy
            Kxy += w * (Gx.T @ C @ Gy + Gy.T @ C @ Gx)
            Fx += w * Gx.T @ C
            Fy += w * Gy.T @ C
    return Kxx, Kyy, Kxy, Fx, Fy


def _edof_periodic(nx: int, ny: int) -> np.ndarray:
    """Element dof table (ne, 8) with periodic node identification.

    Element (i, j) -> index e = i*ny + j; nodes counter-clockwise from the
    lower-left corner; dofs [u1 v1 u2 v2 u3 v3 u4 v4].
    """
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i = i.ravel()
    j = j.ravel()
    ip = (i + 1) % nx
    jp = (j + 1) % ny
    nodes = np.stack([i * ny + j, ip * ny + j, ip * ny + jp, i * ny + jp], axis=1)
    edof = np.empty((nodes.shape[0], 8), dtype=np.int64)
    edof[:, 0::2] = 2 * nodes
    edof[:, 1::2] = 2 * nodes + 1
    return edof


def _homogenize_rectilinear(hx: np.ndarray, hy: np.ndarray, scale: np.ndarray,
                            C0: np.ndarray, dhx: Optional[list] = None,
                            dhy: Optional[list] = None):
    """Homogenise a periodic rectilinear mesh.

    hx (nx,), hy (ny,) : element sizes; cell area = sum(hx) * sum(hy)
    scale (nx, ny)     : stiffness multiplier of each element (0 removes it)
    dhx, dhy           : lists of size-derivative arrays (one per parameter)

    Returns CH (3,3) and a list of dCH/dp (3,3) per parameter.
    """
    nx, ny = hx.size, hy.size
    ne = nx * ny
    Kxx, Kyy, Kxy, Fx, Fy = _unit_square_integrals(C0)
    edof = _edof_periodic(nx, ny)
    HX = np.repeat(hx, ny)
    HY = np.tile(hy, nx)
    s = scale.reshape(ne)
    act = s > 0
    ea = np.flatnonzero(act)
    r1 = (HY / HX)[ea] * s[ea]
    r2 = (HX / HY)[ea] * s[ea]
    ke = (r1[:, None, None] * Kxx + r2[:, None, None] * Kyy
          + s[ea][:, None, None] * Kxy)                        # (na, 8, 8)
    fe = ((HY[ea] * s[ea])[:, None, None] * Fx
          + (HX[ea] * s[ea])[:, None, None] * Fy)               # (na, 8, 3)
    ed = edof[ea]
    ndof = 2 * ne
    iK = np.repeat(ed, 8, axis=1).ravel()
    jK = np.tile(ed, (1, 8)).ravel()
    K = sparse.coo_matrix((ke.ravel(), (iK, jK)), shape=(ndof, ndof)).tocsc()
    F = np.zeros((ndof, 3))
    for c in range(3):
        np.add.at(F[:, c], ed.ravel(), fe[:, :, c].ravel())
    used = np.zeros(ndof, dtype=bool)
    used[ed.ravel()] = True
    # pin one node of an active element to remove the rigid translations
    pin = ed[0, :2]
    free = used.copy()
    free[pin] = False
    fidx = np.flatnonzero(free)
    chi = np.zeros((ndof, 3))
    lu = splu(K[fidx][:, fidx].tocsc())
    chi[fidx] = lu.solve(F[fidx])
    area = float(hx.sum() * hy.sum())
    solidA = float(np.sum(HX[ea] * HY[ea] * s[ea]))
    CH = (solidA * C0 - F.T @ chi) / area
    CH = 0.5 * (CH + CH.T)

    dCH = []
    if dhx is not None:
        chie = chi[ed]                                           # (na, 8, 3)
        Qxx = np.einsum("eai,ab,ebj->eij", chie, Kxx, chie)
        Qyy = np.einsum("eai,ab,ebj->eij", chie, Kyy, chie)
        Px = np.einsum("eai,aj->eij", chie, Fx)                 # chi_i^T Fx e_j
        Py = np.einsum("eai,aj->eij", chie, Fy)
        for dx_, dy_ in zip(dhx, dhy):
            DX = np.repeat(dx_, ny)[ea]
            DY = np.tile(dy_, nx)[ea]
            hxa, hya, sa = HX[ea], HY[ea], s[ea]
            dr1 = (DY * hxa - hya * DX) / hxa ** 2 * sa
            dr2 = (DX * hya - hxa * DY) / hya ** 2 * sa
            dA = np.sum((DX * hya + hxa * DY) * sa)
            chiTdKchi = (np.einsum("e,eij->ij", dr1, Qxx)
                         + np.einsum("e,eij->ij", dr2, Qyy))
            chiTdF = (np.einsum("e,eij->ij", DY * sa, Px)
                      + np.einsum("e,eij->ij", DX * sa, Py))    # [i,j]=chi_i^T dF_j
            d = (dA * C0 - chiTdF - chiTdF.T + chiTdKchi) / area
            dCH.append(0.5 * (d + d.T))
    return CH, dCH


# --------------------------------------------------------------------------- #
# Voxel discretisation (as used for the published training data)
# --------------------------------------------------------------------------- #
def voxel_effective_theta(theta, nel: int = 100):
    """Hole size actually represented on the ``nel`` voxel grid.

    An element is void when its centre lies strictly inside the hole, hence
    theta is rounded to a multiple of 2/nel.
    """
    theta = np.asarray(theta, dtype=float)
    c = (2 * np.arange(nel // 2) + 1) / nel                # centres in (0, 1)
    n_void = 2 * np.sum(c[None, :] < theta.reshape(-1, 1), axis=1)
    return (n_void / nel).reshape(theta.shape)


def voxel_CH(theta1: float, theta2: float, nel: int = 100, E: float = 1.0,
             nu: float = 0.3, Emin: float = 1e-9) -> np.ndarray:
    """Homogenised tensor [C11, C12, C22, C66] on the voxel grid."""
    C0 = plane_strain_C(E, nu)
    c = -1 + (2 * np.arange(nel) + 1) / nel                # centres in (-1, 1)
    vx = np.abs(c) < theta1
    vy = np.abs(c) < theta2
    scale = np.where(vx[:, None] & vy[None, :], Emin / E, 1.0)
    h = np.full(nel, 1.0 / nel)
    CH, _ = _homogenize_rectilinear(h, h, scale, C0)
    return np.array([CH[0, 0], CH[0, 1], CH[1, 1], CH[2, 2]])


# --------------------------------------------------------------------------- #
# Body-fitted discretisation with exact parameter derivatives
# --------------------------------------------------------------------------- #
def _graded(n: int, grading: float, side: str) -> np.ndarray:
    """Weights (sum 1) of ``n`` intervals refined towards ``side``.

    Algebraic grading: the k-th node sits at ``(k/n)**grading`` (measured from
    the refined end).  ``grading`` = 1 is a uniform partition.  The re-entrant
    corners of the hole produce an ``r**0.545`` displacement singularity, for
    which uniform bilinear elements converge only like ``h**1.09`` in energy;
    ``grading`` > 1/0.545 restores the optimal ``h**2`` rate.
    """
    if n == 1 or grading == 1.0:
        return np.full(n, 1.0 / n)
    if side == "both":
        m = n // 2
        if n % 2:
            raise ValueError("an even number of elements is needed across the hole")
        half = np.diff((np.arange(m + 1) / m) ** grading)      # small first
        w = np.concatenate([half, half[::-1]])
    else:
        w = np.diff((np.arange(n + 1) / n) ** grading)         # small first
        if side == "right":
            w = w[::-1]
    return w / w.sum()


def _strip_sizes(theta: float, ns: int, nh: int, grading: float):
    """Element sizes along one axis and their derivative w.r.t. theta."""
    wl = _graded(ns, grading, "right")     # solid wall, refined towards the hole
    wr = wl[::-1]
    wh = _graded(nh, grading, "both")      # hole strip, refined towards both edges
    h = np.concatenate([wl * (1 - theta) / 2, wh * theta, wr * (1 - theta) / 2])
    dh = np.concatenate([-wl / 2, wh, -wr / 2])
    return h, dh


def fitted_CH(theta1: float, theta2: float, ns: int = 24, nh: int = 48,
              grading: float = 2.5, E: float = 1.0, nu: float = 0.3,
              void_scale: float = 0.0, derivatives: bool = True):
    """Homogenised tensor and its exact parameter derivatives on a body-fitted
    mesh with ``ns`` elements across each solid wall and ``nh`` across the hole.

    Returns
    -------
    C  : (4,)   [C11, C12, C22, C66]
    dC : (4, 2) dC/dtheta1, dC/dtheta2   (only if ``derivatives``)
    """
    if not (0.0 < theta1 < 1.0 and 0.0 < theta2 < 1.0):
        raise ValueError("fitted_CH needs 0 < theta < 1 (the mesh degenerates)")
    C0 = plane_strain_C(E, nu)
    hx, dhx = _strip_sizes(theta1, ns, nh, grading)
    hy, dhy = _strip_sizes(theta2, ns, nh, grading)
    nxy = 2 * ns + nh
    hole = np.zeros(nxy, dtype=bool)
    hole[ns:ns + nh] = True
    scale = np.where(hole[:, None] & hole[None, :], void_scale, 1.0)
    zx, zy = np.zeros_like(hx), np.zeros_like(hy)
    if derivatives:
        CH, dCH = _homogenize_rectilinear(hx, hy, scale, C0,
                                          dhx=[dhx, zx], dhy=[zy, dhy])
    else:
        CH, dCH = _homogenize_rectilinear(hx, hy, scale, C0)
    pick = lambda M: np.array([M[0, 0], M[0, 1], M[1, 1], M[2, 2]])
    C = pick(CH)
    if not derivatives:
        return C
    return C, np.stack([pick(dCH[0]), pick(dCH[1])], axis=1)


def fitted_CH_converged(theta1: float, theta2: float, levels=(12, 24, 48),
                        grading: float = 2.5, **kw):
    """Mesh-refinement study at one point.

    Returns the finest-level values, a Richardson-extrapolated estimate using
    the observed convergence order, and the estimated discretisation error
    (|finest - extrapolated|).
    """
    Cs, dCs = [], []
    for n in levels:
        C, dC = fitted_CH(theta1, theta2, ns=n, nh=2 * n, grading=grading, **kw)
        Cs.append(C)
        dCs.append(dC)
    out = {}
    for name, seq in (("C", Cs), ("dC", dCs)):
        a, b, c = seq[-3], seq[-2], seq[-1]
        num, den = (b - a), (c - b)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / den                                   # = 2**p
        ok = np.isfinite(r) & (r > 1.05) & (r < 16)
        extrap = np.where(ok, c + den / (np.where(ok, r, 2.0) - 1.0), c)
        out[name] = c
        out[name + "_extrap"] = extrap
        out[name + "_err"] = np.abs(extrap - c)
        out[name + "_order"] = np.where(ok, np.log2(np.where(ok, r, 2.0)), np.nan)
    return out
