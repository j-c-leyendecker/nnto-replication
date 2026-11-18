"""Training of surrogate networks in NumPy.

Supports plain value regression (as in the paper) and Sobolev training, in
which the input-derivatives of the network are fitted to derivative labels as
well (Czarnecki et al. 2017).  The loss gradient with respect to the weights
is obtained with the package's own AD engine: the Jacobian of the network is
propagated in forward mode (one sweep per input dimension) through a graph
whose weights are reverse-mode tape nodes ("forward-over-reverse").

Loss (relative, per component, so that no output or derivative dominates):

    L = sum_k mean((y_k - Y_k)^2) / var(Y_k)
        + lam * sum_{k,j} mean((dy_k/dx_j - G_kj)^2) / var(G_kj)

Optimisers: full-batch L-BFGS (fast and deterministic; default) or mini-batch
Adam with the paper's hyper-parameters (learning rate 1e-3, batch 32).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import optimize

from .autodiff import Dual, Tape, Var
from .networks import (Affine, ChebyKAN, CholeskyHead, Dense, FourierFeatures,
                       Network, Residual, SoftFloor)

__all__ = ["init_mlp", "train", "TrainResult", "predict_ensemble"]


def _glorot(rng, n_in, n_out):
    lim = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-lim, lim, size=(n_in, n_out))


def init_mlp(xdim: int, ydim: int, hidden: Sequence[int] = (64,),
             activation: str = "sigmoid", X=None, Y=None, seed: int = 0,
             head: str = "denorm", residual: bool = False,
             fourier: Optional[int] = None, fourier_scale: float = 2.0,
             lo: float = 1e-9, siren_omega0: Optional[float] = None,
             kan_degree: Optional[int] = None) -> Network:
    """Glorot-initialised network with input standardisation.

    head : 'denorm'    y = z * std_y + mean_y            (proper de-normalisation)
           'original'  y = (z - mean_y) / std_y          (what the published code does)
           'softfloor' y = lo + softplus(z) * std_y      (positive by construction)
           'cholesky'  admissible orthotropic tensor (ydim must be 4)
           'none'
    """
    rng = np.random.default_rng(seed)
    blocks: List[object] = []
    if X is not None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        blocks.append(Affine(1.0 / sd, -mu / sd))
    width_in = xdim
    if fourier:
        blocks.append(FourierFeatures(fourier_scale * rng.normal(size=(xdim, fourier))))
        width_in = 2 * fourier
    if kan_degree is not None:
        # Chebyshev KAN: every layer (including the last) is a KAN layer
        widths = list(hidden) + [ydim]
        for h in widths:
            K = kan_degree
            std = 1.0 / np.sqrt(width_in * (K + 1))
            blocks.append(ChebyKAN([rng.normal(0.0, std, size=(width_in, h)) for _ in range(K + 1)]))
            width_in = h
        hidden = ()
    for i, h in enumerate(hidden):
        if siren_omega0 is not None:
            # SIREN initialisation (Sitzmann et al. 2020) with omega0 folded
            # into the weights: sin(omega0 * (x W + b)) == sin(x W' + b')
            if i == 0:
                W = rng.uniform(-1.0 / width_in, 1.0 / width_in, size=(width_in, h)) * siren_omega0
            else:
                lim = np.sqrt(6.0 / width_in)
                W = rng.uniform(-lim, lim, size=(width_in, h))
            b = rng.uniform(-1.0, 1.0, size=h) / np.sqrt(width_in) * (siren_omega0 if i == 0 else 1.0)
            blocks.append(Dense(W, b, "sin"))
            width_in = h
            continue
        if residual and i > 0 and h == width_in:
            blocks.append(Residual([Dense(_glorot(rng, width_in, h), np.zeros(h), activation),
                                    Dense(0.1 * _glorot(rng, h, h), np.zeros(h), "linear")]))
        else:
            blocks.append(Dense(_glorot(rng, width_in, h), np.zeros(h), activation))
        width_in = h
    if kan_degree is None:
        blocks.append(Dense(_glorot(rng, width_in, ydim), np.zeros(ydim), "linear"))
    if Y is not None:
        my, sy = Y.mean(axis=0), Y.std(axis=0)
    else:
        my, sy = np.zeros(ydim), np.ones(ydim)
    if head == "denorm":
        blocks.append(Affine(sy, my))
    elif head == "original":
        blocks.append(Affine(1.0 / sy, -my / sy))
    elif head == "softfloor":
        blocks.append(SoftFloor(np.full(ydim, lo)))
        blocks.append(Affine(sy, np.zeros(ydim)))
    elif head == "cholesky":
        if ydim != 4:
            raise ValueError("the Cholesky head needs ydim = 4")
        blocks.append(CholeskyHead(np.ones(4)))
    elif head != "none":
        raise ValueError(head)
    return Network(blocks, xdim, ydim, {"head": head, "activation": activation,
                                        "hidden": tuple(hidden), "seed": seed})


# --------------------------------------------------------------------------- #
def _swap_params(net: Network, new: List[object]) -> Network:
    """Shallow copy of ``net`` whose dense layers use the objects in ``new``."""
    it = iter(new)

    def conv(blk):
        if isinstance(blk, Dense):
            return Dense(next(it), next(it), blk.activation)
        if isinstance(blk, Residual):
            return Residual([conv(b) for b in blk.body])
        if isinstance(blk, ChebyKAN):
            return ChebyKAN([next(it) for _ in blk.C])
        return blk
    return Network([conv(b) for b in net.blocks], net.xdim, net.ydim, dict(net.meta))


def _loss_and_grad(net, shapes, flat, X, Y, G, wy, wg, lam):
    tape = Tape()
    params, pos = [], 0
    for shp in shapes:
        n = int(np.prod(shp))
        params.append(Var(flat[pos:pos + n].reshape(shp), tape))
        pos += n
    vnet = _swap_params(net, params)
    if lam > 0 and G is not None:
        loss = None
        for j in range(net.xdim):
            t = np.zeros_like(X)
            t[:, j] = 1.0
            out = vnet.forward(Dual(X, t))
            if j == 0:
                r = out.v - Y
                loss = (r * r * wy).sum()
            rj = out.t - G[:, :, j]
            loss = loss + (rj * rj * (lam * wg[:, j])).sum()
    else:
        r = vnet.forward(X) - Y
        loss = (r * r * wy).sum()
    loss.backward()
    grad = np.concatenate([(p.grad if p.grad is not None else np.zeros(p.v.shape)).ravel()
                           for p in params])
    value = float(loss.v)
    # the tape and its nodes reference each other; break the cycles now so the
    # (large) intermediate arrays are freed immediately instead of waiting for
    # the cyclic garbage collector
    for node in tape.nodes:
        node.parents = ()
        node.grad = None
    tape.nodes.clear()
    return value, grad


@dataclass
class TrainResult:
    net: Network
    loss_history: np.ndarray
    seconds: float
    n_iter: int
    info: dict


def train(net: Network, X, Y, G=None, lam: float = 0.0, method: str = "lbfgs",
          maxiter: int = 3000, epochs: int = 200, batch: int = 32, lr: float = 1e-3,
          seed: int = 0, tol: float = 1e-12, verbose: bool = False) -> TrainResult:
    """Fit ``net`` to values ``Y`` (N, ydim) and, if ``lam`` > 0, to derivative
    labels ``G`` (N, ydim, xdim).  The network is modified in place."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N = X.shape[0]
    params = net.params()
    shapes = [p.shape for p in params]
    flat0 = np.concatenate([p.ravel() for p in params])
    wy = 1.0 / (np.maximum(Y.var(axis=0), 1e-300) * N)
    wg = None
    if G is not None and lam > 0:
        G = np.asarray(G, dtype=float)
        wg = 1.0 / (np.maximum(G.var(axis=0), 1e-300) * N)        # (ydim, xdim)
    hist: List[float] = []
    t0 = time.time()

    if method == "lbfgs":
        def fg(flat):
            l, g = _loss_and_grad(net, shapes, flat, X, Y, G, wy, wg, lam)
            hist.append(l)
            return l, g
        res = optimize.minimize(fg, flat0, jac=True, method="L-BFGS-B",
                                options=dict(maxiter=maxiter, maxfun=int(maxiter * 1.25),
                                             ftol=tol, gtol=1e-12, maxcor=30))
        flat, nit, info = res.x, res.nit, {"message": str(res.message)}
    elif method == "adam":
        rng = np.random.default_rng(seed)
        flat = flat0.copy()
        m = np.zeros_like(flat); v = np.zeros_like(flat)
        b1, b2, eps = 0.9, 0.999, 1e-7
        step = 0
        for ep in range(epochs):
            perm = rng.permutation(N)
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                nb = idx.size
                Gb = None if G is None else G[idx]
                # per-batch mean (as Keras' MSE), same relative weights
                l, g = _loss_and_grad(net, shapes, flat, X[idx], Y[idx], Gb,
                                      wy * N / nb, None if wg is None else wg * N / nb, lam)
                step += 1
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                flat -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
            hist.append(l)
            if verbose and ep % 10 == 0:
                print(f"epoch {ep}: batch loss {l:.3e}")
        nit, info = step, {"epochs": epochs}
    else:
        raise ValueError(method)

    pos = 0
    for p in params:                       # write back in place
        n = p.size
        p[...] = flat[pos:pos + n].reshape(p.shape)
        pos += n
    return TrainResult(net, np.array(hist), time.time() - t0, nit, info)


def predict_ensemble(nets: Sequence[Network], x):
    """Mean / standard deviation of values and Jacobians over an ensemble."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    Ys = np.stack([n.forward(x) for n in nets])
    Js = np.stack([n.jacobian_nnj(x) for n in nets])
    return Ys.mean(0), Ys.std(0, ddof=1), Js.mean(0), Js.std(0, ddof=1)
