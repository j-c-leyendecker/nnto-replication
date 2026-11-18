"""Run the original repository's functions *verbatim*.

The published scripts execute training / optimisation at import time and need
TensorFlow, so they cannot simply be imported.  Instead the module-level
``def``/``class`` statements are extracted with :mod:`ast` and compiled into a
fresh namespace that provides the same global names the scripts use
(``np``, ``npm``, ``sp``, ``coo_matrix``, ``csc_matrix``, ``spsolve``, ``ceil``
and a ``tf`` stand-in).  No source line is modified.
"""
from __future__ import annotations

import ast
from math import ceil
from typing import Dict, Iterable, Optional

import numpy as np
import scipy as sp
import scipy.optimize  # noqa: F401  (scripts use sp.optimize.*)
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from .keras_shim import make_tf_module

try:                                    # numpy.matlib is deprecated but present
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import numpy.matlib as npm
except Exception:                       # pragma: no cover - future NumPy
    import types
    npm = types.SimpleNamespace(repmat=lambda a, m, n: np.tile(np.atleast_2d(a), (m, n)))


def load_original_namespace(script_path: str,
                            only: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Compile the top-level functions/classes of ``script_path``."""
    with open(script_path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=script_path)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and (only is None or n.name in set(only))]
    module = ast.Module(body=keep, type_ignores=[])
    ns: Dict[str, object] = {
        "np": np, "npm": npm, "sp": sp, "coo_matrix": coo_matrix,
        "csc_matrix": csc_matrix, "spsolve": spsolve, "ceil": ceil,
        "tf": make_tf_module(), "__name__": "original_script",
    }
    exec(compile(module, script_path, "exec"), ns)
    return ns


def setup_mbb(ns: Dict[str, object], nelx=60, nely=30, volfrac=0.5, rmin=2.7,
              ft=1, maxiter=1000, tolx=1e-3, hbto=False):
    """Replicate the 'RUN ... MBB EXAMPLE' section of the scripts."""
    model = ns["model"]
    model.nelx, model.nely = nelx, nely
    model.volfrac, model.rmin, model.ft = volfrac, rmin, ft
    model.maxiter, model.tolx = maxiter, tolx
    if hbto:
        x0 = (np.ones((nely * nelx, 2)) * np.sqrt(1 - volfrac)).flatten("F")
    else:
        x0 = (np.ones((nely * nelx, 1)) * volfrac).flatten()
    model.x = x0
    model.edofMat, model.iK, model.jK = ns["feaprep"](model)
    model.H, model.Hs = ns["filterprep"](model)
    dofs = 2 * (nelx + 1) * (nely + 1)
    F = coo_matrix((np.array([-1]), (np.array([1]), np.array([0]))), shape=(dofs, 1))
    model.dofs, model.F = dofs, F
    ns["dofs"] = dofs                     # Run_DBTO_SIMP.py reads the global
    fxd1 = np.arange(0, 2 * (nely + 1), 2)
    fxd2 = 2 * (nelx + 1) * (nely + 1) - 1
    model.fixeddofs = np.block([fxd1, fxd2])
    model.freedofs = np.setdiff1d(np.arange(0, dofs), model.fixeddofs)
    return model, x0
