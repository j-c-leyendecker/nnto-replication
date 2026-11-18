"""Helpers shared by the experiment scripts."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
# NNTO_RESULTS redirects inputs and outputs to another folder, e.g. to rerun an
# experiment without overwriting the shipped results (added 2026-11-18).
RESULTS = os.path.abspath(os.environ.get("NNTO_RESULTS", os.path.join(ROOT, "results")))
os.makedirs(RESULTS, exist_ok=True)


def find_upstream():
    cands = [os.environ.get("NNTO_UPSTREAM", ""),
             os.path.join(ROOT, "nn-derivative-to-main", "NN_derivatives_TO_v3_final"),
             os.path.join(ROOT, "..", "nn-derivative-to-main", "NN_derivatives_TO_v3_final"),
             os.path.join(ROOT, "nn-derivative-to", "NN_derivatives_TO_v3_final"),          # git clone
             os.path.join(ROOT, "..", "nn-derivative-to", "NN_derivatives_TO_v3_final"),
             os.path.join(ROOT, "..", "upstream", "nn-derivative-to-main", "NN_derivatives_TO_v3_final")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "Run_DBTO_NN.py")):
            return os.path.abspath(c)
    raise SystemExit("original repository not found: set NNTO_UPSTREAM to the folder "
                     "containing Run_DBTO_NN.py")


def model_ref(path):
    """Portable form of a model path, as stored in the result files:
    'models/<file>' for networks trained here (under results/models),
    'UPSTREAM/<name>' for the authors' saved models."""
    if not path:
        return path
    p = os.path.abspath(path)
    models = os.path.join(RESULTS, "models")
    if p.startswith(models + os.sep):
        return "models/" + os.path.relpath(p, models).replace(os.sep, "/")
    if os.path.basename(p).startswith("NN_model_"):
        return "UPSTREAM/" + os.path.basename(p)
    return path


def resolve_model(ref):
    """Inverse of model_ref. Also accepts absolute paths written on another
    machine by older versions of the scripts."""
    if not ref:
        return ref
    if ref.startswith("UPSTREAM/"):
        return os.path.join(find_upstream(), ref.split("/", 1)[1])
    if ref.startswith("models/"):
        return os.path.join(RESULTS, *ref.split("/"))
    if not os.path.exists(ref):
        norm = ref.replace("\\", "/")
        if "/results/models/" in norm:
            return os.path.join(RESULTS, "models", norm.rsplit("/", 1)[1])
        if norm.rsplit("/", 1)[-1].startswith("NN_model_"):
            return os.path.join(find_upstream(), norm.rsplit("/", 1)[1])
    return ref


def mean_ci(v, level=0.95):
    """Mean, sample SD and Student-t confidence half-width."""
    from scipy import stats
    v = np.asarray(v, dtype=float)
    n = v.size
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    hw = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return m, sd, hw


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
    return path
