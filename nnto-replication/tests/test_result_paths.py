"""The result files must not depend on the machine they were written on:
model paths are stored relative to results/ (or to the original repository)
and resolved when loaded."""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import common  # noqa: E402


def test_model_ref_round_trip():
    p = os.path.join(common.RESULTS, "models", "new_x__y__s0.npz")
    ref = common.model_ref(p)
    assert ref == "models/new_x__y__s0.npz"
    assert common.resolve_model(ref) == p
    assert common.model_ref("") == ""


def test_absolute_paths_from_another_machine_are_resolved():
    old = "/some/other/machine/work/results/models/new_x__y__s0.npz"
    assert common.resolve_model(old) == os.path.join(common.RESULTS, "models", "new_x__y__s0.npz")
    assert common.model_ref("/elsewhere/NN_model_HBTO_1e+04") == "UPSTREAM/NN_model_HBTO_1e+04"


def test_shipped_result_files_hold_no_absolute_paths():
    for f in glob.glob(os.path.join(common.RESULTS, "*.json")):
        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == "path" and isinstance(v, str) and v:
                        assert not os.path.isabs(v), (f, v)
                        if v.startswith("models/"):
                            assert os.path.exists(common.resolve_model(v)), (f, v)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        with open(f) as fh:
            walk(json.load(fh))
