"""Original vs corrected: runs the authors' unmodified dfeafun/dcfun (via the
NumPy stand-in for Keras in ../nnto) with predict casting to float32 (as
published) and float64 (with the fix).

usage: python fix_float64/compare_before_after.py <folder containing Run_DBTO_NN.py>"""
import os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from nnto.networks import load_keras_mlp
from nnto.baseline.keras_shim import ShimModel
from nnto.baseline.original import load_original_namespace, setup_mbb

UP = sys.argv[1]
for name, script, hb in (("DBTO", "Run_DBTO_NN.py", False), ("HBTO", "Run_HBTO_NN.py", True)):
    net = load_keras_mlp(os.path.join(UP, f"NN_model_{name}_1e+04"))
    ns = load_original_namespace(os.path.join(UP, script))
    model, x0 = setup_mbb(ns, hbto=hb)
    rng = np.random.default_rng(0)
    xr = rng.uniform(0.2, 1.0, x0.size)
    for label, x in (("initial design", x0), ("random design ", xr)):
        out = {}
        for fx in ("float32", "float64"):
            model.nn_model = ShimModel(net, floatx=fx)
            for der in (1, 2):                     # 1 = NNJ, 2 = CFD
                model.der_type = der
                out[(fx, der)] = ns["dfeafun"](x.copy(), model)
        rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
        print(f"{name} {label}: max rel |CFD - NNJ|  as published {rel(out[('float32', 2)], out[('float32', 1)]):.2e}"
              f"   with fix {rel(out[('float64', 2)], out[('float64', 1)]):.2e}")
