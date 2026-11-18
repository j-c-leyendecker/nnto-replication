"""The original code (run verbatim through the Keras stand-in) and the
vectorised port both reproduce the numbers the authors saved -- and they do
so only if ``Model.predict`` is taken to round its inputs to float32."""
import os

import numpy as np
import pytest

from nnto.baseline.keras_shim import ShimModel
from nnto.baseline.original import load_original_namespace, setup_mbb
from nnto.fea import mbb_half, ke_basis
from nnto.materials import NNMaterial, SimpMaterial
from nnto.topopt import ComplianceProblem

pytestmark = pytest.mark.filterwarnings("ignore")

DBTO = {1: "NNJ", 2: "CFD", 3: "CSM", 4: "AD"}
HBTO_FINAL = {1: "it1225final", 2: "it1105final", 3: "it1371final", 4: "it1342final"}


@pytest.fixture(scope="module")
def dbto_ns(upstream, dbto_net):
    ns = load_original_namespace(os.path.join(upstream, "Run_DBTO_NN.py"))
    model, x0 = setup_mbb(ns)
    return ns, model, x0


@pytest.fixture(scope="module")
def hbto_ns(upstream, hbto_net):
    ns = load_original_namespace(os.path.join(upstream, "Run_HBTO_NN.py"))
    model, x0 = setup_mbb(ns, hbto=True)
    return ns, model, x0


def test_element_matrix_matches_original(upstream):
    ns = load_original_namespace(os.path.join(upstream, "Run_DBTO_NN.py"), only=["kefun_C"])
    c = np.array([1.3, 0.4, 0.9, 0.35])
    C = np.array([[c[0], c[1], 0], [c[1], c[2], 0], [0, 0, c[3]]])
    assert np.allclose(ns["kefun_C"](C), np.einsum("k,kab->ab", c, ke_basis()), atol=1e-15)


def test_simp_reference_reproduced(upstream, paper_results):
    ns = load_original_namespace(os.path.join(upstream, "Run_DBTO_SIMP.py"))
    model, x0 = setup_mbb(ns)
    model.penal = 3.0
    ref = np.load(os.path.join(paper_results, "DBTO_SIMP_result", "DBTO_SIMP.npz"))
    dc0 = ns["dfeafun"](x0, model)
    assert np.allclose(dc0, ref["dc0"], rtol=1e-9)
    assert abs(ns["feafun"](ref["xopt"].flatten("F"), model) - float(ref["comp"])) < 1e-8
    # vectorised port
    prob = ComplianceProblem(mbb_half(), SimpMaterial())
    assert np.allclose(prob.gradient(x0), ref["dc0"], rtol=1e-9)
    assert abs(prob.compliance(x0) - 356.4817168) < 1e-6          # authors' log: 3.5648e+02


@pytest.mark.parametrize("der", [1, 2, 3, 4])
def test_dbto_initial_sensitivities(dbto_ns, dbto_net, paper_results, der):
    ns, model, x0 = dbto_ns
    ref = np.load(os.path.join(paper_results, f"DBTO_NN_result_dy{der}", f"DBTO_NN_dy{der}.npz"))
    model.der_type = der
    model.nn_model = ShimModel(dbto_net, floatx="float32")
    dc32 = ns["dfeafun"](x0, model)
    model.nn_model = ShimModel(dbto_net, floatx="float64")
    dc64 = ns["dfeafun"](x0, model)
    scale = np.abs(ref["dc0"]).max()
    assert np.abs(dc32 - ref["dc0"]).max() / scale < 2e-9         # reproduced with the cast
    if der == 2:                                                  # CFD: only with the cast
        assert np.abs(dc64 - ref["dc0"]).max() / scale > 1e-2
        ratio = ref["dc0"] / dc64
        assert np.allclose(ratio, 1.01327896, atol=2e-7)
    else:
        assert np.abs(dc64 - ref["dc0"]).max() / scale < 2e-9


@pytest.mark.parametrize("der", [1, 2, 3, 4])
def test_hbto_initial_sensitivities_and_final_compliance(hbto_ns, hbto_net, paper_results, der):
    ns, model, x0 = hbto_ns
    ref = np.load(os.path.join(paper_results, f"HBTO_NN_result_dy{der}",
                               f"HBTO_NN_dy{der}_{HBTO_FINAL[der]}.npz"))
    xf = np.concatenate([ref["xopt1"].flatten("F"), ref["xopt2"].flatten("F")])
    model.der_type = der
    model.nn_model = ShimModel(hbto_net, floatx="float32")
    assert np.abs(ns["dfeafun"](x0, model) - ref["dc0"]).max() / np.abs(ref["dc0"]).max() < 5e-9
    c32 = ns["feafun"](xf.copy(), model)
    model.nn_model = ShimModel(hbto_net, floatx="float64")
    c64 = ns["feafun"](xf.copy(), model)
    assert abs(c32 - float(ref["comp"])) < 2e-9                   # 11 digits with the cast
    assert abs(c64 - float(ref["comp"])) > 1e-9                   # only ~8 digits without it


def test_hbto_initial_compliance_signature(hbto_ns, hbto_net):
    ns, model, x0 = hbto_ns
    stored = 255.6131764903404                                    # HBTO_NN_dy*_it1.npz
    model.nn_model = ShimModel(hbto_net, floatx="float32")
    assert abs(ns["feafun"](x0, model) - stored) < 2e-9
    model.nn_model = ShimModel(hbto_net, floatx="float64")
    assert abs(ns["feafun"](x0, model) - stored) > 1e-5


@pytest.mark.parametrize("kind,key", [("density", "NN_model_DBTO_1e+04"), ("hole", "NN_model_HBTO_1e+04")])
def test_vectorised_port_matches_original_on_random_designs(upstream, kind, key, rng):
    from nnto.networks import load_keras_mlp
    net = load_keras_mlp(os.path.join(upstream, key))
    script = "Run_DBTO_NN.py" if kind == "density" else "Run_HBTO_NN.py"
    ns = load_original_namespace(os.path.join(upstream, script))
    model, _ = setup_mbb(ns, hbto=(kind == "hole"))
    model.nn_model = ShimModel(net, floatx="float32")
    mesh = mbb_half()
    # densities below 0.081 are clipped to Emin = 1e-9 by the original code; a
    # random field of such elements makes K singular to working precision
    # (compliance ~1e9) and two direct solvers then agree to ~1e-5 only, so
    # the comparison is made on a well-posed design.
    lo = 0.15 if kind == "density" else 0.0
    x = rng.uniform(lo, 1, mesh.ne * (1 if kind == "density" else 2))
    for der, name in [(1, "NNJ"), (2, "CFD-f32"), (3, "CSM"), (4, "AD-rev")]:
        model.der_type = der
        prob = ComplianceProblem(mesh, NNMaterial(net, derivative=name), kind=kind)
        g0 = ns["dfeafun"](x.copy(), model)
        assert abs(prob.compliance(x) - ns["feafun"](x.copy(), model)) < 1e-8 * abs(prob.compliance(x))
        assert np.abs(prob.gradient(x) - g0).max() < 2e-9 * np.abs(g0).max()
    assert np.isclose(prob.volume(x), ns["nlcon"](x.copy()))
    assert np.allclose(prob.volume_jac(x), ns["dnlcon"](x.copy()))


def test_published_figure_numbers(paper_results):
    """Fig. 8: MSE of initial sensitivities CFD vs NNJ = 1.11e-3, and it is a
    uniform scaling by the float32 step ratio."""
    a = np.load(os.path.join(paper_results, "DBTO_NN_result_dy1", "DBTO_NN_dy1.npz"))["dc0"]
    b = np.load(os.path.join(paper_results, "DBTO_NN_result_dy2", "DBTO_NN_dy2.npz"))["dc0"]
    assert np.isclose(np.mean((a - b) ** 2), 1.113e-3, rtol=2e-3)
    assert np.allclose(b / a, 1.01327896, atol=1e-7)
    predicted = (1.01327896 - 1.0) ** 2 * np.mean(a ** 2)
    assert np.isclose(np.mean((a - b) ** 2), predicted, rtol=1e-5)
