# [Re] Evaluation of Neural Network-Based Derivatives for Topology Optimization

Replication package for a ReScience C replication of Najmon and Tovar (2024),
by Daksh Anand (Department of Computer Science and Engineering, Thapar Institute
of Engineering and Technology, Patiala, India; ORCID
[0009-0003-6100-9295](https://orcid.org/0009-0003-6100-9295)).

* **Article:** `article/article.pdf`; LaTeX sources and `metadata.yaml` in `article/`.
* **Code:** the `nnto` package described below, MIT licence (`LICENSE`).
* **Not included:** the original code, trained networks and data. Their
  repository has no licence, so they are read from a separate copy (see *Reproduce*).

# nnto: verified neural-network derivatives for topology optimization

A NumPy/SciPy re-implementation, audit harness and extension of

> J. Najmon and A. Tovar, "Evaluation of Neural Network-Based Derivatives for
> Topology Optimization", ASME J. Mech. Des. 146(7), 071704 (2024),
> code: https://github.com/edrl-purdue/nn-derivative-to

No TensorFlow is needed. The authors' saved models are read directly from
their checkpoint files, and the original functions are executed unmodified
against a small stand-in for the Keras calls they make.

**Start with `SUMMARY.md`**: every finding and result on a few pages, with the
script and result file behind each number.

## Layout

| Path | Purpose |
|---|---|
| `SUMMARY.md` | Summary of all findings: paper errors, code bugs, the float64 fix, the HBTO ground truth, older vs newer networks, research gaps, limitations |
| `reports/` | Report 1 (paper review), Report 2 (code audit), Report 3 (literature and research gaps), and `figures/` (fig5 = older vs newer designs) |
| `fix_float64/` | The one-line fix for the authors' scripts (`float64_fix.patch`) and two demos |
| `nnto/tf_checkpoint.py` | TensorFlow-free reader for the saved models (CRC-checked) |
| `nnto/functional.py` | Overflow-safe, complex-safe activations with f, f', f'' |
| `nnto/autodiff.py` | Forward-mode (dual / hyper-dual) and reverse-mode AD engine |
| `nnto/networks.py` | Dense, residual, Fourier-feature and Chebyshev-KAN blocks (SIREN via `init_mlp(..., siren_omega0=...)`); positive and Cholesky output heads; analytic Jacobian and Hessian |
| `nnto/derivatives.py` | NNJ, CFD (fixed, optimal-step, Ridders), CSM, forward and reverse AD; six Hessian methods; Hessian-vector products |
| `nnto/training.py` | L-BFGS / Adam training, Sobolev (derivative-label) loss, ensembles |
| `nnto/homogenization.py` | Voxel homogenization (reproduces the published data set) and body-fitted homogenization with exact parameter derivatives |
| `nnto/fea.py`, `nnto/materials.py`, `nnto/topopt.py` | Vectorised FE analysis, material models, problem definition; `trust-constr`, OC and MMA drivers. Every quirk of the original code is a named option, so the original behaviour stays available as a baseline |
| `nnto/baseline/` | The original code, run verbatim (`original.py`) through a Keras stand-in (`keras_shim.py`) |
| `tests/` | 107 unit tests, including reproduction of the authors' saved numbers |
| `experiments/` | The scripts that produced every number in `SUMMARY.md` and `reports/` |
| `results/` | JSON/NPZ outputs, trained networks (`models/`), ground-truth tables |
| `FIXES_2026-11-18.md` | The analysis fixes of 18 November 2026 and the final numbers |
| `nnto/scoring.py` | Scoring helpers added with those fixes: sensitivity bases, hole-size convention, volume projection, absolute error summaries |
| `verification/` | A complete rerun of the package (`full_rerun_2026-11-18/`) and its line-by-line comparison with the console output in `improvemt_document.docx` (`README.md`) |

## Reproduce

```bash
# 1. dependencies, and the original repository (code, trained networks, data, results)
pip install -r requirements.txt
git clone https://github.com/edrl-purdue/nn-derivative-to
export NNTO_UPSTREAM=$PWD/nn-derivative-to/NN_derivatives_TO_v3_final
# (or download it from GitHub as a ZIP: the folder is then nn-derivative-to-main/)

# 2. tests (about 40 s)
python -m pytest tests -q

# 3. figures from the saved results (no retraining)
python experiments/make_figures.py
python experiments/make_comparison_figure.py  # fig5: older vs newer designs

# 4. experiments (run from this folder; each writes results/*.json)
python experiments/gt_voxel_lattice.py        # ~12 min, data-generating model
python experiments/gt_fitted_grid.py          # ~17 min, converged ground truth + derivatives
python experiments/gt_offgrid.py              # 300 independent test points
python experiments/exp_precision.py           # step size / float32 study
python experiments/exp_surrogate_truth.py     # surrogate and derivative errors vs truth
python experiments/exp_higher_order.py        # Hessians
python experiments/exp_architectures.py       # architectures, Sobolev training, data size
python experiments/exp_hbto_surrogates.py     # improved HBTO surrogates, deep ensemble
python experiments/exp_efficiency.py          # timings (~10 min)
python experiments/exp_dbto_ensemble.py       # round-off ensemble (long)
python experiments/exp_dbto_part2.py          # more seeds + ablation
python experiments/exp_patch_check.py         # float64 fix: before/after optimization runs
python experiments/analyse_patch_check.py

# newer networks (about 1 hour on 2 cores, plus 1-3 min per optimization run)
python experiments/exp_newer_models.py 3 2    # 6 architectures x 4 training regimes x 3 seeds
python experiments/exp_newer_models_to.py --out=to_baselines "reference*3" "published*2"
python experiments/exp_newer_models_to.py "Deep MLP 3x64 tanh | converged + Sobolev" "Deep MLP 3x64 tanh | paper data" ...
python experiments/analyse_newer_models.py

# added 18 November 2026 (FIXES_2026-11-18.md)
python experiments/rescore_final_designs.py   # re-scores every saved final design, no re-optimisation
python experiments/exp_published_long.py 30000 1000   # published network for 30,000 MMA iterations (~16 min)
```

To rerun without touching `results/`, point `NNTO_RESULTS` (and `NNTO_FIGURES`) at
another folder; every script reads its inputs from and writes its outputs to that
folder. `verification/full_rerun_2026-11-18/` was produced this way.

`experiments/exp_hbto_runs.py` runs the consistent HBTO formulation with
`trust-constr`. At more than 2 s per iteration it was not run to completion
here; the HBTO comparisons in `SUMMARY.md` use MMA instead.

Model paths in `results/*.json` are stored as `models/<file>` (networks trained
here) or `UPSTREAM/<name>` (the authors' models) and resolved by
`experiments/common.py::resolve_model`, so the results work from any folder.

Requirements: Python >= 3.9, NumPy, SciPy, Matplotlib (figures), pytest (`requirements.txt`).
Developed with Python 3.13, NumPy 2.x, SciPy 1.18.

## Using the corrected pieces in the original TensorFlow scripts

The fixes that matter for the published scripts:

```python
tf.keras.backend.set_floatx('float64')            # before load_model(): removes the 1.33 % CFD error
dy = tape.batch_jacobian(yn_tape, xn_tape)        # instead of reduce_sum(tape.jacobian(...), axis=[2])
dv = np.ones(xPhys.shape) / xPhys.size            # Run_DBTO_*.py::dnlcon, Jacobian of mean(x) - Vt
```

Also pass a sparse constraint Jacobian with `options={'sparse_jacobian': True, ...}`
to `minimize(..., method='trust-constr')`. On the 60 x 30 mesh this is 111x faster
per iteration in the clean rerun (15.9 s to 0.14 s,
`verification/full_rerun_2026-11-18/efficiency.json`; 174x in `results/efficiency.json`),
with the same iterates to round-off (designs within 1.2e-10 after 25 iterations on a
30 x 15 mesh and 2.1e-12 after 10 iterations on the 60 x 30 mesh,
`verification/extra_runs/sparse_vs_dense_iterates.json`).

`fix_float64/float64_fix.patch` applies only the first fix, because it is the
only one verified against the authors' saved numbers. Apply it from the root of
the original repository with `patch -p0 < /path/to/fix_float64/float64_fix.patch`.
It leaves the shipped settings of `Run_HBTO_NN.py` (`maxiter = 50`,
`der_type = 2`) unchanged; the article used other settings.

`experiments/tf_crosscheck_float32.py` checks the float32 behaviour with real
TensorFlow. It could not be executed in the development environment.
