# Report 2 — Source code audit

## 2.1 How the audit was done

TensorFlow could not be installed in the working environment (no network access for package installs), so the original scripts were executed in the following way, which keeps every original line untouched:

1. `nnto/tf_checkpoint.py` reads the saved models (`variables.index` / `variables.data-*`) directly and verifies the CRC32C of every tensor.
2. `nnto/baseline/original.py` extracts the module-level functions of `Run_DBTO_NN.py`, `Run_DBTO_SIMP.py` and `Run_HBTO_NN.py` with `ast` and compiles them unchanged.
3. `nnto/baseline/keras_shim.py` supplies the handful of Keras/TensorFlow calls those functions make (`predict`, layer calls, `GradientTape.jacobian`, `reduce_sum`).

**Validation of this set-up against the authors' saved results** (`tests/test_original_reproduction.py`):

| Quantity saved by the authors | Reproduced to |
|---|---|
| SIMP initial sensitivities `dc0` (1800 values) | 5e-13 relative |
| SIMP final compliance 81.9717019556 at their saved design | 11 digits |
| DBTO initial sensitivities, NNJ / CSM / AD | 1e-11 relative |
| DBTO initial sensitivities, CFD | 3e-10 relative **only** with the float32 cast of `predict`; 1.3e-2 off without it |
| HBTO initial compliance 255.6131764903 | 7e-11 with the cast; 1.8e-5 off without it |
| HBTO initial sensitivities, CFD | 2e-9 with the cast; 1.3e-2 off without it |
| HBTO final compliances at their four saved designs | 1e-11 with the cast; 1e-8 without it |
| Table 1 HBTO test MSE 1.47e-3 | 1.468e-3 |
| Fig. 8 / Fig. 11 sensitivity MSE, CFD vs NNJ (1.11e-3 / 1.07e-4) | 1.1126e-3 / 1.0722e-4 |
| First 6 (DBTO) and 20 (HBTO) optimizer iterations in the authors' logs | all printed digits |

The last row needs a caveat: after iteration 6 (DBTO) and between iterations 20 and 30 (HBTO) my `trust-constr` runs leave the authors' logged trajectory. SciPy changed between 1.10.1 (authors) and 1.18 (here), so this departure cannot be attributed to round-off alone; sensitivity to round-off is tested separately, inside one environment (`SUMMARY.md` §7).

What could **not** be verified: anything that needs TensorFlow itself. The float32 behaviour of `Model.predict` is inferred from the numbers above (three independent signatures, each matching to 6–11 digits) and from the Keras data-adapter source; `experiments/tf_crosscheck_float32.py` checks it directly with TensorFlow but was not run here.

## 2.2 Findings

Severity reflects the effect on the paper's results and conclusions. Line numbers refer to the files in `NN_derivatives_TO_v3_final`.

| ID | File | Function | Identified issue | Label | Severity | Proposed correction |
|---|---|---|---|---|---|---|
| A1 | `Run_DBTO_NN.py` l.367, `Run_HBTO_NN.py` (CFD branch of `dcfun`) | `dcfun`, `der_type == 2` | `nn_model.predict(x ± h)` is called without `set_floatx('float64')`. Keras casts NumPy inputs to float32, so the perturbed inputs are rounded and the step actually taken at θ = 0.5 or √0.5 is 1.01327896e-6 instead of 1e-6. Published CFD sensitivities are 1.328 % too large at the initial designs; over θ ∈ [0.5, 1) the error is −1.65 % or +1.33 % | Confirmed implementation bug | **High** | `tf.keras.backend.set_floatx('float64')` before `load_model`, or evaluate with `nn_model(x)`; or divide by the step actually taken. `nnto.derivatives.cfd(..., exact_step=True)` |
| A1b | `Run_*_NN.py` l.176, 228 | `feafun`, `dfeafun` | Same cast on every prediction: the objective is evaluated at float32-rounded design variables while NNJ/AD derivatives are evaluated at the float64 values | Confirmed implementation bug | Low | as A1 |
| A2 | `Run_DBTO_NN.py` l.418–425, `Run_DBTO_SIMP.py` | `dnlcon` | Returns `ones` as the Jacobian of `nlcon = mean(x) − Vt`; the Jacobian is `1/ne`. Off by a factor 1800. (`Run_HBTO_NN.py` has the 1/ne.) | Confirmed implementation bug | Low (direction correct; see ablation) | `dv = np.ones(xPhys.shape) / xPhys.size` |
| A3 | `Run_*_NN.py` l.182, 235 | `feafun`, `dfeafun` | Predictions are clipped to [Cmin, Cmax] but the derivative is not masked, so the supplied gradient is not the derivative of the evaluated objective wherever the clip is active. DBTO: active for θ < 0.0811; 33–34 % of the variables of the authors' final designs are there (the spurious part is 0.4–0.6 % of the gradient norm). HBTO: 21.6–25.5 % of the cells of the final designs have a clipped component | Confirmed implementation bug | Medium | Zero the derivative where clipped (`clip="consistent"`), or remove the need to clip (A4) |
| A4 | trained models | — | Surrogates are not physically admissible. DBTO: E_NN < 0 on [0, 0.0811] (E_NN(0) = −3.75e-4), dE_NN/dθ < 0 on [0, 0.0289]. HBTO: 2.9 % of [0,1]² has a negative component, 3.0 % is not positive definite, 0.5 % is not positive definite even after clipping; 0.9–2.2 % of the cells in the final designs | Confirmed defect of the trained models | Medium | Positive / Cholesky output head (`SoftFloor`, `CholeskyHead`) |
| A5 | `Run_*_NN.py` l.137–159, 276–279 | `filterfun` | The heuristic sensitivity filter is used as the gradient in a trust-region SQP method. It is not a gradient: cosine similarity with the true gradient at the authors' final designs is 0.86 (DBTO) and **0.17–0.21 (HBTO)**. In HBTO it is applied to hole sizes with the weights and the `max(1e-3, x)` division designed for densities; 17–20 % of the HBTO variables are below 1e-3, where filtered values are amplified (largest filtered entry 6–12 × the largest true-gradient entry) | Methodological limitation | **High** for HBTO | Linear (density-type) filter with the exact chain rule (`filter="density"`) |
| A6 | `HBTO_CH_2D_N10000_100x100.mat` | data | 10,000 samples but only 2,455 distinct targets: the hole snaps to the 100 × 100 voxel grid, so the target is piecewise constant in θ (steps of 0.02). Measured consequence is small: label error vs the converged reference has MSE 3.7e-5, about 7 % of the surrogate's own error; the voxel mesh is stiffer than the converged value by 1.5e-3 on average | Confirmed data property; hypothesis that it limits accuracy **rejected** | Low | Body-fitted homogenization with exact derivative labels (`fitted_CH`) |
| A7 | `Train_*_NN.py` l.68/59, `NN_derivatives_examples.py` l.212 | model definition | The "de-normalization" layer is `Normalization(axis=1, invert=False)`: the output is (z − mean_y)/std_y, a second normalization. The derivative code divides by `std_out` consistently, so derivatives are correct for the model as built, but the dense layers must then produce targets with scale std_y² instead of 1 | Confirmed discrepancy between code and stated method | Low | `invert=True` |
| A8 | `Train_DBTO_NN.py` l.73–74, `Train_HBTO_NN.py` l.64–65 | training loop | `EarlyStopping` is created but not passed to `fit`; the saved optimizer counter is 1,095,000 = 5000 × 219, i.e. all 5000 epochs ran. It monitors the training loss. The validation split is never used for model selection | Confirmed implementation bug | Low | `callbacks=[stop_early]`, `monitor='val_loss'`, `restore_best_weights=True` |
| A8b | `Train_*_NN.py` l.57/48, l.41 | preprocessing, model selection | Normalization statistics are adapted on train + validation samples (saved count 8500). With `reps > 1` the best model is chosen on the test MSE | Potential issue (data leakage) | Low | Adapt on the training split; select on validation |
| A9 | `Run_*_NN.py` `dact_fn`, `NN_derivatives_examples.py` l.242 | `dact_fn` (sigmoid) | `exp(y) / (1 + exp(y))**2` is `inf/inf = nan` for y > 709.8. Latent: the largest pre-activation of the DBTO model on [0,1] is 8.1 | Potential numerical issue | Low | `s = sigmoid(y); s * (1 - s)` with an overflow-free sigmoid |
| A10 | same, l.238 | `dact_fn` (relu) | `max(y, 0) / y` is 0/0 at y = 0 | Potential numerical issue | Low | `(y > 0)` |
| A11 | `Run_*_NN.py` l.405, examples l.335 | `dcfun`, `der_type == 4` and `== 1` | AD forms the full cross-batch Jacobian (N, ydim, N, xdim) and sums it: 26 MB (DBTO) and 207 MB (HBTO) for values that fit in 14 kB / 115 kB. NNJ loops over samples in Python | Confirmed inefficiency | Medium (explains Table 2) | `tape.batch_jacobian`; vectorised chain rule |
| A12 | `Run_*_NN.py` l.444–455 | `DBTO` / `HBTO` | The constraint Jacobian is passed dense, so `trust-constr` uses dense factorizations of a (1 + 2n) × 3n system each iteration. Authors' logs: 17 s per iteration (DBTO), 90–109 s (HBTO, 30–36 hours per run) | Confirmed inefficiency | **High** for run time | Return a sparse Jacobian and set `sparse_jacobian=True`: 111× faster per iteration on the 60 × 30 mesh in the clean rerun (15.9 s → 0.14 s, `verification/full_rerun_2026-11-18/efficiency.json`; 174× in `results/efficiency.json`); the iterates agree to round-off (designs within 1.2e-10 after 25 iterations on a 30 × 15 mesh, `verification/extra_runs/sparse_vs_dense_iterates.json`) |
| A13 | `Run_*_NN.py` l.168–169, 220–221 | `feafun`, `dfeafun` | Solid tensor hard-coded to four digits (relative error up to 4e-5); the comments on `Cmax`/`Cmin` are swapped | Confirmed minor inaccuracy | Low | Compute from E and ν |
| A14 | `Run_*_NN.py` | `feafun`, `dfeafun` | A 64-statement element routine is called once per element per call; objective and gradient each assemble and solve the same system | Confirmed inefficiency | Low | Element basis matrices and a cached solve (`nnto.fea`) |
| A15 | `Run_*_NN.py` | `feafun` | With Emin = 1e-9 and surrogate predictions clipped to it, K is close to singular for designs with scattered clipped elements; two direct solvers then agree to 1e-5 only | Potential numerical issue | Low | Larger Emin, or an admissible surrogate |
| A16 | `Train_*_NN.py` | — | No random seeds; stated in the README | Reproducibility | Low | Seed NumPy, TensorFlow and the sampler |
| A17 | `Run_DBTO_NN.py` | `nlcon`/`dnlcon` with `ft == 2` | Density-filter branch: volume evaluated on unfiltered variables, Jacobian filtered. Not used in the paper | Potential issue (unused path) | Low | Evaluate the volume on the filtered field |

## 2.3 The key bug in detail (A1)

**Original code** (`Run_DBTO_NN.py`, lines 361–368; `Run_HBTO_NN.py` is identical in structure):

```python
h_cfd = 1e-6  # step size
...
    h_mat[:, [d]] = np.ones((Nt, 1)) * h_cfd  # perturbation array
    dy_2[:, :, d] = (
            (nn_model.predict(x_test + h_mat, verbose=0) - nn_model.predict(x_test - h_mat, verbose=0))
            / (2 * h_cfd)).reshape(Nt, ydim)  # derivative via CFDA
```

**Why it is wrong.** `x_test ± h_mat` is a float64 array. `Model.predict` hands NumPy input to Keras' data adapter, which converts floating arrays to `keras.backend.floatx()`. `NN_derivatives_examples.py` calls `set_floatx('float64')` (line 198); the `Run_*` scripts do not, so `floatx` is float32. Near 0.5 the float32 spacing is 5.96e-8 above and 2.98e-8 below, so 0.5 + 1e-6 becomes 0.5 + 17 ulp and 0.5 − 1e-6 becomes 0.5 − 34 half-ulp: the step actually taken is 1.01327896e-6 on each side, but the code divides by 2e-6.

**Consequence.** Every CFD sensitivity in the paper is scaled by 1.01327896 at the initial designs. The authors' saved arrays show exactly this:

| | Predicted from float32 rounding | In the authors' saved `dc0` |
|---|---|---|
| CFD / NNJ ratio, DBTO (θ = 0.5) | 1.0132789612 | 1.0132789611 (std 8e-10 over 1800 elements) |
| CFD / NNJ ratio, HBTO (θ = 0.7071) | 1.0132789612 | 1.0132790309 (std 5e-8 over 3600 values) |
| MSE(CFD, NNJ), DBTO — Fig. 8 prints 1.11e-3 | 1.1126002e-3 | 1.1126001e-3 |
| MSE(CFD, NNJ), HBTO — Fig. 11 prints 1.07e-4 | 1.0721997e-4 | 1.0722082e-4 |

Away from the initial design the error depends on the float32 binade of θ: −1.65 % or +1.33 % on [0.5, 1), −0.16 % to +1.33 % on [0.25, 0.5), below 0.1 % for θ < 0.03.

**Corrected implementation** (`nnto/derivatives.py::cfd`): evaluate in double precision, or, if single-precision inputs cannot be avoided, divide by the step actually taken.

**Original vs corrected** (max relative error against the analytic Jacobian, 4000 random points, `exp_precision.py`):

| Step h | Double precision | As published (float32 inputs) | float32 inputs, actual step |
|---|---|---|---|
| 1e-8 | 2.3e-8 | 2.0 (198 %) | not representable |
| 1e-7 | 2.5e-9 | 0.19 | 3.7e-8 |
| **1e-6 (paper)** | **2.3e-10** | **1.65e-2** | **3.1e-8** |
| 1e-5 | 5.7e-11 | 1.6e-3 | 3.0e-8 |
| 1e-4 | 3.6e-9 | 1.7e-4 | 3.2e-8 |
| 1e-3 | 3.6e-7 | 1.7e-5 | 3.8e-7 |

(DBTO network; the HBTO network gives the same picture, e.g. 1.1e-9 vs 1.5e-2 at h = 1e-6.) See `figures/fig1_cfd_precision.png`. *Corrected 18 November 2026: the double-precision column and the HBTO value previously held numbers from an earlier run (e.g. 2.8e-10, 1.3e-9); the values above are those in `results/precision.json` and in `comprehensive_results.docx` §2.1–2.2.*

## 2.4 Tests that demonstrate the findings

`python -m pytest tests -q` → 100 passed. Tests that document a defect assert the defective behaviour explicitly, so they fail if the claim is wrong:

| Finding | Test |
|---|---|
| A1 | `test_float32_input_quantisation_reproduces_published_cfd_error`, `test_float32_cfd_error_values`, `test_dbto_initial_sensitivities[2]`, `test_hbto_initial_sensitivities_and_final_compliance[2]`, `test_published_figure_numbers` |
| A1b | `test_hbto_initial_compliance_signature` |
| A2 | `test_dbto_volume_jacobian_original_is_off_by_ne` |
| A3 | `test_clip_gradient_consistency[original/consistent/smooth]` |
| A4 | `test_published_dbto_surrogate_predicts_negative_modulus` |
| A5 | `test_sensitivity_filter_is_not_a_gradient` |
| A6 | `test_published_targets_are_piecewise_constant`, `test_voxel_model_reproduces_published_dataset` |
| A7, A8, A8b | `test_shapes_and_values` (saved normalization statistics, step counter 1,095,000, count 8500) |
| A9, A10 | `test_original_sigmoid_derivative_formula_overflows`, `test_original_relu_derivative_formula_is_nan_at_zero` |
| A13 | `test_hard_coded_constants_are_rounded` |
| P1–P5 | `tests/test_paper_equations.py`, `tests/test_fea.py`, `tests/test_materials.py` |
