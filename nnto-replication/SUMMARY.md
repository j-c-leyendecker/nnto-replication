# Summary: auditing and improving Najmon & Tovar (2024)

**Paper:** J. Najmon and A. Tovar, "Evaluation of Neural Network-Based Derivatives for Topology Optimization", *ASME J. Mech. Des.* 146(7), 071704 (2024), doi:10.1115/1.4064243.
**Original code:** https://github.com/edrl-purdue/nn-derivative-to

**Status (13 November 2026).** Reports 1–3 (paper review, code audit, literature and gaps) are in `reports/`. This file summarises every result, including the experiments that Reports 4–6 would draw on. Report 4 (research proposal) and Report 6 (publication readiness) are not written yet. Findings are labelled as in the project brief: confirmed mathematical error, confirmed implementation bug, potential numerical issue, methodological limitation, research opportunity, unverified hypothesis.

**Fixes (18 November 2026).** Four analysis errors were fixed and the affected numbers recomputed from the saved results (`FIXES_2026-11-18.md`): the method-vs-network ratio (§5) is about 600×, not 1,160×; the authors' designs are 8.5–10.3 % worse than reported depending on how holes narrower than 1 % are counted (§5); model errors are summarised by their absolute values (§6); runs that stop short of the volume target are also compared at equal material (§6). Reruns on a second machine reproduce the shipped results (`verification/`).

## 1. In one paragraph

The paper's first-order claim holds: the analytic network Jacobian (NNJ), the complex step (CSM) and automatic differentiation (AD) give the same derivatives of a smooth network, to round-off. Its other conclusions do not survive checking. The finite-difference (CFD) discrepancy it reports is a float32 rounding bug in the code, not a property of the method. The ranking of methods rests on one run each, and those outcomes flip under 1e-10 perturbations. For the multiscale (HBTO) problem the network itself, not the differentiation method, is the dominant error: its derivatives are off by 78 % over the parameter domain (15 % in the interior) on the 51 × 51 reference grid, or 64 % (14 %) at the 300 independent test points used in §6 and in the manuscripts, and the published designs are 8.5–10.3 % less stiff than reported. A deeper network trained on mesh-converged data with derivative labels removes that error. Its designs are within 0.1 % of the reference optimum, and it predicts their compliance to within 0.1 %. They are about 4 % stiffer than the published designs; roughly half of that comes from the better network and half from a consistent formulation and optimizer, with which the published network still never converges.

## 2. Errors in the paper's equations (Report 1)

| ID | Label | Finding |
|---|---|---|
| P1 | Confirmed mathematical error | Eq. (11) misses a minus sign: ∂φ/∂θe = −uᵀ(∂K/∂θe)u. The code is right. |
| P2 | Confirmed mathematical error | Eq. (15) has the wrong sign: dE/dθ = +p(Emax − Emin)θ^(p−1). The code is right. The two printed sign errors cancel for SIMP, not for a network derivative. |
| P3 | Confirmed inconsistency | Eq. (1) is written for column vectors; Eqs. (3)–(6) are only valid for row vectors, which Appendix B and the code use. |
| P4 | Confirmed paper–code inconsistency | Sec. 4.2 and Fig. 4(a) describe a cantilever; the code and the published topologies are a half MBB beam. |
| P5 | Typographical errors | Three sign typos in the Appendix B intermediate vectors; the final value 1.623 is correct. |

## 3. Code audit (Report 2, selection)

| ID | Where | Issue | Label | Severity |
|---|---|---|---|---|
| A1 | `Run_*_NN.py`, `dcfun`, CFD branch | `Model.predict` rounds inputs to float32, so the step actually taken is 1.0133e-6 instead of 1e-6. Every published CFD sensitivity is 1.33 % too large at the start (−1.65 % to +1.33 % elsewhere). This accounts for the whole "CFD is less accurate" result (Fig. 8 and Fig. 11 MSEs reproduced to 5–7 digits). | Confirmed implementation bug | High |
| A5 | `filterfun` | The heuristic sensitivity filter is handed to `trust-constr` as if it were the gradient. Its cosine with the true gradient is 0.86 (DBTO) and 0.17–0.21 (HBTO) at the authors' final designs. | Methodological limitation | High for HBTO |
| A12 | `DBTO` / `HBTO` drivers | Dense constraint Jacobian: 17 s per iteration (DBTO) and 90–109 s (HBTO) in the authors' logs, 30–36 h per HBTO run. A sparse Jacobian is 111–174× faster per iteration on the 60 × 30 mesh, depending on the machine (§8). | Confirmed inefficiency | High for run time |
| A3 | `feafun`, `dfeafun` | Predictions are clipped but their derivatives are not masked; 21.6–25.5 % of the cells of the HBTO designs are affected. | Confirmed implementation bug | Medium |
| A4 | trained models | Not physically admissible: the DBTO network predicts a negative modulus for θ < 0.081; 3 % of the HBTO input space gives a stiffness that is not positive-definite (401 × 401 grid over [0, 1]²; 1.9 % on the 197 × 197 grid over [0.01, 0.99]² used in §6). | Confirmed model defect | Medium |
| A11 | `dcfun`, AD and NNJ branches | AD builds the full N × N cross-batch Jacobian (207 MB for HBTO instead of 115 kB); NNJ loops in Python. This, not the methods, is what Table 2 times. | Confirmed inefficiency | Medium |
| A2 | `Run_DBTO_*.py`, `dnlcon` | Volume-constraint Jacobian too large by a factor of 1,800. | Confirmed implementation bug | Low |
| A8 | `Train_*_NN.py` | Early stopping is created but never used; the validation split never selects a model; normalisation is fitted on training + validation data. | Confirmed bug / potential leakage | Low |
| A6 | data set | 10,000 samples but only 2,455 distinct targets (voxel snapping). Tested: **not** a limiting factor. | Hypothesis rejected | Low |

A7, A9, A10 and A13–A17 (all low severity) are in `reports/part2_audit.md`. Each finding has a unit test; `python -m pytest tests -q` gives 107 passed (100 before the fixes of 18 November 2026).

## 4. The one-line float64 fix (A1) and what it changes

The fix is `tf.keras.backend.set_floatx('float64')` before `load_model` (`fix_float64/float64_fix.patch`; apply it with `patch -p0` from the root of the original repository).

* **Derivatives.** The maximum CFD error at h = 1e-6 drops from 1.65e-2 to 2.3e-10 (4,000 random points; 1.5e-2 to 1.1e-9 for the HBTO network). Along an optimization path the CFD sensitivities go from 1.3–1.7 % error to round-off.
* **Optimization paths.** Measured against the exact-derivative run, the published CFD run departs (design difference above 1e-3) at iteration 3; the fixed CFD run follows it until iteration 71.
* **Final designs** (density problem, 6 runs each). True compliance is 81.03 ± 0.57 before the fix and 81.20 ± 0.67 after (95 % CI half-widths), Welch p = 0.63. **There is no measurable change in the final designs**, because the outcome is decided by the path noise described in §7.

## 5. A ground truth for the multiscale (HBTO) problem

The paper had no reference for HBTO. This package builds one:
* body-fitted homogenization of the rectangular-hole cell, with exact parameter derivatives;
* Richardson-extrapolated to mesh convergence on a 51 × 51 grid;
* checked at 300 independent points: values agree to 5.6e-5 and derivatives to 0.7 %.

The authors' voxel model is also reproduced: it regenerates their data set.

Against this reference:
* **Network error.** The published network's Jacobian has a relative RMS error of 78 % over the domain and 15 % in the interior, measured on the 51 × 51 reference grid (`surrogate_truth.json`); at the 300 independent Latin-hypercube test points used for every network in §6 and in the manuscripts it is 64 % and 14 % (`newer_models.json`). The grid weights the domain edge, where the error is largest, more heavily. Its initial-design sensitivities are 22 % off in relative L2 norm.
* **Differentiation method vs network error.** On the filtered sensitivities the paper compares in Fig. 11, the difference between derivative methods (MSE 1.07e-4, itself the float32 artefact) is about 600 times smaller than the error all four methods share (MSE 0.064; 1.3 % vs 25 % relative, 19×). On raw gradients the figures are 2.8e-4 and 0.125 (444×; 17× relative). *Corrected 18 November 2026: the earlier "1,160×" divided the raw-gradient MSE by the paper's filtered one.* The authors' stored `dc0` is the filtered sensitivity: it matches the recomputed filtered value to 2e-12.
* **Published designs.** The authors' four final designs are 8.5–10.3 % more compliant than the paper reports: 8.5–9.7 % under their own voxel model, 9.2–10.3 % under the converged reference. About 30 % of their cells have a hole side below 0.01. The converged reference (tabulated on 0.01–0.99) scores these as 1 % slits; counted as solid, as in the authors' voxel data, the converged reference also gives 8.5–9.7 %. The optimizer settles where the network over-predicts stiffness.
* **New observation: the parameterization is singular at its edge.**
  * A thin slit is much softer than solid material: C11 drops from 1.346 to 0.857 for a 5 % × 50 % slit.
  * The data set labels holes narrower than 1 % as solid, so the training target jumps at the edge of the domain. The network's derivative error exceeds 200 % there.
  * 14–20 % of the cells in the published designs are slit cells.

## 6. Older vs newer networks (HBTO)

**The networks**, tested on the same 300 independent points:

| | Older (published) | Newer (3 × 64 tanh, trained on converged data with derivative labels) |
|---|---|---|
| Error on stiffness values | 4.1 % | 0.025 % |
| Error on stiffness derivatives (whole range / interior) | 64 % / 14 % | 0.51 % / 0.28 % |
| Points with a stiffness that is not positive-definite | 1.9 % | 2.1 % (0 % with the Cholesky output head) |
| Size | 452 parameters | 8,772 parameters, 142 s to train |
| At the starting design: compliance error / gradient error | −8.8 % / 22 % | +1.0 % / 3.1 % |

**The final designs.** "True" compliance comes from the converged reference. The reference optimum, found by optimizing with the reference model itself, is 78.85; lower is better.

| | Older, as published | Older network, same optimizer as newer | Newer |
|---|---|---|---|
| True compliance | 82.65 (best of the authors' four; 82.11 if holes < 1 % count as solid) | 81.96 (81.88–82.05); 81.87 at the exact volume target | 78.88 (78.83–78.95) |
| Above the reference optimum | +4.8 % (other three: +5.0 to +7.4 %) | +3.8 to +4.1 % | −0.02 to +0.12 % |
| Network's own prediction | off by −7.8 to −8.4 % (depending on the hole convention) | off by −5.0 to −5.2 % | off by −0.08 to +0.03 % (mean absolute 0.035 %) |
| Iterations | 1,105–1,371 | stopped at the 3,000 limit, not converged | 694–1,148, converged |
| Time per run | 30–36 h (authors' logs, trust-constr) | about 85 s | 67–113 s |

* **Size of the improvement.** Against the authors' published designs the newer designs have 3.9–4.6 % lower compliance (depending on how holes narrower than 1 % are counted). Against the older network with the same optimizer the gap depends on the iteration budget, because the older network never converges (`exp_published_long.py`):
  * at the 3,000-iteration cap used above: 3.7–3.8 % (gap 3.08, 95 % CI 2.91–3.26, Welch p = 2.5e-6; 3.66 % with both hole fields scaled to the exact volume target, which two of the three capped runs miss by 0.2–0.3 %);
  * after 10,000 iterations: 80.50–80.62 (3 runs), about 2.1 %;
  * from 11,000 to 30,000 iterations it oscillates around 80.5 (80.45–80.96) without converging: about 2 %.
  So roughly half of the 4 % over the paper comes from the better network and half from the consistent formulation and optimizer (the older network gets from 82.1–82.65 to about 80.5 with them).
* **Model error.** Its sign varies between runs, so it is summarised by the absolute value: 0.035 % on average and 0.079 % at worst for the newer network, against 7.8–8.4 % for the authors' best design (about 100–240× smaller).
* **Run time.** The newer network costs about 3.5× more per iteration but converges, so total run time is about the same.
* **Figure.** `reports/figures/fig5_older_vs_newer.png` shows the four designs side by side.

**What made the difference** (ablation, 2–3 runs each, all scored by the reference):

| Change | True compliance | Converged within 3,000 iterations |
|---|---|---|
| Paper's small 1 × 64 sigmoid network, retrained on converged data with derivative labels | 79.20–79.39 | yes (623–1,118), 12–23 s per run |
| Deep network trained on the authors' own data | 79.07–79.46 | no |
| Deep network, converged data, values only | 78.82–78.91 | no |
| Deep network, converged data + derivative labels (the "newer" above) | 78.83–78.95 | yes (694–1,148) |
| Same with the Cholesky (always valid) output head | 78.85–78.86 | yes (1,541–1,823) |
| Fourier features, ResNet, SIREN, Chebyshev KAN (same data and labels) | 78.83–78.89 | yes (990–1,657) |

* **No single change accounts for all of it.**
  * Better data with derivative labels, or more capacity, each gets within 0.3–0.8 % of the reference optimum.
  * Together they get within 0.1 %.
* **Derivative labels mainly buy convergence**, not a better final design.
* **Newer architectures add nothing** over a plain deep network on this problem.

## 7. The density (SIMP) problem: nothing left to recover

Runs with the published network and runs with the exact SIMP law land in the same two outcomes, at true compliance ≈80.75 or ≈82.0 (1.5 % apart).
* Which outcome a run reaches is decided by gradient perturbations as small as 1e-10, not by the derivative method.
* The paper's ranking of methods, from one run each, is therefore not supported.
* The network's own error on the compliance of its designs is at most 0.1 %.

`reports/figures/fig3_dbto_runs.png` shows where the single runs end.

## 8. Other results

* **Precision.** In float32 only the complex step and NNJ/AD stay at round-off (3e-7). Central differences need h ≈ 1e-2 to 1e-3 and reach 5e-5 at best.
* **Second derivatives.**
  * The analytic chain rule, hyper-dual AD and the complex step of the Jacobian agree to 1e-15.
  * Second differences of values reach 4e-8 at best and 4e-4 at the paper's h = 1e-6.
  * The published DBTO network's error grows from 0.11 % (value) to 0.22 % (first derivative) to 2.5 % (second derivative).
* **Architectures.**
  * NNJ = CSM = AD to 1e-15 for deep tanh, residual swish, Fourier-feature and constrained-output networks, and to 1e-12 for the Chebyshev KAN (unit test). The paper's first-order statement generalizes to them.
  * It does not generalize to ReLU: CFD is wrong near kinks (7e-3 error on a trained network) and second derivatives vanish.
* **Uncertainty.** A five-member deep ensemble trained on the published data has a spread that correlates with the true error (Spearman 0.43 for values, 0.60 for derivatives). It flags the domain edge but under-states the error there: spread 0.18 against an actual error of 1.09.
* **Cost** (machine-dependent: `results/efficiency.json` comes from a slower machine; the clean rerun is `verification/full_rerun_2026-11-18/efficiency.json`; numbers corrected on 18 November 2026, they previously came from an earlier run). The paper's Table 2 times the implementation, not the methods.
  * **Derivatives.** Vectorised, all five derivative routines take 1.3–6.0 ms for the 1,800 elements in the clean rerun (1.0–60 ms in `results/`). The original AD routine (run through the NumPy stand-in, so without TensorFlow overhead) takes 0.61 s (DBTO) and 2.0 s (HBTO) in the clean rerun (1.1 s and 5.9 s in `results/`), and its cross-batch Jacobian needs 207 MB where 115 kB suffice.
  * **Objective plus gradient.** One evaluation drops from 0.77 s to 0.011 s in the clean rerun (2.4 s to 0.014 s in `results/`).
  * **The optimizer dominates.** A `trust-constr` iteration on the 60 × 30 mesh takes 15.9 s with the published dense constraint Jacobian and 0.14 s with a sparse one in the clean rerun (111×; 538 s and 3.1 s, 174×, in `results/`). The iterates are the same to round-off: the designs agree to 1.2e-10 after 25 iterations on a 30 × 15 mesh and to 2.1e-12 after 10 iterations on the 60 × 30 mesh (`verification/extra_runs/sparse_vs_dense_iterates.json`).
  * **Many inputs.** For a scalar function of many inputs only reverse-mode AD stays cheap. At 256 inputs it takes 0.7 ms in the clean rerun, against 155 ms for forward mode, 173 ms for CFD, 475 ms for CSM and 1.1 s for NNJ.

## 9. Research gaps (Report 3)

1. **Surrogate fidelity, not the differentiation method, dominates sensitivity error.** This is the strongest gap, with evidence in §5–6.
2. **No noise floor.** Methods are compared through single optimization runs.
3. **Gradient/objective consistency.** The heuristic filter and unmasked clipping are combined with a general NLP solver.
4. **Accidental mixed precision** in the published comparison.
5. **Second derivatives and second-order optimizers.**
6. **Operator surrogates** (CNN, FNO, PINN): the question there is cost, not accuracy. The gap is closed for MLP-type networks.
7. **Uncertainty-aware optimization**, so the optimizer cannot exploit network error.
8. **The singular rectangular-hole parameterization.**

3-D problems, multiple loads, nonlinear and dynamic problems (direction C) were not run.

**Strongest direction (basis for Report 4).** Rank the sources of sensitivity error in NN-based multiscale topology optimization (differentiation method, precision, filter consistency, surrogate fidelity, training data). Follow each one to the true compliance of the final design, using a converged reference and a noise floor from repeated runs. The results above are the pilot study:
* the network dominates;
* derivative labels and capacity fix it;
* new architectures do not matter.

Novelty caveat: Sobolev training (White et al. 2019; Black & Najafi 2025) and Cholesky heads (Gupta et al. 2026) already exist. What the literature search did not find is this controlled ranking with an end-to-end check against a converged reference.

## 10. Limitations of this work

* **TensorFlow was not available**, so the authors' scripts ran unchanged through a NumPy stand-in for the Keras calls. That set-up reproduces their saved numbers to 6–11 digits; the float32 behaviour of `Model.predict` is inferred from three independent numerical signatures. `experiments/tf_crosscheck_float32.py` checks it with real TensorFlow but was not run.
* **SciPy versions differ** (1.10.1 for the authors, 1.18 here). The `trust-constr` paths leave the authors' logged paths after iteration 6 (DBTO) and between iterations 20 and 30 (HBTO).
* **The HBTO comparisons use MMA**, not the authors' `trust-constr`, which takes more than 2 s per iteration here. The consistent filter and the change of optimizer were introduced together, so their effects are not separated.
* **The newer networks were trained on different data**: 2,601 converged solutions instead of the 7,000 training rows of the authors' 10,000 voxel samples (the other rows are their validation and test sets).
* **Sample sizes and scope.**
  * 2–3 runs per HBTO configuration and 4–6 per DBTO configuration.
  * Only the 60 × 30 half MBB beam.
  * No 3-D problems, other load cases, or CNN/FNO/PINN surrogates.
* **The newer networks cover only the multiscale problem.** None were built for the density problem (§7 shows there was nothing to recover there).

## 11. Where each number comes from

| Section | Script(s) | Output in `results/` |
|---|---|---|
| §2–3 equations and audit | `tests/`, `experiments/exp_audit_extras.py` | `audit_extras.json` |
| §4 float64 fix | `exp_precision.py`, `exp_patch_check.py`, `analyse_patch_check.py` | `precision.json`, `patch_check*.json` |
| §5 ground truth, published network | `gt_voxel_lattice.py`, `gt_fitted_grid.py`, `gt_offgrid.py`, `exp_surrogate_truth.py` | `fitted_grid.npz`, `offgrid_truth.npz`, `surrogate_truth.json` |
| §6 newer networks | `exp_newer_models.py`, `exp_newer_models_to.py`, `analyse_newer_models.py`, `make_comparison_figure.py` | `newer_models*.json`, `to_baselines.json`, `models/` |
| §5–6 re-scoring of every final design (fixes of 18 Nov 2026) | `rescore_final_designs.py` | `final_designs_rescored.json` |
| §7 noise floor | `exp_dbto_ensemble.py`, `exp_dbto_part2.py` | `dbto_part2.json`, `dbto_ensemble*.npz`, `dbto_ensemble.log` |
| §8 second derivatives, architectures, ensemble, cost | `exp_higher_order.py`, `exp_architectures.py`, `exp_hbto_surrogates.py`, `exp_efficiency.py` | `higher_order.json`, `architectures.json`, `hbto_surrogates.json`, `efficiency.json` |
