# Verification of your results (18 November 2026)

**Reference:** the console output you pasted into `improvemt_document.docx` and your final `comprehensive_results.docx`.
**Rerun:** every script of the package, from scratch, with the fixed code, on Linux (Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, 2 cores), into `full_rerun_2026-11-18/` (outputs, `logs/`, `figures/`). About 2 h of compute.
**Not rerun:** `exp_dbto_ensemble.py` and `exp_dbto_part2.py` (density-problem noise-floor ensembles, about 1.5 h; `exp_patch_check.py` answers the same question and was rerun), `exp_hbto_runs.py` (trust-constr multiscale runs, never run to completion by design), `tf_crosscheck_float32.py` (needs TensorFlow).

Files dropped from `full_rerun_2026-11-18/` because they are bit-identical to `results/`: the 63 trained networks (`models/`), the optimisation designs (`*.npz`) and `patch_check.npz` (now in `results/`).

## 1. `improvemt_document.docx`, section by section

Each line of the document was matched to the same line of the rerun's console output; "round-off-sized" means both values are below 1e-6 (errors of exact methods, CFD at tiny steps).

| § | Script | Numbers compared | Result |
|---|---|---|---|
| 1 | `analyse_newer_models.py` | 315 | All accuracy and optimisation numbers identical; only the training-time column differs (machine) |
| 2 | `analyse_patch_check.py` | 10 | Identical |
| 3 | `exp_architectures.py` | 212 | Different networks: training on your Mac and on Linux takes different paths (e.g. value error 7.9e-5 vs 4.7e-5). Linux repeats bit for bit; conclusions identical |
| 4 | `exp_audit_extras.py` | 67 | Equal to 3.8e-4 (finite-difference directional checks) |
| 5 | `exp_higher_order.py` | 78 | Exact methods agree to ~1e-15 in both; round-off-dominated second differences vary up to 2× |
| 6 | `exp_newer_models.py` | 6 | Identical |
| 7 | `exp_newer_models_to.py` | 25 | Identical |
| 8 | `exp_precision.py` | 229 | Deterministic columns (the float32 bug, 1.65e-2) identical; round-off columns (float64 CFD, all-float32) vary up to 2× |
| 9 | `exp_surrogate_truth.py` | 146 | Identical to 2.3e-14 |
| 10–11 | `gt_offgrid.py`, `gt_voxel_lattice.py` | progress lines only | Outputs: test points equal to 3.4e-7, voxel table to 2.6e-6 |
| 12 | `make_comparison_figure.py` | 9 | Identical |
| 13 | `make_figures.py` | — | All five figures drawn (`full_rerun_2026-11-18/figures/`) |

## 2. `comprehensive_results.docx`

Its tables are drawn from the result files. Against the rerun:

| Section | Result |
|---|---|
| §2 precision | as §8 above |
| §3 surrogate truth | identical to 6e-11 |
| §4 patch check (your 10 runs, on your Mac) | different paths from the same start (chaotic), same two groups: yours 80.77–80.88 and 82.06–82.16; Linux (19 runs) 80.72–81.03 and 81.94–82.12. The Linux rerun repeats the earlier Linux summary exactly (899 of 901 values identical, the rest to 4e-16). Your runs are kept in `your_patch_check_runs_from_comprehensive_results.json` |
| §5 efficiency | machine-dependent. Your times are 10–600× slower than this clean run (e.g. 538 s vs 16 s per dense trust-constr iteration), so your machine was probably busy; the speed-ups point the same way (dense vs sparse 174× yours, 111× here) |
| §6 higher order | as §5 above |
| §7 architectures | as §3 above (your Mac numbers) |
| §8 newer models | all 300 per-network accuracy values identical; all 60 networks retrained bit-identical |
| §9 HBTO surrogates | identical (72 of 72 values) |

## 3. Optimisation runs and final designs

* All 31 multiscale optimisation runs (reference, published network, 10 newer configurations) rerun from scratch: true compliance, surrogate compliance, iterations and volume identical (largest difference 0).
* `rescore_final_designs.py` on the rerun gives the same headline numbers as on the uploaded results (improved 78.8755, published 81.9590, 3.762 % lower; differences below 1e-10, from the voxel table).
* `extra_runs/`: the reference, published and improved runs repeated with the **unmodified** code (identical), and the published network for 10,000 iterations (80.50–80.62, not converged). `../results/published_long.json`: the published network for 30,000 iterations (≈ 80.5 from 10,000 iterations on, never converged).

## 4. Independent check

A separate finite-element implementation (written for this check, not using the package; same mesh, supports, load, element and filter as `Run_HBTO_NN.py`) reproduces:
* the start-design compliances (280.18687 and 255.61319);
* the authors' saved start-design sensitivities (`dc0`) to 1.4e-7, which shows they are the filtered sensitivities;
* the true compliance of the authors' four designs (both references) and of your optimisation runs, to 3 decimals;
* the network-vs-method ratio on the paper's filtered basis: 601×.

The paper's Fig. 8 (10 θ-MSE and 10 sensitivity-MSE values) was also recomputed from the authors' saved density designs: all 20 values agree to 3 significant figures.
