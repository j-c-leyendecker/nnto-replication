# Your final results vs the original paper

**Paper:** J. Najmon and A. Tovar, "Evaluation of Neural Network-Based Derivatives for Topology Optimization", *ASME J. Mech. Des.* 146(7), 071704 (2024): Tables 1–2, Figs. 5, 6, 8, 9, 11, Sec. 5.
**Your results (the reference):** `comprehensive_results.docx` (your final results) and, for the multiscale optimisation runs that it does not include, the console output in `improvemt_document.docx`. Section numbers like "§3.9" point to `comprehensive_results.docx`.
**Checked against:** the authors' saved results, an independent re-implementation of the finite-element scoring, and a complete rerun of the package (`verification/`).

## 1. In short

* **Confirmed:** NNJ, CSM and AD give the same derivatives (to about 1e-15), and the paper's printed Table 1 errors, Fig. 8 and Fig. 11 values come out of the authors' saved models and designs exactly.
* **Overturned:** the paper's "CFD is less accurate" result is a float32 bug in the code, not the method; and its multiscale (HBTO) compliances are 8.5–10.3 % too optimistic, because the network overestimates the stiffness.
* **Not supported:** the paper's ranking of methods on the density problem (Fig. 5) is within run-to-run noise: repeated runs of one method land 1.5 % apart.
* **Your improvement:** a deeper network trained on mesh-converged data with derivative labels gives designs with 3.9–4.6 % lower true compliance than the paper's, within about 0.1 % of the best design found with the exact material model. About half of that gain is the better network and half the consistent formulation and optimizer.

## 2. The paper's results, one by one

| # | Paper result | Paper | Your result | Verdict |
|---|---|---|---|---|
| 1 | Table 1: density (DBTO) network test MSE, 10,000 samples | 1.87 × 10⁻⁷ | 1.866 × 10⁻⁷ against the exact SIMP law (§3.1) | ✅ Reproduced |
| 2 | Table 1: multiscale (HBTO) network test MSE | 1.47 × 10⁻³ | 1.468 × 10⁻³ on the authors' own test rows (§3.3) | ✅ Reproduced |
| 3 | Table 1: 100-sample network (1.44 × 10⁻⁴) and training times (7–41 min) | — | not rerun: the authors' saved networks were used | — Not tested |
| 4 | NNJ, CSM and AD give identical sensitivities | "to working precision" | agree to 7.5 × 10⁻¹⁶ – 1.5 × 10⁻¹⁵ (§2.4), and for six architectures (§7.8) | ✅ Confirmed |
| 5 | CFD differs from the others (Figs. 8, 11) "most likely" because of the finite step | MSE 1.11 × 10⁻³ (DBTO), 1.07 × 10⁻⁴ (HBTO) | same MSEs, predicted exactly by float32 rounding of the step, which makes it 1.0133 × 10⁻⁶ instead of 10⁻⁶ (§2.8). With float64, CFD's error drops from 1.65 × 10⁻² to 2.3 × 10⁻¹⁰ (§2.1) | ❌ A code bug, not the method |
| 6 | Fig. 8: differences between the density designs (10 pairs) and their starting sensitivities | θ-MSE 7.85 × 10⁻⁴ – 7.58 × 10⁻²; sensitivity MSE 0 – 2.84 × 10⁻³ | all 20 values recomputed from the authors' saved designs to 3 significant figures (check made here) | ✅ Reproduced |
| 7 | Fig. 5: density-problem compliance, one run per method | SIMP 81.97, NNJ 82.16, CFD 80.84, CSM 82.69, AD 80.96 | the saved designs give exactly these values; but 10 repeated runs (§4) land in two groups, ≈ 80.8 and ≈ 82.1, whatever the method | ❌ The ranking is noise |
| 8 | Fig. 5: iterations | 287–657 | 220–405 (§4; 220–343 for the patch-check runs) | ⚠ Different SciPy version, different paths |
| 9 | Fig. 6: 100-sample network gives 86.38 | 86.38 | not rerun | — Not tested |
| 10 | Fig. 9: multiscale (HBTO) final compliance | NNJ 75.70, CFD 76.78, CSM 76.14, AD 75.70 | the same designs scored with a converged material model: 82.16–84.25 (authors' voxel model), 82.65–84.72 (converged reference) (§3.9) | ❌ 8.5–10.3 % too optimistic |
| 11 | Fig. 9: iterations; compliance at iterations 10 and 100 | 1,105–1,371; 172.68–173.83; 105.19–106.98 | not rerun (30–36 h per run with the original optimizer) | — Not tested |
| 12 | Fig. 11: differences between the multiscale designs (6 pairs) | θ-MSE 3.23 × 10⁻³ – 4.12 × 10⁻² | all six reproduced (with the 7.5 % bound snapping of the Fig. 9 caption) | ✅ Reproduced |
| 13 | Table 2: seconds per iteration for the sensitivities | CSM fastest (0.088 / 0.180 s), NNJ and AD slowest (2.7–8.6 s) | original routines without TensorFlow: CFD fastest (2–4 ms), AD slowest (1.1–5.9 s); vectorised, all five take 1–60 ms (§5.2). Your §5.1 repeats the paper's Table 2; it is not a measurement | ⚠ Times measure the implementation, not the method |
| 14 | Sec. 5: NNJ and AD recommended | — | all four are equally accurate once CFD runs in float64; only reverse-mode AD stays cheap with many inputs (§5.6) | ⚠ Partly |
| 15 | Sec. 5: a more accurate network gives a better design (Fig. 6) | DBTO only | holds for HBTO too: the improved network's designs have 3.9–4.6 % lower true compliance | ✅ Confirmed and quantified |
| 16 | Sec. 4.3: no ground truth for HBTO | — | built one: the published network's derivative error is 64 % / 14 % (whole range / interior) | New |

## 3. What you improved, and by how much

| Metric | Paper / original | Your final result | Change |
|---|---|---|---|
| CFD derivative error at h = 10⁻⁶ (DBTO / HBTO) | 1.65 × 10⁻² / 1.53 × 10⁻² (float32 step) | 2.3 × 10⁻¹⁰ / 1.1 × 10⁻⁹ (float64) | > 10⁷× lower |
| HBTO network error on stiffness values (300 independent points) | 4.08 % | 0.025 % | ~160× lower |
| HBTO network error on stiffness derivatives, whole range / interior | 64 % / 14 % | 0.51 % / 0.28 % | 127× / 52× lower |
| Stiffness predictions that are not positive-definite | 1.9 % | 0 % with the Cholesky output head (2.1 % without) | removed |
| Starting-design sensitivity error (raw gradients) | 22 % | 3.1 % | 7× lower |
| **True compliance of the final multiscale design** (lower = stiffer) | 82.11–82.65 for the paper's best design (reported as 75.70) | **78.88 ± 0.15** (3 trained networks) | **3.9–4.6 % lower** |
| The paper's network with your optimizer | 81.96 after 3,000 iterations; about 80.5 from 10,000 to 30,000 iterations, never converging | 78.88, converged in 694–1,148 iterations | 3.8 % / about 2 % lower |
| Distance from the reference optimum (78.85) | +4.1 to +4.8 % | −0.02 to +0.12 % | ≈ 99 % of the gap closed |
| Network's error on its own design's compliance | −7.8 to −8.4 % | within ±0.08 % (mean absolute 0.035 %) | 99–238× smaller |
| One multiscale optimisation | 1,105–1,371 iterations, 30–36 h (authors' logs) | 694–1,148 iterations, 1–2 min | — |
| Objective + gradient, one evaluation | 2.41 s (yours; 0.77 s in the clean rerun) | 0.014 s (0.011 s) | 170× (67×) faster |
| trust-constr iteration, 60 × 30 mesh | 538 s with the dense Jacobian (yours; 16 s clean) | 3.1 s with a sparse one (0.14 s) | 174× (111×) faster |
| Density (DBTO) problem | 80.84–82.69, one run per method | 80.77–82.16 over repeated runs, in two groups | no measurable gain |

Values come from `comprehensive_results.docx`, except the multiscale optimisation rows and the improved network's starting-design error (`improvemt_document.docx` §1, §7, §12), the 10,000–30,000-iteration runs (`exp_published_long.py`, run here) and the time per run (rerun here).

**Where the 4 % comes from** (`improvemt_document.docx` §1; true compliance, your optimizer):

| Configuration | True compliance | Converged |
|---|---|---|
| Paper's network | 81.96 at 3,000 iterations; ≈ 80.5 later | never |
| Paper's 1 × 64 network retrained on converged data with derivative labels | 79.31 ± 0.24 | yes |
| Deep 3 × 64 network on the paper's data | 79.24 ± 0.49 | no |
| Deep network, converged data, values only | 78.86 ± 0.12 | no |
| **Deep network, converged data + derivative labels** | **78.88 ± 0.15** | **yes** |
| ResNet, Fourier, SIREN, KAN (same data and labels) | 78.83–78.87 | yes |
| Exact material model (reference) | 78.85 | yes |

* Of the 3.9–4.6 % over the paper's design, 2.0–2.6 % comes from the consistent formulation and optimizer (the paper's network reaches ≈ 80.5 with them) and about 2.0 % from the better network (80.5 → 78.88).
* Better data with derivative labels, or a deeper network, each recover most of the network's share; together they also converge.
* Newer architectures add nothing over a plain deep MLP.

## 4. Corrections to `comprehensive_results.docx`

These are labelling and interpretation errors in the document; the numbers themselves come out of the result files correctly.

| Where | What it says | What it is |
|---|---|---|
| Key findings | "CFD optimal step size is ~1e-7 to 1e-8 in float64" | Your own §2.1–2.2 tables have the smallest float64 error at h ≈ 1e-5 (DBTO: 5.7e-11) and 1e-6–1e-5 (HBTO: 1.1e-9). |
| Key findings | "float32 suffers catastrophic cancellation below 1e-6" | All-float32 CFD is already 1.4 % wrong at h = 1e-5 and 18 % at 1e-6 (DBTO); its best step is ≈ 1e-2. |
| Key findings | "Vectorized implementations provide 10–100x speedup" | 1.2× (CFD) to about 3,000× (AD, HBTO) in §5.2; 170× for objective + gradient (§5.4). |
| Key findings | "Newer architectures … show improved performance over paper MLP" | True against the paper's 1 × 64 network, but none beats a plain deep MLP: the deep 3 × 64 tanh MLP has the lowest derivative error (§8.3), and in the optimisation all of them reach the same compliance (78.83–78.88). |
| §2.1–2.5 | columns "rel_err / abs_err" | The largest and the RMS relative error (both scaled by the largest derivative), not a relative and an absolute error. |
| §2.6 | "CFD Optimal Step: 4.4926e-11" | The error reached with the optimal step, not the step size; likewise for Ridders. |
| §2.7 | "Mean Error / Std Error" per binade | The smallest and largest relative error in each binade (a range). |
| §3.8 | "Sensitivity MSE NN vs Truth 0.1245" next to "Sensitivity MSE CFD f32 vs NNJ Paper 1.07e-4" | Different bases: 0.1245 is on raw gradients, the paper's 1.07e-4 on filtered ones. On the same filtered basis the network error is 0.064, about 600× the paper's CFD difference (19× in relative terms). |
| §3.9 | "Underest. vs Converged" 9.2–10.3 % | About 30 % of the cells in the authors' designs have a hole side below 0.01, which the converged reference treats as a thin slit. Counted as solid, as in the authors' own data, it is 8.5–9.7 %, the same as the voxel model. |
| §4 | "Patch check … for HBTO optimization" | It is the density (DBTO) problem. The table holds 10 of the 19 planned runs. |
| §5.1 | "Paper Table 2 Reproduction" | These are the paper's own Table 2 values, copied. Your measured times are §5.2–5.6. |
| §5.2–5.6 | timings | Much slower than a clean run (538 s vs 16 s per dense trust-constr iteration; 63 ms vs 0.3 ms for a one-input reverse-mode gradient), so the machine was probably busy with other runs. The ratios point the same way as in the clean rerun (dense vs sparse 174× vs 111×). §5.6 is in seconds. |
| §7 | "Comparison of 12 architectures" | Six architectures, each trained two ways (values only, Sobolev). |
| — | missing | The multiscale optimisation results with the newer networks (true compliance of the final designs), the main improvement. They are in `improvemt_document.docx` §1 and §7 and in §3 above. |

## 5. How far your results reproduce

The whole package was rerun from scratch on Linux (`verification/`). Your documents were made on your Mac.

* **Identical, or equal to round-off:**
  * the mesh-converged reference (bit-identical), the voxel model (≤ 2.6 × 10⁻⁶) and the 300 test points (≤ 3.4 × 10⁻⁷);
  * every number in the surrogate-truth output (≤ 2.3 × 10⁻¹⁴ against your document), the paper-figure checks (§2.8) and the network accuracy table (§8);
  * all 60 newer networks: retrained bit-identical to the ones in your package;
  * all 31 multiscale optimisation runs: identical true compliance, iterations and volume.
* **Different, with the same conclusions:**
  * Round-off-sized numbers (10⁻¹⁶ to 10⁻⁹, e.g. CSM/AD agreement, CFD at tiny steps) vary by up to about 2× between runs and machines. Quote their order of magnitude.
  * The 1-D architecture study (§7) trains to different networks on your Mac and on Linux (e.g. value error 7.9 × 10⁻⁵ on yours, 4.7 × 10⁻⁵ on Linux); Linux runs repeat bit for bit. The conclusions (derivative labels cut the first-derivative error by roughly an order of magnitude; ReLU breaks second derivatives; NNJ = CSM = AD) hold on both.
  * The density-problem runs (§4) are chaotic. From the same start your Mac and Linux take different paths (CFD published, seed 2: 82.16 on yours, 80.77 on Linux; NNJ published: 80.77 vs 81.96), but both land in the same two groups, ≈ 80.7–80.9 and ≈ 81.9–82.2. That is the spread of the paper's Fig. 5, produced by round-off alone. The Linux rerun (19 runs) repeats the earlier Linux run exactly.
  * Timings depend on the machine and its load (§5).

## 6. Caveats

* **Not rerun from the paper:** Table 1 training times and the 100-sample network (Fig. 6); the authors' multiscale runs with their own optimizer (Fig. 9 iterations, Fig. 10; 30–36 h each); Appendix B.
* **Different optimizer:** your improvement uses MMA with a consistent density filter and hole sizes in 0.01–0.99; the paper used trust-constr with a heuristic sensitivity filter. The comparison with the paper's network under your optimizer isolates the network's share (about 2 %).
* **Different training data:** the improved networks use 2,601 mesh-converged samples with derivative labels; the paper's network used 7,000 training rows of its 10,000-sample voxel data set.
* **Sample sizes:** 2–3 runs per configuration; the paper's network is a single trained instance.
* **Not rerun here:** the long density-problem ensembles (`exp_dbto_ensemble.py`, `exp_dbto_part2.py`, about 1.5 h); the patch check covers the same question.
