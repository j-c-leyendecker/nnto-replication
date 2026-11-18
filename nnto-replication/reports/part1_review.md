# Report 1 — Critical paper review

**Paper:** J. Najmon and A. Tovar, "Evaluation of Neural Network-Based Derivatives for Topology Optimization", *ASME J. Mech. Des.* 146(7), 071704 (2024), doi:10.1115/1.4064243.
**Code:** https://github.com/edrl-purdue/nn-derivative-to (the uploaded `nn-derivative-to-main.zip`, folder `NN_derivatives_TO_v3_final`).

## 1.1 What the paper does

The paper compares four ways of differentiating a trained multilayer perceptron (MLP) with respect to its inputs — the analytic chain rule ("neural network Jacobian", NNJ), central finite differences (CFD, h = 1e-6), the complex-step method (CSM, h = 1e-12) and TensorFlow automatic differentiation (AD) — and uses each to supply material-model derivatives to two compliance topology-optimization (TO) formulations on a 60 × 30 half MBB beam:

* **DBTO**: a 1-64-1 sigmoid MLP replaces the SIMP law E(θ) = Emin + (Emax − Emin)θ³; the exact SIMP law gives a ground truth.
* **HBTO**: a 2-64-4 sigmoid MLP maps the width and height of a rectangular hole in a square cell to the homogenized tensor entries [C11, C12, C22, C66]; no ground truth is available in the paper.

Both networks are trained on 10,000 Latin-hypercube samples (70/15/15 split, Adam, 5000 epochs). The optimizer is SciPy `trust-constr` with a sensitivity filter. One optimization run is made per derivative method. Conclusions: NNJ, CSM and AD agree to working precision; CFD differs slightly; NNJ and AD are "the most consistent" and are recommended; NN-based sensitivities are "sufficient" for DBTO and HBTO.

## 1.2 Strengths

* A clear, self-contained statement of the four methods and a worked Jacobian example.
* An exact reference (SIMP) for the DBTO case.
* Code, trained models, data set, optimizer logs and result arrays are public. That is what made this audit possible: every number below that refers to "the authors' saved results" comes from their own files.
* The authors state several limitations themselves: MLP only, no HBTO ground truth, timings "not optimized", slow trust-region convergence.
* The first-order statement that NNJ, CSM and AD return the same derivative of a smooth MLP is correct, and this audit confirms it for five further architectures.

## 1.3 Errors in the published formulation

Each item is checked by a unit test in `tests/` (name in brackets).

| # | Label | Finding |
|---|---|---|
| P1 | **Confirmed mathematical error** | Eq. (11) prints ∂φ/∂θe = uᵀ(∂K/∂θe)u. From φ = fᵀu and Ku = f, ∂u/∂θ = −K⁻¹(∂K/∂θ)u, so ∂φ/∂θe = **−**uᵀ(∂K/∂θe)u. The code has the correct sign (`dc = -1 * ceUdKU`). [`test_paper_eq11_sign_is_wrong`, `test_gradient_passes_taylor_test`] |
| P2 | **Confirmed mathematical error** | Eq. (15) prints dE/dθ = −p(Emax − Emin)θ^(p−1). The derivative of Eq. (14) is **+**p(Emax − Emin)θ^(p−1). The code is correct. Used together, the two sign errors cancel for SIMP; applied as printed to a network derivative (which is positive) Eq. (11) gives the wrong sign. [`test_simp_derivative_sign_and_value`] |
| P3 | **Confirmed mathematical inconsistency** | Eq. (1) defines a layer as F(W y + B) (column convention) but Eqs. (3)–(6) multiply the layer Jacobians in the order that is valid only for F(y W + B) (row convention, used in Appendix B and in the code). With Eq. (1) as written the factors of Eq. (6) are not conformable for non-square layers. Eq. (2) also carries W_NL and B_NL at the outermost level where W_NL+1 and B_NL+1 are meant. [`test_eq4_needs_the_row_vector_convention`] |
| P4 | **Confirmed paper–code inconsistency** | Sec. 4.2 and Fig. 4(a) describe a clamped left edge (u = 0) with a load at the bottom-right corner — a cantilever. The code applies symmetry rollers on the left edge, a load at the top-left node and a roller at the bottom-right node — the half MBB beam — and the published topologies are MBB topologies. Initial SIMP compliance: 356.48 for the code's boundary conditions (matches the authors' log), 330.31 for the cantilever as drawn. [`test_boundary_conditions_code_vs_figure`] |
| P5 | **Confirmed typographical errors** | Appendix B: the third entry of the vector in (B15b) should be −0.913, the second entry of F1(·) in (B16c) should be +0.290, the third entry of the row vector in (B18b) should be −0.218. As printed, (B18b) would give 0.66 instead of 1.623. The final value 1.623 is correct. [`test_appendix_b_intermediate_vectors_have_sign_typos`] |

## 1.4 Experimental and methodological weaknesses

| # | Label | Finding | Evidence |
|---|---|---|---|
| P6 | **Methodological limitation** | The stated stopping rule ("maximum change in θ less than 10⁻³") is not what stops the runs. `tol=1e-3` sets `xtol`, `gtol` and `barrier_tol` of `trust-constr`. The authors' logs show `gtol` termination for four DBTO runs, but `xtol` termination — the trust region collapsing while the optimality measure is still 2.7e-3 to 7.1e-3 — for DBTO-CSM and for the three HBTO runs with a log. "Iterations to convergence" in Figs. 5 and 9 are therefore partly iteration counts at trust-region collapse. | authors' `.txt` logs |
| P7 | **Methodological limitation** | One run per derivative method. NNJ, CSM and AD return identical sensitivities, so differences between their runs are path noise; the paper says as much ("accumulation of numerical errors") and then ranks the methods by those differences. The perturbation runs (`SUMMARY.md` §7) show the ranking does not survive a 1e-10 perturbation of the gradient, nor a change of software environment. | `exp_dbto_ensemble.py`, `exp_dbto_part2.py` |
| P8 | **Methodological limitation** | Final compliances of NN-based designs are evaluated with the surrogate that produced them. For DBTO the effect is small (≤ 0.1 %). For HBTO the four saved designs are 8.5–10.3 % more compliant under an independent material model than reported. | `exp_surrogate_truth.py` |
| P9 | **Methodological limitation** | "Neural network-based sensitivity coefficients are sufficiently accurate" is asserted for HBTO without a reference. Against a mesh-converged reference the surrogate's Jacobian has a relative RMS error of 78 % over the parameter domain (15 % in the interior [0.1, 0.9]²), and the initial-design compliance sensitivities are off by 22 % in relative L2 norm. | `exp_surrogate_truth.py` |
| P10 | **Methodological limitation** | The CFD discrepancy is attributed to "the approximation that occurs from its small finite step". It is a single-precision artefact of the implementation (finding A1 in Report 2); in double precision CFD at h = 1e-6 agrees with the other methods to 3e-10. The paper also notes that its findings "differ slightly" from the authors' earlier conference study, where all four methods agreed; the example script used there sets `floatx` to float64, the TO scripts do not. | `exp_precision.py`, reproduction tests |
| P11 | **Methodological limitation** | Table 2 times the implementation, not the methods: NNJ loops over samples in Python with two eager layer calls per sample; AD builds an N × N cross-batch Jacobian. More importantly, the per-iteration cost of the published runs (17 s for DBTO, about 100 s for HBTO, from the logs) is dominated by dense linear algebra inside `trust-constr`, so the derivative method accounts for roughly 0.5–16 % of an iteration (0.09–2.7 s of 17 s). | logs, `exp_efficiency.py` |
| P12 | **Claim not supported** | CFD and CSM are recommended for "third-party networks" whose details are not accessible. The CSM implementation re-implements the forward pass in NumPy from the extracted weights (Keras dense layers with real kernels do not accept complex input), i.e. it needs full white-box access. Also, "holomorphic activations" is stricter than necessary — a ReLU implemented by branching on the real part gives the correct one-sided derivative — while the real danger is silent failure for modulus-type operations. | `test_relu_network_methods`, `test_complex_step_fails_for_modulus_based_activation` |
| P13 | **Reproducibility gap** | The 100-sample DBTO model of Fig. 6 and Table 1 is not in the repository; training scripts are unseeded; `Run_HBTO_NN.py` ships with `maxiter = 50` and `der_type = 2`, not the paper's settings; the θ-MSE values of Fig. 11 are computed after the 7.5 % thresholding that the caption of Fig. 9 mentions only for plotting (recovered exactly once the thresholding is applied; the raw arrays give values about 6–10 % smaller). | `exp_audit_extras.py` |

## 1.5 What was not investigated in the paper

Second derivatives; architectures other than a one-hidden-layer sigmoid MLP; single precision (the default on GPUs); surrogate admissibility (positive-definite, monotone stiffness); the effect of derivative labels in training (mentioned in the introduction, not used); any second benchmark, mesh, load case or 3-D problem; any repetition of a run; any optimizer other than `trust-constr`; any check of the gradient handed to the optimizer against the objective it evaluates.

## 1.6 Does the paper's main conclusion hold?

* "NNJ, CSM and AD give identical sensitivities to working precision": **holds**, and extends to deep tanh, residual swish, Fourier-feature and constrained-output networks, and to second derivatives (analytic, hyper-dual AD and complex step of the analytic Jacobian agree to 1e-15).
* "CFD is slightly less accurate because of its finite step": **does not hold as stated**. The measured 1.33 % discrepancy is a float32 rounding artefact of `Model.predict`.
* "NNJ and AD are the most consistent and are recommended over CSM": **not supported**. The evidence is one optimization path per method, and paths differ by more under a 1e-10 gradient perturbation than between methods.
* "NN-based sensitivities are sufficient for HBTO": **not supported**. The differences between derivative methods are about three orders of magnitude smaller (in MSE of the initial sensitivities) than the error all of them share with respect to the true sensitivities.
