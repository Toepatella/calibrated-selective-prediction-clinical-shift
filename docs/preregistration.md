# Pre-registration (applied evaluation protocol)

*Pre-registers **what we measure and report** for the trustworthy selective-prediction pipeline of [method_note.md](method_note.md) — the single source of truth for all notation, weights, thresholds, and budgets reused below. This document fixes the datasets, the shift construction and its diagnostic, the baselines, the metrics, and the operating-point search **before any result is seen**. It pre-registers no certificate. Following the honesty discipline of the method note, every component guarantee is cited only as a property of that method under its own assumptions; where realistic clinical shift violates those assumptions, the protocol below **measures and reports** the degradation rather than claiming it away. `α_acc` and `α_ood` are named, **separately measured** operating budgets, not a certified additive split; the small labeled target slice `D_tar^lab` exists only to **measure** residual degradation and fit a simple scalar correction, never to identify a nuisance parameter or certify a combined-shift case.*

**What pre-registration means here.** We keep the discipline — the grid, the budgets, the subgroups, the diagnostics, and their cutoffs are all fixed *before* we look at any held-out target result — but the object being fixed is a **measurement and reporting protocol**, not a claim. The point of fixing it in advance is to prevent the operating point, the subgroups, or the diagnostic cutoffs from being chosen *after* seeing what makes the numbers look good. Nothing below is certified; everything below is measured on held-out target data and reported with its sampling uncertainty.

---

## 1. Objective of the evaluation

We **want** the answered-case target risk

```
R_T^accept := E_{P_T}[ ℓ(Y, ŷ(X)) | A(X) ]
```

to be low under realistic clinical distribution shift, over the in-scope/answered event `A(x) = {g(x)=1} ∩ {o(x) ≤ t̂_ood} ∩ {ŵ_cov(x) ≤ w_max}` (method note §1.5.1). We do **not** certify `R_T^accept`. The pre-registered protocol calibrates all thresholds on source / unlabeled-target data, applies the weighted correction of method note §3, and then **measures** the realized selective risk, coverage, and OOD leakage on held-out target data — reporting the degradation overall and by subgroup. This is the measure-and-report stance set by the two impossibility results the method note cites: OOD detection is not distribution-free learnable in the unrestricted setting (Fang et al., NeurIPS 2022), and finite-sample distribution-free conditional coverage under covariate shift is not attainable once the density-ratio weight is estimated (Yang, Kuchibhotla & Tchetgen Tchetgen, JRSS-B 2024). These are limits on what **any** method can promise, not on our engineering; the protocol is built to quantify the residual, not to claim it away.

---

## 2. Datasets and the source → target deployment

Two benchmarks, each giving a labeled **source** distribution `P_S` (on which the pipeline is calibrated) and a held-out **target** distribution `P_T` (on which everything is measured). Splits and folds follow the disjointness discipline of method note §1.7; all fold boundaries are fixed before any target result is seen.

### 2.1 CAMELYON17-WILDS (covariate-dominant, cross-hospital)

- **Task.** Binary tumor-vs-normal patch classification (`K = 2`), the WILDS CAMELYON17 benchmark.
- **Shift.** Cross-**hospital** acquisition shift (scanner, staining, center). This is the **covariate-dominant** benchmark: the dominant mechanism moves `p(x)` (appearance), with class-conditional appearance drift `p(x∣y)` also present at the harder pairs — which is exactly why we measure residual degradation rather than assume pure covariate shift.
- **Source / target.** Source = the WILDS in-distribution training hospitals; target = the held-out hospital(s) (the official OOD validation/test centers). Each source→target **hospital pair** is evaluated and reported separately, never pooled.

### 2.2 CheXpert ↔ MIMIC-CXR (mixed covariate + label shift, cross-site)

- **Task.** Chest-radiograph multi-label/finding classification on the shared label schema between the two corpora; `K` and the finding set are fixed before results.
- **Shift.** Cross-**site** shift (institution, scanner/protocol, *and* finding prevalence). This is the **mixed** benchmark: both `p(x)` (site appearance) and `p(y)` (prevalence) move, plus appearance changes that move `p(x∣y)` and violate the pure-shift premises of the correction methods.
- **Source / target — both directions.** We evaluate **CheXpert → MIMIC-CXR** and **MIMIC-CXR → CheXpert** as two separate deployments, reported separately.

### 2.3 Folds (fixed before results, disjointness asserted in code)

Per benchmark and per source→target pair, the mutually **disjoint** folds of method note §1.7:

| Fold | Draw | Role |
|---|---|---|
| `D_cal` | labeled `∼ P_S` | accepted-region risk calibration of `(τ, λ)` |
| `D_bbse^src` | labeled `∼ P_S` (disjoint from `D_cal`) | source confusion matrix `Ĉ_S` |
| `D_disc` | source-vs-target covariates | domain discriminator `d(x)` (also the shift-type diagnostic, §3) |
| `D_tar` | unlabeled `∼ p_T(·)` | `ŵ_cov`, BBSE/MLLS `q̂_T`, setting `t_ood` inputs |
| `D_tar^lab` | labeled `∼ P_T`, `m_lab ≈ 50–200` | **measure** residual degradation; fit a simple scalar correction (method note §1.7) |
| `O` | OOD-exposure set | set `t_ood`; measure far-OOD leakage |
| held-out **test** | labeled `∼ P_T` | final measured metrics |

`m = |D_tar|` is a primary stress axis (small-`m` sweep, §6). `D_tar^lab` is disjoint from `D_tar` and from every other fold; its role is **measurement and a scalar correction only** — it identifies nothing.

---

## 3. Shift construction and the shift-type diagnostic

### 3.1 Shift construction (fixed before results)

The shift is the **natural** cross-hospital / cross-site shift the benchmarks already carry; we do not synthesize covariate shift on top of it. Where we deliberately **stress** a mechanism, we do so on pre-registered axes only (§6): the unlabeled-target size `m`, the choice of source→target pair/direction, and — for the mixed benchmark — a **label-prevalence re-weighting sweep** of the target sample that varies `p_T(y)` over a fixed grid of prevalence vectors while leaving the class-conditional images untouched. The prevalence grid is fixed before any result is seen.

### 3.2 The per-pair domain-discriminator shift-type diagnostic and its fixed cutoff

For each source→target pair we fit the domain discriminator `d(x)` of method note §1.5 on `D_disc` (source-vs-target, on the frozen embedding `φ(x)`, never raw pixels) and report its **AUROC** separating source from target features. This AUROC is the pre-registered **shift-type diagnostic**:

- **AUROC near chance** (`≤ τ_disc`) ⇒ the pair is in the **label-dominant / small-covariate regime**: the covariate correction is near-inert and the label correction is doing the work. This is the cleanest regime (fewest assumptions; method note §3.5/§7.6) — reported as such, **not** as a certified corollary.
- **AUROC above the cutoff** (`> τ_disc`) ⇒ the pair is in the **combined covariate+label regime**: the combined weight `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` (combined via `Ẑ`, **not** the product; method note §3.3) is in force, and the measured residual from `D_tar^lab` is reported alongside.

**The cutoff `τ_disc` is FIXED BEFORE ANY RESULT IS SEEN.** Its only role is to **label which regime a pair is reported under**, so that the cleaner label-dominant result is never silently substituted for the harder combined case. It is a **reporting/regime tag**, not a gate that selects a different guarantee — there is no guarantee in either branch. We pre-register `τ_disc` as a fixed AUROC value (e.g. AUROC `= 0.60`, the exact value recorded in the run config before results) together with the fitting/evaluation protocol for `d` (fold `D_disc`, embedding `φ`, classifier family, base-rate constant `ĉ = n_S/n_T`).

We additionally report, for every pair, the discriminator AUROC itself (a continuous shift-severity diagnostic) and the BBSE consistency check `q̂_T` vs `Ĉ_S p̂_T` (method note §3.4) — both as continuous diagnostics, regardless of which side of `τ_disc` the pair falls.

---

## 4. Baselines (fixed before results)

Every baseline is run on the same folds, the same frozen base model `f`, and the same held-out target test set, so differences are attributable to the method and not the data split. We pre-register four:

1. **Naive split conformal / split-RCPS (no shift correction).** The selective layer of method note §2 calibrated on source `D_cal` with **unweighted** risk and applied directly to target — i.e. `ŵ ≡ 1`. This is the "ignore the shift" reference; it carries the RCPS `(α_acc, δ)` property *only under exchangeability* (Bates et al. 2021), which the target deployment breaks, so we report its **realized** target risk and coverage, expecting degradation.
2. **Temperature scaling (post-hoc calibration only).** The frozen `f` recalibrated by temperature scaling on source, with selection by the recalibrated confidence and **no** importance weighting and **no** OOD routing. Isolates how much a pure calibration fix buys against the shift-aware pipeline.
3. **TRUECAM-style empirical pipeline.** A conformal selective pipeline with OOD **detection-and-removal** (suspected-OOD inputs dropped before conformal calibration/inference), reproduced as an *empirical* baseline. Per method note's positioning, this restores validity by removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and does not claim a maintained distribution-free guarantee in deployment; we run it to show what the **measured residual leakage** the detect-and-remove framing leaves unquantified actually is on `O`.
4. **Label-shift baseline (BBSE label weighting).** The BBSE estimator `ŵ_lab = Ĉ_S⁻¹ q̂_T` (Lipton, Wang & Smola 2018) applied as the **only** correction (no covariate weight, no `Ẑ` combine, no OOD routing). This is both the label-shift reference and the consistency diagnostic of method note §3.5. (Our full pipeline uses MLLS+BCTS as the primary prevalence estimator; BBSE is retained here as the baseline/diagnostic.)

The **full pipeline** of method note §§3–5 (MLLS+BCTS label weight, discriminator covariate weight, `Ẑ`-combine, OOD routing, LTT-wrapped grid, optional `D_tar^lab` scalar correction) is the method under test against these four.

---

## 5. Metrics (fixed before results)

All metrics are **measured on the held-out target test set** (and, where a residual/weight correction is needed, on `D_tar^lab`), reported per benchmark, per source→target pair/direction, and per pre-registered subgroup. Each is a reported number with its sampling uncertainty — none is a certificate.

### 5.1 Primary selective-prediction metrics

- **Realized selective (accepted) risk** `R̂_T^accept` — empirical accepted-case error `E_{P_T}[ ℓ ∣ A(X) ]` on labeled target; with importance weights it is the self-normalized (Hájek) estimator `R̂_w = Σ ŵ·ℓ / Σ ŵ` (method note §1.7). Reported with the betting / hedged-capital interval (Waudby-Smith & Ramdas 2024) at the pre-registered confidence `δ` — the confidence of the *reported interval*, not a PAC promise about the pipeline.
- **Coverage / answer rate** `coverage = P_T(A(X))`, **decomposed** into the three routing outcomes: abstain-on-weight (`ŵ_cov > w_max`), route-on-OOD (`o > t̂_ood`), defer-on-uncertainty (`u > τ̂`).
- **AURC** — area under the risk–coverage curve, summarizing the selective-risk/coverage trade-off across operating points.
- **ECE** — expected calibration error of the (recalibrated) posterior `σ̃(f(x))` on target, since the displayed confidence is the clinician-facing signal. The recalibrator `σ̃` is fit on source and may itself drift under shift (method note §6.5, §7.6); target ECE **measures** that drift rather than assuming it away.

### 5.2 OOD metrics (measured on the exposure set `O`)

- **OOD AUROC** and **FPR95** of the score `o(x)` separating in-scope from `O`, for **each** detector (Mahalanobis primary; energy, kNN ablations; method note §4).
- **Measured far-OOD leakage** — the fraction of `O` scoring `≤ t̂_ood` at the operating threshold, reported against the `α_ood` budget the threshold was set to spend.
- **Exposure-set sensitivity** — how `t̂_ood` and the leakage estimate move when the (swappable) `O` is changed, since `O` is a modeling choice, not ground truth.

### 5.3 Subgroup / fairness audit metrics

For each pre-registered subgroup `s` (§7), on labeled target (`D_tar^lab` where available):

- **Subgroup selective risk** `R̂^accept_s` — the Hájek weighted accepted-in-scope risk restricted to `s`, with its finite-sample interval (wide where the subgroup labeled sample is small, and the panel says so).
- **Subgroup abstention / defer rate** — so low subgroup risk *bought by* high abstention is not mistaken for good performance (silent under-service is surfaced explicitly).
- **Subgroup OOD-routing and weight-routing rates** — `P̂(o > t̂_ood ∣ s)` and `P̂(ŵ_cov > w_max ∣ s)`.

### 5.4 Reported reliability diagnostics (not gates)

Reported alongside every evaluation, never used to silently emit or withhold a number (method note §1.7, §7.5):

- **Effective sample size** `n_eff = (Σ ŵ_i)² / Σ ŵ_i²` (Kish) — flags low-effective-sample regimes; low-`n_eff` configurations are reported as low-reliability, not averaged in silently.
- **BBSE conditioning** `κ(Ĉ_S)` — label-shift error grows as `Ĉ_S` becomes ill-conditioned; worst-class error reported with it.
- **BBSE consistency check** — `q̂_T` vs `Ĉ_S p̂_T`, to flag gross anti-causal / appearance-shift violation.
- **Per-pair discriminator AUROC** — the §3.2 shift-severity diagnostic, reported continuously.
- **Clip-induced abstention rate** — the mass routed by `ŵ_cov > w_max`, since clipping itself biases the weights and its effect is reported empirically.

### 5.5 Explanation faithfulness & stability (Decline-Attribution Record)

Additive measurements of the per-case **Decline-Attribution Record (DAR)** — the faithful-by-construction explanation of *why the model declined* (which reliability gates fired, with the signed margin to each threshold; method note §5.1, §6.3, §6.6, and explainability_framing §3). "Faithful by construction" here means **faithful to the routing decision, not to the diagnosis**: the checks below verify the explanation matches the predicate the pipeline actually evaluated to route the case, *not* that the routing or the class prediction is correct. These checks introduce **no new claim**: they audit an artifact the pipeline already computes, the thresholds `τ̂`, `t̂_ood`, `w_max` remain tuned operating points (not bounds), and the margins remain signed distances in score space, **never** per-case error probabilities (this respects §8 — in particular the no-per-case-guarantee item). All are measured on the held-out target test set, reported with sampling uncertainty, and stratified by the pre-registered subgroups (§5.3, §6.3). Checks (A)–(B) are executable invariants over already-computed gate scores; (C)–(E) reuse the existing augmentation family and calibration folds; (F) is planned/IRB-gated.

- **(A) Construction-faithfulness invariant** — for every declined case, assert in code that the DAR's reported firing set equals the set of gate predicates the routing rule actually evaluated false, and that each reported margin equals the recomputed score-minus-threshold (`m_u = τ̂ − u(x)`, `m_ood = t̂_ood − o(x)`, `m_w = w_max − ŵ_cov(x)`). Report the **discrepancy rate; target exactly 0** — a non-zero rate is a bug, not a modelling residual. No analogous exact check exists for the saliency baseline (E).
- **(B) Counterfactual / margin validity, conditioned on firing-set cardinality** — the by-construction counterfactual flip is stated *conditionally* to avoid overclaiming. For **single-gate declines** (exactly one violated gate): report (i) the flip-rate when the lone binding score is moved to its threshold (target high, by construction — sufficiency) and (ii) decision-invariance under non-binding perturbations within their passing region (target high — necessity of the named gate). For **multi-gate declines** (≥2 violated gates): the flip is defined only when **all** violated scores are moved to their thresholds simultaneously (report flip-rate, target high), with single-score moves reported as *not* expected to flip. The rate and composition of multi-gate declines is itself the informative number. This is the model-sensitivity logic of the saliency sanity checks (Adebayo et al. 2018) applied to the decline explanation.
- **(C) Input-perturbation stability vs distance-to-threshold** — apply clinically plausible, label-preserving perturbations (small rotations/translations, intensity/stain jitter, mild noise/compression within a pre-registered acquisition-realistic family) and report (i) the binding-gate flip rate and (ii) the distribution of `|Δ margin|` for each of `m_u`, `m_ood`, `m_w`, **stratified by distance-to-threshold**, so instability is attributed to genuinely borderline cases rather than to explanation noise. Stability is **measured, not guaranteed**; configurations where it is low are reported as such (method note §6.5), and the OOD/weight scores are themselves known to degrade under shift (method note §4.4, §7.3).
- **(D) Re-estimation / retraining stability** — re-estimate the thresholds across calibration folds, and retrain `f` (or vary seeds); report agreement of the binding-gate label and rank-correlation of margins for matched cases. This is the reproducibility analogue of the Arun et al. (2021) cross-retraining criterion.
- **(E) Saliency-baseline contrast under the same sanity checks** — for the same declined cases, generate a post-hoc saliency explanation over the input (an attribution map for the binding score or the classifier logit) and subject it to the *same* model-parameter-randomization and label-randomization sanity checks (Adebayo et al. 2018) and the Arun et al. (2021) repeatability/reproducibility metrics. The pre-registered expectation, grounded in that work, is that saliency may be insensitive to randomization (its documented failure mode) whereas (A) is exact and (B) is sensitive by design. This is reported honestly as a faithfulness/stability **asymmetry**, not a head-to-head accuracy race — the DAR and saliency explain different objects (a gatekeeper's choice vs a classifier's pixel attention). *(Concrete saliency baselines — Grad-CAM (Selvaraju et al. 2017) and SHAP (Lundberg & Lee 2017) — are verified in explainability_framing §3.4(E) / ledger; named here as the saliency-contrast baselines.)*
- **(F) Clinician comprehension (planned / IRB-gated)** — a small reader study testing whether clinicians correctly infer the firing gate and the appropriate action (act / defer / out of scope) from the rendered DAR. If not run for this paper, it is stated as planned and the clinician-facing claim is conceded to be argued from design (method note §6.4), not yet measured. *(The contestable-AI actionability anchor — Ploug & Holm 2020 — is verified in explainability_framing §3.4(F) / ledger.)*

---

## 6. What is FIXED before results are seen

The following are recorded in the run config **before any held-out target metric is computed**. This is the load-bearing pre-registration discipline: the operating point and the reporting structure cannot be chosen after seeing what flatters the numbers.

### 6.1 The operating-point grid and its multiple-looks wrapper

- **The `(τ, λ, t_ood)` grid `G`** — the fixed, finite grid over the selection threshold `τ`, the risk-control threshold `λ`, and the OOD threshold `t_ood` (method note §1.6, §5.4). The grid bounds and spacing are fixed in advance.
- **The LTT family wrapping `G`** — the Learn-Then-Test (Angelopoulos et al.) multiple-testing family used as the **multiple-looks correction** over `G`, applied as published. This makes the recorded operating point reproducible and free of grid-selection bias on the calibration set; we claim **no new validity guarantee** survives the shift from it.
- The risk threshold `λ̂` is selected on calibration by the RCPS `inf`-rule (method note §2.2); its realized risk is then **measured** on target, not re-asserted.

### 6.2 The operating budgets (reporting budgets, not a certified split)

- **`α_acc`** — the accepted in-distribution selective-risk budget the risk threshold `λ̂` is tuned to on calibration, and against which realized accepted risk is then reported.
- **`α_ood`** — the far-OOD leakage budget the OOD threshold `t̂_ood` is set to spend on `O` (method note §4.3), and against which realized leakage is then reported.
- Both target values are fixed in advance. They are **separately measured operating budgets**: we present their sum `α = α_acc + α_ood` *only* as a single convenience headline for "total tolerated error," and we **do not** claim it is a proven additive bound on deployed target risk, nor a union-bound certificate combining the accepted risk and the audited leakage. There is no certified `α = α_acc + α_ood` split.
- **`δ`** — the confidence level of the **reported interval** (the betting / hedged-capital UCB used inside RCPS, Bates et al. 2021 §3.1.3; Waudby-Smith & Ramdas 2024), fixed in advance. It is the confidence of the reported CI, not a PAC promise about the deployed pipeline.

### 6.3 The pre-registered subgroups

The subgroup axes for the §5.3 / §6.2-of-the-method-note audit are fixed before results: acquisition **site / scanner / hospital** (per benchmark), and the clinically salient strata available per corpus (e.g. sex, age band, self-reported race where ethically collected, and case-type strata such as biopsy-confirmed vs. screening). The subgroup audit is **reporting**, not a multiplicity-corrected guarantee — we do not wrap it in a new certificate and call it a fairness guarantee.

### 6.4 The minimum-coverage floor

A pre-registered **minimum-coverage floor** `coverage_min`: operating points whose realized answer rate falls below `coverage_min` are reported as **degenerate** (risk driven down only by answering almost nothing) rather than presented as a favorable risk number. The floor is fixed in advance so that "low risk" cannot be manufactured after the fact by collapsing coverage. It is a reporting guard on the risk–coverage trade-off, not a guarantee.

### 6.5 Fixed routing and weight controls

Also fixed before results: the weight clip / routing cap `w_max`; the base-rate constant `ĉ = n_S/n_T`; the label-weight floor/ceiling `[w_lab,min, w_lab,max]` (`w_lab,min > 0`); the primary OOD detector (plain tied-covariance Mahalanobis, with energy and kNN as ablations); the primary prevalence estimator (MLLS+BCTS, with BBSE as the baseline/diagnostic); the recalibrator `σ̃` (temperature/Platt, fit on a held-out source fold); and the stress axes of §6.6.

### 6.6 Stress axes (pre-registered)

The deliberate stress sweeps, each on a fixed grid: small **unlabeled-target size `m`**; **source→target pair / direction** (each CAMELYON17 hospital pair; both CheXpert↔MIMIC directions); and, for the mixed benchmark, the **target-prevalence sweep** of §3.1. These define the failure-mode catalog the experiments stress; the cells are fixed before results.

---

## 7. The role of the labeled target slice `D_tar^lab` (measurement only)

`D_tar^lab` (`m_lab ≈ 50–200`, disjoint from all other folds, drawn before calibration) exists for **two measurement purposes only**, stated plainly so it is not over-read:

1. to **measure** the residual degradation of the shift correction (plug-in risk under `ŵ` vs the empirical risk on the labeled slice), reported with its own slice-size uncertainty; and
2. to **fit a simple scalar correction** to the combined weight and **report the measured residual gap** after it.

It does **not** identify any nuisance parameter, it does **not** certify the combined covariate+label case, and the factorizable-shift premise it accompanies is a **modeling choice** that is not identifiable from unlabeled target alone (method note §3.3, §7.4) — never an identification result. If `D_tar^lab` is unavailable for a pair, the combined-shift result is reported **uncorrected** (method note §3.4 step 3), not certified.

---

## 8. What is explicitly NOT claimed

- **No certificate of `R_T^accept`.** We do not certify the answered-case target risk; we measure it on held-out target and report the degradation. No component guarantee (RCPS `(α,δ)`; weighted-conformal `1−α` marginal coverage; BBSE/MLLS consistency) is attributed to the deployed pipeline — each is cited only as a property of its own method under its own assumptions, which clinical shift typically violates.
- **No certified `α = α_acc + α_ood` additive split.** `α_acc` and `α_ood` are separately measured operating budgets; their sum is a reporting convenience, not a union-bound certificate.
- **No identification from the labeled slice.** `D_tar^lab` only **measures** residual degradation and fits a scalar correction; it identifies and certifies nothing (§7). There is no Factorizable-Joint-Shift identification claim and no "labeled-slice-as-identification" step.
- **No distribution-free OOD guarantee.** OOD routing is a measured screen with a reported leakage rate on a stated, swappable exposure set `O`; distribution-free OOD detection is provably not learnable in the unrestricted setting (Fang et al. 2022).
- **No new guarantee from LTT under shift.** LTT is used as a plain multiple-looks correction over the grid `G`; we claim no new validity guarantee survives the source→target shift.
- **No per-case guarantee.** The clinician-facing panel exposes a calibrated case-level signal whose population reliability is measured; it asserts no finite-sample per-patient coverage or risk bound.
- **The Decline-Attribution Record explains the routing, not the diagnosis.** The DAR is faithful-by-construction to *which reliability gate fired and by how much* (faithful to the routing, not the diagnosis); it explains neither why the classifier assigned its label, nor the disease, biology, or causal dynamics of the patient. Its margins are signed distances to tuned thresholds, not calibrated error probabilities, and the gated scores inherit the component-method limits of method note §7.

*The contribution this protocol pre-registers is a **measurement / auditability discipline** — a fixed-in-advance protocol for what we measure, report, and stratify under realistic shift — not a certificate. Every number it produces is measured on held-out target data and reported with its uncertainty and its reliability diagnostics.*
