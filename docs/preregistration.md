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

**Disjointness is asserted at the group id, not the instance.** All fold disjointness is enforced at the **group level** — the **WSI slide id** for CAMELYON17 and the **patient id** for CheXpert/MIMIC — so no slide's patches and no patient's studies straddle two folds. Each run emits a **per-fold-pair set-intersection = 0 artifact** over the group ids (a printed, checked-in-code invariant), recorded before any target metric.

**Deliberate leakage-probe run.** As a calibrated sensitivity figure, we run a deliberate **leakage probe**: inject a known set of patient/slide cases that also appear in another fold into `D_cal`, and report **how much the realized accepted risk moves** relative to the clean (group-disjoint) run. This **quantifies** the optimism that group-level leakage would introduce — it is a measured sensitivity, **not** a bound on leakage and not a guarantee that the clean run is leakage-free.

---

## 3. Shift construction and the shift-type diagnostic

### 3.1 Shift construction (fixed before results)

The shift is the **natural** cross-hospital / cross-site shift the benchmarks already carry; we do not synthesize covariate shift on top of it. Where we deliberately **stress** a mechanism, we do so on pre-registered axes only (§6): the unlabeled-target size `m`, the choice of source→target pair/direction, and — for the mixed benchmark — a **label-prevalence re-weighting sweep** of the target sample. This sweep varies `p_T(y)` over a fixed grid of prevalence vectors **by re-weighting the target sample only** — each prevalence cell is produced by sub/re-weighting whole cases, with the **class-conditional images left untouched** (no pixel synthesis, no class-conditional edit), so only `p_T(y)` moves. The prevalence grid is fixed before any result is seen.

Each prevalence cell is **tagged by its domain-discriminator AUROC** (§3.2) as evidence that covariate shift is held **≈ constant** while `p_T(y)` varies: because re-weighting does not touch `p(x∣y)`, the `d`-AUROC should be ~flat across the cells, and we report it per cell to show the label axis is being moved in isolation. We **pre-commit** the following downgrade: if the separation between the **covariate-only** path and the **`Ẑ`-combined** path (the gain attributable to the label-aware correction) does **not exceed the betting intervals**, the real-data label-aware claim is **downgraded** to "**shown on synthetic + prevalence-reweighted only**" — i.e. the label-correction benefit is reported as demonstrated only under this constructed sweep, not on the natural cross-site prevalence shift.

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

## 4A. Pre-registered ablations (component swaps within the full pipeline)

Distinct from the §4 baselines (reference *methods*), these are **component swaps inside the full pipeline** that isolate one design choice while holding everything else fixed. Each is run on the same folds, the same frozen `f`, and the same held-out target test set; each is reported against the primary on the §5.1 metrics at a **matched total answer-rate** with betting intervals; and a "win" follows the **non-overlapping-interval rule** of §6.7. None of these ablations converts a measured quantity into a guarantee, and none recovers a certificate under shift (§8).

1. **MAPLS label-shift ablation (small-`m`–targeted).** Swap the primary prevalence estimator MLLS+BCTS (method note §3.4 step 2) for **MAPLS** — maximum-a-posteriori label-shift estimation, i.e. MLLS with a Dirichlet prior on `p_T(y)` — holding every other component fixed (discriminator covariate weight, `Ẑ`-combine, OOD routing, LTT-wrapped grid).
   - *What it isolates.* Whether Dirichlet-regularized MAP estimation of `p_T(y)` reduces label-weight error in the **small-`m`** regime, where the MLLS likelihood term is highest-variance (few unlabeled target points form `q̂_T`). Evaluated across the pre-registered small-`m` grid of §6.6. Across that grid we report `κ(Ĉ_S)` and the **effective rank** of `Ĉ_S`, so the regime where BBSE is ill-conditioned (inadmissible) and MAPLS's prior is doing the lifting is visible, not implicit.
   - *Fixed before results.* The Dirichlet prior — centered on the source prevalence `p_S(y)` with a fixed concentration — plus a small pre-registered concentration sweep reported as a sensitivity; the **same** `D_bbse^src` (for `Ĉ_S`) and `D_tar` (for `q̂_T`) folds as the primary; reporting on the §5.1 selective-risk metrics plus `κ(Ĉ_S)` and `n_eff` (§5.4).
   - *Honest framing.* MAPLS shares the label-shift + confusion-matrix-invertibility identifiability conditions of MLLS/BBSE (Garg et al. 2020); the prior trades variance for bias and carries **no finite-sample certificate**. The deliverable is a **measured** selective-risk delta vs MLLS+BCTS with betting intervals; if it does not clear the intervals in the small-`m` cells we report "**no measured small-`m` advantage**," not a null guarantee.

2. **Doubly-robust (DR) covariate-correction — key secondary analysis.** Replace the Hájek self-normalized accepted-risk estimator `R̂_w` (method note §3.4 step 5) with a **doubly-robust / augmented (AIPW-style)** estimator that pairs the discriminator weight `ŵ_cov` with an **outcome model** `m̂(x) ≈ E_S[ ℓ(Y, ŷ(X)) ∣ φ(x) ]` fit on labeled source, of the schematic form
   ```
   R̂_DR^accept  ∝  Σ_{A} [ m̂(xᵢ) + ŵᵢ ( ℓᵢ − m̂(xᵢ) ) ]   (self-normalized over the accepted in-scope set)
   ```
   holding all other components fixed.
   - *What it isolates.* Whether augmenting the **estimated** covariate weight with a source outcome model **shrinks the measured residual** degradation (the plug-in-vs-empirical gap on `D_tar^lab`, §7) — the partial robustness the DR construction buys: consistency if **either** `ŵ_cov` **or** `m̂` is correct (Yang, Kuchibhotla & Tchetgen Tchetgen 2024; Qiu, Dobriban & Tchetgen Tchetgen 2023). This is the empirical answer to "why only measure the residual rather than shrink it first" (method note §7.1).
   - *Fixed before results.* The outcome-model family for `m̂` (a light regressor on the frozen `φ(x)`), fit by **K-fold cross-fitting** so each accepted point is scored by an `m̂` trained on held-out folds — the standard nuisance-model hygiene the DR/AIPW asymptotics assume — on a **disjoint** fitting fold (labeled source, disjoint from `D_cal`, `D_bbse^src`, and test); the self-normalization convention; and that DR is run **primarily on the covariate-dominant path** (CAMELYON17, and the covariate factor of the mixed benchmark), since under *joint* covariate+label shift the pure-covariate DR construction does not transfer unmodified — on the mixed benchmark it is an exploratory, explicitly caveated arm.
   - *Honest framing.* DR validity here is **asymptotic**, not a finite-sample deployment certificate, and `m̂` is a **second estimated object** that can itself be misspecified — so the **Hájek single-model estimate remains the auditable primary headline**, and DR is reported as the residual-shrinkage diagnostic, showing the DR-vs-Hájek delta and both betting intervals. A smaller DR residual is a measured improvement, **not** a bound (§8).

3. **Weight-combination decomposition (the `Ẑ`-combine ablation).** Hold the frozen `f`, the folds, and every threshold fixed and swap **only how the covariate and label weights are combined**, sweeping the full decomposition ladder on the §5.1 metrics at a matched answer-rate: (a) **no reweighting** (`ŵ ≡ 1`), (b) **covariate-only** (`ŵ_cov`), (c) **label-only** (`ŵ_lab` — the §4 BBSE baseline's correction run inside the full pipeline), (d) **naïve product** (`ŵ_cov·ŵ_lab`), and (e) **`Ẑ`-divide** (`ŵ_lab·ŵ_cov/Ẑ`, the primary; method note §3.3).
   - *What it isolates.* Whether the double-count corrector `Ẑ` (method note §3.3, §7.4) actually buys a measured improvement **on real cross-site data** — i.e. whether (e) beats (d), and whether the combined (e) beats the single-shift rungs (b)/(c) — rather than only in the §3.3 synthetic worked example. This is the load-bearing real-data check for the combined weight the rest of the pipeline rests on, and the single most-requested addition from methodological review.
   - *Fixed before results.* The five arms; the matched answer-rate operating points (§6.7); and the recalibrated `σ̃` feeding `Ẑ`. Reported with betting intervals; a "win" for `Ẑ`-divide over the naïve product follows the §6.7 non-overlap rule. If (e) does **not** clear (d)'s interval we report "**no measured advantage of the `Ẑ`-combine on this pair**," not a claim that the correction helped.
   - *Honest framing.* None of the five arms is certified under combined shift (§8); the ablation **measures** which combine has the lowest realized weighted risk — it bounds none of them.

4. **MLLS-without-BCTS label-shift ablation.** Swap the primary MLLS+BCTS prevalence estimator (method note §3.4 step 2) for **MLLS on the raw, uncalibrated softmax** (BCTS removed), holding every other component fixed.
   - *What it isolates.* Whether the BCTS recalibration step is **load-bearing** — MLLS is consistent only when `f`'s outputs are calibrated (Garg et al. 2020), so this tests empirically whether removing that calibration degrades the prevalence estimate and the downstream selective risk, rather than asserting that it does.
   - *Fixed before results.* The same `D_bbse^src` / `D_tar` folds and the §5.1 metrics, plus the **calibration quality of `f` before vs. after BCTS** reported as **ECE, NLL, and Brier** on a held-out source fold and on target, so the mechanism (BCTS improves calibration ⇒ MLLS improves) is visible rather than assumed.
   - *Honest framing.* This isolates a calibration-quality effect, not a guarantee; the deliverable is a **measured** prevalence-error and selective-risk delta with betting intervals.

---

## 5. Metrics (fixed before results)

All metrics are **measured on the held-out target test set** (and, where a residual/weight correction is needed, on `D_tar^lab`), reported per benchmark, per source→target pair/direction, and per pre-registered subgroup. Each is a reported number with its sampling uncertainty — none is a certificate.

### 5.1 Primary selective-prediction metrics

- **Realized selective (accepted) risk** `R̂_T^accept` — empirical accepted-case error `E_{P_T}[ ℓ ∣ A(X) ]` on labeled target; with importance weights it is the self-normalized (Hájek) estimator `R̂_w = Σ ŵ·ℓ / Σ ŵ` (method note §1.7). Reported with the betting / hedged-capital interval (Waudby-Smith & Ramdas 2024) at the pre-registered confidence `δ` — the confidence of the *reported interval*, not a PAC promise about the pipeline.
- **Matched-answer-rate reporting (fixed before results).** Every headline realized selective-risk number is reported at a **matched total answer-rate** between the full pipeline and each of the four baselines (§4), so that a lower risk bought by answering less is never compared against a higher-coverage operating point as if equal. Because pipeline abstention is **split across three mechanisms** (abstain-on-weight / route-on-OOD / defer-on-uncertainty), the match is on **total** answer-rate — the decision-relevant quantity — and is therefore **approximate** (coverage-matching cannot equate the three mechanisms case-for-case). Beside every matched number we print (i) the **three-way routing decomposition** of that answer rate, (ii) **AURC** as the threshold-free companion (so the comparison does not hinge on the matched point alone), and (iii) the **frozen-`f` raw target-vs-source error** — no abstention, no weighting — as the **base-rate anchor** in the same cell. The matched-answer-rate operating points and the base-rate anchor are fixed before any result is seen (recorded in §6).
- **Coverage / answer rate** `coverage = P_T(A(X))`, **decomposed** into the three routing outcomes: abstain-on-weight (`ŵ_cov > w_max`), route-on-OOD (`o > t̂_ood`), defer-on-uncertainty (`u > τ̂`).
- **AURC** — area under the risk–coverage curve, summarizing the selective-risk/coverage trade-off across operating points.
- **ECE, NLL, and Brier** — calibration of the (recalibrated) posterior `σ̃(f(x))` on target, since the displayed confidence is the clinician-facing signal (ECE as the headline, with NLL and Brier as proper-scoring-rule companions that ECE's binning can hide). Reported **per benchmark, per subgroup, and at the deployed operating point** (the accepted-case slice the panel actually shows), with reliability diagrams of `σ̃`. The recalibrator `σ̃` is fit on source and may itself drift under shift (method note §6.5, §7.6); target ECE **measures** that drift rather than assuming it away. We pre-register a **display response** when measured ECE exceeds a fixed threshold `ECE_max`: the clinician panel **widens / greys / annotates** the displayed confidence for that benchmark/subgroup/operating point. This is explicitly a **display rule** — it changes how the confidence is rendered, never whether a case is answered — **not** a certification gate and not a bound.
- **Per-study (slide / patient) aggregation.** Alongside the patch-/token-level metrics, accepted-risk and coverage are also reported at the **slide level** (WSI slide id, CAMELYON17) and **patient/study level** (CheXpert/MIMIC) via a fixed positivity rule over the accepted patches/tokens (e.g. a study is called positive if ≥ a pre-registered fraction of its accepted patches are positive; the rule is fixed in §6). At the headline operating point we additionally report the **realized routed workload** — studies/day flagged for human review — as the deployment-facing quantity.
- **Severity-stratified report.** Accepted-error and OOD-leakage are additionally broken down by **FN vs FP on the clinically positive class**, and we report a **severity-weighted loss** alongside the 0–1 selective risk. The severity weights are stated as a **reporting choice** (a fixed cost matrix recorded in §6), **not** a certified cost bound — the severity-weighted number is a measured re-weighting of the same accepted cases, carrying no guarantee.

### 5.2 OOD metrics (measured on the exposure set `O`)

The exposure set `O` is split into **two tiers**, reported separately and never pooled: a **far-OOD** tier (clearly out-of-domain inputs) and a **near-OOD** tier (held-out scanner/site, or wrong-but-adjacent anatomy — inputs that look in-domain but are not). Both tiers are drawn **disjoint from the held-out test set and from `D_tar^lab`**.

- **OOD AUROC** and **FPR95** of the score `o(x)` separating in-scope from `O`, for **each** detector (Mahalanobis primary; energy, kNN ablations; method note §4), reported **separately per tier and per benchmark**.
- **Measured leakage** — the fraction of `O` scoring `≤ t̂_ood` at the operating threshold, reported against the `α_ood` budget the threshold was set to spend, **separately for the far-OOD and near-OOD tiers**. Near-OOD leakage is **reported and expected to be worse**, never certified: this is the unrestricted-detection regime where distribution-free OOD detection is provably not learnable (Fang et al. 2022), so the near-OOD number is a measured screen, not a bound.
- **TRUECAM-style head-to-head on the same `O`** — the detect-and-remove baseline (§4 item 3) is run on the **same** two-tier `O` as the full pipeline, so the measured residual leakage the detect-and-remove framing leaves unquantified is reported side-by-side per tier.
- **Exposure-set sensitivity** — how `t̂_ood` and the leakage estimate move when the (swappable) `O` is changed, since `O` is a modeling choice, not ground truth.

### 5.3 Subgroup / fairness audit metrics

For each pre-registered subgroup `s` (§7), on labeled target (`D_tar^lab` where available). The audit **leads with the abstention / coverage disparity across subgroups** — so unequal *service* (who gets answered at all) is read before any per-subgroup risk number, and a subgroup whose risk looks good only because it is rarely answered is surfaced first:

- **Realized labeled-target n** — the realized labeled-target sample size is **printed per subgroup AND per subgroup × class**; strata below a fixed `n_subgroup,min` (recorded in §6) carry an explicit **"unpowered — demonstrated on pooled target only"** tag and their risk numbers are not presented as standalone subgroup findings.
- **Subgroup abstention / defer rate** — reported **first** (the coverage-disparity lead above), so low subgroup risk *bought by* high abstention is not mistaken for good performance (silent under-service is surfaced explicitly).
- **Subgroup selective risk** `R̂^accept_s` — the Hájek weighted accepted-in-scope risk restricted to `s`, with its finite-sample interval (wide where the subgroup labeled sample is small, and the panel says so).
- **Subgroup OOD-routing and weight-routing rates** — `P̂(o > t̂_ood ∣ s)` and `P̂(ŵ_cov > w_max ∣ s)`.

### 5.4 Reported reliability diagnostics (not gates)

Reported alongside every evaluation, never used to silently emit or withhold a number (method note §1.7, §7.5):

- **Effective sample size** `n_eff = (Σ ŵ_i)² / Σ ŵ_i²` (Kish) — flags low-effective-sample regimes; low-`n_eff` configurations are reported as low-reliability, not averaged in silently.
- **Overlap / positivity diagnostic.** Source→target support overlap — the silent failure mode of every importance-weighted estimator — is summarized by four numbers reported together: (i) the per-pair **discriminator AUROC** (AUROC → 1 ⇒ near-disjoint supports, a positivity violation rather than merely large shift); (ii) the **routed-tail fraction** `P̂(ŵ_cov > w_max)` — the share of target mass whose density ratio is untrustworthy, i.e. the empirical positivity-violation rate; (iii) `n_eff`; and (iv) the **weight-distribution summary** — the median, 95th and 99th percentile, and max of `ŵ_cov` (and of the combined `ŵ`), with per-class and per-site weight **histograms**, because two weightings with identical `n_eff` can hide very different tails and `n_eff` alone will not reveal a single dominating point or a class/site whose mass is concentrated. We state explicitly that the **doubly-robust estimator (§4A) and the residual-correction CI (method note §6.1) both degrade as overlap worsens**, and every weighted result is read against these four; poor-overlap pairs are flagged, never silently averaged into a headline.
- **Detector feature-geometry diagnostic** — the **condition number `κ(Σ̂)` and effective rank** of the Mahalanobis tied covariance `Σ̂` (method note §4.1). A near-singular or low-effective-rank `Σ̂` is the quantitative signature of the **feature-collapse** failure mode (method note §7.6 (v)), where the task-trained embedding `φ(x)` has discarded the directions a novel input would occupy so the detector can silently accept it. This is the detector-side analogue of the BBSE conditioning diagnostic `κ(Ĉ_S)`; reported, not gated.
- **Factorization-entanglement diagnostic.** On the small labeled target slice `D_tar^lab` we additionally estimate a **direct joint weight** `ŵ_joint(x,y)` and report its divergence from the factorized weight `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` (mean-squared log-ratio and an estimated KL), as a measured check on the factorizable-shift premise (method note §3.3, §7.4). Because the joint weight is **not identifiable from unlabeled target alone** and is high-variance at `m_lab ≈ 50–200`, `ŵ_joint` is used **only** as this slice-level divergence diagnostic, never in the deployed pipeline. A large divergence on a pair means the factorization is failing there (prevalence and acquisition are entangled, `p(x∣y)` is moving); for such pairs the combined-shift claim is **downgraded** and the residual reported uncorrected, in the same spirit as the §3.1 covariate-vs-`Ẑ` downgrade rule. This makes the factorizable-shift premise **falsifiable per pair**, not assumed.
- **BBSE conditioning** `κ(Ĉ_S)` — label-shift error grows as `Ĉ_S` becomes ill-conditioned; worst-class error reported with it. We report it **as a function of the unlabeled-target size `m`**, alongside the **effective rank** of `Ĉ_S`, so the boundary where BBSE becomes inadmissible — and the MAPLS ablation (§4A) is doing the lifting — is visible rather than implicit.
- **BBSE consistency check** — `q̂_T` vs `Ĉ_S p̂_T`, to flag gross anti-causal / appearance-shift violation.
- **Per-pair discriminator AUROC** — the §3.2 shift-severity diagnostic, reported continuously.
- **Clip-induced abstention rate** — the mass routed by `ŵ_cov > w_max`, since clipping itself biases the weights and its effect is reported empirically.
- **`Ẑ(x)`-sensitivity of the headline weighted risk** — the headline Hájek weighted risk (with its interval) is recomputed under three settings of the posterior feeding `Ẑ`: **raw softmax**, **BCTS-recalibrated `σ̃`**, and a **temperature-perturbation band** around the recalibrated point. We report the **shift** in the headline number across these settings and **flag the rare / high-`w_lab` classes** where the `Ẑ(x)` denominator residual is largest (these dominate the combine in method note §3.3 and are where the correction is least trustworthy). This is a diagnostic on estimator sensitivity, not a correction.
- **Discriminator-shortcut audit** — the domain-discriminator AUROC `d`-AUROC is reported **before vs after** standard de-shortcutting of the inputs to `d`: cropping/masking burned-in text, image borders, and laterality markers, plus intensity-normalization. We inspect the **top-weighted cases** for acquisition artifacts driving the separation. For any pair whose `d`-AUROC **collapses after de-shortcutting**, the representativeness chip and the §3.2 regime tag for that pair are **caveated** — the measured covariate shift may be an acquisition shortcut rather than genuine appearance shift. (Cross-referenced from §3.2.)

### 5.5 Explanation faithfulness & stability (Decline-Attribution Record)

Additive measurements of the per-case **routing-faithful Decline-Attribution Record (DAR)** — the explanation of *why the model declined* (which reliability gates fired, with the signed margin to each threshold; method note §5.1, §6.3, §6.6, and explainability_framing §3). "Routing-faithful" here means **faithful to the routing decision, not to the diagnosis**: the checks below verify the explanation matches the predicate the pipeline actually evaluated to route the case, *not* that the routing or the class prediction is correct. These checks introduce **no new claim**: they audit an artifact the pipeline already computes, the thresholds `τ̂`, `t̂_ood`, `w_max` remain tuned operating points (not bounds), and the margins remain signed distances in score space, **never** per-case error probabilities (this respects §8 — in particular the no-per-case-guarantee item). All are measured on the held-out target test set, reported with sampling uncertainty, and stratified by the pre-registered subgroups (§5.3, §6.3). Checks (A)–(B) are executable invariants over already-computed gate scores; (C)–(E) reuse the existing augmentation family and calibration folds; (F) is planned/IRB-gated.

- **(A) Construction-faithfulness invariant** — for every declined case, assert in code that the DAR's reported firing set equals the set of gate predicates the routing rule actually evaluated false, and that each reported margin equals the recomputed score-minus-threshold (`m_u = τ̂ − u(x)`, `m_ood = t̂_ood − o(x)`, `m_w = w_max − ŵ_cov(x)`). Report the **discrepancy rate; target exactly 0** — a non-zero rate is a bug, not a modelling residual. No analogous exact check exists for the saliency baseline (E).
- **(B) Counterfactual / margin validity, conditioned on firing-set cardinality** — the by-construction counterfactual flip is stated *conditionally* to avoid overclaiming. For **single-gate declines** (exactly one violated gate): report (i) the flip-rate when the lone binding score is moved to its threshold (target high, by construction — sufficiency) and (ii) decision-invariance under non-binding perturbations within their passing region (target high — necessity of the named gate). For **multi-gate declines** (≥2 violated gates): the flip is defined only when **all** violated scores are moved to their thresholds simultaneously (report flip-rate, target high), with single-score moves reported as *not* expected to flip. The rate and composition of multi-gate declines is itself the informative number. This is the model-sensitivity logic of the saliency sanity checks (Adebayo et al. 2018) applied to the decline explanation.
- **(C) Input-perturbation stability vs distance-to-threshold** — apply clinically plausible, label-preserving perturbations (small rotations/translations, intensity/stain jitter, mild noise/compression within a pre-registered acquisition-realistic family) and report (i) the binding-gate flip rate and (ii) the distribution of `|Δ margin|` for each of `m_u`, `m_ood`, `m_w`, **stratified by distance-to-threshold**, so instability is attributed to genuinely borderline cases rather than to explanation noise. Stability is **measured, not guaranteed**; configurations where it is low are reported as such (method note §6.5), and the OOD/weight scores are themselves known to degrade under shift (method note §4.4, §7.3).
- **(D) Re-estimation / retraining stability** — re-estimate the thresholds across calibration folds, and retrain `f` (or vary seeds); report agreement of the binding-gate label and rank-correlation of margins for matched cases. This is the reproducibility analogue of the Arun et al. (2021) cross-retraining criterion.
- **(E) Saliency-baseline contrast under the same sanity checks** — for the same declined cases, generate a post-hoc saliency explanation over the input (an attribution map for the binding score or the classifier logit) and subject it to the *same* model-parameter-randomization and label-randomization sanity checks (Adebayo et al. 2018) and the Arun et al. (2021) repeatability/reproducibility metrics. The pre-registered expectation, grounded in that work, is that saliency may be insensitive to randomization (its documented failure mode) whereas (A) is exact and (B) is sensitive by design. This is reported honestly as a faithfulness/stability **asymmetry**, not a head-to-head accuracy race — the DAR and saliency explain different objects (a gatekeeper's choice vs a classifier's pixel attention). *(Concrete saliency baselines — Grad-CAM (Selvaraju et al. 2017) and SHAP (Lundberg & Lee 2017) — are verified in explainability_framing §3.4(E) / ledger; named here as the saliency-contrast baselines.)*
- **(F) Clinician comprehension (planned / IRB-gated)** — a small reader study testing whether clinicians correctly infer the firing gate and the appropriate action (act / defer / out of scope) from the rendered DAR. If not run for this paper, it is stated as planned and the clinician-facing claim is conceded to be argued from design (method note §6.4), not yet measured. *(The contestable-AI actionability anchor — Ploug & Holm 2020 — is verified in explainability_framing §3.4(F) / ledger.)*
- **(G) Accept-side trust-signal co-variation** — the counterpart to the decline-side checks (A)–(F), audited on the **answered** in-scope cases. Bin answered cases by **representativeness-chip level**, by **`n_eff` regime**, and by **shift-regime badge** (§3.2), and report the **measured accepted error per bin** with betting (Hájek) intervals. We pre-register the **monotone expectation** that heavily-up-weighted / low-`n_eff` / combined-regime bins show **higher** realized accepted error. To have power for the per-bin error this is computed on the **larger labeled target test split**, not only `D_tar^lab`. This validates the accept-side trust signals as **population co-variation** — that the chips/badges track realized error in aggregate — and asserts **no per-case guarantee**: a green chip on an individual answered case is not a per-patient correctness claim (§8).

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
- **What `δ` integrates over (fixed in advance).** We state explicitly what randomness each headline interval is **marginal over**: the **calibration / weight-estimation draw**, the **test sample**, or **both**. We also state whether resampling **re-estimates the covariate weight `ŵ_cov` and the label weight `ŵ_lab` per draw**. If, for cost, the weights are **held fixed** across resamples, the interval is labelled "**conditional on the estimated weights**" and a **separate weight-resampling band** (re-estimating `ŵ_cov`, `ŵ_lab` per draw on the full pipeline) is reported as a diagnostic beside it, so the weight-estimation uncertainty is not silently dropped from the headline.

### 6.3 The pre-registered subgroups

The subgroup axes for the §5.3 / §6.2-of-the-method-note audit are fixed before results: acquisition **site / scanner / hospital** (per benchmark), and the clinically salient strata available per corpus (e.g. sex, age band, self-reported race where ethically collected, and case-type strata such as biopsy-confirmed vs. screening). The subgroup audit is **reporting**, not a multiplicity-corrected guarantee — we do not wrap it in a new certificate and call it a fairness guarantee.

### 6.4 The minimum-coverage floor

A pre-registered **minimum-coverage floor** `coverage_min`: operating points whose realized answer rate falls below `coverage_min` are reported as **degenerate** (risk driven down only by answering almost nothing) rather than presented as a favorable risk number. The floor is fixed in advance so that "low risk" cannot be manufactured after the fact by collapsing coverage. It is a reporting guard on the risk–coverage trade-off, not a guarantee.

### 6.5 Fixed routing and weight controls

Also fixed before results: the weight clip / routing cap `w_max`; the base-rate constant `ĉ = n_S/n_T`; the label-weight floor/ceiling `[w_lab,min, w_lab,max]` (`w_lab,min > 0`); the primary OOD detector (Mahalanobis++ — the L2-normalized tied-covariance score of Müller et al. 2025, built from `mueller-mp/maha-norm`, keeping the in-distribution-only normalization and dropping only the OOD-coupled add-ons — with the plain Lee et al. 2018 tied-covariance score, energy, and kNN as ablations); the primary prevalence estimator (MLLS+BCTS, with BBSE as the baseline/diagnostic); the recalibrator `σ̃` (temperature/Platt, fit on a held-out source fold); and the stress axes of §6.6.

Also fixed before results, for the §5 reporting and display rules: the **severity cost matrix** — the FN/FP cost weights of the severity-weighted loss (§5.1); the **calibration display threshold `ECE_max`** above which the panel widens / greys / annotates the displayed confidence (§5.1); the **per-study positivity rule** — the fraction of accepted patches/tokens at which a slide / study is called positive in the per-study aggregation (§5.1); and the **minimum subgroup size `n_subgroup,min`** below which a subgroup (or subgroup × class) stratum is tagged "unpowered — pooled only" (§5.3). These are reporting / display constants — `ECE_max` triggers a display rule, never a gate on whether a case is answered, and none gates a guarantee.

### 6.6 Stress axes (pre-registered)

The deliberate stress sweeps, each on a fixed grid: small **unlabeled-target size `m`**; **source→target pair / direction** (each CAMELYON17 hospital pair; both CheXpert↔MIMIC directions); a **weight-clip `w_max` sweep** (over a fixed `w_max` grid, report the Hájek weighted risk + its interval + `n_eff` at each clip; cells where the headline number is **non-flat** across the grid go to the **failure-mode catalog**, since clip-sensitivity means the reported risk is an artifact of the clip choice); and, for the mixed benchmark, the **target-prevalence sweep** of §3.1. These define the failure-mode catalog the experiments stress; the cells are fixed before results.

**Backbone-robustness check (scoped, fixed before results).** Because every OOD/detector result is explicitly **backbone-conditional** (method note §4.1), the **headline selective-risk and the OOD-detector comparison** are run on **two architecturally distinct frozen backbones** — one ImageNet-pretrained CNN and one domain/medical foundation-model (ViT) encoder — so the headline cannot be a single-backbone artifact (our pathology foil TRUECAM likewise spans multiple foundation backbones). To stay within scope before the deadline, the **full ablation grid (MAPLS, DR, prevalence sweep) runs on the primary backbone only**; the second backbone covers the headline + OOD comparison. This scoping is **disclosed, not silent** — we report exactly which results are single- vs two-backbone.

### 6.7 Interval, comparison, and matched-operating-point protocol (pre-registered)

Recorded in the run config before any held-out target metric is computed:

- **Matched operating points and base-rate anchor.** The **matched total answer-rate** operating points used for the §5.1 head-to-heads, and the **frozen-`f` base-rate anchor** (raw target-vs-source error, no abstention, no weighting) printed beside each cell, are **fixed before results**.
- **Resamples and seeds.** The number of **calibration/test resamples** and the number of **seeds** each headline interval is **marginal over** is fixed in advance (and is the object §6.2 declares the interval conditional/unconditional over).
- **Weighted-path UCB is unit-tested as exercised.** The weighted-path WSR upper confidence bound is **unit-tested to confirm it is actually exercised** on the weighted estimator — i.e. there is **no silent range-bound fallback** quietly substituting a trivial `[0,1]`-type bound for the betting interval. The test is part of the pre-registered code.
- **Non-overlap win rule.** A pipeline-vs-baseline comparison is called a **win only if the pipeline and baseline betting intervals do not overlap**. This is a deliberately **conservative** test (interval non-overlap), **not** a p-value and not a certified superiority claim.
- **Below-`n_eff`-floor exclusion.** Cells below the pre-registered `n_eff` floor (already **flagged** as low-reliability, §5.4) are **excluded from headline aggregation** — this formalizes the flag into an explicit exclusion so low-`n_eff` cells cannot drift into a headline average.

### 6.8 Empirical interval-coverage check on synthetic data (pre-registered)

A pre-registered **empirical coverage check** of the WSR self-normalized (Hájek) interval, run on **synthetic data with known ground-truth weighted risk and known weights** (so the true weighted risk the interval is supposed to cover is available). Across the benchmarks' `n_eff` regimes we report **realized coverage vs the nominal `1 − δ`**. If the interval **under-covers** in a regime, it is **relabelled "nominal interval, coverage validated empirically"** wherever it is reported in that regime. This check is **synthetic** and is **not gated on data credentialing** — it tests the estimator's interval behavior, independently of any clinical-data access. It is a measurement of the interval's realized coverage, not a guarantee of it.

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
- **The Decline-Attribution Record explains the routing, not the diagnosis.** The routing-faithful DAR is faithful to *which reliability gate fired and by how much* (faithful to the routing, not the diagnosis); it explains neither why the classifier assigned its label, nor the disease, biology, or causal dynamics of the patient. Its margins are signed distances to tuned thresholds, not calibrated error probabilities, and the gated scores inherit the component-method limits of method note §7.

*The contribution this protocol pre-registers is a **measurement / auditability discipline** — a fixed-in-advance protocol for what we measure, report, and stratify under realistic shift — not a certificate. Every number it produces is measured on held-out target data and reported with its uncertainty and its reliability diagnostics.*
