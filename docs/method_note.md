# Method Note — Trustworthy Selective Prediction for Clinical Imaging Under Distribution Shift

*This note realizes the Methods section of the applied paper: a trustworthy, auditable selective-prediction pipeline for clinical image classification under realistic distribution shift. It is the **single source of truth for notation**, and it composes existing, citable methods — weighted conformal / RCPS, BBSE and MLLS+BCTS label-shift estimation, post-hoc OOD detection, and a clinician-facing audit layer. Each method's guarantee is cited as a property **of that method under its own stated assumptions**; we **claim no new guarantee** for the combined system. Where realistic clinical shift violates a method's assumptions, we **measure and report** the resulting degradation rather than claim it away.*

**Figure.** A one-page overview of the full pipeline — frozen classifier → shift-aware calibration → gated inference → clinician audit — is in [`figures/pipeline_overview.svg`](figures/pipeline_overview.svg).

**Conventions.** `monospace` denotes a symbol or variable, identical to the names used in the `conformal/` and `ood/` code. A hat (`x̂`) **always** denotes a quantity *estimated from finite data*; the un-hatted symbol is the population truth. `1[·]` = 1 if the bracket is true else 0 · `≤/≥` = at most/at least · `∝` = proportional to · `Σ` = sum · `∘` = elementwise product. Section 1 is the single source of truth for every symbol, weight, threshold, and budget reused below; later sections add no new notation.

---

## 1. Notation and problem setting

This is an *applied* pipeline assembled from existing, citable methods. Each method's guarantee is a property *of that method* under *its* stated assumptions; we invoke none of them as a guarantee of the combined system. Where a method's assumptions are violated by realistic clinical shift, we **measure and report** the resulting degradation rather than claim it away.

### 1.1 Objective and what we report

We classify clinical images with a frozen model, abstain on uncertain cases, route suspected out-of-distribution scans to a human, and correct calibration for covariate and label (prevalence) shift between a labeled **source** and a deployment **target**. The quantity we care about is the answered-case target risk

```
R_T^accept  :=  E_{P_T}[ ℓ(Y, ŷ(X)) | A(X) ]
```

over the cases the pipeline answers and judges in-scope (the event `A(x)`, defined in §1.5.1). We **do not certify** `R_T^accept`. We calibrate operating thresholds on source data, apply a weighted correction, and then **measure** realized selective risk, coverage, and OOD-leakage on held-out target data, reporting the degradation. This is a deliberate stance, not an omission: distribution-free OOD detection is *not* learnable in the unrestricted setting (Fang et al., NeurIPS 2022), and finite-sample distribution-free *conditional* coverage under covariate shift is not attainable — once the density-ratio weight is estimated, only asymptotic / doubly-robust validity is available (Yang, Kuchibhotla & Tchetgen Tchetgen, JRSS-B 2024). These results are the reason we report measured degradation rather than promise a distribution-free guarantee. Prior work (e.g. TRUECAM) restores validity by detecting and removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and does not claim a maintained distribution-free guarantee in deployment; we instead measure and report the residual leakage.

### 1.2 Spaces, distributions, and the shift taxonomy

- `𝒳` — input space (an image). `𝒴 = {1,…,K}` — label space, `K` classes. `Δ^{K−1}` — the probability simplex over `𝒴`.
- `P_S`, `P_T` — the **s**ource and **t**arget joint distributions over `𝒳 × 𝒴`. The pipeline is calibrated on `P_S` and deployed on `P_T`.
- `p_S(x), p_T(x)` — input (covariate) marginals · `p_S(y), p_T(y)` — label (prevalence) marginals · `p_S(y∣x), p_T(y∣x)` — posteriors · `p_S(x∣y), p_T(x∣y)` — class-conditionals.
- **Shift taxonomy** — which factor moves decides which weight is correct (§1.5):
  - *Covariate shift* — `p(x)` changes, `p(y∣x)` invariant ⇒ the correct weight depends on `x` only.
  - *Label / prevalence shift* — `p(y)` changes, `p(x∣y)` invariant ⇒ the correct weight depends on `y` only.
  - *Both at once* — neither single weight suffices, and the combined weight is **not** the product of the two (§1.5, the double-counting correction). Realistic clinical shift (CAMELYON17 scanner/center change, CheXpert↔MIMIC-CXR site change) typically mixes both, plus appearance changes that move `p(x∣y)` and so violate the pure-shift premises of the methods below — which is exactly why we measure residual degradation.

### 1.3 Frozen base model and its outputs

- `f` — the **frozen** base model, `f : 𝒳 → ℝ^K` (logits). Trained on the source *train* split; parameters fixed before any calibration. The selective/conformal layer is base-model-agnostic; `f` only buys efficiency.
- `σ(f(x)) ∈ Δ^{K−1}` — softmax probabilities · `ŷ(x) = argmax_k f(x)_k` — the predicted label.
- `φ(x) ∈ ℝ^p` — penultimate **feature embedding** (`p` = feature dim), optionally spectral-normalized. The OOD score `o(·)` and the domain discriminator `d(·)` both act on `φ(x)`, never on raw pixels.
- `u(x) ∈ ℝ` — scalar **uncertainty** (high = less certain), e.g. softmax-response `u(x) = 1 − max_k σ(f(x))_k` or energy `u(x) = −log Σ_k exp f(x)_k`.

### 1.4 Loss, selection, and selective risk

- `ℓ(y, ŷ) ∈ [0,1]` — bounded **loss**; default 0–1 loss `ℓ = 1[y ≠ ŷ]`. Boundedness in `[0,1]` is what the RCPS / betting concentration bound (§1.6) requires. Although the machinery admits **any** bounded `[0,1]` loss, results are reported **both** as the 0–1 selective risk **and** severity-stratified — false negatives vs. false positives on the positive class reported separately, plus a severity-weighted loss. The severity weights are a **reporting choice**, not a certified clinical cost.
- `τ` — **selection threshold** on `u`. `g(x) = 1[u(x) ≤ τ]` — the **selection gate**: `g=1` ⇒ answer, `g=0` ⇒ abstain (defer).
- `coverage = P(g(X)=1)` — fraction answered. **Selective (accepted) risk** on a distribution `P`:
  ```
  R^accept_P  :=  E_P[ ℓ(Y, ŷ(X)) | g(X)=1 ]
  ```
  Selection makes the answered subset non-exchangeable with the full calibration set, so the controlling threshold is calibrated *on the accepted region* (§2).

### 1.5 Distribution-shift weights

- **Master importance-weighting identity.** The Radon–Nikodym derivative
  ```
  w(x,y)  :=  dP_T/dP_S (x,y)  =  p_T(x,y) / p_S(x,y)
  ```
  transports any expectation from source to target: `E_{P_T}[ h(X,Y) ] = E_{P_S}[ w(X,Y) · h(X,Y) ]`. `w(x,y)` is the **combined weight**.

- `w_cov(x) := p_T(x) / p_S(x)` — **covariate density ratio**; equals the full `w(x,y)` *iff* the shift is pure covariate. Estimated from a domain discriminator `d(x) = P(domain = T ∣ x)` via the conditional odds ratio:
  ```
  ŵ_cov(x) = clip( (d(x) / (1 − d(x))) · c , 0 , w_max ),     c = π_S / π_T   (domain base-rate correction)
  ```
  where `π_S, π_T` are the source/target proportions in the discriminator's training pool (`c ≈ n_S/n_T`). This is the discriminator density-ratio estimator of Tibshirani, Foygel Barber, Candès & Ramdas (NeurIPS 2019, their eq. 12), with the Remark-3 base-rate correction (the constant `c` cancels in their normalized weights, so it matters only through clipping). Their Corollary 1 establishes finite-sample `1−α` *marginal* coverage for weighted conformal prediction **under pure covariate shift with a known (oracle) likelihood ratio**; we invoke this only as a property of *their* method under *its* assumptions. Because our `ŵ_cov` is *estimated* (no finite-sample guarantee carries over — the paper itself only shows good *average* coverage empirically with estimated weights) and clinical shift is not pure covariate shift, we make no coverage claim for the deployed weight and instead measure residual degradation. The discriminator is unreliable in the far tail, so we clip at `w_max` and route the overflow to abstain (§2); clipping itself biases the weights, so its effect is reported empirically.

- `w_lab(y) := p_T(y) / p_S(y)` — **label / prevalence ratio**; equals the full `w(x,y)` *iff* the shift is pure label. Estimated by **BBSE** (Lipton, Wang & Smola, ICML 2018) from the source confusion matrix and target predictions:
  ```
  ŵ_lab = Ĉ_S^{−1} q̂_T,      then   p̂_T(y) = ŵ_lab(y) · p_S(y)
  ```
  - `C_S` — **source confusion matrix**, `C_S[i,k] = P_S(ŷ = i, y = k)`. Invertibility / good conditioning of `Ĉ_S` is the BBSE **identifiability** condition; estimation error grows as `Ĉ_S` becomes ill-conditioned.
  - `q̂_T` — distribution of `f`'s **predicted** labels on the unlabeled target sample, `q̂_T[i] = P_T(ŷ = i)`.
  - BBSE is a *consistent* estimator of `w_lab` **under label shift** (`p(x∣y)` invariant, the anti-causal regime) given an invertible `Ĉ_S` — consistency, not a finite-sample certificate, and it does not require the base classifier to be calibrated. As the primary prevalence estimator we use MLLS with bias-corrected temperature scaling (Alexandari, Kundaje & Shrikumar, ICML 2020), which the authors find empirically hard to beat among label-shift estimators; because MLLS is consistent only when `f`'s outputs are calibrated, BCTS supplies that calibration. Garg, Wu, Balakrishnan & Lipton (NeurIPS 2020) show MLLS and BBSE share the same confusion-matrix-invertibility identifiability condition and that MLLS additionally requires the classifier to be calibrated. BBSE is retained as the baseline and as the consistency diagnostic of §3.

- `Z(x)` — **per-`x` double-count corrector** used to combine the two weights *without* multiplying them:
  ```
  Z(x) := Σ_{y'} w_lab(y') · p_S(y' ∣ x) = E_{p_S(·∣x)}[ w_lab ]      (estimated  Ẑ(x) = Σ_{y'} ŵ_lab(y') · σ̃(f(x))_{y'})
  ```
  with `σ̃` a **temperature/Platt-recalibrated** softmax (raw softmax is a biased denominator). The combined weight is then
  ```
  w(x,y) = w_lab(y) · w_cov(x) / Z(x)      (NOT  w_cov(x) · w_lab(y))
  ```
  This is the standard label-plus-covariate combination (cf. Tasche); `Z(x)` removes the double-count the naïve product introduces (worked example in §3.3). It is an estimation correction, not a new identification result: under combined shift the combined weight is not identifiable from unlabeled target data alone, so we use the small labeled target slice `D_tar^lab` to **measure** the residual and fit a simple scalar correction (§1.7), not to certify the combined case. ⚠ **`w(x,y) ≠ w_cov(x) · w_lab(y)`** in general.

### 1.5.1 OOD score, routing, and the in-scope event

- `o(x) ∈ ℝ` — **OOD score**, a feature-space distance on `φ(x)` (Mahalanobis primary, kNN ablation); high = more out-of-distribution. We use these as established post-hoc scoring rules to *route* suspected OOD scans to a human; none carries a distribution-free guarantee (Fang et al. 2022).
- `t_ood` — **OOD threshold**: `o(x) > t_ood` ⇒ route to abstain. Set on a held-out exposure set `O` to spend an OOD-leakage budget `α_ood` (§1.6), whose realized value is then measured.
- `w_max` — clip cap / **routing cap** on the covariate weight: where `ŵ_cov(x) > w_max` the density ratio is untrustworthy ⇒ route to abstain. This removes the tail mass that would otherwise destabilize the weighted estimate.
- **In-scope / answered event:**
  ```
  A(x)  :=  { g(x) = 1 }  ∩  { o(x) ≤ t_ood }  ∩  { ŵ_cov(x) ≤ w_max }.
  ```

### 1.6 Operating thresholds and budgets

These are **tuned** on calibration data, not certified. Their realized performance is measured on held-out target data.

- `λ` — **risk-control threshold** indexing a nested family of decision rules; `R(λ)` is the risk as a function of `λ`. The operating value `λ̂` is selected on the calibration set by the RCPS rule (Bates et al. 2021); RCPS gives finite-sample `(α,δ)` control *when calibration and test are exchangeable* — broken under shift — so we treat `λ̂` as an operating threshold whose realized risk is then measured on target.
- `τ` — selection threshold (§1.4); `t_ood` — OOD threshold (§1.5.1).
- `α_acc` — operating budget for accepted in-distribution error · `α_ood` — operating budget for far-OOD leakage missed by the detector, **measured on `O`**. These are two *separately measured* operating budgets; we do **not** claim they compose into a single certified `α = α_acc + α_ood` union-bound.
- `δ` — confidence level of the reported interval (an upper confidence bound from a betting/concentration method; we use the variance-adaptive hedged-capital bound presented for non-binary losses in RCPS — Bates et al. 2021, §3.1.3, the Waudby-Smith–Ramdas bound — analyzed in Waudby-Smith & Ramdas 2024). It is the confidence of the *reported CI*, not a PAC promise about the deployed pipeline. The reported interval integrates over the sampling randomness of the labeled evaluation draw (the target slice / held-out target points). When the estimated weights `ŵ` are held **fixed** across resamples, the interval is **conditional on the estimated weights**; the weight-estimation error is then **not** in the interval and is instead carried by the `n_eff` and `κ(Ĉ_S)` reliability diagnostics (whose prereg sweep is `docs/preregistration.md` §6.2).
- Jointly-tuned knobs `(τ, λ, t_ood)` are searched on one calibration set; the multiple-looks correction uses LTT (Angelopoulos et al.). We apply these bounds as published and report the resulting numbers; we claim no new validity guarantee.

### 1.7 Calibration data, folds, and estimate convention

- `D_cal = {(X_i, Y_i)}_{i=1}^n`  iid `∼ P_S` — **labeled source calibration set**; `n = |D_cal|`.
- `D_tar = {X̃_j}_{j=1}^m`  iid `∼ p_T(·)` — **unlabeled target sample** (covariates only) for estimating `ŵ_cov`, running BBSE/MLLS, and setting `t_ood`; `m` = target sample size. Small `m` is a primary stress axis.
- `D_tar^lab = {(X̃_k, Ỹ_k)}_{k=1}^{m_lab}`  iid `∼ P_T` — **small labeled target slice** (`m_lab ≈ 50–200`), disjoint from `D_tar`. Its role is to **empirically measure residual degradation** of the shift correction and to **fit a simple scalar correction**; it does *not* identify any nuisance parameter or certify the combined-shift case.
- `D_disc` — discriminator-fitting fold · `D_bbse^src` — labeled source fold for `Ĉ_S` (disjoint from `D_cal`) · `O` — OOD-exposure set (for setting `t_ood` and measuring leakage).
- `R̂_w(τ,λ) = Σ (ŵ · ℓ) / Σ ŵ` — **self-normalized (Hájek) weighted accepted-in-scope risk estimator** (over accepted, in-scope points), used so the unknown `E[w]=1` cancels; its `(τ,λ)` dependence is via the selection gate and the decision rule. This is the only rendering used everywhere (no `R_w-hat` variant).
- `n_eff = (Σ w_i)² / Σ w_i²` — **Kish effective sample size**, *reported as a reliability diagnostic* to flag low-effective-sample regimes; it is **not** a certification gate. Importance weights inflate variance and deflate `n_eff`, which is why §1.6 uses a variance-adaptive bound.
- `w_lab,min / w_lab,max` — floor/ceiling applied to `ŵ_lab` after simplex projection (`w_lab,min > 0`), to keep an ill-conditioned `Ĉ_S` from blowing up the combined weight.
- **Disjointness (leakage discipline).** The folds for fitting `d`, estimating `Ĉ_S` and `q̂_T`, fitting `o`, calibrating `(τ, λ, t_ood)`, the exposure set `O`, the labeled slice `D_tar^lab`, and the final test set are mutually **disjoint** — asserted in code. In particular `Ĉ_S` is fit on a *source* fold disjoint from `D_cal`. Disjointness is asserted at the **group id** (slide / patient), not the patch / token, so that patches from one slide or reads from one patient cannot straddle two folds; the assertion emits a printed **set-intersection = 0** artifact over the group ids of every fold pair.
- **Hat = estimate.** `ŵ_cov, ŵ_lab, p̂_T, q̂_T, Ĉ_S, λ̂, τ̂, t̂_ood, Ẑ` are finite-sample estimates of their un-hatted population counterparts; their residual errors are *measured* on `D_tar^lab` and via the diagnostics of §3, not bounded by a certificate.

### 1.8 Quick-reference table

| Symbol | Type | Meaning |
|---|---|---|
| `P_S, P_T` | distribution | source / target joint over `𝒳×𝒴` |
| `f, σ(f(x)), ŷ(x)` | model | frozen logits · softmax · predicted label |
| `φ(x)` | vector `∈ ℝ^p` | penultimate features (input to `o`, `d`) |
| `u(x)` | scalar | uncertainty (high = unsure) |
| `ℓ(y,ŷ) ∈ [0,1]` | scalar | bounded loss (default `1[y≠ŷ]`) |
| `g(x)=1[u(x)≤τ]`, `τ` | gate / threshold | selection: 1 = answer, 0 = abstain |
| `R^accept_P`, `coverage` | scalar | selective risk `E_P[ℓ∣g=1]` · fraction answered |
| `R_T^accept` | scalar | answered-case target risk (**measured**, not certified) |
| `w(x,y)=dP_T/dP_S` | weight | combined weight (**≠** product) |
| `w_cov(x)=p_T/p_S(x)`, `d(x)`, `c` | weight / prob. / const | covariate density ratio (discriminator) · domain posterior · base-rate const `π_S/π_T` |
| `w_lab(y)=p_T/p_S(y)`, `C_S`, `q̂_T` | weight / matrix / vector | label ratio (BBSE / MLLS+BCTS) · source confusion · target predicted-label dist. |
| `Z(x)`, `σ̃` | scalar / vector | per-`x` double-count corrector · recalibrated softmax |
| `o(x)`, `t_ood`, `w_max` | scalar / thresholds | OOD score · OOD cutoff · weight-clip / routing cap |
| `A(x)` | event | answered & in-scope: `g=1 ∧ o≤t_ood ∧ ŵ_cov≤w_max` |
| `λ, R(λ)` | thresh. / fn | operating risk-control threshold · risk |
| `α_acc, α_ood` | budgets | separately-measured accepted-error · OOD-leakage operating budgets |
| `δ` | scalar | confidence level of the reported interval |
| `n, m, m_lab` | counts | source-cal size · unlabeled target-sample size · labeled-slice size |
| `D_cal, D_tar, D_tar^lab` | datasets | labeled source cal · unlabeled target · small labeled target slice |
| `D_disc, D_bbse^src, O` | datasets | discriminator fold · BBSE source fold · OOD-exposure set |
| `R̂_w(τ,λ)`, `n_eff` | scalar | Hájek self-normalized weighted accepted-in-scope risk estimator (`(τ,λ)`-dependent) · Kish effective sample size (diagnostic) |
| `w_lab,min / w_lab,max` | bounds | floor / ceiling on `ŵ_lab` after simplex projection |

---

## 2. Selective prediction and risk-controlled abstention

Abstention is the trust mechanism of this pipeline. Rather than force a label on every scan, the system answers only where it has evidence it can stand behind, and *defers* the rest to a human reader. This section specifies the selection gate, the bounded loss it controls, and how a single threshold is tuned so that the risk *among accepted cases* is held at a target level on exchangeable calibration data. We state the controlling guarantee strictly as a property of the cited method (RCPS; conformal risk control) under its own exchangeability assumption, and we are explicit that this assumption is broken by the distribution shift the rest of the paper addresses — under shift we *measure and report* the realized accepted-case risk on held-out target data rather than re-asserting the certificate. All symbols are the single-source definitions of §1; this section adds none.

### 2.1 The selection gate

Let `f : 𝒳 → ℝ^K` be the frozen base classifier with softmax `σ(f(x)) ∈ Δ^{K−1}` and prediction `ŷ(x) = argmax_k f(x)_k` (§1.3). Attach to each input a scalar **uncertainty** score `u(x) ∈ ℝ` (higher = less certain) — by default the softmax-response `u(x) = 1 − max_k σ(f(x))_k`, with an energy score `u(x) = −log Σ_k exp f(x)_k` as an ablation.

A **selection threshold** `τ` on `u` induces the **selection gate**

```
g(x) = 1[ u(x) ≤ τ ]          g = 1 ⇒ answer,   g = 0 ⇒ abstain / defer.
```

We deliberately frame `g = 0` as *deferral to a clinician*, not silent failure: the accepted set is the region where the model takes responsibility, and the abstained set is routed to a human. The fraction the model answers is the

```
coverage = P( g(X) = 1 ),
```

and the quantity we want to control is the **selective (accepted) risk** on a distribution `P`,

```
R^accept_P  :=  E_P[ ℓ(Y, ŷ(X)) | g(X) = 1 ],
```

where `ℓ(y, ŷ) ∈ [0,1]` is a bounded loss (default 0–1 loss `ℓ = 1[y ≠ ŷ]`). Boundedness in `[0,1]` is not incidental: it is exactly the precondition the concentration bound below requires. There is an intrinsic trade-off — lowering `τ` answers fewer, easier cases and drives `R^accept` down at the cost of coverage; raising `τ` answers more at higher risk. The role of the calibration step is to pick the operating point on this risk–coverage curve that meets a clinician-specified risk target.

This selective-classification / learning-to-abstain setup is the one introduced by Geifman and El-Yaniv (2017, 2019): the prediction head `f` paired with a selection head `g`, with selective risk and coverage defined exactly as above. We use SelectiveNet (Geifman and El-Yaniv, 2019) as the architecture/objective when an end-to-end abstention head is trained, but we attach no statistical guarantee to it. SelectiveNet is a *training objective* for the risk–coverage trade-off with a soft coverage constraint; the authors themselves note the target coverage can be violated on the test set, and the calibrated test-time selective-risk bound is supplied by the *separate* SGR (Selection with Guaranteed Risk) procedure of Geifman and El-Yaniv (2017), not by the SelectiveNet architecture itself. All distribution-free risk control in this pipeline therefore comes from the calibration layer of §2.2, not from the abstention architecture.

### 2.2 Choosing the threshold so accepted-case risk is controlled

We tune `τ` (and, in later sections, the companion thresholds that handle shift and OOD) using **risk-controlling prediction sets (RCPS)** — equivalently, conformal risk control — applied to the selective-risk functional above.

Index a *nested* family of decision rules by a scalar threshold `λ`, so that the risk `R(λ)` is monotone non-increasing in `λ` and the rules grow with `λ` (here `λ` enters through `τ`; the nested structure is what RCPS requires). On a labeled calibration set `D_cal = {(X_i, Y_i)}_{i=1}^n` drawn i.i.d. from the calibration distribution, form an upper confidence bound `UCB_δ(λ)` on `R(λ)` and select the operating threshold by the RCPS selection rule

```
λ̂ = inf{ λ : UCB_δ(λ′) < α_acc   for all λ′ ≥ λ }.
```

This `inf`-rule is the *mechanism* by which RCPS chooses an operating point on calibration data; on real target data we *measure* the realized risk of `λ̂` rather than re-assert its calibration-time guarantee (see below).

For the upper confidence bound we use a betting / hedged-capital concentration method that is variance-adaptive rather than range-only (Waudby-Smith and Ramdas, 2024); this is the same betting bound recommended as the UCB in RCPS (Bates et al., 2021, §3.1.3). The practical reason to prefer it here is that the importance weights introduced in later sections inflate the variance of the risk estimate and deflate the effective sample size, so a bound that adapts to the realized variance is materially tighter than a Hoeffding–Bentkus range bound.

**The guarantee, named with its method and its assumption.** When the calibration set `D_cal` and the deployment/test points are **exchangeable** (i.i.d. from the *same* distribution), the loss is bounded, and the risk is monotone over the nested family, RCPS gives the finite-sample, distribution-free PAC guarantee

```
P( R(λ̂) ≤ α_acc ) ≥ 1 − δ        (Bates et al., 2021, Definition 1 + Theorem 1),
```

i.e. with probability at least `1 − δ` over the calibration draw, the true risk of the selected rule is at most the target `α_acc`. This is a property *of RCPS under exchangeability between calibration and test* — Definition 1 fixes what an `(α_acc, δ)`-RCPS *is*, and Theorem 1 shows the UCB rule above *produces* one. Here `α_acc` is the clinician-specified accepted-error budget and `δ` is the confidence level of the reported bound — read it as "the certificate holds with probability `1 − δ` over the calibration draw," not as a promise about any individual case. A less conservative alternative, **conformal risk control** (Angelopoulos et al., 2024), controls the *expectation* `E[ℓ_{n+1}(λ̂)] ≤ α_acc` of a monotone bounded exchangeable loss (tight to `O(1/n)`) but carries no `δ` / tail control; we report it as a comparison row, since a clinician operating-point promise is more naturally a high-probability tail (RCPS) than an in-expectation statement.

**Where the assumption breaks — what we do instead.** The RCPS / CRC certificate is a statement *between calibration and test from one distribution*. Its own scope is narrow: Bates et al. (2021, Remark 2) permit only the *initial model* `f` to have been trained on a different distribution — the calibration set and the deployment set must still share a distribution. Clinical deployment violates exactly this: target scans differ from the source calibration data by covariate (scanner/stain/site) and label/prevalence shift, so the accepted subset under `P_T` is not exchangeable with `D_cal`. We do not re-assert the `(α_acc, δ)` certificate in that regime. Instead:

- we calibrate `λ̂` on source data, applying the weighted correction developed in §3 to the accepted region, and
- we **measure** the realized in-scope answered target risk `R_T^accept := E_{P_T}[ ℓ(Y, ŷ(X)) | A(X) ]` on held-out target data — where `A(X)` is the answered & in-scope event of §1.5.1 — reporting any degradation from the source-calibrated target, rather than claiming it is bounded.

Two results are the principled reason we report rather than promise here, and they recur throughout the paper: finite-sample distribution-free conditional coverage under covariate shift is not attainable — once the weight is estimated, only asymptotic / doubly-robust validity is available (Yang, Kuchibhotla and Tchetgen Tchetgen, 2024) — and out-of-distribution detection is not distribution-free learnable in the unrestricted setting (Fang et al., 2022). Because the corrections of §3 and §4 rest on *estimated* weights and an *imperfect* detector — neither of which the exchangeability theorem covers — the honest object is a measured residual, not a recovered certificate.

**Objective and what we report.** To summarize the contract of this section: we *want* the answered-case target risk `R_T^accept` to be low under shift, and we select the operating threshold `λ̂` to that end on calibration data; we do *not* certify `R_T^accept ≤ α_acc` in deployment. We invoke the RCPS guarantee only as a property of that method under exchangeability, and on real target data we report the realized selective risk, the coverage, and the gap between them. The accepted budget `α_acc` and the OOD-leakage budget `α_ood` of §4 are named, *separately measured* operating budgets — not a proven additive union-bound split. This is also where our framing departs from prior work such as TRUECAM, which restores validity by detecting and removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and does not claim a maintained distribution-free guarantee in deployment: rather than leave that gap unquantified, we measure and report the residual leakage on a stated exposure set.

---

## 3. Shift-aware calibration (covariate and label shift)

Split-conformal prediction and risk-controlling prediction sets need calibration and test points to be **exchangeable**. Clinical deployment breaks this: the model is calibrated on a source distribution `P_S` (one set of hospitals, scanners, prevalence) and deployed on a target `P_T` (different sites, acquisition, case mix). This section names the exact way each shift breaks exchangeability and describes the **practical** corrections we apply — weighted conformal for covariate shift, BBSE/MLLS+BCTS for label shift, and a principled combine step. Throughout, **each correction's guarantee is stated as a property of the method that owns it, under that method's own assumptions**; where clinical shift violates those assumptions we **measure and report** the residual degradation rather than claim a guarantee. All symbols, weights, and budgets are the single-source definitions of §1.

### 3.1 Objective and what we report

We **want** the answered-case target risk to be low under shift:
```
R_T^accept := E_{P_T}[ ℓ(Y,ŷ(X)) | A(X) ]   small.
```
We do **not** certify it. Instead we calibrate on source, apply the weighted correction below, and **measure** realized selective risk, coverage, and OOD-leakage on a held-out target set, **reporting the degradation** (overall and by subgroup) for the clinician's audit.

This is a deliberate, principled stance, not a concession of weakness — two impossibility results explain why, for the two quantities below, the honest move is to **measure** rather than **promise**:

- **OOD detection is not distribution-free learnable** in the unrestricted setting (Fang et al., NeurIPS 2022). No post-hoc detector (Mahalanobis, energy, kNN) can carry a distribution-free guarantee on far-/near-OOD leakage, so we audit leakage empirically on a stated exposure set `O` rather than promise an OOD certificate (§4).
- **Finite-sample distribution-free conditional coverage under covariate shift is not attainable**; once the density-ratio weight is estimated, only asymptotic / doubly-robust validity is available (Yang, Kuchibhotla & Tchetgen Tchetgen, JRSS-B 2024). The covariate correction is therefore **reported, not guaranteed**: we measure residual coverage/risk degradation on `D_tar^lab` instead of attaching a finite-sample certificate to `ŵ_cov`.

**Positioning (relative to TRUECAM).** Prior work that restores validity by detecting and removing OOD inputs does not claim a maintained distribution-free guarantee in deployment. Our contribution is not a stronger guarantee but a **measurement discipline**: we budget the detector's miss-rate against a stated, swappable exposure model `O` and **report the residual leakage** as one auditable number, instead of leaving the OOD gap unquantified.

### 3.2 Four exchangeability-breakers and their practical neutralizers

Each mechanism below breaks the exchangeability that split-conformal / RCPS need between calibration and a fresh test point. Each has a **practical neutralizer**; what each neutralizer leaves behind is a **measured diagnostic / reported residual**, not a certified quantity.

| Breaker | Why it breaks exchangeability | Practical neutralizer | What we measure |
|---|---|---|---|
| **Selection** `g=1[u≤τ]` answers only low-uncertainty cases | A statement about the *marginal* risk does **not** bound the *conditional* accepted risk `P(·∣g=1)` | Calibrate **on the accepted region**: form the weighted risk and tune `(τ,λ)` using only cal points with `g=1` and in-scope `A=1`, wrapped in LTT for the multiple looks | Realized accepted risk + coverage on held-out target; whether enough effective sample remains (`n_eff`, §3.5) |
| **Covariate shift** `p(x)` moves, `p(y∣x)` fixed | Cal `∼P_S`, test `∼P_T`: unweighted risk is biased; exchangeability becomes *weighted* exchangeability (Tibshirani et al. 2019) | Reweight every cal contribution by clipped `ŵ_cov`; clip + route the far tail (`ŵ_cov>w_max`) to abstain | Residual coverage degradation on `D_tar^lab`; discriminator AUROC per hospital pair as an overlap diagnostic |
| **Label / prevalence shift** `p(y)` moves, `p(x∣y)` fixed | Covariate-only reweighting cannot capture a *label-marginal* change; naïvely multiplying `w_cov·w_lab` **double-counts** | BBSE / MLLS+BCTS `ŵ_lab`, **combined via `Z(x)`** — not the product (§3.3) | BBSE conditioning diagnostic `κ(Ĉ_S)`; the `q̂_T` vs `Ĉ_S p̂_T` anti-causal sanity check; prevalence-correction error on `D_tar^lab` |
| **Far-OOD + imperfect detector** `p_S≈0`, huge density ratio, or a leaky detector | On target-only support `dP_T/dP_S` is undefined/∞; one huge weight collapses the effective sample; a leaky detector passes far-OOD into the answered set | **Route** to abstain on `ŵ_cov(x)>w_max` OR `o(x)>t_ood`; set `t_ood` on disjoint `O` to spend the OOD-leakage budget `α_ood` (§4) | Audited far-OOD leakage rate on `O`; per-benchmark degradation of the detector |

Two routing mechanisms are **decoupled**: `w_max` is a variance/boundedness control on the weighted estimator (its routed mass costs only coverage), while `t_ood` is the OOD guard whose miss-rate is measured against `α_ood`. The routing rule is `ŵ_cov(x) > w_max`. Clipping itself biases the weights, so its effect is reported empirically rather than treated as exact. The full routing logic is in §4.

### 3.3 Combining the two weights (do not multiply — divide by `Z`)

Under both shifts, the combined weight is the label ratio times a **corrected** covariate factor, **not** the product. The correction follows directly from the master importance-weighting identity of §1.5 (apply the law of total probability to `dP_T/dP_S`); it is a standard estimation step (cf. Tasche on factorizable joint shift), not a new theorem.

- **The combine identity.** With the per-`x` normalizer `Z(x) := Σ_{y'} w_lab(y') · p_S(y'∣x) = E_{p_S(·∣x)}[ w_lab ]` (§1.5),
  ```
  w(x,y) = w_lab(y) · w_cov(x) / Z(x).
  ```
  `Z(x)` is exactly the double-count corrector the naïve product omits. It collapses correctly to `w_lab(y)` under pure label shift (`w_cov = Z`) and to `w_cov(x)` under pure covariate shift (`w_lab ≡ 1`, `Z ≡ 1`).
- **Why the product is wrong (worked example).** Even under *pure* label shift (truth `w = w_lab(y)`, `x`-independent), the prevalence change moves `p(x)`, so `w_cov(x) = Z(x)` is non-constant and the naïve product inflates by exactly `Z(x)`. Take a 2-class problem, `p_S(y) = (.5, .5) → p_T(y) = (.9, .1)`, with invariant class-conditionals. Then `w_lab = (1.8, 0.2)`. At an `x` where class 1 dominates, `p_S(1∣x) ≈ 1`, so `Z(x) ≈ 1.8`: the naïve product gives `w̃(x,1) ≈ 1.8 · 1.8 = 3.24`, whereas the truth is `1.8`. Dividing by `Z(x)` removes the inflation exactly.

**Why we adopt the factorizable-shift modeling premise.** Combining the two corrections presumes the two shift mechanisms can be treated as separable. We adopt a **factorizable (independent-mechanism) shift** premise as a **modeling choice** for dense imaging shift, and do **not** adopt Sparse Joint Shift (Chen, Zaharia & Zou 2022) because its sparsity premise — only a few features shift — does not fit raw-image covariate shift, which is dense. The premise is **not identifiable from unlabeled target data alone** (which is why the labeled slice exists); we therefore fit a **simple scalar correction** on `D_tar^lab` and **report the residual gap empirically** (§3.4). There is no exactness claim and no certified combined weight.

### 3.4 Calibration and inference algorithm

```
CALIBRATION   (mutually DISJOINT folds: D_disc · D_bbse^src (labeled source) · D_tar (unlabeled target)
               · D_tar^lab (small labeled target slice) · O (OOD-exposure) · D_cal (labeled source,
               accepted-region) · held-out test)

  1. DISCRIMINATOR.   Fit d(x) on D_disc (source-vs-target).
                      ŵ_cov(x) = clip( (d/(1−d))·ĉ , 0 , w_max ),   ĉ = n_S/n_T.
                      Route to abstain where ŵ_cov(x) > w_max.
                      Report discriminator AUROC per hospital pair as an overlap diagnostic.

  2. LABEL SHIFT.     Ĉ_S on D_bbse^src (labeled source, disjoint from D_cal);  q̂_T on D_tar.
                      Primary: MLLS+BCTS prevalence estimate (recalibrate f with BCTS first).
                      Baseline/diagnostic: ŵ_lab = Ĉ_S⁻¹ q̂_T.
                      PROJECT p̂_T onto the simplex and FLOOR ŵ_lab ∈ [w_lab,min , w_lab,max], w_lab,min>0
                      (an engineering fix so an ill-conditioned Ĉ_S cannot make Z explode or go negative).
                      DIAGNOSTICS: report κ(Ĉ_S) (conditioning); compare q̂_T vs Ĉ_S p̂_T to flag gross
                      anti-causal / appearance-shift violation.

  3. COMBINE (not multiply).  Ẑ(x) = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}, with σ̃ a TEMPERATURE/Platt-recalibrated
                      softmax fit on a held-out source fold (raw softmax is a biased plug-in for p_S(y∣x)).
                      ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x).
                      OPTIONAL: fit a simple scalar recalibration on D_tar^lab and REPORT the measured
                      residual gap (plug-in risk under ŵ vs empirical risk on the slice). If D_tar^lab is
                      unavailable, skip the scalar fit and report combined-shift results as uncorrected.

  4. OOD.            Fit o(x) on (spectral-norm) features.  On O, set t_ood to the cutoff whose measured
                      far-OOD leakage rate ≤ α_ood (monotone scan over the exposure set).  [Detail in §4.]

  5. RISK GRID.      On accepted+in-scope cal pts {g=1, o≤t_ood, ŵ_cov≤w_max}, form the self-normalized
                      (Hájek) weighted risk
                          R̂_w(τ,λ) = Σ ŵ·ℓ(λ) / Σ ŵ.
                      Tune (τ,λ,t_ood) over a fixed grid; correct for the multiple looks with LTT.
                      Report n_eff = (Σ ŵ)²/Σ ŵ² (Kish) as a reliability diagnostic and FLAG low-effective-
                      sample regimes (do not treat n_eff as a certificate gate).

  6. RECORD  (τ̂, λ̂, t̂_ood, w_max, ĉ, [w_lab,min,w_lab,max], α_acc, α_ood, δ, realized n_eff, κ(Ĉ_S)).

INFERENCE on target x:
  if ŵ_cov(x) > w_max  or  o(x) > t̂_ood :   abstain (suspected OOD / unreliable weight)
  elif u(x) > τ̂ :                           abstain (defer to clinician)
  else :                                     answer ŷ(x); emit per-case risk + audit.
                                             Realized accepted risk is MEASURED on held-out target.
```

Two notes on what these steps do and do not claim. First, `λ̂` is the **operating threshold selected on calibration**; we report its realized risk on target rather than asserting it transfers. Second, the LTT wrapper is used purely as a **multiple-looks correction** over the `(τ, λ, t_ood)` grid — we apply the published procedure and report the resulting selective-risk numbers; we do not claim a new validity guarantee survives the shift.

### 3.5 Methods, tools, and reliability diagnostics

The pipeline composes existing, citable methods. Each is used as published, and each guarantee below is a property **of that method under its own assumptions** — none survives the source-to-target shift as a deployment certificate.

- **Weighted (split) conformal / weighted RCPS** for the covariate-shift correction and the operating risk threshold. Tibshirani et al. (2019, Corollary 1) prove finite-sample `1−α` **marginal** coverage for weighted conformal **under pure covariate shift with a known (oracle) likelihood ratio**; Bates et al. (2021) give the `(α,δ)` RCPS risk guarantee **when calibration and test are exchangeable**. We invoke these as properties of their methods. Because our weight is *estimated* (not oracle) and clinical shift includes label shift that violates the conditional-invariance premise, we make **no** finite-sample claim for the deployed pipeline; we measure residual degradation on `D_tar^lab`.
- **Variance-adaptive concentration bound.** Importance weights inflate the risk-estimate variance and deflate the effective sample size, so a variance-adaptive bound is preferable to a range-only one. We use the betting / hedged-capital UCB (Waudby-Smith & Ramdas 2024; also the recommended UCB inside RCPS, Bates et al. 2021 §3.1.3) and the self-normalized (Hájek) weighted risk so the unknown `E[w]=1` cancels. We apply these bounds as published and report the resulting numbers; we do **not** claim a new validity guarantee under shift.
- **Label-shift estimation.** BBSE (Lipton et al. 2018) is a **consistent** estimator of `w_lab` under label shift given an invertible `Ĉ_S`; MLLS+BCTS (Alexandari et al. 2020) is empirically hard to beat when source softmax is miscalibrated; Garg et al. (2020) unify the two and name the shared invertibility identifiability condition. These are consistency/optimization results, not finite-sample certificates. More recent estimators now challenge MLLS+BCTS — ELSA (semiparametric-efficient label-shift estimation) and MAPLS (Bayesian Dirichlet-regularized MLLS, stronger in the limited-target-sample regime); we keep MLLS+BCTS as the primary for its simplicity and reproducibility but acknowledge it is no longer *uniquely* state-of-the-art, and we **pre-register MAPLS as an ablation** (prereg §4A) in our **small-`m`** stress regime (§1.7), where its limited-data advantage is most relevant.
- **Multiple-looks correction.** LTT (Angelopoulos et al.) handles the joint `(τ, λ, t_ood)` search; we use it as a plain multiple-testing correction.
- **Comparison baseline.** CRC (Angelopoulos et al. 2024) is kept as the less-conservative comparison row — it controls the *expected* loss (no `δ` tail), and its weighted variant needs the same estimated importance weights.
- **Repository references.** `aangelopoulos/rcps`, `gostevehoward/confseq`, `aangelopoulos/ltt`, the label-shift code in `kundajelab/labelshiftexperiments` (the `abstention` package), and for the OOD detectors `mueller-mp/maha-norm` (Mahalanobis++, Müller et al. 2025 — deployed *with* its in-distribution-only L2-normalization as the primary; the plain Lee et al. 2018 tied-covariance score, normalization removed, is the canonical-baseline ablation) and `deeplearning-wisc/knn-ood` (kNN ablation).

Rather than a theorem, we **report** a small set of diagnostics that flag when a correction is unreliable:

- **Effective sample size** `n_eff = (Σ ŵ)² / Σ ŵ²` (Kish) is reported per evaluation as a reliability diagnostic; we flag low-`n_eff` regimes where the weighted estimate is dominated by a few points.
- **BBSE conditioning.** The label-shift error grows as `Ĉ_S` becomes ill-conditioned, so we report `κ(Ĉ_S)`; near-singular `Ĉ_S` (weak base model, hard-to-separate or rare classes) is where the estimate is least trustworthy.
- **Softmax plug-in bias.** The denominator `Ẑ(x)` uses a plug-in for `p_S(y∣x)`; because raw softmax is miscalibrated, we recalibrate `f` (temperature/Platt) before forming `Ẑ`. This is a measured bias-reduction step, reported, not a guarantee. The plug-in bias is **non-vanishing** — recalibration reduces but does not remove it — and it is **amplified on rare / high-`ŵ_lab` classes**, where a small denominator error scales a large weight. We therefore do **not** assume the bias reduced: its residual effect on the headline weighted risk is **quantified** by a pre-registered sensitivity (raw softmax vs BCTS-recalibrated vs a temperature band), and the `w_max` clip bias is **swept** the same way (prereg `docs/preregistration.md` §5.4 / §6.6).
- **Anti-causal invariance check.** The `p(x∣y)`-invariance premise of all label-shift estimators fails under genuine cross-scanner appearance change; we report the discriminator-AUROC diagnostic per hospital pair and the `q̂_T` vs `Ĉ_S p̂_T` mismatch to flag gross violations.
- **Discriminator-shortcut audit.** The domain discriminator `d(x)` can separate source from target by keying on an **acquisition shortcut** — burned-in text, padding, laterality markers, a global intensity / stain offset — rather than clinical appearance, in which case a high discriminator AUROC overstates real covariate shift. We run a reported shortcut audit (e.g. masking/perturbing the suspected shortcut region and re-measuring discriminator AUROC). Where the discriminator keys on such a shortcut, the **representativeness chip** (§6.1) and the **shift-regime tag** (§6.3) are **caveated**, because the covariate weight then reflects an artifact, not clinical case mix.

**The label-shift regime is the cleanest setting we evaluate.** Under pure label shift the combined weight collapses to `w(x,y) = w_lab(y) = p̂_T(y)/p_S(y)`, the covariate correction is inert, and only **one** untestable assumption (anti-causal `p(x∣y)`-invariance) is in play rather than two. This is therefore the regime where **our correction needs the fewest assumptions and where we observe the lowest measured realized risk** — stated as an empirical observation, **not** a certified corollary. The combined covariate+label case is the stress extension: it carries the extra covariate-weight error and the FJS modeling residual, both **measured** on `D_tar^lab`.

### 3.6 Honest limitations and the empirical failure-mode catalog

The following are the substantive limitations of the corrections above, framed as measured limitations — not as repairs to any certificate. They feed directly into the failure-mode catalog (the experiments deliberately stress each one) and the discussion (§7).

**Why we measure rather than guarantee (the cited reasons).**
- *Combined shift is not identifiable* from unlabeled target alone — covariate and label shift cannot be disentangled without some labeled target signal, so we **measure** the residual on `D_tar^lab` and report it.
- *OOD detection is not distribution-free learnable* in the unrestricted setting (Fang et al. 2022). Near-OOD (same anatomy, shifted acquisition) overlaps ID support — exactly the regime their impossibility result covers — so we report the residual leakage on `O`, we do not certify it.
- *Finite-sample distribution-free conditional coverage under covariate shift is not attainable* — once the weight is estimated, only asymptotic / doubly-robust validity is available (Yang, Kuchibhotla & Tchetgen Tchetgen 2024), which is the reason the covariate correction is reported, not guaranteed with a finite-sample certificate.

**Where the assumptions actually break under clinical shift, and what we do.**
- *Anti-causal invariance fails under appearance change.* Cross-scanner/stain/site shift (CAMELYON17 hospitals; CheXpert↔MIMIC) alters `p(x∣y)`, violating the premise of every label-shift estimator; BBSE/MLLS then silently misattribute the change in predicted-label histogram to prevalence. We surface this with the discriminator-AUROC diagnostic per hospital pair and the `q̂_T` vs `Ĉ_S p̂_T` check.
- *Ill-conditioned `Ĉ_S` inflates label-shift error*, worst for rare (clinically important minority) classes; reported via `κ(Ĉ_S)` and worst-class error.
- *Estimated covariate weights are uncertified*; high-dimensional medical images make the discriminator hard to calibrate, and clipping at `w_max` biases the weights. We report measured coverage degradation on `D_tar^lab` and the clip-induced abstention rate.
- *Overlap / effective-sample collapse.* Genuinely OOD scans violate support overlap; large tail weights collapse `n_eff`. We route the non-overlap tail to a human (Fang et al. 2022 motivates routing rather than claiming coverage on it) and report `n_eff`.

**Genuine engineering fixes (applied, then measured).** Floor `ŵ_lab` onto a positive simplex range to avoid blow-up; recalibrate softmax before forming `Ẑ`; report `n_eff`; report the BBSE conditioning diagnostic; keep the exposure set `O` swappable and report its hardness. None of these convert a measured quantity into a guaranteed one — they reduce, and then quantify, the residual.

---

## 4. Out-of-distribution routing

The reweighting of §3 corrects for shift *within* the source support: it assumes the target point lies in a region the source covered, so that the density ratio `w(x,y) = dP_T/dP_S` is finite and the calibration scores carry information about it. Genuinely out-of-distribution scans — a stain or scanner the model never saw, an anatomy or artifact outside the training population, a corrupted acquisition — violate that premise. There the ratio is undefined or astronomically large, the discriminator `d(x)` can be confidently wrong, and no amount of reweighting recovers a calibrated answer. This section adds a **screen** that routes such cases to a human reviewer before they reach the answer path, and is explicit that the screen is an empirically evaluated filter, **not** a guarantee. All symbols are the single-source definitions of §1.

### 4.1 The OOD score `o(x)`

We score each case by a feature-space distance on the frozen penultimate embedding `φ(x) ∈ ℝ^p` (§1.3), never on raw pixels, so the detector reuses the representation the classifier already learned. High `o(x)` = more out-of-distribution.

- **Primary — Mahalanobis.** Following Lee et al. (2018), we fit class-conditional Gaussians to `φ(x)` on source training features, with per-class empirical means `μ̂_c` and a single **tied** empirical covariance `Σ̂`, and score by the squared Mahalanobis distance to the nearest class-conditional Gaussian,
  ```
  o(x) = min_c (φ(x) − μ̂_c)ᵀ Σ̂⁻¹ (φ(x) − μ̂_c).
  ```
  This is (the negative of) the confidence of a generative classifier *under a Gaussian-discriminant-analysis (GDA) model of the features* — that GDA model is the assumption it rests on (Lee et al. 2018), and we adopt the convention that larger distance means more out-of-distribution. We **adopt the L2 feature-normalization of Mahalanobis++** (Müller et al. 2025) for the deployed primary score: it is a *parameter-free, in-distribution-only* transform — each `φ(x)` is scaled by its own norm, and the per-class means `μ̂_c` and tied covariance `Σ̂` are then fit on normalized **source** features — so it requires **no OOD exposure**, keeps the exposure set `O` disjoint and swappable (§4.3), and empirically sharpens separation. We do **not** adopt the paper's other headline add-ons — the FGSM-style input perturbation and the per-layer logistic-regression feature ensemble — because *those* are tuned on a validation set that **contains OOD/adversarial samples**, which would couple the detector to the very exposure set we evaluate it on and is not OOD-agnostic. The exposure-coupling objection therefore applies to the perturbation/ensemble add-ons, **not** to the L2-normalization (which sees no OOD); that is why we keep the normalization and drop only the add-ons. Implementation: we build on the `mueller-mp/maha-norm` repository (Müller et al. 2025) and deploy the L2-normalized tied-covariance score as the primary `o(x)`; the **plain Lee et al. 2018 tied-covariance score** (normalization removed) is retained as the canonical-baseline ablation.

- **Alternative — energy.** The energy score (Liu et al. 2020) is the negative temperature-scaled log-partition function, `o(x) = −T · log Σ_k exp(f(x)_k / T)`, reported as `u(x)`'s OOD-facing twin (large `o(x)` = high energy = more out-of-distribution). Its justification is heuristic: under a Gibbs view `exp(−o(x)/T)` tracks the model density up to a constant (Liu et al. 2020). But this density-alignment is heuristic: it degrades under high-dimensional clinical shift when `f`'s logits are themselves unreliable, so energy can read confidently in-distribution on near-OOD scans — a failure mode we measure, not one the original paper rules out.

- **Alternative — kNN.** The non-parametric deep-nearest-neighbour score (Sun et al. 2022) L2-normalizes `φ(x)` to the unit hypersphere and takes the distance to the `k`-th nearest stored in-distribution feature. It makes **no distributional assumption** about the feature space — its appeal relative to Mahalanobis's tied-Gaussian model — at the cost of storing an ID feature bank and choosing `k`. Its level-set / density justification is **asymptotic** (large ID sample, suitable `k`), not a finite-sample bound (Sun et al. 2022).

- **Ablation — ReAct (wraps any `o(·)`).** As a cheap, OOD-agnostic add-on, we also evaluate **ReAct** (Sun et al. 2021): rectify (clip) the penultimate activations `φ(x)` at an upper percentile *estimated on source features* before scoring, suppressing the anomalously large activations that otherwise produce confident-but-wrong scores on OOD inputs. The clip threshold sees no OOD, so it preserves the disjoint-`O` discipline, and it composes with Mahalanobis, energy, or kNN alike.

We report the L2-normalized Mahalanobis score as the primary detector, with energy, kNN, and ReAct-wrapped variants as ablations. Crucially, **no single post-hoc detector dominates across backbones** — activation-shaping scores (e.g. ASH, SCALE) tend to lead on some CNN backbones but degrade on transformers, while the Mahalanobis family tends to lead on ViT/Swin — and OOD detection is not distribution-free learnable in any case (§4.4). We therefore treat the primary detector as a **backbone-dependent, empirical** choice and lean on the fact that **the routing logic below is identical for any choice of `o(·)`**: the detector is a swappable component, not a load-bearing assumption.

### 4.2 The routing rule

A case is routed to human review when its OOD score exceeds a threshold, **or** when its estimated covariate weight is so large that the density ratio is untrustworthy:
```
route to human  ⟺  o(x) > t_ood   OR   ŵ_cov(x) > w_max.
```
The two cutoffs are **decoupled** and guard different failure modes (cf. §3.2):

- `t_ood` is the **sole far-OOD guard**. The OOD score is what catches genuinely novel inputs, because on such inputs the discriminator can satisfy `d(x) ≈ 0` (the input looks like neither training pool) and so `ŵ_cov` does *not* flag it — `w_max` alone cannot catch far-OOD.
- `w_max` is a **near-OOD / variance guard** on the in-scope weighted estimator. Where `ŵ_cov(x) > w_max` the density ratio is finite but enormous; retaining it would let one point dominate the self-normalized weighted risk (collapse the effective sample size `n_eff`, reported as a reliability diagnostic in §1.7). Routing this tail is a deliberate bias–variance choice, and because clipping/routing itself perturbs the weights its effect is reported empirically, not assumed harmless.

A routed case is never scored for an answer; it is handed to the clinician. The answered-and-in-scope event is exactly `A(x) = {g(x)=1} ∩ {o(x) ≤ t_ood} ∩ {ŵ_cov(x) ≤ w_max}` from §1.5.1.

**A high-precision metadata pre-screen precedes the learned score.** A feature-space detector built on a *task-trained* `φ(x)` can be blind to inputs the base model was never trained to represent — wrong modality, wrong body-part, wrong view, or a corrupt acquisition can project into a dense in-distribution region of `φ` (the feature-collapse failure mode of §7.6). We therefore consult **DICOM/header metadata first, but as a high-precision flag, not a hard reject** — body-part/modality tags are frequently **wrong or missing across sites**, so a deterministic reject would itself drop valid scans. A case is routed out **only** when its tags *confidently* indicate out-of-trained-scope (e.g. wrong modality or body-part); **missing or ambiguous tags fall through** to the learned `o(x)` rather than being auto-rejected. This keeps the screen rule-based, exposure-free, and regulator-legible while not penalizing a scan for a bad tag. We additionally **log the image-vs-metadata disagreement rate** — cases where the metadata flag and the learned OOD score `o(x)` disagree — as its own reported number, because that rate is a direct, auditable measure of cross-site tag reliability. What the metadata flag removes is reported separately from the learned-score leakage on `O`.

### 4.3 Setting `t_ood`, and what we report

We set `t_ood` on the disjoint OOD-exposure set `O` (§1.7), held out from discriminator fitting, BBSE, risk calibration, the labeled target slice, and test. `t_ood` is chosen as the smallest cutoff whose measured far-OOD leakage on `O` meets a stated operating budget `α_ood`. We treat `α_ood` as a **named, separately-measured operating budget**, not as a certified additive term in any union-bound risk envelope: there is no `α = α_acc + α_ood` certificate here, and `α_ood` makes no probabilistic promise about deployment far-OOD beyond what we observe on `O`.

What we report for the screen is empirical, evaluated per benchmark shift (CAMELYON17 cross-hospital, CheXpert↔MIMIC-CXR cross-site):

- **AUROC / FPR95** of `o(x)` separating in-scope from the exposure set `O`, for each detector (Mahalanobis, energy, kNN);
- **Measured leakage** — the fraction of `O` cases scoring below `t_ood` (far-OOD that the screen passed into the answer path) — at the operating `t_ood`;
- **Sensitivity to the exposure set**: `O` is swappable, and we report how `t_ood` and the leakage estimate move when it is changed, because the exposure set is a modelling choice rather than ground truth.

### 4.4 Why this is a measured screen, not a guarantee

OOD detection is, formally, **not distribution-free learnable**. Fang et al. (2022) prove that no algorithm PAC-learns OOD detection over the unrestricted domain space (their Theorem 4 / total space), and overlap between ID and OOD support is what breaks the necessary condition (their Condition 1); restricting to disjoint ID/OOD supports (the 'separate space') is necessary but not sufficient — even there OOD detection remains unlearnable in general (their Theorem 5), and learnability is recovered only under further restrictions such as a finite feature space tied to the hypothesis class (their Theorems 6/10), a finite ID-distribution space (their Theorem 8), or a density-based space with bounded density and finite reference measure (their Theorems 9/11) — none of which we can assert for clinical imaging, where near-OOD (same anatomy, shifted acquisition) overlaps the in-distribution support and is exactly the regime the impossibility covers. We therefore make **no** distribution-free claim for the detector and present `o(x)` as an effective heuristic screen whose residual miss-rate we measure on `O` and report under each shift. The three detectors carry no finite-sample or distribution-free certificate: Mahalanobis is a GDA-model confidence (Lee et al. 2018), energy a density-alignment heuristic up to a constant (Liu et al. 2020), and kNN an asymptotic level-set argument (Sun et al. 2022) — each a property of *that* method under *its* assumptions.

This is the honest counterpart to prior pipelines (e.g. TRUECAM) that restore validity by detecting and removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and do not claim a maintained distribution-free guarantee in deployment: rather than leave the OOD gap unbounded or claim a certificate we cannot earn, we **measure and report the residual leakage** on a stated, swappable exposure set, and route the flagged tail to a human.

**Honest limitations of the screen, measured not assumed.** Under realistic clinical shift each detector's grounding assumption is the first thing to break. Mahalanobis's tied-Gaussian feature model transfers poorly across sites; energy's density alignment degrades precisely when `f` is miscalibrated under shift, so it can read confidently in-distribution on near-OOD scans; kNN depends entirely on whether the embedding still separates near-OOD after the in-distribution manifold has itself moved. We expose these through the per-hospital-pair discriminator-AUROC and detector-AUROC diagnostics, report `o(x)` performance separately for each benchmark shift, and state plainly that the screen **reduces** far-OOD leakage but cannot **certify** it.

---

## 5. Integrated decision rule and empirical operating budget

This section assembles the components of §§2–4 into a single accept rule, states the quantities we report for each deployment, and explains how the two operating budgets `α_acc` and `α_ood` are used for **reporting** — not as a certified additive error bound. Every threshold below is **tuned on calibration data and its realized effect MEASURED on held-out target data**; nothing here is a finite-sample guarantee on the deployed pipeline. Symbols are the single-source-of-truth definitions of §1.

### 5.1 The combined accept rule

For a target input `x`, the pipeline answers only when all three gates pass: the case is confident enough to answer, in-distribution enough to trust, and inside the region where the covariate weight is reliable. Formally, define the **answered & in-scope event**

```
A(x)  :=  { g(x) = 1 }  ∩  { o(x) ≤ t̂_ood }  ∩  { ŵ_cov(x) ≤ w_max },
```

with `g(x) = 1[u(x) ≤ τ̂]` the selection gate (§1.4), `o(x)` the OOD score routed at `t̂_ood` (§4), and `ŵ_cov(x)` the clipped covariate weight routed at `w_max` (§3.2). The decision is:

```
DECISION(x):
  if  ŵ_cov(x) > w_max        → ABSTAIN / route   (covariate weight untrustworthy; near-OOD tail)
  elif o(x) > t̂_ood          → ABSTAIN / route to human   (suspected far-OOD)
  elif u(x) > τ̂              → DEFER   (in-distribution but too uncertain)
  else  (A(x) holds)         → ANSWER ŷ(x);  emit per-case risk + audit record
```

The three routing outcomes are **distinct clinical actions, not one bucket**: a high covariate weight or high OOD score routes the scan **out of the model's scope** (to a human or to manual review), whereas a high uncertainty `u(x)` **defers** a case the model considers in-scope but cannot answer confidently. We keep the two scope guards `w_max` and `t̂_ood` decoupled on purpose (§3.2): `w_max` is a variance/boundedness control on the in-scope weighted estimator — its routed mass costs only coverage — while `t̂_ood` is the sole far-OOD guard whose miss-rate we charge to the reported `α_ood`. The order is immaterial to the final action (the gates are conjunctive), but we evaluate the scope guards first so that an obviously out-of-scope scan is never "deferred" as if it were a borderline in-distribution case.

### 5.2 What we report: an empirical accounting

For every benchmark shift we **measure on a held-out target split** (and, where prevalence/weight residuals are needed, on the small labeled target slice `D_tar^lab`, §1.7) the following quantities. None is a certificate; each is a reported number with its sampling uncertainty.

- **Realized selective risk on answered cases.** `R_T^accept = E_{P_T}[ ℓ(Y, ŷ(X)) ∣ A(X) ]`, estimated on labeled target as the empirical accepted-case error. This is the quantity the pipeline aims to keep low; we report its measured value, **not** a guarantee that it sits below any budget. Realized accepted-risk **and** coverage are aggregated **both** at the patch/token level **and** at the clinical **decision unit** (slide / patient; the per-study read), since that decision unit is where a human acts: for **CAMELYON17** the decision unit is the slide and the human integration point is **pre-read slide triage**; for **CheXpert / MIMIC-CXR** the decision unit is the per-study read and the integration point is a **flag-on-report second-reader**. Each risk is reported **both** as 0–1 selective risk **and** severity-stratified (FN vs FP on the positive class, with a severity-weighted loss whose weights are a reporting choice, not a certified cost; §1.4).
- **Coverage / answer rate.** `coverage = P_T(A(X))` and its decomposition into the three routing outcomes (abstain-on-weight, route-on-OOD, defer-on-uncertainty), so a clinician sees how often, and why, the model declined.
- **OOD leakage.** The fraction of exposure-set far-OOD cases that pass the detector (`o(x) ≤ t̂_ood`), measured on the disjoint exposure set `O` and reported as the realized leakage rate against the `α_ood` budget the threshold was set to spend. This is a measured rate on `O`, not a certified deployment bound.
- **Subgroup audit.** All of the above stratified by the clinically relevant axes (e.g. source hospital / scanner for CAMELYON17, site/protocol for CheXpert↔MIMIC), plus the per-hospital-pair discriminator AUROC as a shift-severity diagnostic, the BBSE conditioning diagnostic `κ(Ĉ_S)` and the `q̂_T`-vs-`Ĉ_S p̂_T` consistency check (§3.4), and the Kish effective sample size `n_eff = (Σ ŵ_i)² / Σ ŵ_i²` flagged where it is small (low-`n_eff` regimes have noisy weighted risk estimates and are reported as low-reliability, not silently averaged in).

The honest framing throughout: we **calibrate thresholds on source / unlabeled-target data, apply the weighted correction of §3, and then measure the realized risk, coverage, and leakage on held-out target, reporting any degradation** under each shift. Where the assumptions of a component method are violated by clinical shift — covariate-conditional invariance for weighted conformal/RCPS (Tibshirani et al. 2019; Bates et al. 2021), label-conditional invariance `p_T(x∣y)=p_S(x∣y)` for BBSE/MLLS (Lipton et al. 2018; Alexandari et al. 2020), the parametric/heuristic groundings of the OOD detectors (Lee et al. 2018; Liu et al. 2020; Sun et al. 2022) — we report the measured gap rather than claiming the guarantee survives.

### 5.3 The operating budgets `α_acc` and `α_ood` are reporting knobs, not a certified budget

We track **two named operating budgets**:

- `α_acc` — the budget against which the **accepted in-distribution selective risk** is tuned (via the risk-control threshold `λ̂` selected on calibration, §3.4 step 5), and against which the realized accepted risk is then reported.
- `α_ood` — the budget against which the **OOD threshold** `t̂_ood` is set on the exposure set `O` (§4.3), and against which the realized far-OOD leakage is then reported.

We sometimes present their sum `α = α_acc + α_ood` **only** as a convenient single headline number for "total tolerated error," and we are explicit that this sum is **not** a proven additive error bound on the deployed target risk. We do **not** claim a union-bound certificate that the plug-in accepted risk and the audited leakage jointly fall under their budgets with confidence `1 − δ`, nor any deployment inequality bounding the true answered-case target risk by `α_acc + α_ood` plus residual bias terms. Each budget is a knob we tune and a number we then measure separately:

- The accepted-risk side inherits no finite-sample guarantee under shift. RCPS (Bates et al. 2021) certifies `P(R ≤ α_acc) ≥ 1 − δ` **only when calibration and test are exchangeable from the same distribution** — its Remark 2 permits the *initial model* to have been trained elsewhere, but **not** a shifted calibration/test pair. Weighted conformal (Tibshirani et al. 2019) gives finite-sample `1 − α` **marginal** coverage **only under pure covariate shift with the oracle likelihood ratio** (their Corollary 1). Our deployment has discriminator-*estimated* weights and combined covariate+label shift, for which neither paper provides a finite-sample guarantee; we therefore treat `λ̂` as **the operating threshold selected on calibration whose realized accepted risk we then MEASURE on target** — we do not re-assert the `(α_acc, δ)` certificate or the weighted-coverage guarantee for the deployed pipeline.
- The OOD side cannot carry a distribution-free certificate at all: Fang et al. (2022) prove OOD detection is **not distribution-free PAC-learnable** in the unrestricted setting (where ID/OOD supports can overlap, as near-OOD clinical shift does), and Yang, Kuchibhotla & Tchetgen Tchetgen (2024) show **finite-sample distribution-free conditional coverage under covariate shift is not attainable** — once the weight is estimated, only asymptotic / doubly-robust validity remains. These results are precisely **why we report a measured leakage rate against `α_ood` rather than promise a guaranteed OOD bound**.

So `α_acc` and `α_ood` are operating budgets we spend during tuning and account for during evaluation; their sum is a reporting convenience, and the realized risk and leakage are the load-bearing numbers — reported with their measured degradation under each shift.

### 5.4 Tuning on one calibration set: a practical multiplicity note

The three knobs `(τ, λ, t_ood)` are searched jointly over a fixed grid `G` on the calibration data. Selecting the best-looking configuration from a grid on a single calibration set is a **multiple-looks / selection-bias** problem: an unadjusted risk estimate at the selected configuration is optimistically biased. As a lightweight, standard safeguard we wrap the grid search in a **Learn-then-Test (LTT)–style multiple-testing correction** (Angelopoulos et al.), so that the configuration we record is chosen under a multiplicity-aware procedure rather than by naïvely picking the empirical minimum.

This is a **practical note, not a theorem**. We use LTT as published to avoid grid-selection bias on the calibration set and to make the recorded operating point reproducible; we do **not** claim it restores any distribution-free validity guarantee on the shifted target deployment, for the reasons in §5.3. Two related diagnostics are reported rather than gated on: where the weighted risk estimate is variance-dominated, we prefer a variance-adaptive concentration bound (the betting / hedged-capital bound of Waudby-Smith & Ramdas 2024, also used inside RCPS §3.1.3 of Bates et al. 2021) over a range-only bound, because importance weights inflate the risk-estimate variance and deflate `n_eff`; and we **report `n_eff`** as a reliability flag, treating low-effective-sample configurations as unreliable rather than as certified.

### 5.5 Positioning relative to prior heuristic-OOD pipelines

Prior trustworthy-prediction pipelines such as TRUECAM restore validity by detecting and removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and do not claim a maintained distribution-free guarantee in deployment. Our contribution at this layer is **not** a stronger guarantee — it is a more honest accounting: rather than dropping suspected OOD with an unquantified residual, we route on an explicit, swappable exposure model and **measure and report the residual far-OOD leakage** against a stated budget `α_ood`. The previously-unquantified OOD gap becomes a reported number with a named exposure set, while every component method's guarantee continues to be cited only as a property of that method under its own assumptions.

---

## 6. Auditability and clinician-facing interpretability layer

*This section turns the uncertainty quantification of §§2–5 into an **auditable trust signal a clinician can act on**. Nothing here introduces a new guarantee: every number the layer exposes is a finite-sample **estimate** (`x̂`) or a **measured** diagnostic, and every guarantee sentence below names the method it belongs to and the assumption that method needs. The layer's job is to make the residual degradation that §3 warns about **visible at the point of care**, not to hide it behind a single certificate.*

The deployed pipeline (§3.4) already produces, for each target case `x`, the three signals a clinician needs: the routing decision over the in-scope/answered event `A(x) = {g(x)=1} ∩ {o(x)≤t̂_ood} ∩ {ŵ_cov(x)≤w_max}` (§1.5.1), the recalibrated posterior `σ̃(f(x))`, and the per-case combined weight `ŵ(x,y)`. This section specifies how these are surfaced. It has three components: (6.1) a **per-case calibrated risk panel**, (6.2) a **subgroup / fairness audit** of selective risk, and (6.3) **explicit OOD and shift flags** with their measured-degradation context. Section 6.4 gives worked clinical reads; §6.5 states the honesty boundaries the panel must show so it is not itself misread.

### 6.1 Per-case calibrated risk exposure

For an in-scope answered case (`A(x)=1`, `u(x)≤τ̂`), the panel exposes four things, all defined in §1:

- **Predicted label and recalibrated confidence** — `ŷ(x) = argmax_k f(x)_k`, and the temperature/Platt-recalibrated posterior `σ̃(f(x)) ∈ Δ^{K−1}`. We display `σ̃`, **not** the raw softmax `σ(f(x))`, because the recalibration step is the genuine estimation fix that makes the displayed probability mean what a clinician reads it to mean — and because the same `σ̃` enters the double-count corrector `Ẑ(x)` (§3.3). The displayed confidence is `max_k σ̃(f(x))_k`; the abstention threshold `τ̂` is shown alongside as the reference line the case cleared. Because `σ̃` is fit on source and its calibration can drift under shift, the displayed confidence carries the **ECE-triggered display behavior** of §6.5: where measured target ECE (reported per subgroup and per operating point) is poor, the panel widens the band, greys out the point confidence, and annotates "confidence unreliable for this site" — a display rule, never a gate.
- **A per-case risk band, not a point** — the operating accepted-risk budget `α_acc` is the cohort-level operating point the threshold `λ̂` was tuned to (RCPS; Bates et al. 2021). What we expose **per case** is the recalibrated error probability `1 − max_k σ̃(f(x))_k`, framed as *this case's individual uncertainty relative to the cohort the system was tuned to answer at `α_acc`*. We **do not** claim `α_acc` as a per-case guarantee: RCPS controls a population risk at `(α_acc, δ)` **only when calibration and test are exchangeable** (Bates et al. 2021), which clinical shift breaks (§3). The panel therefore labels `α_acc` as the *cohort operating point* and the per-case number as a *calibrated case-level uncertainty*, never as a certified individual bound. On the panel surface the per-case error number `1 − max_k σ̃(f(x))_k` is **never rendered beside `α_acc`**: the cohort budget `α_acc` lives in the demoted reliability footer (§6.4–6.5), so the two are not juxtaposed in a way that invites reading the per-case number as a draw against the budget.
- **The weight context `ŵ(x,ŷ)`** — the combined importance weight `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` (§3.3) tells the clinician *how representative this case is of the source calibration data*. A large `ŵ(x,ŷ)` means the case lives in a region the source set under-covered, so its contribution to the weighted target-risk estimate is amplified and its *representativeness* is low — the displayed confidence should therefore be read with extra caution, since the cohort the system was calibrated and audited on contained few cases like it. (This is a representativeness flag, not a claim that `σ̃` is miscalibrated for this individual case.) The panel renders this as a three-level chip (typical / up-weighted / heavily up-weighted relative to `w_max` and the realized weight distribution), with the numeric `ŵ` available on demand.
- **The reliability diagnostic `n_eff`** — the Kish effective sample size `n_eff = (Σ ŵ_i)² / Σ ŵ_i²` (§1.7) for the accepted in-scope cohort is shown as a **cohort-level reliability flag**, not a per-case quantity: when `n_eff` is small, importance weighting has concentrated the effective evidence on few points and *all* displayed risk numbers for that target site are less stable. This is a **reported reliability diagnostic, not a certification gate** — we surface it so a low-effective-sample regime is visible, exactly the regime Tibshirani et al. (2019, §2.3) note widens and destabilizes weighted coverage (their analysis attributes the overdispersion to the reduced effective sample size, via the Kish `n_eff` heuristic they credit to Gretton et al. 2009 and Reddi et al. 2015).
- **The residual-correction uncertainty** — when a scalar shift correction is fit on the small labeled target slice `D_tar^lab` (§3.4 step 3), that fit is itself an **estimate with its own sampling variance**, which shrinks with `m_lab` and is *not* captured by `n_eff` (which reflects weight concentration — a different thing). The panel therefore surfaces the **confidence-interval width of the residual correction** as a second cohort-level reliability flag: an `n_eff`-healthy cohort can still carry a wide residual CI when `m_lab` is small, and the clinician should see that the *correction itself* is uncertain, not only that the weights are well-spread.

**What the per-case panel does NOT claim.** It does not assert finite-sample coverage or risk control for the individual case. The weighted-conformal coverage result of Tibshirani et al. (2019, Corollary 1) is finite-sample and distribution-free **only under pure covariate shift with a known (oracle) likelihood ratio, and only marginally over the calibration draw** — not conditional on `x`. Our weight `ŵ` is **estimated**, and clinical shift includes prevalence shift that violates their conditional-invariance premise; so the panel presents a calibrated *case-level signal whose population reliability we measure*, never a per-patient certificate. This restriction is shown to the clinician as the panel's footnote, not buried.

### 6.2 Subgroup / fairness audit of selective risk

The audit re-computes the selective-prediction quantities **within each pre-registered subgroup** `s` (e.g. acquisition site / scanner, sex, age band, self-reported race where ethically collected, and clinically salient strata such as biopsy-confirmed vs. screening cases). For each subgroup it reports, on the held-out **labeled** evaluation data (target slice `D_tar^lab` where available, source otherwise):

- **Selective risk** `R̂^accept_s` — the self-normalized (Hájek) weighted accepted-in-scope risk `R̂_w = Σ_{i∈s} ŵ_i ℓ_i / Σ_{i∈s} ŵ_i` (§1.7) restricted to subgroup `s`. This is a weighted **estimate** of the accepted error a clinician in that subgroup's population experiences — an estimate because the weights `ŵ` are themselves estimated; we report it with the interval below rather than as an exact realized rate.
- **Coverage** `P̂(g(X)=1 ∣ s)` and **abstention/defer rate** — the fraction of `s` the system answers vs. defers. A subgroup that is disproportionately abstained on is being *silently under-served* even if its accepted risk looks low; the audit surfaces the coverage gap explicitly so low risk bought by high abstention is not mistaken for good performance.
- **OOD-routing rate** `P̂(o(X) > t̂_ood ∣ s)` and **weight-routing rate** `P̂(ŵ_cov(X) > w_max ∣ s)` — which subgroups are most often routed to a human, and via which mechanism (far-OOD detector vs. untrustworthy covariate weight).
- **Per-subgroup shift diagnostics** — the domain-discriminator AUROC and BBSE conditioning number `κ(Ĉ_S)` *for that subgroup's source-vs-target pair*, so a subgroup where the correction is least trustworthy is visible (see §6.3).

**Honest framing of the audit numbers.** The subgroup selective risks `R̂^accept_s` are **measured (estimated) realized risks on labeled data**, reported with finite-sample confidence intervals (the betting/concentration interval of Waudby-Smith & Ramdas 2024, applied as published) — they are **diagnostics, not certified per-subgroup risk guarantees**. We deliberately do **not** wrap the per-subgroup search in a new multiplicity-corrected certificate and call it a fairness guarantee; the multiple-looks correction (LTT; Angelopoulos et al.) is applied to the *operating-threshold selection* in §3.4, whereas the subgroup audit is *reporting*, and we present it as such. Where a subgroup's labeled target sample is small, the interval is wide and the panel says so rather than implying a precision it does not have. The audit's value is making **disparate selective risk and disparate abstention visible and attributable** — to a scanner, a prevalence gap, or an ill-conditioned `Ĉ_S` — not promising parity. This **subgroup-attributable disparate-abstention finding** is one of the two outputs that are **load-bearing only at the audit layer** — i.e. they exist *because* of the audit and have no counterpart in the deployed accept rule of §5.1; the other is the **shift-regime badge** of §6.3, which prevents the cleaner label-shift result from being silently substituted for the harder combined case. Both are reporting outputs, not gates or guarantees.

### 6.3 Explicit OOD and shift flags

Three flags accompany every case and every cohort report. Each is an **operational routing signal with a measured leakage rate**, never a guarantee of correctness.

- **Far-OOD flag** — raised when `o(x) > t̂_ood`. The OOD score `o(x)` is a feature-space distance on `φ(x)` (Mahalanobis primary, kNN ablation; Lee et al. 2018, Sun et al. 2022). The threshold `t̂_ood` is set on a disjoint exposure set `O` to achieve a measured leakage rate `α_ood` (§4.3), reported as a separately-measured operating budget rather than a certified one. The panel shows the **measured** residual far-OOD leakage on `O` next to the flag, i.e. the fraction of known-OOD exposure cases the detector let through at `t̂_ood`. We attach the principled reason this is a *measured* number and not a certificate: distribution-free OOD detection is **provably not learnable in the unrestricted setting** (Fang et al., NeurIPS 2022), so no detector — Mahalanobis, energy, or kNN — can carry a distribution-free OOD guarantee; the pipeline therefore audits leakage empirically rather than promising coverage on novel inputs.
- **Out-of-support / untrustworthy-weight flag** — raised when `ŵ_cov(x) > w_max`. This routes cases whose covariate density ratio is too large to trust (the overlap / effective-sample-collapse failure mode of weighted conformal; Tibshirani et al. 2019, §2.3, where the weighted-coverage overdispersion is explained by the reduced effective sample size — the Kish `n_eff` heuristic they attribute to Gretton et al. 2009 and Reddi et al. 2015). It is a **separate** mechanism from the OOD flag: a case can be in-distribution yet so far up-weighted that its individual contribution would destabilize the weighted estimate. The clip-and-route mitigation is engineering, not covered by any theorem (clipping itself biases the weights), so its effect is **reported empirically** — the panel logs how much mass each site routes this way.
- **Shift-regime flag** — a cohort-level badge stating which correction regime is actually in force for this target site, decided by the per-pair diagnostics: (i) **pure-label-shift regime** when the domain-discriminator AUROC is near chance (covariate shift small) — the regime where our label correction (BBSE; Lipton et al. 2018, with MLLS+BCTS; Alexandari et al. 2020) needs the fewest assumptions, and where we observe the lowest measured realized selective risk; or (ii) **combined covariate+label regime** when the discriminator shows real covariate shift — here the combined weight `ŵ(x,y)=ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` (combined via the per-`x` corrector `Ẑ`, **not** the product; §3.3) is in force, and a **measured residual** from the small labeled target slice `D_tar^lab` is reported. The badge prevents the cleaner label-shift result from being silently substituted for the harder combined case. With the subgroup-attributable disparate-abstention finding of §6.2, this shift-regime badge is one of the two **audit-layer-only load-bearing outputs**: it has no counterpart in the deployed accept rule and exists purely so the reader cannot mistake the easier regime's numbers for the combined case.

**Why these are flags and not guarantees.** The label-shift estimators (BBSE; MLLS+BCTS) are **consistent under the label-shift assumption `p_T(x∣y)=p_S(x∣y)` given an invertible source confusion matrix** (Lipton et al. 2018; Garg et al. 2020 for the shared identifiability condition) — they are *not* finite-sample or distribution-free, and the assumption is exactly what cross-scanner appearance change violates. We therefore (a) run the BBSE consistency diagnostic (compare `q̂_T` vs `Ĉ_S p̂_T`) to flag gross violation, (b) report the discriminator-AUROC per hospital pair so the reader sees how far the pair is from "no covariate shift," and (c) **measure** the residual prevalence- and shift-correction error on `D_tar^lab` rather than claiming it is zero. The reason we report rather than promise is the same pair of results we cite throughout: finite-sample distribution-free conditional coverage under covariate shift is not attainable once the weight is estimated (Yang, Kuchibhotla & Tchetgen Tchetgen, JRSS-B 2024), and OOD detection is not distribution-free learnable (Fang et al. 2022).

### 6.4 Worked clinical reads

The layer is designed so a clinician reads three signals in sequence — *answered or routed?* → *how confident, and how representative?* → *which regime and how trustworthy here?* Four representative cases:

- **Confident answer, typical case.** `A(x)=1`, `max_k σ̃(f(x))_k = 0.97`, `ŵ(x,ŷ) ≈ 1.0` (typical chip), cohort `n_eff` healthy, shift badge = *pure-label-shift*. Read: a routine, well-represented case answered with high calibrated confidence in the regime where the correction needs the fewest assumptions — act on the prediction.
- **Answered but up-weighted — the "trust, but note" case.** `A(x)=1`, `max_k σ̃ = 0.91`, but `ŵ(x,ŷ)` is in the *heavily up-weighted* band and the cohort badge = *combined covariate+label*. Read: the case is in-scope and the displayed confidence is decent, **but** this case lives in a region the source set under-covered, so it is poorly represented in the cohort the system was calibrated and audited on; the panel directs the clinician to weight their own judgement more heavily and consult the per-subgroup audit for this case's scanner.
- **Deferred case.** `A(x)=1` (in-scope) but `u(x) > τ̂` ⇒ defer. Read: the model is too uncertain to answer at the cohort operating point `α_acc`; this is the system working as designed — a human decides, and the deferral is logged for the coverage audit so it cannot masquerade as a low-risk answer.
- **Routed-to-human (OOD).** `o(x) > t̂_ood` ⇒ far-OOD flag. Read: this scan looks unlike anything in the calibration distribution (e.g. a scanner or stain the system never saw); it is sent to a human, and the panel shows the **measured** OOD-leakage rate on `O` so the clinician knows the detector's empirical miss-rate at this threshold — there is no claim that all such cases are caught, only a reported rate.

In every read the panel co-displays the operating budgets `α_acc` (cohort accepted-risk operating point) and `α_ood` (OOD-leakage rate) as the **separately measured operating budgets they are** — not as a single certified total, since we make no union-bound guarantee that combines them.

### 6.5 Honesty boundaries the panel must display

So the audit layer is not itself a source of overconfidence, the interface carries these statements wherever the corresponding number appears:

- **`α_acc` is a tuned cohort operating point, not a per-case or post-shift certificate.** RCPS control at `(α_acc, δ)` is a property of the *method under exchangeability between calibration and test* (Bates et al. 2021); under clinical shift that premise is broken, so the realized accepted risk is **measured on held-out target**, and the panel shows that measured number alongside the budget.
- **The displayed confidence is a recalibrated estimate `σ̃(f(x))`,** whose calibration was fit on source and may itself degrade when the label-shift premise `p_T(x∣y)=p_S(x∣y)` is violated by cross-scanner appearance change (the anti-causal invariance break of §3.5/§7.3, not a result of Garg et al. 2020); the per-subgroup audit is where that degradation becomes visible. This is enforced by a **concrete panel display behavior**, not a footnote: target-site calibration is **measured** (the ECE of `σ̃` on held-out target, reported per subgroup and per operating point; the measurement protocol is preregistered, `docs/preregistration.md` §5.1), and where measured target ECE is poor the panel **widens** the displayed confidence band, **greys out** the point confidence, and **annotates** "confidence unreliable for this site." This is a **display rule** — it changes how the number is shown so it is read with appropriate caution; it is **not** a gate, a recalibration, or a bound on the case.
- **Subgroup selective risks are measured diagnostics with finite-sample intervals,** widening where labeled target data are scarce — not certified per-group guarantees.
- **OOD and weight flags reduce, but cannot certify, leakage** (Fang et al. 2022); their residual miss-rates are reported on a stated, swappable exposure set `O`, with the set's composition disclosed so its difficulty can be judged.

The contribution of this layer, stated honestly, mirrors the project's overall stance: **prior work (e.g. TRUECAM) restores validity by detecting and removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and does not claim a maintained distribution-free guarantee in deployment; we instead expose the routing decision, the calibrated per-case signal, the subgroup selective-risk audit, and the measured residual leakage as one coherent, auditable interface — surfacing where the correction is least trustworthy rather than papering over it.** That visibility, not a new certificate, is the explainability/trustworthiness hook.

### 6.6 Decline-Attribution Record (DAR): a routing-faithful explanation of the routing decision (faithful to the routing, not the diagnosis)

The routing logic of §5.1 already decides, for every target case, whether to answer, defer, or route. This subsection **names and surfaces** what that logic produces — it adds no method and no guarantee. We call the central explanatory artifact the **Decline-Attribution Record (DAR)**: a per-case, **routing-faithful** explanation of *why the model declined to answer*, expressed as the set of reliability gates that fired together with the signed margin to each gate's threshold. Throughout, **routing-faithful** means **faithful to the routing decision, not to the diagnosis** — the explanation matches the predicate the pipeline actually evaluated to route the case; it does **not** assert that the routing, or the underlying class prediction `ŷ(x)`, is correct. We keep this scope clause attached to the term wherever it appears.

**What the DAR records.** The accept rule answers `x` only when the in-scope/answered event `A(x) = {g(x)=1} ∩ {o(x)≤t̂_ood} ∩ {ŵ_cov(x)≤w_max}` of §1.5.1 holds (§5.1). When the case is deferred or routed instead, the DAR reports, for that `x`:

- **The firing set** — which of the three predicates failed: uncertainty (`u(x) > τ̂`, *defer to clinician*), suspected far-OOD (`o(x) > t̂_ood`, *route out of scope*), or untrustworthy covariate weight (`ŵ_cov(x) > w_max`, *out of support*). These are the **distinct clinical actions** of §5.1, not one "rejected" bucket.
- **The signed margin to each threshold** — `m_u = τ̂ − u(x)`, `m_ood = t̂_ood − o(x)`, `m_w = w_max − ŵ_cov(x)`. A negative margin marks a violated gate (a reason for the decline); the magnitudes show *by how much* the binding gate was crossed and *how close* the case sat to a different action.
- **The measured operating context of the binding gate** — for an OOD decline, the measured far-OOD leakage rate of `t̂_ood` on the exposure set `O` (§4.3); for an uncertainty decline, the cohort operating budget `α_acc` the threshold `λ̂` was tuned to (§5.3); for a weight decline, the cohort `n_eff` and the clip-induced routing rate (§6.3).

**Routing-faithful by construction (faithful to the routing, not the diagnosis), contrasted with post-hoc saliency.** Faithfulness is the property that an explanation reflects the model's true decision process rather than a plausible substitute for it (Jacovi & Goldberg, 2020). Post-hoc feature attribution (saliency, Grad-CAM, SHAP) builds a *second* artifact that approximates the model and so cannot in general guarantee this — Rudin (2019) argues such an artifact "must be wrong," and Adebayo et al. (2018) show several saliency methods are insensitive to randomising the model's parameters or labels, while Arun et al. (2021) found that all eight saliency methods they tested on two chest-radiograph datasets failed at least one trustworthiness criterion. The DAR has no such fidelity gap, because it is **not an approximation of the decision — it is the decision predicate**: the system declines a case iff some gate predicate in `A(x)` evaluated false (§5.1), and the DAR reports precisely the predicate(s) that evaluated false and the numerical margins the routing rule itself computed. There is no surrogate model and no learned attribution, so the explanation and the computation are the same conditional. This is the inherently-interpretable route Rudin (2019) advocates, applied not to the diagnostic classifier `f` (which remains a black box) but to the **reliability gating** layered on top of it — a small, transparent, monotone rule over three named scores whose output is its own faithful explanation. Note the scope: each margin `m = threshold − score` is an exact affine function of its score, so it introduces *no instability beyond that score's own*; this is **not** a stability guarantee, since the OOD and weight scores themselves degrade under shift (§4.4, §7.3), and it is **not** a head-to-head accuracy contest, since the DAR and saliency explain different objects.

**How it is surfaced in the panel.** The DAR sits in the clinician-facing surface of §6.3–6.4. For a declined case the panel renders, in sequence: (i) the **plain-language firing reason** for the binding gate ("too uncertain to answer — deferred to you"; "looks unlike the calibration data — routed for review"; "outside the trusted weight region — routed"); (ii) a **distance-to-threshold** for the binding gate, rendered as "how close to being answered," with the non-binding margins available on demand so the clinician sees which constraint the case was next-closest to tripping; and (iii) the **resulting clinical action**. The measured operating context above and the statistician-facing diagnostics (`σ̃` calibration, `n_eff`, `κ(Ĉ_S)`, per-pair discriminator AUROC) are placed in the demoted reliability footer of §6.4–6.5, not on the explanation surface. For an *answered* case the panel instead shows `σ̃(f(x))`, the representativeness chip from `ŵ(x,ŷ)`, and the cohort `n_eff` flag (§6.1), with `τ̂` shown as the reference line the case cleared. The DAR is thus the routing logic of §5.1 and the per-case flags of §6.3 turned outward: the same predicates the pipeline evaluates to make the decision are the predicates it reports.

### 6.7 Panel failure modes and what is not validated

The panel is an interface, and interfaces have failure modes of their own — distinct from the statistical failure modes of §§3–4. We name them here so they are not mistaken for solved problems. The DAR fidelity checks establish that the panel *faithfully reports the routing decision* (§6.6), but **fidelity is not decision-validation**: whether a clinician reads the panel correctly and acts better with it is an open, IRB-gated question.

- **Automation bias from a green panel.** A confident answer with a *typical* representativeness chip and a healthy `n_eff` can induce over-reliance — the clinician defers to the model precisely where it looks safe. A faithful DAR does not prevent this; it only makes the routing reasoning visible.
- **Alarm fatigue from frequent up-weighted / route chips.** Under heavy shift the *up-weighted* / *heavily up-weighted* chips and the route-to-human flags fire often; a surface that flags constantly trains the reader to ignore the flag, eroding exactly the caution the chip is meant to add.
- **Cohort-vs-per-patient misreading of `α_acc` / `n_eff`.** `α_acc` is a **cohort** operating point and `n_eff` is a **cohort** reliability flag (§6.1); both can be misread as per-patient quantities. The panel labels them as cohort-level and keeps `α_acc` off the per-case surface (§6.1), but the misreading risk is a property of the reader, not removable by labelling alone.

**What is and is not validated.** The interface is **fidelity-measured** — the DAR checks of §6.6 confirm the explanation matches the predicate the pipeline evaluated — but it is **not yet decision-validated**: we have **not** shown that it improves clinician decisions or reduces error in practice. The clinician-facing decision study that would test these failure modes is **IRB-gated and planned**, not completed; until it is run, every claim in this section is about the panel's reporting fidelity, never about its effect on care.

---

## 7. Assumptions, scope, and honest limitations

This section states, component by component, the assumptions each method needs, where it degrades under realistic clinical shift, and *why* we **measure and report** that degradation rather than promise a guarantee. The discipline throughout: every guarantee sentence names the method it belongs to and the assumption it requires; no guarantee is attributed to the *pipeline as a whole*, and no new certificate is claimed for the combined shift these benchmarks exhibit. The evaluation protocol that operationalizes "measure and report" — the stress axes, the held-out target measurements, the diagnostics — is specified in the preregistration (`docs/preregistration.md`); this section names what those measurements are for.

### 7.1 The two impossibility results that set our stance

Two published impossibility results are the reason this is an *applied, measure-and-report* paper rather than a *certified-guarantee* paper. They are not limitations of our engineering; they are limits on what **any** method in this space can promise.

- **OOD detection is not distribution-free learnable (Fang et al., NeurIPS 2022).** There is no algorithm that PAC-learns out-of-distribution detection over the unrestricted (total) domain space (their Theorem 4); disjoint ID/OOD supports alone do not recover it (the 'separate space' still carries an impossibility, their Theorem 5), and learnability returns only under further restrictions — a finite ID-distribution space (their Theorem 8) or a finite-feature / bounded-density space tied to a specific hypothesis class (their Theorems 6/9/10/11). Near-OOD — same anatomy, shifted acquisition — overlaps in-distribution support and is exactly the regime the theorem rules out. **Consequence for us:** no detector we deploy (Mahalanobis, energy, kNN) can carry a distribution-free OOD certificate, so we set the OOD threshold `t_ood` on a *stated, swappable* exposure set `O` and **report the residual far-OOD leakage `α_ood` measured on `O`** (§4), with degradation under each benchmark shift measured, not guaranteed.
- **No finite-sample distribution-free conditional coverage under covariate shift (Yang, Kuchibhotla & Tchetgen Tchetgen, JRSS-B 2024).** Finite-sample, distribution-free *conditional* coverage under covariate shift is not attainable through the prediction-set score; once the density ratio is estimated, only asymptotic / doubly-robust validity is available. **Consequence for us:** the covariate-shift correction inherits no finite-sample certificate once the weight is *estimated* (`ŵ_cov`) rather than oracle, so we **measure the residual coverage/risk degradation** on the small labeled target slice `D_tar^lab` instead of promising it.

**Estimated weights are not a flat dead-end — and we state why we still only measure.** The same Yang, Kuchibhotla & Tchetgen Tchetgen (2024) result that rules out finite-sample conditional coverage also names what *is* recoverable: with an auxiliary outcome/score model, **doubly-robust** conformal constructions (Yang et al. 2024 itself; Qiu, Dobriban & Tchetgen Tchetgen 2023) retain *asymptotic* validity when **either** the weight model **or** the outcome model is correct, so weight-estimation error is **partially survivable**, not automatically fatal. We engage this rather than framing estimation error as unrecoverable. We nonetheless **choose pure measurement** for this applied pipeline, for stated reasons: (i) a doubly-robust step adds a *second* model that can itself be misspecified, against our base-model-agnostic, minimal-moving-parts stance; (ii) its validity is *asymptotic*, not a finite-sample deployment certificate, so it does not close the gap our honesty stance is about; and (iii) our shift is *joint* covariate+label, where the pure-covariate DR constructions do not transfer unmodified. A doubly-robust **step that shrinks the residual before we measure what remains** is therefore **pre-registered as a comparison ablation** (prereg §4A) — not adopted as primary, but engaged empirically rather than only in prose.

A third, structural fact compounds these: **combined covariate-and-label shift is not identifiable from unlabeled target covariates alone.** This is why `D_tar^lab` (a labeled target slice, `m_lab ≈ 50–200`) exists purely to *empirically measure* residual degradation and fit a simple scalar correction — never to "identify" a nuisance parameter or certify the combined case.

### 7.2 Objective, and what we actually report

We **want** the answered-case target risk `R_T^accept := E_{P_T}[ ℓ(Y, ŷ(X)) | A(X) ]` to be low under shift, over the observable in-scope/answered gate `A(x) = {g(x)=1} ∩ {o(x) ≤ t_ood} ∩ {ŵ_cov(x) ≤ w_max}` (§1.5.1). We do **not** certify `R_T^accept`. Instead we: calibrate thresholds on source data; apply the weighted correction of §3; and **measure the realized selective risk, coverage, and OOD leakage on held-out target data**, reporting the degradation. `α_acc` (accepted in-distribution error) and `α_ood` (far-OOD leakage) are named, **separately measured** operating budgets — not a proven union-bound split, and `δ` is the confidence level of a reported interval, not a PAC promise over the pipeline.

**Positioning, stated honestly.** Prior work such as TRUECAM restores validity by detecting and removing OOD inputs rather than by a guarantee that provably survives in-scope covariate shift, and does not claim a maintained distribution-free guarantee in deployment. We do not improve on that with a new guarantee; we simply **measure and report the residual leakage** against a stated exposure model rather than leaving it unbounded and unmeasured. That is the contribution — auditability, not a certificate. The broader near-neighbor positioning — against the conformal-prediction-for-clinical-deployment cluster (Lu et al. 2022 aggregate performance estimation; Angelopoulos et al. 2024 Conformal Triage; Xu et al. 2025 Selective Conformal Risk Control; Hulsman et al. 2025 CRC for nodule detection; and the concurrent prevalence-shift triage audit of Li et al. 2026) — is in `docs/positioning_memo.md`: the differentiator is composition across joint covariate+label correction, the decoupled OOD/variance double gate, and the audit layer, never a stronger guarantee.

### 7.3 Per-component assumptions and where each degrades

Each row names the method, the assumption its published guarantee needs, and the clinical mechanism that breaks it. The guarantee is a property **of that method under its assumption**; under our benchmarks the assumption is typically violated, so we report the measured residual rather than re-claiming the guarantee.

| Component (method) | Guarantee it carries, and under what assumption | How clinical shift breaks it | What we report instead |
|---|---|---|---|
| **Weighted (split) conformal** (Tibshirani, Foygel Barber, Candès & Ramdas 2019) | Finite-sample `1−α` **marginal** coverage **under pure covariate shift with a *known* (oracle) likelihood ratio** (their Corollary 1) | (i) Clinical shift includes prevalence/label shift, violating the conditional-invariance premise `p(y∣x)` fixed; (ii) our `ŵ_cov` is *estimated* by a domain discriminator, not oracle; (iii) genuinely OOD scans violate overlap `supp(p_T) ⊆ supp(p_S)` | Measured coverage degradation on `D_tar^lab`; we make **no** finite-sample claim for the deployed pipeline |
| **Covariate weight `ŵ_cov`** via domain discriminator (Tibshirani et al. 2019, eq. 12: `ŵ_cov(x) = clip((d(x)/(1−d(x)))·c, 0, w_max)`) | The conditional odds ratio equals the oracle likelihood ratio up to the base-rate constant `c = π_S/π_T`, which cancels in the weighted quantile — **for the oracle weight** | High-dimensional images make `d` hard to calibrate; tail weights inflate `max_i p^w_i`, collapsing the effective sample size; clipping at `w_max` itself biases the weights; the discriminator may key on an **acquisition shortcut** (burned-in text, padding, laterality, intensity) rather than clinical appearance, overstating real covariate shift | Per-hospital-pair discriminator AUROC diagnostic; **discriminator-shortcut audit** (mask/perturb the suspected shortcut, re-measure AUROC) caveating the representativeness chip and shift-regime tag (§3.5); reported `n_eff`; measured effect of clipping (no theorem covers it) |
| **BBSE label weights** `ŵ_lab = Ĉ_S⁻¹ q̂_T` (Lipton, Wang & Smola 2018) | **Consistent** estimator of `w_lab = p_T(y)/p_S(y)` **under label shift** (`p(x∣y)` invariant, anti-causal) with an **invertible** confusion matrix `C_S`; works with a biased/uncalibrated black box | Cross-scanner/stain/site change alters `p(x∣y)`, violating label shift; weak base model or hard classes make `Ĉ_S` ill-conditioned; small `m` makes `q̂_T` noisy and the inverse unstable | BBSE consistency diagnostic (compare `q̂_T` vs `Ĉ_S p̂_T`) to flag gross violation; `Ĉ_S` conditioning diagnostic; simplex-floor `ŵ_lab ∈ [w_lab,min, w_lab,max]` to avoid blow-up |
| **MLLS + BCTS** prevalence estimate (Alexandari, Kundaje & Shrikumar 2020; consistency conditions: Garg, Wu, Balakrishnan & Lipton 2020) | The MLLS objective is **concave** (EM reaches the global optimum); consistency requires (sufficient conditions) the classifier be **calibrated** *and* `C_S` invertible (Garg et al. 2020 prove these are sufficient — Thm 1/2, Prop 1; calibration is shown only *sometimes* necessary, Example 1) | Calibration is fit on source and can degrade under shift — the calibration precondition may fail precisely when needed; the label-shift premise itself fails under appearance shift | Measured residual prevalence-correction error on `D_tar^lab`; we do not claim exact or distribution-free correction |
| **RCPS** threshold selection (Bates, Angelopoulos, Lei, Malik & Jordan 2021) | Finite-sample, distribution-free **`(α,δ)` PAC** control `P(R ≤ α) ≥ 1−δ` of a bounded, monotone, nested risk **when calibration and test are exchangeable** (Remark 2: only the *initial model* may come from a different distribution) | Covariate **or** label shift between calibration and deployment breaks exchangeability — the `(α,δ)` certificate is void; importance weighting introduces weight-estimation error the bare theorem does not cover | `λ̂` is the operating threshold selected on calibration; its realized selective risk is **measured** on target. We apply the hedged-capital UCB (already part of RCPS §3.1.3; also Waudby-Smith & Ramdas 2024) because importance weights inflate variance, and report the numbers |
| **CRC** (comparison baseline) (Angelopoulos, Bates, Fisch, Lei & Schuster 2024) | Controls the **expectation** `E[L] ≤ α` (not a tail) of a monotone bounded exchangeable loss; weighted variant handles covariate shift with known/estimated weights, inheriting their error | Same exchangeability break; gives a clinician no per-deployment tail guarantee, and its weighted form is only as good as the estimated weights | Kept as the less-conservative comparison row; expectation-only and approximate under estimated weights |
| **SelectiveNet** abstention head (Geifman & El-Yaniv 2019) | A **training objective** for the risk-coverage trade-off; **no** standalone test-time certificate — its target coverage can be violated on test (its §5); a calibrated bound comes from the **separate** SGR procedure (Geifman & El-Yaniv 2017) | Assumes i.i.d. training data, has no OOD mechanism; learned coverage and selective risk are not preserved under shifted/OOD scans | Coverage and selective risk re-calibrated and audited empirically on the target slice |
| **OOD detectors** — Mahalanobis (Lee, Lee, Lee & Shin 2018), energy (Liu, Wang, Owens & Li 2020), kNN (Sun, Ming, Zhu & Li 2022) | **None** carries a finite-sample or distribution-free guarantee: Mahalanobis is a generative-classifier confidence under a tied-Gaussian (GDA) assumption; energy aligns with the data density only up to a constant and its grounding is a heuristic the original paper does not certify under shift; kNN's level-set justification is asymptotic | Cross-site shift breaks exactly each grounding assumption: tied-Gaussian transfer fails (and OOD-tuned ensemble weights leak OOD info), energy can read confidently in-distribution on near-OOD scans, kNN depends on the embedding still separating near-OOD; near-OOD overlaps ID support — the Fang et al. regime | Detectors **reduce** but cannot **certify** leakage; residual miss-rate reported on the stated exposure set `O`, with per-benchmark degradation measured. Prefer the OOD-agnostic Mahalanobis-family score — the Maha++ L2-normalization is *in-distribution-only*, so it stays OOD-agnostic — and kNN, so `O` stays disjoint and swappable; drop only the OOD-tuned add-ons (input perturbation, validation-OOD feature ensemble), not the normalization |

### 7.4 Combining the two weights: a measured estimation correction, not a guarantee

Combining covariate and label weights by the **naïve product** `ŵ_cov(x)·ŵ_lab(y)` is *wrong* — it double-counts, because a prevalence change moves the covariate marginal `p(x)` and the discriminator picks that up. The correct combine divides by the per-`x` normalizer, which follows directly from the master importance-weighting identity of §1.5 (law of total probability on `dP_T/dP_S`); it is a standard estimation step (cf. Tasche), not a new result:
```
Z(x) := Σ_{y'} w_lab(y') p_S(y'∣x) = E_{p_S(·∣x)}[ w_lab ],     Ẑ(x) = Σ_{y'} ŵ_lab(y') σ̃(f(x))_{y'}
w(x,y) = w_lab(y) · w_cov(x) / Z(x)
```
*Worked check:* under pure label shift `p_S(y)=(.5,.5) → p_T(y)=(.9,.1)`, the truth at an `x` where class 1 dominates is `w = 1.8`, while the naïve product gives `1.8·1.8 = 3.24`; dividing by `Z(x)` recovers `1.8` (the full derivation is in §3.3). We use the **temperature/Platt-recalibrated** softmax `σ̃` inside `Ẑ` because the raw softmax denominator is a non-vanishing plug-in bias — this is a genuine estimation fix, not a certificate. The bias is **not assumed reduced**: it is **non-vanishing** even after recalibration and is **amplified on rare / high-`ŵ_lab` classes**, so its residual effect on the headline weighted risk is **quantified** by a pre-registered sensitivity (raw vs BCTS vs a temperature band), and the `w_max` clip bias is **swept** likewise (prereg `docs/preregistration.md` §5.4 / §6.6).

**Modeling choice, adopted not proven.** We adopt a factorizable (independent-mechanism) shift premise as a modeling choice; it is not identifiable from unlabeled target data, so we fit a scalar correction on `D_tar^lab` and report the residual. Sparse Joint Shift is not adopted because raw-image shift is dense.

### 7.5 Reported reliability diagnostics

Because the guarantees above are void or approximate under our shift, the following are **reported as diagnostics** (not certification gates):

- **Effective sample size** `n_eff = (Σ ŵ_i)² / Σ ŵ_i²` (Kish) — flags low-effective-sample regimes where importance weights have collapsed the usable calibration mass; reported, never used to silently emit or withhold a bound.
- **BBSE conditioning diagnostic** — the conditioning `κ(Ĉ_S)` / `σ_min(Ĉ_S)`; BBSE error grows as `Ĉ_S` becomes ill-conditioned, and rare clinically-important minority classes are estimated worst.
- **BBSE consistency check** — compare `q̂_T` against `Ĉ_S p̂_T` to flag gross anti-causal / `C_T ≠ C_S` violations (the appearance-shift failure mode).
- **Per-hospital-pair discriminator AUROC** — anti-causal `p(x∣y)`-invariance fails under cross-scanner appearance change; the discriminator AUROC per hospital pair surfaces where covariate reweighting is doing real work versus picking up class-conditional drift.
- **Multiple-looks correction** — tuning `(τ, λ, t_ood)` on one calibration set is multiple looks, so we correct with Learn-Then-Test (Angelopoulos) and report; we claim no new validity guarantee from it.

### 7.6 Cleanest regime, and what remains genuinely open

- **Lowest-assumption regime.** The **pure label-shift** case is where our correction needs the fewest assumptions: with `w_cov ≡ 1` the combined weight collapses to `w(x,y) = w_lab(y) = p̂_T(y)/p_S(y)`, identifiable whenever `Ĉ_S` is invertible, resting on a **single** untestable premise (anti-causal `p(x∣y)`-invariance) rather than two. We expect, and report, the lowest measured realized risk here. *Stated as the cleanest evaluation regime — not a certified corollary.* Its honest caveat remains: anti-causal invariance still fails under genuine cross-scanner appearance change, surfaced by the BBSE consistency diagnostic.
- **Genuinely open edges (named, not bugs).** (i) Residual combined-shift error is **uncertifiable** without more labeled target data — `D_tar^lab` gives a measured point estimate with a CI, not a bound, and its own slice-size variance is reported. (ii) OOD leakage rests on the exposure set `O` being representative — an **untestable** transfer assumption — so `O` must be stated, swappable, and its hardness reported (Fang et al. 2022 is why this cannot be closed by a detector). (iii) Calibration of `f` is fit on source and can itself drift under shift, degrading MLLS exactly when it is needed. These are reported and audited per benchmark; they are not resolved here, and we do not claim otherwise.
- **Frontier gaps we name but do not adopt.** (iv) *Marginal vs. class-conditional control.* RCPS controls a **marginal** risk; under heavy label shift a model can meet the target by serving abundant classes while failing rare pathologies. We **report** per-class / worst-class and severity-stratified risk (§3.6, §6.2) but do **not** calibrate thresholds per class; **Mondrian / class-conditional conformal** is the standard upgrade, named here as future work. (v) *Task-trained embedding / feature collapse.* `φ(x)` is optimized for the source label and can discard task-irrelevant structure, so a genuinely anomalous input (foreign body, novel artifact) may project into a dense ID region and slip past `o(x)`; the deterministic DICOM pre-screen (§4.2) is the first-line mitigation, and using a **frozen medical foundation-model embedding** for the OOD/shift layer (rather than the classifier's own `φ`) is named as an embedding-source ablation, not adopted. (vi) *Temporal / within-target drift.* We correct a **static** source→target shift; drift of the target *over time* (scanner-software updates, seasonal case-mix) is **out of scope** and belongs to post-market surveillance (control charts / CUSUM on realized error), named here so its omission is explicit rather than an oversight.

*Pointer.* The full evaluation protocol — stress axes (small `m`, scanner/site pairs, prevalence sweeps), held-out target measurements, and the per-subgroup audit — is preregistered in `docs/preregistration.md`. This section names the assumptions and failure modes; the preregistration fixes how each is measured.

### 7.7 The Decline-Attribution Record (DAR): what it explains, and what it does not

The DAR of §6.6 faithfully explains **the routing decision** — *which reliability gate failed and by how much* — and nothing more. Its limits are honest scope statements, not repairs to any certificate.

- **It explains the gate decision, not the class prediction or the biology.** The DAR does **not** explain why the classifier `f` assigned its label `ŷ(x)`, and it does **not** explain the disease, the biology, or the causal dynamics of the patient. It also does not explain *why* a gate's underlying score took its value (e.g. why `o(x)` was high), only that it did and that this is why the case was routed.
- **The margins are distances to tuned thresholds, not probabilities or bounds.** The thresholds `τ̂`, `t̂_ood`, `w_max` are **tuned operating points, not certified bounds** (§5.3), so the signed margins `m_u, m_ood, m_w` are distances in score space — **not** calibrated probabilities of error and **not** a per-case guarantee. `α_acc` and `α_ood` remain the separately-measured operating budgets of §5.3/§7.2, never recombined into a certified split by being displayed beside a margin.
- **The gated scores inherit the §7.3 component-method limits.** A faithful attribution to the OOD or weight gate is a faithful report of a *heuristic screen's* decision, not a certificate that the case is truly OOD or truly unrepresentative: the OOD score carries no distribution-free guarantee (Fang et al. 2022; §4.4, §7.1, §7.3), and the covariate weight `ŵ_cov` is *estimated*, not oracle (§7.3). The DAR makes the model's routing reasoning auditable; it does not make the underlying scores infallible, and the panel says so (§6.5).
