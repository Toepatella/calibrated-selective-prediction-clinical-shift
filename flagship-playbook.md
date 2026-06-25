# Flagship — Calibrated Selective Prediction Under Clinical Distribution Shift
### Single journal-grade build — *supersedes the two-phase MVN→flagship split*

> **What changed.** The original `gap3-project-playbook.md` staged a conference MVN (Phase 1) then a flagship superset (Phase 2). This document **collapses that into one journal submission**: the first implementation includes the full method superset — covariate **and** label shift, an **integrated** OOD error budget, a temporal track, formal theory, and a subgroup audit. The two-phase doc is kept intact as a reference; this is the operative plan.
>
> **Two decisions baked in (yours):**
> 1. **Keep the staged validation gates as *internal engineering milestones*.** One paper, but you still build and prove each layer on synthetic data before stacking the next. The gates are debugging infrastructure, not lesser papers. Skipping them is how flagship attempts die with "nothing submitted."
> 2. **Depth over breadth: two imaging benchmarks, full method.** CAMELYON17-WILDS + CheXpert↔MIMIC-CXR, taken all the way (covariate + label shift, integrated OOD, temporal, theory, subgroups). The multi-modality sweep (ECG/retina/2nd-pathology) is explicit, honestly-scoped **future work**, not a v1 obligation.
>
> **Self-contained.** Everything you need is in *this* file: the **PRIMER** (the five ideas), **Appendix A** reading list, **B** dataset cheat-sheet, **C** glossary, **D** paste-in prompts (all rewritten for this scope), **E** repo tree, **F** notation & formulas, **G** environment cheat-sheet, **H** symbol decoder. You don't need to open `gap3-project-playbook.md` anymore — it's retained only as a historical record of the dropped two-phase split.
>
> **Conventions.** `monospace` = file/symbol/variable. "Backbone" = the risk-control layer (RCPS/LTT), not the neural net (that's "base model"). Effort: **S** ≤ a day · **M** a few days · **L** a week+ · **XL** multi-week.

---

## 0. The contribution on one page (flagship)

**The gap.** No clinically validated selective-prediction method simultaneously: **(1)** abstains/defers; **(2)** gives a *distribution-free* guarantee on the error rate of answered cases; **(3)** keeps that guarantee approximately valid under real clinical shift — **covariate *and* label/prevalence shift**; **(4)** routes far-OOD inputs to abstain **with that routing inside the certified budget** (not a bolt-on); **(5)** is benchmarked on real public medical shift data.

**Why it's open (from the audit).** TRUECAM is closest but its shift-validity is *empirical* (OOD-removal + "cannot be guaranteed in deployment"), single-domain, no label shift. SCRC/SCoRE have the guarantee but assume exchangeability. Liang & Sun handle shift+OOD but give *no* distribution-free guarantee. Nobody composes **selection + covariate/label shift + integrated far-OOD** under **one** distribution-free risk-control statement that provably survives real shift.

**Novelty core (never trim):** conditions **(2) + (3) + (4-integrated)** — a distribution-free error guarantee on answered cases that survives real covariate *and* label shift, with OOD routing folded into the same error budget.

| Objective | In v1? | Novelty weight | Notes |
|---|---|---|---|
| (1) Abstain/defer | ✅ | Low (table stakes) | Selection threshold on the accepted region |
| (2) Distribution-free guarantee | ✅ | **Core (spine)** | Weighted RCPS w/ WSR bound; LTT for joint tuning |
| (3a) Covariate shift | ✅ | **Core** | Clipped density-ratio weights |
| (3b) Label/prevalence shift | ✅ | **Core (the step up)** | BBSE-estimated label weights, combined |
| (4) Far-OOD routing, **integrated** | ✅ | **Core (beats TRUECAM)** | Split α across accept-error + OOD-leakage |
| (5) Real public benchmarks | ✅ ×2 | Validation | CAMELYON17 + CheXpert↔MIMIC, both shift types |
| (5′) Many modalities | ⛔ future work | Breadth | Stated as the honest open extension |
| Temporal/continual (DtACI) | ✅ | Medium | Long-run track; complementary promise |
| Subgroup validity | ✅ | Medium (clinical credibility) | Per-group risk + coverage; RLCP optional |

**The honesty constraint (write on a sticky note).** *Covariate **and** label shift, modelled explicitly* + *unlabeled target sample of size m* + *far-OOD routed away so residual density ratio is bounded* + *label proportions estimable (BBSE identifiability)* ⇒ "**approximately valid**." Exact distribution-free validity under *arbitrary* shift with *zero* target info is impossible. Every guarantee sentence carries this.

**Done =** a journal-grade paper + reproducible release whose headline shows the accepted-case risk held at ≤ α under real covariate *and* label shift on both imaging benchmarks — where naïve conformal, and a TRUECAM-style empirical pipeline, visibly break — with OOD error inside the certified budget and a subgroup audit.

---

## 1. Strategy: one paper, staged internally

**The merge.** *Publishing* = a single flagship journal submission. *Engineering* = the same staged ladder as before, with each gate now an **internal milestone** you must pass before adding the next layer. You are not writing two papers; you are de-risking one.

**Attribution discipline (the rule that protects the result).** The headline number is produced with a **standard backbone** + the standard `u(x)` (softmax-response/energy) + spectral-norm Mahalanobis++ `o(x)`. Fancier base models (DDU, SNGP, deep ensembles) appear **only as ablations layered on top**, never as the foundation. Reason: validity comes from the conformal layer and is *base-model-agnostic*; a fancier net only buys efficiency (less abstention at equal guaranteed risk). Building the core on SNGP hands Reviewer 2 the kill — *"is the win just the backbone?"* Keep the win attributable.

**Depth-over-breadth, stated honestly.** Two imaging modalities (histopathology, chest X-ray) × two distinct shift mechanisms (scanner/stain; institution/population) + estimated label shift + temporal drift on one site. Condition (5′) breadth is **future work**; the discussion says so plainly. This is still strictly beyond every audited competitor.

**The internal gate ladder (what must turn green, in order).**
```
G-A  exchangeable selective RCPS  → synthetic: accepted-risk ≤ α straddles α across splits
G-B  + covariate weights          → synthetic shift: true weights restore ≤ α, none visibly violates
G-C  + label-shift weights        → synthetic prevalence shift: covariate-only fails, label-aware restores
G-D  + integrated OOD budget      → imperfect detector: bolt-on leaks, integrated bound holds
G-E  + temporal DtACI             → injected change-point: fixed threshold drifts, DtACI restores long-run risk
G-R  real data headline           → both datasets: ≤ α under real covariate+label shift where baselines break
```
*If a synthetic gate fails, the real result cannot exist — stop and fix the layer before moving on.*

---

## 2. Foundations (was Phase 0) — status & remainder

**Built (verified in `phase0_learning.ipynb`):** steps 0.1.1–0.1.6 — split conformal, CRC, selective + the selection-bias trap, RCPS vs CRC vs LTT, weighted conformal + ACI, OOD (Mahalanobis++/kNN) + distance awareness. Implementations cross-checked against canonical sources (`ryantibs/conformal`, `herbps10/AdaptiveConformal`, `mueller-mp/maha-norm`, `deeplearning-wisc/knn-ood`).

**Remaining foundation work (do before/while coding the method):**
- **2.a Competitor matrix** *(M)* — `analysis/competitor_matrix.csv`, one row per audited work, columns: satisfies (1)–(4)? · exact missing piece · reusable asset. Summarize SCRC's proof structure in your own words (closest ancestor).
- **2.b Re-verify the gap** *(S)* — run Appendix D-2 falsification on mid-2025→today camera-readies; `analysis/reverify_<date>.md`; verdict OPEN/PARTIAL + dated. (Last audit 2026-06-20 = PARTIALLY ADDRESSED/OPEN.)
- **2.c Positioning memo** *(M)* — `docs/positioning_memo.md`: gap + one-sentence contribution + the flagship honesty constraint + residual-contribution sentences.

---

## 3. Method formalization (the full superset, up front)

Write `docs/method_note.md` with the complete method — not a covariate-only subset.

> **3.1 Notation** *(S)*. `P_S,P_T` source/target · `f` frozen base model · `ŷ(x)` prediction · `u(x)` uncertainty · `g(x)=1[u(x)≤τ]` selection · `ℓ(y,ŷ)∈[0,1]` loss · `w_cov(x)=p_T(x)/p_S(x)` covariate density ratio · `w_lab(y)=p_T(y)/p_S(y)` label ratio · `w(x,y)` combined weight · `o(x)` OOD score · thresholds `τ, λ, t_ood` · budgets `α=α_acc+α_ood`, `δ` · target sample `m`.

> **3.2 Target guarantee (integrated)** *(S)*. Box this:
> ```
> R_T^accept := E_{P_T}[ ℓ(Y,ŷ(X)) | g(X)=1, o(X)≤t_ood ] ≤ α_acc
> + a certified bound on residual risk from far-OOD that the detector MISSED,
> so the end-to-end answered-case risk ≤ α = α_acc + α_ood.
> ```
> In words: among target cases the model answered and the detector judged in-scope, expected loss ≤ α_acc; the leakage of undetected far-OOD is separately bounded by α_ood; together ≤ α. **This integration is the direct answer to TRUECAM's "cannot be guaranteed in deployment."**

> **3.3 The four exchangeability-breakers + neutralizers** *(M)*. Four-row table:
> - **Selection** → calibrate the controlling threshold *on the accepted region*.
> - **Covariate shift** → reweight calibration by clipped `ŵ_cov(x)` from a domain discriminator.
> - **Label/prevalence shift** → estimate `p_T(y)` by **MLLS+BCTS** (Alexandari et al. 2020) — calibrate the classifier with Bias-Corrected Temperature Scaling *first*, then EM; this dominates vanilla BBSE under class imbalance. **RLLS** (Azizzadenesheli et al. 2019, regularized BBSE) is the robustness companion / ablation. Vanilla BBSE (Lipton et al. 2018) kept only as the naïve foil. The real lever is calibration, not the solver — MLLS and BBSE are the *same estimator* under different calibration (Garg et al. 2020). Fold `ŵ_lab(y)` in. *Derive the combined weight under a stated shift model — do not naïvely multiply `w_cov·w_lab`* (double-counts when both are present).
> - **Far-OOD tail** → where `ŵ>w_max` or `o(x)>t_ood`, route to abstain; the removed mass is exactly what would blow up the weighted bound. Detector error is *budgeted*, not assumed-zero.
>
> **3.3b The combined weight + the identifiability tension** *(M)* — *the central caveat, write it on the sticky note next to the honesty constraint.* The combined-shift problem has a canonical name: **Generalized Label Shift / Generalized Target Shift** (Zhang et al. 2013 → Gong et al. 2016 → Tachet des Combes et al. 2020). For medical imaging commit to the **anticausal `Y→X`** factorization (disease causes image), so label shift is primary and scanner/stain is a *residual* class-conditional shift. Under that model the honest combined weight is `w(x,y) = w_lab(y) · r(x,y)`, where `r(x,y)=p_T(x|y)/p_S(x|y)`. Writing it as `r(x)` (y-independent) **is the simplifying assumption** ("the new scanner shifts every class identically") — state it as a limitation, don't bury it in notation.
> **The tension (name it in the paper, don't hide it):** *every* label-shift estimator — BBSE, MLLS+BCTS, RLLS, GS-B³SE, FMAPLS — assumes `p(x|y)` is **fixed** to identify `w_lab(y)` from unlabeled target. The residual `r` is exactly the violation of that assumption. So **a fancier label-shift estimator buys robustness to sampling noise / imbalance, NOT robustness to conditional shift** — it cannot rescue Gap-2. Three ways to live with it (pick per dataset, report which): **(a)** assume `r` benign + stress-test by injecting growing conditional shift in G-C; **(b)** use a small labeled-target slice and estimate the joint ratio directly via an `(x,y)`-discriminator (sidesteps identifiability, trades double-counting *bias* for *variance*); **(c)** sensitivity-bound how wrong `w_lab` gets as a function of `‖r‖` and propagate into the conformal budget (the §3.6 route).

> **3.4 Calibration + inference algorithm** *(M)*. Implementable pseudocode:
> ```
> CALIBRATION (source cal set + unlabeled target sample + target predictions):
>   1. Fit domain discriminator d(x); w_cov(x)=clip(d/(1-d)*c, 0, w_max).
>   2. Estimate p_T(y) via MLLS+BCTS (calibrate f first, then EM) from target predictions;
>      w_lab(y)=p_T(y)/p_S(y). Combine: w(x,y)=w_lab(y)*r(x,y) under the anticausal model
>      (NOT w_cov*w_lab). If conditional shift r is non-benign, fall back to a joint
>      (x,y)-discriminator on a small labeled-target slice. (RLLS = robustness ablation.)
>   3. Fit OOD score o(x) on f's spectral-norm features; pick t_ood for the α_ood budget.
>   4. Among NON-OOD calibration points, choose (τ, λ) so the WEIGHTED accepted risk
>      ≤ α_acc via weighted RCPS with the WSR (betting) upper bound.
>      If (τ,λ,t_ood) are tuned JOINTLY, wrap selection in LTT (multiple testing) to stay valid.
>   5. Record (τ, λ, t_ood, w_max, c, α_acc, α_ood).
> INFERENCE on target x:
>   if o(x)>t_ood or w(x)>w_max:  abstain (far-OOD)        # (4)
>   elif u(x)>τ:                  abstain (defer)           # (1)
>   else:                         answer ŷ(x), risk ≤ α     # (2)+(3a)+(3b)
> ```

> **3.5 Backbone decision** *(S)*. **Weighted RCPS** for the deployed `(α,δ)` PAC claim, with the **WSR / hedged-capital upper bound** (`gostevehoward/confseq`, `src/confseq/betting.py`) instead of Hoeffding–Bentkus — tighter, and it matters *more* here because importance weights inflate the risk-estimate variance (which widens the UCB). **LTT** (`aangelopoulos/ltt`) wraps the calibration when you tune `(τ,λ,t_ood)` jointly on one calibration set — a single RCPS bound is *invalid* under multiple looks. **CRC** kept only as the less-conservative comparison row. Reference RCPS impl: `aangelopoulos/rcps` (`core/` concentration bounds + λ̂).

> **3.6 Theorem + formal statement** *(L)*. State the **finite-sample** bound for selective + weighted RCPS under bounded density ratio (after OOD routing), with **explicit constants** — not just a sketch. Characterize the approximation error as a function of (covariate-weight error) + (**label-weight error: MLLS+BCTS estimation error**) + (**residual conditional-shift `‖r‖`**, the term that is *not* identifiable from unlabeled target — the §3.3b tension) + (detector error → α_ood). Give the label-shift corollary. This is the journal-grade theory step the MVN deferred.

> **3.7 Red-team** *(S)*. Run Appendix D-3 against the full note; fix the weakest claim (likely: the combined covariate+label weight identifiability under residual conditional shift §3.3b, or the integrated-budget detector-error term). *Anticipate the reviewer who knows GS-B³SE/FMAPLS and asks "why not just a better label-shift estimator?" — the answer is in §3.3b: the gap is conditional shift, which every such estimator assumes away.*

---

## 4. Lock experimental design *before* coding

> **4.1 Datasets + splits** *(S)* → `docs/design.md §datasets`.
> - **CAMELYON17-WILDS** (`p-lambda/wilds`): 5-hospital scanner/stain shift; official ID vs OOD split; carve an unlabeled *target-calibration* slice disjoint from target test.
> - **CheXpert ↔ MIMIC-CXR**: institution/population shift; shared-pathology label subset (`docs/label_mapping.md`); **both directions** (gives you a second shift instance cheaply). Start credentialing **now** (Stanford AIMI; PhysioNet + CITI) — human approval is the long pole.
> - **Far-OOD probes**: off-modality / wrong-anatomy / non-medical injections to test (4) + α_ood.

> **4.2 Shift scope** *(S)*. Covariate **and** label/prevalence shift (this is the upgrade). Temporal drift handled in the DtACI track (§5, Stage E) by time-ordering one site.

> **4.3 Metrics with formulas** *(M)* → `analysis/metrics.py` + `docs/design.md §metrics`.
> - *Primary*: realized accepted-target risk vs α, as a **distribution over many cal/test splits** (histogram + α line) — the guarantee is marginal over the calibration draw.
> - *Integrated-budget*: accepted risk **with a deliberately imperfect OOD detector** — show the bound still holds (bolt-on leaks).
> - *Label-shift*: realized risk under estimated vs oracle `p_T(y)`.
> - *Operating*: risk–coverage curve + AURC (source & target).
> - *Temporal*: long-run realized risk across a change-point (DtACI vs fixed).
> - *Subgroup*: per-group coverage **and** risk (sex, age, site).
> - *Secondary*: base-model ECE, abstention rate, accepted accuracy.

> **4.4 Baselines** *(S)*. (1) naïve split-conformal selective [the foil]; (2) softmax-response/SelectiveNet; (3) SCRC (guarantee, no shift); (4) weighted conformal w/o selection; (5) a **TRUECAM-style empirical pipeline** (OOD-removal + CRC, no maintained guarantee) — the closest competitor, beat it head-to-head; (6) deep ensemble + threshold [UQ ceiling]; (7) **Wang & Qiao 2025** — "Conformal Prediction Under Generalized Covariate Shift with Posterior Drift" ([arXiv:2502.17744](https://arxiv.org/abs/2502.17744)), **the nearest published conformal treatment of combined shift — now a mandatory baseline.** It delivers *coverage* only; you remain distinct on **risk control (RCPS) + selection + integrated-OOD budget + label-shift estimation**. Position explicitly: their "posterior drift" (`p(y|x)` changes) is a different factorization from your anticausal label-shift + residual model; (8) **yours**.

> **4.5 Ablations** *(S)*. base-model {standard, DDU, SNGP, deep ensemble} (validity unchanged, efficiency moves); **label-shift estimator {MLLS+BCTS [core], RLLS, vanilla BBSE [foil], FMAPLS, GS-B³SE}** — *same logic as the base-model ablation: validity flat, efficiency moves; proves your conformal layer is estimator-agnostic and lets you cite the 2025 bleeding edge without betting the result on it. FMAPLS/GS-B³SE are reimplement-from-scratch (no released code) → run only if time allows, FMAPLS first;* **WSR vs Hoeffding–Bentkus** bound; covariate-only vs covariate+label weights; **bolt-on vs integrated** OOD budget; `w_max` and target-size `m` sensitivity; OOD detector {Maha++, kNN}; backbone RCPS vs CRC.

> **4.6 Pre-register** *(S)* → `docs/preregistration.md`, committed to git with a timestamp *before* results: "Yours holds accepted risk ≤ α under covariate+label shift on both datasets while ≥1 baseline (incl. the TRUECAM-style pipeline under label shift) visibly violates it, at comparable or better coverage, with OOD error inside the budget."

> **4.7 Critique** *(S)*. Run Appendix D-4; add any missing baseline.

---

## 5. Build sequence with internal gates

Repo scaffold (full tree in **Appendix E**): `data/ models/ ood/ conformal/ experiments/ analysis/ tests/ docs/ paper/`. Pin env (**Appendix G**), wire W&B/MLflow, Hydra configs + `set_seed()`, `pytest`. Then build in order — **each stage ends in its synthetic gate; do not proceed past a red gate.**

> **Stage A — exchangeable selective RCPS** *(M)* → `conformal/selective.py`, `conformal/rcps.py` (WSR bound). **Gate G-A**: synthetic exchangeable data, accepted-risk ≤ α straddles α across many splits. Ref: `aangelopoulos/rcps`, `confseq`.

> **Stage B — covariate weights** *(M)* → `conformal/weights.py` (discriminator + clipped ratio), weighted path in `rcps.py`. **Gate G-B**: synthetic known covariate shift — true weights restore ≤ α on target; no-weights visibly violates. Ref: `ryantibs/conformal` `weighted.quantile`.

> **Stage C — label-shift weights** *(M)* → `conformal/label_shift.py`. Primary estimator **MLLS+BCTS** (calibrate `f`, then EM); **RLLS** as the regularized-robustness path; vanilla BBSE as the foil. Combine under the **anticausal `Y→X`** model as `w(x,y)=w_lab(y)·r(x,y)` (§3.3b), *not* `w_cov·w_lab`. **Gate G-C** (now two sub-gates): **(C1)** synthetic known prevalence shift, `p(x|y)` fixed — covariate-only weighting fails, label-aware restores ≤ α. **(C2 — the honest one)** inject *growing residual conditional shift* `r` and watch `w_lab` error degrade: this empirically maps the identifiability tension and tells you when to switch to the labeled-slice joint-discriminator fallback. *Biggest theory step; budget time — the tension in §3.3b is the journal's central caveat.*

> **Stage D — integrated OOD budget** *(M)* → `ood/detector.py` (Maha++/kNN on spectral-norm features) + budget split in the calibration. **Gate G-D**: with a *deliberately imperfect* detector, the bolt-on version leaks risk above α while the integrated-budget bound holds. This is the TRUECAM-beating result *in vitro*.

> **Stage E — temporal DtACI track** *(M)* → `conformal/aci.py`. Time-order one site; stream; update `α_t` from realized error. Use **DtACI** (`isgibbs/DtACI`) for abrupt drift, not vanilla ACI. **Gate G-E**: injected change-point — fixed weighted threshold drifts out of control, DtACI restores long-run risk. State plainly: this is **long-run/amortized**, a *complementary* promise to the finite-sample RCPS bound, and needs label feedback.

> **Stage R — real-data headline** *(M each)*. CAMELYON17 then CheXpert↔MIMIC, all baselines, sweep α, many splits. **Gate G-R**: yours ≤ α under real covariate+label shift where naïve and the TRUECAM-style pipeline break, on **both** datasets. *If unconvincing, iterate the method, not the plot.*

> **Stage S — subgroup / local coverage** *(M)*. Per-subgroup risk + coverage (sex, age, site); selective prediction can *magnify* disparities (Jones et al. 2021). Optionally strengthen toward (approx.) conditional coverage with **RLCP** (`rohanhore/RLCP`).

> **Code review** *(S)*. Run Appendix D-5; fix leakage / off-by-one / clip-bias. All synthetic gates must still pass after fixes.

---

## 6. Run, ablate, red-team (deep, on the two datasets)

- **6.1** Full ablation set (§4.5), tabulated from logs — the base-model {standard/DDU/SNGP/ensemble} efficiency-frontier plot is a flagship highlight (validity flat, efficiency moves).
- **6.2** Confound checks: report coverage *with* risk (validity not bought by near-total abstention); weight-error stress (degrade the discriminator); small-`m` breakdown; BBSE-error stress.
- **6.3** Find and *report* a shift where it fails — a characterized boundary is a feature, not an embarrassment.
- **6.4** Results red-team (Appendix D-6); rewrite the central claim to match the evidence.
- **Done when** every table/figure regenerates from logged runs by one script, and the top 3 alternative explanations are ruled out with specific plots.

---

## 7. Write the journal paper

- **7.1 Venue** *(S)*: Medical Image Analysis / IEEE TMI / npj Digital Medicine / TMLR by theory-vs-clinical balance. Match template now.
- **7.2 Structure** *(S)*: intro (gap + one-sentence contribution + honesty constraint) → related work (your matrix) → method (§3, full superset) → theory (§3.6, with constants) → experiments (§5–6) → subgroup audit → limitations → reproducibility. Lead with the structural finding: *"guarantee xor shift-robustness — nobody has both, and OOD has never been inside the budget."*
- **7.3 Figures** *(M)*: headline risk-vs-α under covariate+label shift; integrated-vs-bolt-on under an imperfect detector; risk–coverage; base-model efficiency frontier; DtACI change-point; subgroup table; OOD-routing illustration.
- **7.4 Prose** *(L)*: every guarantee sentence carries its assumption; quote TRUECAM's *"cannot be guaranteed in deployment"* as the contrast you resolve.
- **7.5 Reviewer-2 pass** *(S)*: Appendix D-7 three-reviewer sim; apply top-3 revisions.
- **7.6 Honest "what remains open"** *(S)*: Appendix D-8 audit — name multi-modality breadth (5′), arbitrary-shift limits, and BBSE identifiability as the open edges.
- **7.7 Release** *(S)*: pinned env, configs, seeds, one script per figure; data-access README (can't redistribute CheXpert/MIMIC — ship splits + steps); Zenodo DOI in the paper.

---

## ● Consolidated gate ladder (honor in order)

- [ ] **G-A** exchangeable accepted-risk ≤ α (synthetic).
- [ ] **G-B** covariate weights restore ≤ α (synthetic).
- [ ] **G-C** label-aware weights restore ≤ α where covariate-only fails (synthetic).
- [ ] **G-D** integrated budget holds under an imperfect detector where bolt-on leaks (synthetic).
- [ ] **G-E** DtACI restores long-run risk across a change-point (synthetic stream).
- [ ] **G-R** headline ≤ α under real covariate+label shift, both datasets, ≥1 baseline (incl. TRUECAM-style) violates.
- [ ] Subgroup audit done; limitations honestly bound the claim.
- [ ] Theory has explicit constants, externally checked (Appendix D-3 + a conformal-literate reader).
- [ ] Manuscript submitted + reproducible release archived.

---

## Risk register (updated for the single-flagship build)

| Risk | Symptom | Mitigation | Stage |
|---|---|---|---|
| **Build-everything-blind** | Headline absent, can't localize which of 8 parts is wrong | The internal gate ladder G-A…G-E — never stack past a red gate | §5 |
| Self-deception on the guarantee | Risk controlled only via near-total abstention | Report coverage with risk; pre-register | 4.6, 6.2 |
| Data leakage | Suspiciously perfect target coverage | Assert disjoint cal/weight/OOD/test indices; code review | 4.1, §5 review |
| Combined-weight double-counting | Holds under one shift, breaks when both present | Derive `w(x,y)=w_lab·r` from the anticausal model, don't multiply `w_cov·w_lab` | 3.3b, Stage C |
| Conditional shift breaks label-weight identifiability | `w_lab` passes C1, silently wrong under real `p(x|y)` drift; no estimator fixes it | State `r⊥y` benign as a limitation; G-C2 stress-test; labeled-slice joint-discriminator fallback; sensitivity-bound into budget | 3.3b, 3.6, Stage C |
| Label-shift estimator mistaken for the novelty | Reviewer: "is the win just FMAPLS/GS-B³SE?" | Core = MLLS+BCTS (canonical, coded); exotic 2025 estimators are ablation-only | 4.5 |
| Backbone confound | Gains attributed to SNGP not the contribution | Standard backbone is the base; DDU/SNGP/ensemble are ablations only | §1, 4.5 |
| Weight / label-shift estimation error | In vitro pass, real-data fail | Clip weights; route high-weight tail; MLLS+BCTS over vanilla BBSE under imbalance; stress-test both | 6.2 |
| Detector error leaks silently | "Maintained risk" with a leaky OOD route | Integrated α_ood budget + G-D imperfect-detector test | Stage D |
| Multiple-testing invalidity | RCPS bound used after tuning 3 thresholds | LTT wrapper when jointly tuning | 3.5 |
| Self-normalized (Hájek) bound goes vacuous under strong shift | Paired numerator-UCB/denominator-LCB composition is *valid* but can blow up when the denominator LCB → 0 under heavy weights | v1 ships the paired bound + `n_eff` abstention gate (correct, just conservative). A **direct betting bound on the ratio itself** would be tighter under strong shift but is genuinely new math, not an off-the-shelf result — optional v2 item, only worth deriving if coverage plots show the v1 bound going vacuous too often. Not a correctness issue. | 3.6, post-G-R |
| Dataset access latency | Stuck on MIMIC/CheXpert | Credential in foundations; lead with CAMELYON17 | 4.1 |
| Overclaiming | "Valid under any shift" / "exact under label shift" | Honesty constraint on every guarantee sentence; Reviewer-2 | 0, 7.5 |
| Breadth creep | Chasing ECG/retina/path#2, nothing submitted | (5′) is explicit future work; two datasets, deep | §1 |

---

## Repo map — which repo at which stage

| Stage / component | Repo | Role |
|---|---|---|
| Risk-control spine (RCPS) | [aangelopoulos/rcps](https://github.com/aangelopoulos/rcps) | concentration bounds + λ̂ reference |
| Tighter UCB (WSR) | [gostevehoward/confseq](https://github.com/gostevehoward/confseq) | `betting.py` hedged-capital bound |
| Joint-tuning validity (LTT) | [aangelopoulos/ltt](https://github.com/aangelopoulos/ltt) | multiple-testing wrapper |
| Less-conservative comparison (CRC) | [aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk) | CRC baseline row |
| Covariate weights | [ryantibs/conformal](https://github.com/ryantibs/conformal) | `weighted.quantile` convention |
| Label-shift estimator (core) | [kundajelab/labelshiftexperiments](https://github.com/kundajelab/labelshiftexperiments) | MLLS+BCTS reference (Alexandari 2020) |
| Label-shift estimator (robustness) | RLLS — Azizzadenesheli 2019 | regularized BBSE; ill-conditioning fix |
| Combined-shift model + code | [microsoft/…Generalized-Label-Shift](https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift) | GLS (Tachet 2020) — the §3.3b factorization |
| Joint-shift fallback (labeled slice) | [stanford-futuredata/SparseJointShift](https://github.com/stanford-futuredata/SparseJointShift) | direct `(x,y)` joint-ratio estimation |
| Temporal track (DtACI) | [isgibbs/DtACI](https://github.com/isgibbs/DtACI) | `DtACI.R` abrupt-drift online CP |
| OOD score (primary/ablation) | [mueller-mp/maha-norm](https://github.com/mueller-mp/maha-norm) · [deeplearning-wisc/knn-ood](https://github.com/deeplearning-wisc/knn-ood) | Maha++ / kNN on spectral-norm features |
| Data loaders | [p-lambda/wilds](https://github.com/p-lambda/wilds) | CAMELYON17 |
| Subgroup local coverage (optional) | [rohanhore/RLCP](https://github.com/rohanhore/RLCP) | approx. conditional coverage |
| Competitor to beat | [iamownt/TRUECAM](https://github.com/iamownt/TRUECAM) | reference + the head-to-head baseline |
| Your repo | [Toepatella/calibrated-selective-prediction-clinical-shift](https://github.com/Toepatella/calibrated-selective-prediction-clinical-shift) | the build |

*Full per-repo notes: `repo_links.md`. This document is now self-contained — Primer, paste-in prompts, and reference tables are in the appendices below.*

---

# PRIMER — the five ideas this project composes

> **Who this is for.** You can run Python and follow a tutorial; you know intro probability/averages. Not assumed: dense math notation, proofs, PyTorch fluency. After this you can read every symbol in the doc. You've already built P.3's five ideas as toy notebooks (0.1.1–0.1.6), so treat this as reference, not a blocker.

## P.1 The math you'll actually need
- **Function `f(x)`** — put `x` (an image) in, get scores out.
- **Conditional probability `P(A | B)`** — how often `A` happens *among the cases where `B` is true*. **The key one:** the whole guarantee is about the error rate *among the cases the model chose to answer*.
- **`P_S` vs `P_T`** — the data "pattern" at the **s**ource vs **t**arget hospital. The project is a promise made on `P_S` still holding on `P_T`.
- **Expectation `E[Z]`** — long-run average. `E[ℓ | answered]` = average loss among answered cases.
- **Quantile** — the value below which a given fraction of data sits; conformal uses a high quantile of "wrongness scores" as a cutoff.
- **`α`** — error rate you tolerate (0.05 = 5%). **`δ`** — small allowed chance the *promise itself* fails (confidence `1−δ`). **"Distribution-free"** — the promise holds whatever the data's shape.
- **Symbols:** `≤`/`≥` at most/least · `∝` proportional to · `⌈·⌉` round up · `inf`/`sup` smallest/largest that works · `1[·]` 1-if-true-else-0 · `x̂` an *estimate*. (Full list: **Appendix H**.)

## P.2 The ML/coding you'll actually need
- **Logits → softmax** — raw scores squashed to probabilities summing to 1.
- **Features (embeddings)** — the vector just before the final layer; the OOD detector and the domain discriminator both work on these, not raw pixels.
- **Train / calibration / test splits** — train fits the model; *calibration* is a separate labeled set used only to set thresholds; test is untouched. **No leakage between them is sacred** — most accidental cheating is a leak here.
- **PyTorch in a paragraph** — you mostly *load a frozen pretrained model*, *run it* to get logits + features, and *save those numbers*. The contribution is math on the saved numbers (runs in seconds on CPU). You don't need to be a deep-learning engineer.

## P.3 The five core ideas (plain English, then gentle math)
1. **Conformal prediction — the honesty thermostat.** Convert shaky confidence into a cutoff with a *real* coverage promise: take a high quantile `q̂` of calibration "wrongness scores," keep every answer less surprising than `q̂`. Kept sets contain the truth ≈`1−α` of the time — *if new data resembles calibration data* (**exchangeability**). Delivers condition **(2)**.
2. **Risk control — CRC vs RCPS vs LTT.** All pick a threshold so error ≤ α. **CRC:** controls the *average*. **RCPS:** `1−δ` confident *this* model's error ≤ α — the clinical-grade promise. **LTT:** tune several knobs (`τ,λ,t_ood`) at once and stay valid via multiple-testing. Spine of **(2)**.
3. **Selective prediction — knowing when to say "I don't know."** A threshold `τ` on uncertainty; answer the confident, abstain on the rest. The catch: by answering only easy cases you *bias* the group, so you must set the cutoff *on the answered group itself*. Condition **(1)**.
4. **Distribution shift — the new-hospital problem (two flavors).** **Covariate shift:** `p(x)` changes — reweight calibration by a density ratio `w_cov(x)` from a source-vs-target discriminator. **Label/prevalence shift:** `p(y)` changes (e.g. the target hospital sees more disease) — estimate the new class proportions with **MLLS+BCTS** (calibrate the model first, then EM; more robust than plain BBSE when classes are imbalanced) and reweight by `w_lab(y)`. *Caveat that runs through the whole project: this estimate is only trustworthy if the disease's appearance `p(x|y)` itself hasn't also shifted — see §3.3b.* **ACI/DtACI:** keep nudging the threshold over time as errors arrive. Condition **(3)** — the novel heart.
5. **OOD detection + distance awareness — "not even the right kind of input."** Measure how *far* an input's features sit from the training cloud (Mahalanobis++/kNN); far = OOD → abstain. **Spectral normalization** keeps the feature space honest so "far" means far. Condition **(4)** — and routing the far tail *protects* (3) by removing the mass where weights blow up.

## P.4 How they combine (the project in one paragraph)
A frozen classifier gives **scores** + **features**. **Selective prediction** decides answer-vs-abstain. **Conformal + RCPS** turn each "answer" into a *promised* error rate. **Covariate + label weighting** keeps that promise valid at a new hospital with a different case-mix; **DtACI** handles drift over time. **OOD routing** discards inputs too strange to promise anything about — and, crucially, its own error is folded into the *same* budget (`α = α_acc + α_ood`) rather than bolted on. The contribution is doing all of this at once with the promise **provably surviving real covariate *and* label shift**, demonstrated deeply on two public medical benchmarks.

## P.5 Optional skill-ramp (search for current free versions)
- **PyTorch:** the official "60-Minute Blitz"; fast.ai.
- **Math intuition:** 3Blue1Brown; StatQuest.
- **Conformal:** Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* — the single best on-ramp.
- **Shift + risk control for this project:** Tibshirani et al. 2019 (weighted), Lipton et al. 2018 (BBSE), Bates et al. 2021 (RCPS), Waudby-Smith & Ramdas (betting bound), Gibbs & Candès 2022 (DtACI).

---

# Appendix A — Curated reading list

**Conformal & risk control**
- Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* (2005) — foundations.
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* — the on-ramp.
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* (ICLR 2024, [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)) — expectation control; CRC comparison backbone.
- Bates, Angelopoulos, Lei, Malik, Jordan — *Distribution-Free, Risk-Controlling Prediction Sets* (JACM 2021, [arXiv:2101.02703](https://arxiv.org/abs/2101.02703)) — **RCPS; the clinical-safety spine.**
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test* (AOAS 2025, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)) — multi-threshold risk control.
- Waudby-Smith & Ramdas — *Estimating Means of Bounded Random Variables by Betting* (JRSS-B 2024, [arXiv:2010.09686](https://arxiv.org/abs/2010.09686)) — **the WSR/hedged-capital upper bound used in weighted RCPS.**

**Conformal under shift**
- Tibshirani, Foygel Barber, Candès, Ramdas — *Conformal Prediction Under Covariate Shift* (NeurIPS 2019, [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)) — weighted conformal.
- Podkopaev & Ramdas — *Distribution-Free Uncertainty Quantification for Classification Under Label Shift* (UAI 2021, [arXiv:2103.03323](https://arxiv.org/abs/2103.03323)) — **the conformal-native label-shift treatment; the glue between the `w_lab` estimator and the weighted-conformal layer.**
- Wang & Qiao — *Conformal Prediction Under Generalized Covariate Shift with Posterior Drift* (2025, [arXiv:2502.17744](https://arxiv.org/abs/2502.17744)) — **nearest published conformal treatment of combined shift; mandatory baseline (§4.4). Coverage only — you add risk control + selection + integrated OOD.**
- Gibbs & Candès — *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021, [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)) — online ACI.
- Gibbs & Candès — *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts* (JMLR 2024, [arXiv:2208.08401](https://arxiv.org/abs/2208.08401)) — **DtACI (Stage E).**

**Label-shift estimation (Stage C — the `w_lab(y)` module)**
- Lipton, Wang, Smola — *Detecting and Correcting for Label Shift (BBSE)* (ICML 2018, [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)) — the foil / baseline estimator.
- Azizzadenesheli, Liu, Yang, Anandkumar — *Regularized Learning for Domain Adaptation under Label Shifts (RLLS)* (ICLR 2019, [arXiv:1903.09734](https://arxiv.org/abs/1903.09734)) — **regularized BBSE; the ill-conditioning fix under imbalance (robustness companion).**
- Alexandari, Kundaje, Shrikumar — *Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation (MLLS+BCTS)* (ICML 2020, [arXiv:1901.06852](https://arxiv.org/abs/1901.06852)) — **the core estimator; calibrate first, then EM.**
- Garg, Wu, Balakrishnan, Lipton — *A Unified View of Label Shift Estimation* (NeurIPS 2020, [arXiv:2003.07554](https://arxiv.org/abs/2003.07554)) — **MLLS≡BBSE under different calibration; the real lever is calibration.**
- Kimura — *Graph-Smoothed Bayesian Black-Box Shift Estimator (GS-B³SE)* (2025, [arXiv:2505.16251](https://arxiv.org/abs/2505.16251)) — *ablation-only; still needs the confusion matrix (graph-prior regularized, HMC/Newton-CG); no released code.*
- Hu & Barria — *Bayesian Online Label Shift Estimation with Dynamic Dirichlet Priors (FMAPLS)* (2025, [arXiv:2511.18615](https://arxiv.org/abs/2511.18615)) — *ablation-only; no matrix inversion, built for long-tailed imbalance, has an online variant; no released code.*

**Combined covariate + label shift (the §3.3b model)**
- Zhang, Schölkopf, Muandet, Wang — *Domain Adaptation under Target and Conditional Shift* (ICML 2013) — **Generalized Target Shift; the original combined model.**
- Gong, Zhang, Liu, Tao, Glymour, Schölkopf — *Domain Adaptation with Conditional Transferable Components* (ICML 2016) — what's identifiable under combined shift.
- Tachet des Combes, Zhao, Wang, Gordon — *Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift (GLS)* (NeurIPS 2020, [arXiv:2003.04475](https://arxiv.org/abs/2003.04475)) — **modern GLS formalization; has Microsoft reference code.**
- Schölkopf, Janzing, Peters, Sgouritsa, Zhang, Mooij — *On Causal and Anticausal Learning* (ICML 2012, [arXiv:1206.6471](https://arxiv.org/abs/1206.6471)) — the `Y→X` framing that fixes the medical-imaging factorization.

**Selective prediction**
- Geifman & El-Yaniv — *Selective Classification for DNNs* (NeurIPS 2017); *SelectiveNet* (ICML 2019, [arXiv:1901.09192](https://arxiv.org/abs/1901.09192)).
- Jones et al. — *Selective Classification Can Magnify Disparities Across Groups* (ICLR 2021, [arXiv:2102.11203](https://arxiv.org/abs/2102.11203)) — **the subgroup-audit motivation (Stage S).**

**OOD detection & deterministic UQ**
- Lee et al. — Mahalanobis OOD (NeurIPS 2018, [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)); Müller et al. — *Mahalanobis++* (2025, the L2-norm fix); Sun et al. — *kNN OOD* (ICML 2022, [arXiv:2204.06507](https://arxiv.org/abs/2204.06507)).
- Liu et al. — *SNGP* (NeurIPS 2020); Mukhoti et al. — *DDU*; Ovadia et al. — *Can You Trust Your Model's Uncertainty?* (NeurIPS 2019, [arXiv:1906.02530](https://arxiv.org/abs/1906.02530)) — ensembles as ceiling.

**Local / conditional coverage (Stage S, optional)**
- Hore & Barber — *Conformal Prediction with Local Weights: Randomization Enables Robust Guarantees* (RLCP, [arXiv:2310.07850](https://arxiv.org/abs/2310.07850)).

**Datasets & calibration**
- Koh et al. — *WILDS* (ICML 2021); Bandi et al. — *CAMELYON17* (IEEE TMI 2018).
- Irvin et al. — *CheXpert* (AAAI 2019); Johnson et al. — *MIMIC-CXR* (Sci Data 2019).
- Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017) — ECE.

**Competitors (from the audit — beat or cite):** TRUECAM ([arXiv:2501.00053](https://arxiv.org/abs/2501.00053)), SCRC, SCoRE, Conformal Triage, Kim 2026 (Sci Rep), Liang & Sun (TMLR 2024, [arXiv:2405.05160](https://arxiv.org/abs/2405.05160)), the "Pitfalls" critique ([arXiv:2506.18162](https://arxiv.org/abs/2506.18162)).

---

# Appendix B — Dataset cheat-sheet

**v1 (build deep — both modalities, both shift mechanisms):**

| Dataset | Modality | Shift | Access | Friction |
|---|---|---|---|---|
| CAMELYON17-WILDS | Histopathology | 5 hospitals/scanners (covariate) | `wilds` package, open | **Low — start here** |
| CheXpert | Chest X-ray | Institution/population (covariate + label) | Stanford AIMI registration | Medium |
| MIMIC-CXR | Chest X-ray | Institution/population (covariate + label) | PhysioNet credentialed + CITI | Medium–High — **start early** |

> Both CXR directions (CheXpert→MIMIC and MIMIC→CheXpert) give two shift instances cheaply. Label/prevalence shift is estimated per dataset via BBSE. Begin MIMIC/CheXpert credentialing during foundations — human approval is the long pole; CAMELYON17 unblocks the build immediately.

**Future-work breadth (condition 5′ — stated as the open extension, *not* a v1 obligation):** PTB-XL (ECG, cross-cohort), EyePACS↔APTOS (retina, population), MIDOG (2nd pathology scanner-shift), ISIC↔Fitzpatrick17k (dermatology, skin-tone). **Far-OOD probes:** off-modality / wrong-anatomy / non-medical injections to test (4) + the α_ood budget.

---

# Appendix C — Glossary (one line each)

- **Exchangeability** — joint distribution invariant to ordering; split-conformal's core assumption. Every failure here is an exchangeability violation.
- **Marginal coverage** — holds on average over the random cal+test draw, not conditional on a given `x`.
- **CRC** — calibrate a threshold so the *expected* bounded loss ≤ α (expectation over the calibration draw too).
- **RCPS** — PAC risk control: `P(R ≤ α) ≥ 1−δ` for your one realized calibration set/model — the stronger clinical statement.
- **LTT** — risk control for non-monotone losses and multiple thresholds via multiple hypothesis testing (keeps the bound valid when you jointly tune `τ,λ,t_ood`).
- **WSR / betting (hedged-capital) bound** — a tight, variance-adaptive upper confidence bound; replaces Hoeffding–Bentkus in weighted RCPS where weight-variance inflates the CI.
- **Selective risk / coverage** — error among answered / fraction answered; the risk–coverage curve / AURC summarizes the tradeoff.
- **Weighted conformal** — reweight calibration by a density ratio to transport the guarantee under shift.
- **Covariate vs label shift** — `p(x)` changes (label rule fixed) vs `p(y)` changes (case-mix); different importance weights.
- **Density ratio `w_cov(x)`** — `p_T(x)/p_S(x)`; from a source-vs-target discriminator; unreliable in the far tail → route to abstain.
- **Label weight `w_lab(y)`** — `p_T(y)/p_S(y)`; from BBSE-estimated target proportions.
- **BBSE (Black-Box Shift Estimation)** — estimates target label proportions `p_T(y)` from the source confusion matrix + target predictions, no target labels needed.
- **Integrated OOD budget** — split total `α = α_acc + α_ood`: accepted in-distribution error + a certified bound on residual far-OOD that the detector missed. Makes OOD part of the guarantee, not a bolt-on.
- **ACI / DtACI** — online α adjustment for long-run coverage under drift; needs label feedback; DtACI auto-tunes the step size for abrupt drift. Long-run (not per-case finite-sample) control.
- **Far-OOD** — inputs unlike training; extreme shift tail where weighting fails; the abstain-route is both safety and math.
- **Distance-aware UQ (SNGP/DDU)** — single-pass models whose uncertainty grows with feature-distance; improves selection/OOD efficiency, not validity.
- **Spectral normalization** — constrains layer Lipschitz constants → approx. bi-Lipschitz features → feature-distance OOD becomes meaningful (resists feature collapse).
- **RLCP (randomly localized conformal)** — local-weight + randomization for approximately *conditional* coverage; optional Stage-S strengthening of the subgroup audit.

---

# Appendix D — All paste-in prompts (updated for the flagship scope)

> Drop into a search-enabled assistant as a second brain. All reflect the v1 scope: covariate **+** label shift, integrated OOD budget, WSR bound, LTT, two-datasets-deep, journal venue.

**D-1 · Concept tutor / steelman** — §0.1 / PRIMER
```
You are a conformal-prediction and selective-classification expert. I'm building a
selective medical classifier with a distribution-free risk guarantee on accepted cases
that must survive covariate AND label/prevalence shift, with far-OOD inputs routed to
abstain and the routing error folded into the SAME risk budget.

Teach me, precisely and with exact theorem assumptions, how these compose:
(1) split conformal + the risk-control layer — CRC vs RCPS vs Learn-then-Test (expectation
    vs PAC (alpha,delta) vs multi-threshold control; assumption; finite-sample vs asymptotic),
    and why the WSR/betting upper bound is tighter than Hoeffding-Bentkus under importance
    weighting;
(2) selective prediction and WHY selection breaks exchangeability, and how selective risk
    control restores a guarantee on the accepted subset;
(3) weighted conformal for COVARIATE shift AND BBSE/label-shift correction — what each needs,
    how to COMBINE the two importance weights without double-counting, and exactly where each
    fails;
(4) integrating far-OOD routing into the risk budget: splitting alpha into accepted-error and
    OOD-leakage, and bounding the residual when the detector is imperfect.

For each: the precise quantity guaranteed, the assumption it rests on, and the single most
common way practitioners violate it without noticing. Then give me three exam questions and
grade my answers. Cite real papers with links; flag anything you're unsure exists.
```

**D-2 · Falsification re-check** — §2.b
```
ROLE: Meticulous research-gap auditor for trustworthy ML in healthcare. Use live literature
search. Default to skepticism; actively try to prove the gap below is ALREADY SOLVED.

GAP: A selective-prediction method for medical models that simultaneously (1) abstains/defers;
(2) gives a distribution-free guarantee on the error rate of ACCEPTED cases; (3) keeps that
guarantee approximately valid under REAL clinical COVARIATE AND LABEL/PREVALENCE shift; (4)
routes far-OOD inputs to abstain WITH THAT ROUTING INSIDE THE CERTIFIED RISK BUDGET (not a
separate heuristic), evaluated on public medical shift benchmarks.

CLOSED only if ONE work demonstrably does ALL of: abstains; distribution-free error/coverage
guarantee on accepted cases; that guarantee holds under a real (not synthetic) medical shift
that includes label/prevalence change, not only covariate change; far-OOD routed to abstain
with the detector error accounted for in the guarantee; evaluated on >=1 public medical shift
dataset.

FOCUS: 2025-06 to today. Prioritize MICCAI, MLHC, CHIL, ML4H, NeurIPS/ICML/ICLR, TMLR, Medical
Image Analysis, IEEE TMI, npj Digital Medicine. Check explicitly: TRUECAM, SCRC, SCoRE,
Conformal Triage, Kim 2026, weighted/adaptive conformal in medicine, any "selective conformal
under shift" or "label-shift conformal risk control" work.

OUTPUT: Verdict OPEN / PARTIALLY ADDRESSED / CLOSED + one-line justification; an evidence table
(Title|Year|Venue|Link|which of (1)-(4) satisfied|what's missing); the closest competitor and
exactly what it leaves unsolved; the precise residual contribution still novel. Cite every
claim with a working link + date; separate peer-reviewed from preprints; if a search fails,
say so rather than guessing.
```

**D-3 · Method red-team / proof check** — §3.7
```
You are a conformal-prediction theorist and Reviewer 2. Below is my method and its claimed
guarantee [PASTE notation, target guarantee R_T^accept <= alpha = alpha_acc + alpha_ood,
algorithm, assumptions].

Do five things, harshly:
1. State the exact conditions under which the accepted-case risk <= alpha holds. Finite-sample
   or asymptotic? Marginal over what randomness?
2. Find every place selection, covariate shift, LABEL shift, or far-OOD routing breaks
   exchangeability that I have NOT neutralized. Give a concrete counterexample for each hole.
3. Pressure-test the COMBINED covariate+label importance weight: is it identifiable? When does
   naive multiplication w_cov*w_lab double-count? What structural shift model makes it valid?
4. Pressure-test the INTEGRATED OOD budget: does alpha_ood actually bound the residual risk
   from an IMPERFECT detector, or does it leak? Prove or refute. Also: is the WSR-bound +
   weighting combination valid, and does LTT correctly cover my joint (tau, lambda, t_ood)
   tuning, or have I invalidated the bound with multiple looks?
5. Name the single weakest claim a reviewer attacks first, and the smallest change that makes
   it defensible. Cite the exact theorems (weighted conformal, BBSE, RCPS/LTT, WSR betting,
   selective conformal), with links. Do not be reassuring; if the guarantee is only
   "approximate," characterize the error as a function of weight + BBSE + detector error.
```

**D-4 · Experiment-design critic** — §4.7
```
You are an experienced ML-for-health reviewer. Critique this experimental plan for a
selective-prediction-under-shift JOURNAL paper [PASTE: two imaging datasets taken deep,
covariate + label shift, integrated-OOD-budget, temporal DtACI, metrics, baselines (incl. a
TRUECAM-style empirical pipeline), base-model ablation, success criterion].

Tell me: (a) which baseline is missing such that a reviewer says "but did you compare to X?";
(b) whether my metrics demonstrate the (2)+(3)+(4-integrated) guarantee or just correlate with
it — especially: does my plot actually show the integrated budget holding under a DELIBERATELY
imperfect OOD detector, and label-aware weighting beating covariate-only under prevalence
shift?; (c) the right way to show a finite-sample, calibration-draw-marginal guarantee
empirically (how many splits, what plot, what statistic); (d) the most likely confound (base
model just worse on target; gains from the backbone not the guarantee layer) and how to rule
it out via the {standard, DDU, SNGP, ensemble} ablation; (e) given I deliberately chose DEPTH
on two imaging datasets over breadth, is that defensible for a top medical-imaging/ML journal,
and if a reviewer demands breadth, which single extra modality is the cheapest credible
rebuttal?
```

**D-5 · Conformal code review** — §5 (code review)
```
You are a careful research engineer who has shipped conformal-prediction code. Review this
implementation of selective + weighted (covariate+label) conformal risk control + integrated
OOD routing [PASTE code]. Check specifically:
- off-by-one in the conformal quantile / CRC threshold / RCPS lambda-hat;
- whether the WSR (betting / hedged-capital) upper confidence bound is implemented correctly
  and is actually being used in the weighted path (not silently falling back to Hoeffding);
- data leakage between calibration, weight-fitting, BBSE estimation, OOD-fitting, and test;
- whether covariate weights AND label weights are normalized correctly and COMBINED without
  double-counting; whether clipping biases the risk estimate (log the clip fraction);
- whether the alpha split (alpha_acc + alpha_ood) is enforced and the OOD-leakage term is
  actually subtracted from the budget, not ignored;
- whether selection is calibrated on the ACCEPTED region, and whether LTT multiple-testing
  wraps the joint (tau, lambda, t_ood) search.
Then propose the minimal synthetic-data unit tests that PROVE each guarantee (G-A exchangeable,
G-B covariate, G-C label, G-D imperfect-detector), with exact expected numbers so I know if a
test passes.
```

**D-6 · Results red-team** — §6.4
```
You are a skeptical senior co-author. Here are my results [PASTE tables/figures/setup]. Try to
explain my positive result WITHOUT my method being correct: confounds, abstention inflation,
leakage, lucky splits, weak baselines, cherry-picked alpha, dataset quirks, the base model
just being worse on target, or the win coming from the backbone rather than the guarantee
layer. Specifically attack: (i) is the "label-shift" win just covariate weighting in disguise?
(ii) does the "integrated budget" actually hold when the OOD detector is imperfect, or did I
test it only with a near-perfect detector? (iii) is validity bought by near-total abstention?
For each alternative explanation, give the exact additional analysis or plot that rules it out.
Then tell me which claims the data supports vs. which I'm overstating, and rewrite my central
claim sentence to match the evidence.
```

**D-7 · Reviewer 2 simulation** — §7.5
```
You are three reviewers for [JOURNAL: Medical Image Analysis / IEEE TMI / npj Digital Medicine
/ TMLR]: a conformal theorist, a clinical-ML practitioner, and a skeptical generalist. Here is
my paper draft [PASTE]. Each writes: summary, strengths, weaknesses, questions, score, and the
one change that would most raise it. The theorist checks the guarantee's assumptions, the
combined covariate+label weight identifiability, the integrated-OOD-budget bound and its
explicit constants, and whether the empirics demonstrate them. The clinician asks whether the
abstention behavior is usable in a real workflow, whether the shifts are clinically realistic,
and whether the subgroup audit is adequate. The generalist hunts for overclaiming, missing
baselines, and whether two datasets suffice or breadth is needed. End with the top 3 revisions,
ranked by impact.
```

**D-8 · Flagship gap-coverage audit** — §7.6
```
You are auditing whether my results close the original gap. The gap requires a single method
that: (1) abstains; (2) gives a distribution-free guarantee on accepted cases; (3) holds under
real covariate AND label/prevalence shift; (4) routes far-OOD to abstain WITH that step inside
the guarantee; (5) is validated on real public medical shift data; (5') across multiple diverse
modalities. Here are my results [PASTE]. For each of (1)-(5'), state SATISFIED / PARTIAL / NOT,
with exact evidence and the exact residual gap. I deliberately scoped (5') to future work (two
imaging datasets, deep) — confirm whether (1)-(5) are fully closed on those two, and write the
most honest one-paragraph "what remains open" statement for my discussion (naming multi-modality
breadth, arbitrary-shift limits, and BBSE identifiability as the open edges).
```

---

# Appendix E — Repo tree reference

```
calibrated-selective-prediction-clinical-shift/
├── README.md                 # overview + exact data-access steps
├── requirements.txt / env    # pinned deps + lockfile (incl. confseq, wilds)
├── configs/                  # one YAML per experiment (Hydra)
├── data/
│   ├── camelyon17.py         # WILDS loaders + split manifest
│   ├── cxr.py                # CheXpert/MIMIC loaders + harmonization
│   └── splits/               # frozen index files (cal/test disjoint)
├── models/                   # backbones, training, spectral-norm option
├── ood/
│   └── detector.py           # Mahalanobis++ / kNN on spectral-norm features
├── conformal/
│   ├── selective.py          # selection + accepted-region calibration
│   ├── rcps.py               # RCPS + WSR (betting) bound + weighted path
│   ├── crc.py                # CRC comparison backbone
│   ├── ltt.py                # Learn-then-Test multiple-testing wrapper
│   ├── weights.py            # domain discriminator + clipped covariate ratio
│   ├── label_shift.py        # BBSE / EM target-proportion estimation
│   ├── budget.py             # integrated alpha = alpha_acc + alpha_ood split
│   └── aci.py                # DtACI adaptive conformal (Stage E)
├── experiments/              # run scripts (method × dataset × seed)
├── analysis/
│   ├── metrics.py            # risk, coverage, AURC, ECE, OOD AUROC, subgroup
│   ├── figures.py            # headline + integrated-budget + frontier + change-point
│   └── competitor_matrix.csv
├── tests/                    # synthetic gates G-A..G-E (pytest)
├── docs/                     # positioning_memo, method_note, design, preregistration
└── paper/                    # manuscript + figures
```

---

# Appendix F — Notation & key formulas (quick reference)

**Notation:** `P_S,P_T` source/target · `f` frozen base model · `ŷ(x)` prediction · `u(x)` uncertainty · `g(x)=1[u(x)≤τ]` selection · `ℓ∈[0,1]` loss · `w_cov(x)=p_T(x)/p_S(x)` covariate ratio · `w_lab(y)=p_T(y)/p_S(y)` label ratio · `w(x,y)` combined weight · `o(x)` OOD score · `τ,λ,t_ood` thresholds · `α=α_acc+α_ood` budget · `δ` RCPS confidence · `m` target sample size.

- **Split-conformal quantile:** `q̂ = ⌈(n+1)(1−α)⌉`-th smallest calibration score. Marginal coverage ≥ `1−α` under exchangeability.
- **CRC:** choose `λ̂` so `E[ℓ(λ̂)] ≤ α` (monotone bounded loss; expectation includes the calibration draw).
- **RCPS:** `λ̂ = inf{λ : UCB_δ(R(λ′)) < α ∀ λ′≥λ}` ⇒ `P(R(λ̂) ≤ α) ≥ 1−δ`. Use the **WSR betting UCB** (tighter than Hoeffding–Bentkus, variance-adaptive — matters under weighting).
- **Weighted conformal (covariate):** reweight calibration point `i` by `w_cov(x_i)/Σ_j w_cov(x_j)` (+ test-point term); transports coverage under covariate shift.
- **Density ratio from a discriminator:** `w_cov(x) ∝ d(x)/(1−d(x))`, `d=`P(target|x); clip at `w_max`.
- **BBSE (label shift):** estimate `p̂_T(y)` by solving `Ĉ_S · p̂_T(y) = q̂_T`, where `Ĉ_S` = source confusion matrix and `q̂_T` = distribution of `f`'s predictions on the target sample; then `w_lab(y)=p̂_T(y)/p_S(y)`.
- **Combined weight:** under the stated shift model, fold `w_cov` and `w_lab` into one `w(x,y)` — derive it, don't multiply naïvely (double-counts when both shifts coexist).
- **Selective risk / coverage:** `risk = E[ℓ | g=1]`, `coverage = P(g=1)`; risk–coverage curve / AURC summarizes the tradeoff.
- **Integrated target guarantee:** `E_{P_T}[ ℓ(Y,ŷ(X)) | g(X)=1, o(X)≤t_ood ] ≤ α_acc`, with residual far-OOD leakage bounded by `α_ood`, so answered-case risk `≤ α = α_acc + α_ood`.

---

# Appendix G — Environment & command cheat-sheet

```bash
# env
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision wilds numpy scipy scikit-learn pandas matplotlib wandb pytest confseq

# data (confirm current WILDS API)
python -c "from wilds import get_dataset; get_dataset('camelyon17', download=True)"

# run an experiment from a config
python -m experiments.run --config configs/camelyon17_ours.yaml seed=0

# the synthetic gates (must pass IN ORDER before real data)
pytest tests/ -k "G_A_exchangeable or G_B_covariate or G_C_label or G_D_imperfect_ood or G_E_changepoint" -v

# regenerate every figure/table from logged runs
python -m analysis.figures --from-logs runs/ --out paper/figures/
```

> **Compute note (single GPU):** cache features once (Stage R prep) so conformal calibration reruns in seconds. Deep-ensemble baselines = K sequential trainings — schedule overnight. WSR/BBSE/budget math runs on cached numbers on CPU, fast.

---

# Appendix H — Math & notation decoder (plain meanings)

*Keep open while reading. Every symbol the doc uses, in plain words.*

**Core symbols**
- `x`, `y` — an input (image) and its true label.
- `f`, `f(x)` — the frozen model, and the scores it outputs.
- `ŷ(x)` — the model's predicted label. `x̂` (a "hat") — an *estimate* from data, not the truth.
- `P_S`, `P_T` — the data pattern at **s**ource vs **t**arget hospital.

**Probability & averages**
- `P(A | B)` — probability of `A` given `B` ("among the `B` cases").
- `E[Z]` — long-run average. `E[ℓ | g=1]` — average loss among answered cases.

**Knobs and budgets**
- `α` — error rate you allow; here split as `α = α_acc + α_ood`.
- `α_acc` — budget for accepted in-distribution error. `α_ood` — budget for far-OOD the detector missed.
- `δ` — small chance the *promise* fails; RCPS confidence `1−δ`.
- `ℓ(y,ŷ)` — loss, scaled to `[0,1]`. `u(x)` — uncertainty (high = unsure).
- `g(x)=1[u(x)≤τ]` — answer (1) if uncertainty below `τ`, else abstain. `τ` — selection threshold.
- `λ` — the risk-control threshold tuned on calibration. `t_ood` — OOD cutoff; above it, abstain.
- `m` — how many unlabeled target points you have to calibrate the shift correction.

**Shift & OOD**
- `w_cov(x)` — covariate density ratio `p_T(x)/p_S(x)`: how *target-like* `x` looks.
- `w_lab(y)` — label ratio `p_T(y)/p_S(y)`: how the *case-mix* changed (from BBSE).
- `w(x,y)` — the combined importance weight under the stated shift model.
- `o(x)` — OOD score: how *unlike training* `x` is (a feature-space distance).

**Operators**
- `≤`/`≥` at most/least · `∝` proportional to · `⌈·⌉` round up · `inf`/`sup` smallest/largest meeting a condition · `1[·]` 1-if-true-else-0 · `Σ` sum.

**Concepts in two words**
- **exchangeable** (≈ i.i.d. here) — points "drawn the same way," order doesn't matter; the resemblance conformal needs.
- **distribution-free** — the promise holds regardless of the data's shape.
- **finite-sample vs asymptotic** — holds with the data you have vs only as data → ∞.
- **PAC** — "probably approximately correct": with prob ≥ `1−δ`, error ≤ `α` (RCPS's promise).
- **BBSE** — black-box estimate of the target class proportions from the confusion matrix + target predictions.
- **integrated budget** — OOD-detector error is *inside* `α`, not a separate heuristic.
- **risk–coverage curve / AURC** — error vs how much you answer; AURC summarizes (lower = better).
- **ECE** — expected calibration error: how honest the raw probabilities are.
- **bi-Lipschitz** — a mapping that doesn't squash distances, so "far" stays far (why spectral norm helps OOD).

---

*End of flagship playbook. Work top to bottom; honor the gate ladder G-A→G-R; run each relevant Appendix-D prompt against your actual artifact before calling a stage done.*
