# Adversarial Stress-Test Findings — Applied Trustworthy Selective-Prediction Paper

*Output of the 6-lens adversarial reviewer panel (`docs/stress_test_prompt.md`) run against the current design + preregistration package on **2026-06-27**.*

**Artifact reviewed (pre-results):** `docs/method_note.md`, `docs/positioning_memo.md`, `docs/preregistration.md`, `docs/explainability_framing.md`, `flagship-playbook.md`, `analysis/competitor_matrix.csv`. No experiments have run (`analysis/metrics.py`/`figures.py` are stubs), so all attacks target the **design, protocol, and positioning** as they would be carried into submission.

**Panel:** R1 clinical realism · R2 reproducibility/confounds · R3 venue/explainability · R4 honesty/novelty · R5 scope/feasibility · R6 measurement soundness. Each lens attacked in parallel; an independent skeptic verified every finding and fix; a chair deduped across lenses and ranked by blocking-likelihood.

**Run stats:** 13 agents · 40 findings → **37 VALID, 3 discarded as out-of-scope** · ~504 s.

**Verdict: REVISE** — no fatal design flaw and no smuggled guarantee; risks are two confound-control gaps + execution/scheduling discipline, all closable before 2026-10-05.

---

## 1. Prioritized weakness list

*Ranked by likelihood of blocking acceptance; lenses that raised each are tagged; cross-lens convergence raised confidence.*

### 🔴 Blocking

**1. [Blocking · R2+R6] No matched-coverage comparison is pre-registered.** The pipeline's three routing mechanisms (`w_max`, `t_ood`, `τ`) hand the baselines a strictly larger, dirtier accepted set, so any headline accepted-risk win at the gate↔baseline seam is readable as *bought by abstaining more*, not by shift correction.
- *Why it blocks:* The single most damaging cheap-confound reading for a conformal-literate referee; `coverage_min` (a degeneracy floor) and coverage-beside-risk reporting do **not** rule it out, so the whole quantitative contribution gets discounted.
- *Fix:* Pre-register a **matched-total-answer-rate** protocol — operate baselines and pipeline at equal answer-rate for every headline cell, print the 3-way routing decomposition beside every number, report **AURC** as the threshold-free companion, and add frozen-`f` raw target/source error as the base-rate anchor.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (re-run on cached logits). Caveat: match is approximate across a routing vs routing-free harness — pre-register it as matched-*total*-answer-rate with the decomposition shown.

**2. [Blocking · R3] The accept-side trust signals have zero quantitative validation.** The representativeness chip, `n_eff` regime, and shift-regime badge — the interface that *is* the contribution — are never measured: prereg §5.5 checks A–F all operate on the *decline* record, and the only answered-case route (check F) is IRB-gated and may miss the deadline.
- *Why it blocks:* For a trust-interface paper at an XAI venue, the headline accept-side signal a clinician acts on is asserted from design and never measured — and this is **not disclosed anywhere** (§8 concedes no `R_T^accept` certificate, but not that the answered-case signals are unvalidated), so the interface leg reads as unsupported.
- *Fix:* Add a no-IRB accept-side faithfulness check to §5.5 — bin **answered** in-scope cases by chip level / `n_eff` regime / shift-regime badge and report measured accepted error per bin with betting/Hájek intervals, pre-registering that heavily-up-weighted / low-`n_eff` / combined-regime bins show higher realized error. Bin on the larger labeled test split to mitigate `D_tar^lab` thinness.
- *Fix verified:* in-scope ✓ (asserts no per-case guarantee) · closes ✓ (converts the interface from decoration to a measured deliverable) · feasible ✓ (reuses computed quantities). Caveat: thin bins → wide intervals per §5.3.

### 🟠 Major

**3. [Major · R2] The label-aware-beats-covariate-only claim is confounded on real data.** On the mixed benchmark prevalence is moved by label-subsampling the target, which also moves the covariate marginal, so a real-data Z-combined win *with covariate shift held constant* is not forced by the protocol.
- *Why it blocks:* One of the two load-bearing measurement claims; if it's demonstrable only on synthetic data or a co-moved target, "label-aware weighting earns its keep" is dismissable and the composition novelty thins.
- *Fix:* Pre-register prevalence-**reweighting** (not subsampling) across the fixed §3.1 grid as the decisive real-data panel, **tag each cell with its discriminator-AUROC** to show covariate shift held ~constant while `p_T(y)` varies, and pre-commit a downgrade to "shown on synthetic + prevalence-reweighted only" if separation doesn't exceed the betting intervals.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (pre-results). Caveat: reweighting can't perfectly hold the realized covariate marginal fixed — the AUROC tag is the honest disclosure; mixed-benchmark-credentialing-gated.

**4. [Major · R2] The OOD-cuts-leakage claim risks being shown only where it's trivially true.** The exposure set `O` defaults to far-OOD, yet the docs themselves (§7.1) name **near-OOD** as the regime where the cited impossibility (Fang) actually bites.
- *Why it blocks:* A reviewer dismisses the screen as untested exactly where it matters; the honest TRUECAM contrast rests on quantifying residual leakage in the *hard* regime.
- *Fix:* Pre-register a **two-tier exposure set** (far-OOD **and** near-OOD: held-out scanner/site or wrong-but-adjacent anatomy), report AUROC/FPR95 and leakage vs `α_ood` per tier and benchmark, run the TRUECAM head-to-head on the **same** `O`, and state near-OOD leakage is reported-as-worse, never certified.
- *Fix verified:* in-scope ✓ (measuring near-OOD leakage ≠ a near-OOD guarantee) · closes ✓ · feasible ✓ from CAMELYON17 alone (cross-hospital = built-in near-OOD); CXR tier credentialing-gated. Caveat: draw the near-OOD tier disjoint from test/`D_tar^lab`.

**5. [Major · R6] The headline interval's validity is unbacked at the WSR-UCB↔Hájek-ratio seam.** The betting UCB bounds the *mean* of bounded iid variables but is invoked on a self-normalized *ratio* with a random denominator and estimated weights — with no derivation/citation that `1−δ` coverage transfers and no empirical interval-coverage check.
- *Why it blocks:* For a measure-and-report paper the headline deliverable is a defensible number with an honest interval; the self-normalized form may under-cover. (Major not Blocking — it doesn't revive a guarantee.)
- *Fix:* Pre-register an **empirical interval-coverage check on synthetic data** with known ground-truth weighted risk and weights — report realized coverage of the WSR Hájek interval vs nominal `1−δ` across `n_eff` regimes; if it under-covers, relabel as "nominal interval, coverage validated empirically."
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓✓ — **purely synthetic, NOT gated on credentialing** (do this early).

**6. [Major · R6] A favorable headline risk could be an operating-point artifact via two un-swept knobs.** The `w_max` clip/route biases the *retained* weighted risk by trimming the heavy tail, and the `σ̃`/`Ẑ` plug-in denominator carries a non-vanishing bias amplified on rare/high-`w_lab` classes — neither is in the pre-registered stress axes.
- *Why it blocks:* The combined-shift headline divides every accepted-case weight by `Ẑ` and routes at `w_max`; without sweeps a reviewer argues the low number was bought by tail-trimming or a biased denominator.
- *Fix:* Promote a **`w_max` sweep** into the pre-registered stress axes (report `R̂_w` + interval + `n_eff` across a fixed grid; non-flat cells → failure-mode catalog) **and** add a **`Ẑ`-sensitivity** report (`R̂_w` under raw-softmax vs BCTS vs a temperature-perturbation band, flagging rare/high-`w_lab` classes). Fix the grids now.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (re-runs on existing folds). *(Couples with rank 16.)*

**7. [Major · R2+R1] Fold disjointness is asserted "in code" but the unit is unfixed.** CAMELYON17 patches share WSI slides and CheXpert/MIMIC images share patients, so a slide/patient can land in both `D_cal` and test while row indices stay disjoint — no leakage probe bounds the optimism.
- *Why it blocks:* Patch/image-level splitting is a known clinical-imaging inflation route; a reviewer suspecting group leakage discounts the whole realized-risk table.
- *Fix:* Pre-register disjointness at the **group id** (WSI slide id; patient id) with a printed set-intersection = 0 artifact per fold pair, and add one deliberate **leakage-probe run** (inject a known patient into `D_cal`) reporting how much realized risk moves.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (CAMELYON17-WILDS ships grouped splits). Caveat: the probe quantifies optimism, doesn't bound it — frame as a sensitivity number.

**8. [Major · R2+R1] The domain discriminator is never probed for keying on an acquisition shortcut.** `d(φ(x))` drives the covariate weight, the per-pair AUROC, the regime badge, *and* the clinician-facing representativeness chip — but is never checked for keying on burned-in text, padding, laterality markers, or intensity/window rather than clinical appearance.
- *Why it blocks:* If `d` keys on a scanner artifact, every downstream audit object and the point-of-care chip audit the wrong thing; a trustworthy-ML reviewer treats a shortcut-driven trust interface as actively misleading.
- *Fix:* Pre-register a **discriminator-shortcut audit** — report `d`-AUROC before vs after standard de-shortcutting (crop/mask burned-in text and borders, intensity-normalize), inspect top-weighted cases for artifacts, and caveat the chip/regime-tag where AUROC collapses after de-shortcutting.
- *Fix verified:* in-scope ✓ (diagnostic on an already-computed artifact; `f` stays frozen) · closes ✓ · feasible ✓ (CXR text/laterality de-shortcutting is the cheap credible version).

**9. [Major · R6] "Marginal over what?" is unspecified for `δ`.** `δ` is called "the confidence of the reported interval" but the docs never say which randomness it integrates over (calibration draw / weight-estimation draw / test sample / all), nor whether resampling re-estimates `ŵ_cov`/`ŵ_lab` per draw — an interval holding estimated weights fixed understates uncertainty.
- *Why it blocks:* "Marginal over what" is exactly what a UQ methodologist polices; an ambiguous/weight-fixed interval reads as an over-claim.
- *Fix:* State explicitly in method_note §1.6 / prereg §6.2 what `δ` integrates over; pre-register whether resampling re-estimates the weights per draw; if held fixed for cost, label the interval "conditional on the estimated weights" and report a separate weight-resampling band as a diagnostic.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (text + a resampling band on existing folds).

**10. [Major · R1] Displayed-confidence (`σ̃`) is least trustworthy exactly when it matters, with only a footnote guardrail.** Target ECE is pre-registered only at target level — no per-subgroup or per-operating-point ECE — and the panel's response to poor ECE is a §6.5 footnote.
- *Why it blocks:* A green miscalibrated "0.97" under shift drives automation bias; an unguarded confidence panel is a direct point-of-care over-claim.
- *Fix:* Pre-register ECE / reliability diagrams of `σ̃` per benchmark, per subgroup, at the deployed operating point, **plus a concrete display response** when ECE exceeds a fixed threshold (widen/grey-out/annotate "confidence unreliable for this site") — not a footnote.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓. Caveat: keep the ECE-triggered response a *display* rule, never a certification gate (invariants 1/4).

**11. [Major · R3] No headline number is uniquely produced by the audit/explainability layer (§6).** Delete §6 and the §§2–5 measured-risk/coverage/leakage/diagnostic results all stand, so an XAI-collection reviewer reads the explainability leg as decoration on a trustworthy-UQ paper.
- *Why it blocks:* At a venue centred on Explainable AI, an excisable explanatory layer invites an off-theme read — the core venue-fit risk.
- *Fix:* Elevate two audit-**only** outputs the docs already specify to named pre-registered headlines: (1) the **subgroup disparate-abstention finding** (low accepted risk bought by high gate-attributed abstention) and (2) the **shift-regime badge** reported per CheXpert↔MIMIC pair as preventing silent substitution of the easier label-shift result for the combined case.
- *Fix verified:* in-scope ✓ (badge measures/tags only) · closes ✓ · feasible ✓ (design-level pre-registration now). *(Reinforces ranks 2 & 12.)*

**12. [Major · R3] The trustworthiness and explainability offers overlap almost entirely.** They share the same risk/leakage/diagnostic numbers; the only separable explanatory deliverables are the per-case DAR and the §5.5(B) margin-validity counterfactual, so the explanatory leg is thin.
- *Why it blocks:* A reviewer who sees both legs sharing every number narrows the contribution to relabeled UQ.
- *Fix:* Present the explicit **two-column trustworthiness-vs-explainability table** and give the DAR its own separable measured deliverable by pairing §5.5(B) margin-validity with the rank-2 accept-side check; if separation isn't convincing post-results, honestly narrow the venue claim to "trustworthy/auditable ML with a faithful decline-attribution component."
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓. Caveat: depends on rank 2; otherwise rests on §5.5(B) alone.

**13. [Major · R1] Metrics live at patch/token level; nothing rolls up to the unit a clinician acts on.** Coverage/selective-risk are measured at CAMELYON17 patches and CheXpert/MIMIC tokens, but no doc aggregates to a slide/patient call or per-study read, so "point-of-care trust" is argued on an object no clinician decides about.
- *Why it blocks:* A clinical-AI referee downgrades the trust claim to algorithmic-component telemetry; the abstention lever's operational benefit is undemonstrated without a named integration point.
- *Fix:* Pre-register a **slide-/patient-level (per-study) aggregation** of accepted-risk and coverage alongside the patch/token metrics (fixed positivity rule over accepted patches), name the per-benchmark **integration point** (CAMELYON17 = pre-read slide triage; CXR = flag-on-report second-reader), and report realized routed workload (studies/day flagged). Otherwise scope the abstract to "algorithmic-component reliability."
- *Fix verified:* in-scope ✓ (post-hoc rollup; no new labels) · closes ✓ · feasible ✓. Major not Blocking — patch-level WILDS is the standard task; only the framing is at risk.

**14. [Major · R1] Risk is symmetric 0-1 only — a missed tumour scores like a misread normal.** No FN-vs-FP stratification on the clinically positive class and OOD leakage is a flat rate.
- *Why it blocks:* At an oncology/radiology venue the trust argument turns on consequence asymmetry; an acceptable-looking 0-1 leakage rate can hide an unacceptable false-negative profile.
- *Fix:* Add a pre-registered **severity-stratified report** — accepted-error and OOD-leakage broken down by FN vs FP on the positive class, plus a severity-weighted loss alongside 0-1 selective risk (severity weights stated as a reporting choice, not a certified cost bound).
- *Fix verified:* in-scope ✓ (post-hoc; `ℓ∈[0,1]` already admitted) · closes ✓ · feasible ✓ (no new data).

**15. [Major · R1] No panel failure-mode discussion.** The genuine unhandled delta (comprehension being IRB-gated is already conceded) is the absence of any discussion of automation bias from a green panel, alarm fatigue from frequent up-weighted/route chips, and cohort-vs-per-patient misreading of `α_acc`/`n_eff`.
- *Why it blocks:* For a trust-interface paper, naming how the interface is misread is part of the honesty contribution; its absence lets a reviewer treat the interface as naively optimistic.
- *Fix:* Add a **panel failure-mode subsection** (automation bias, alarm fatigue, cohort-vs-per-patient misreading) as a measured-limitation discussion, and state plainly the interface is fidelity-measured (DAR checks A–E) but **not** decision-validated.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (discussion-only, no human subjects).

**16. [Major · R6] `Ẑ(x)` plug-in bias is unquantified.** Its bias doesn't vanish with sample size and is amplified on rare/high-`w_lab` classes; §3.5/§7.4 say BCTS "reduces" it but never quantify the residual or its effect on the headline `R̂_w` under combined shift.
- *Why it blocks:* The combined-shift headline divides every accepted-case weight by `Ẑ`; an unquantified non-vanishing denominator bias means the reported number isn't provably the claimed quantity. *(Coupled to rank 6.)*
- *Fix:* Add a pre-registered **`Ẑ`-sensitivity report** isolating the denominator — recompute headline `R̂_w` (with interval) under raw-softmax vs BCTS vs a temperature-perturbation band, report the shift, flag rare/high-`w_lab` classes.
- *Fix verified:* in-scope ✓ (invariant 5 respected) · closes ✓ · feasible ✓. Caveat: residual is bounded empirically and disclosed, not zeroed — the honest outcome.

**17. [Major · R2+R6] The weighted-path betting interval is under-specified.** No pre-committed #splits/#seeds the interval is marginal over, no unit-test that the weighted-path UCB is actually exercised (vs a silent range-bound fallback), and no decision rule when pipeline and baseline intervals **overlap**.
- *Why it blocks:* Without pre-committed power and a non-overlap criterion, "shift-aware holds up under measurement" is attackable as unpowered or interval-overlapping.
- *Fix:* Pre-register per cell: #resamples and seeds; a **unit-test of the weighted-path UCB** against a brute-force reference (no range-bound fallback); a "win only if betting intervals don't overlap" rule (a conservative test, not a p-value); exclude below-`n_eff`-floor cells from headline aggregation.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (cached logits). Partly pre-existing (playbook mandates the UCB check; prereg §5.4 excludes low-`n_eff` cells) — this formalizes them + adds #splits/#seeds + the non-overlap rule.

**18. [Major · R5] The credentialing GO/NO-GO trigger is week-keyed and soft, not a dated artifact-in-hand check.** "If CXR credentialing slips past Wk 4" lets a solo author silently defer the fallback decision week to week until *neither* the two-benchmark nor the fallback paper finishes cleanly.
- *Why it blocks:* Not a scientific flaw — the standard way a two-benchmark plan with a weeks-long human-approval long pole drifts into a half-finished one and misses 2026-10-05.
- *Fix:* Replace with a **dated, artifact-keyed rule**, e.g. "By 2026-07-31, GO two-benchmark iff the PhysioNet credentialed-approval email **and** the Stanford AIMI CheXpert access link are both in hand; else COMMIT to CAMELYON17-primary, CXR drop-in-if-it-arrives." Record CITI-cert / PhysioNet-approval / AIMI-registration dates as run-config checkboxes.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓✓ (planning-doc edit). Caveat: sanity-check 2026-07-31 against actual turnaround.

**19. [Major · R5] No explicit CORE-vs-future scope cut, and the deepest block is back-loaded.** Milestones A–S are one ordered block; the deepest experimental work (CheXpert↔MIMIC both directions + full ablations + prevalence sweep + Stage S) sits in the Wk5–7 window that only starts *after* credentialing, with panel/figures/draft/reviewer-pass stacked on a ~4-day buffer.
- *Why it blocks:* A behind-schedule solo author risks finishing ten things at 70% instead of a defensible core cleanly.
- *Fix:* Add a one-page **CORE-vs-nice-to-have table** (CORE = full pipeline + 4 baselines on the primary benchmark, headline risk-coverage, OOD screen, subgroup audit with intervals, DAR check (A), standard backbone; NICE = DDU/SNGP/ensemble frontier, energy/kNN ablations, DAR (C)–(E), 2nd CXR direction) and rebalance so the CAMELYON17 Stage R/S harness **and** the audit-panel renderer are built/debugged by Wk5 (credentialing-independent), making the mixed-benchmark block a *re-run*, not a build.
- *Fix verified:* in-scope ✓ (CORE keeps all 4 baselines + both benchmarks when available) · closes ✓ · feasible ✓.

**20. [Major · R5] The reproducibility-from-logs promise rests on confirmed one-line stubs.** `metrics.py`/`figures.py` are stubs, and no release **contract** binds cached frozen-`f` logits/features + `σ̃` recalibrator + `σ`/`Ẑ` correctors *versioned with* the frozen split manifests under one tag.
- *Why it blocks:* Discover Computing weights reproducibility heavily; a release that can't regenerate the audit numbers undercuts the paper's own auditability thesis.
- *Fix:* Pre-register the **release manifest now** (frozen patient/slide split indices, cached per-image logits+`φ`, fitted `σ̃` recalibrator and `Ẑ` corrector, run logs — one version tag tied to the split hash) and build `metrics.py`/`figures.py` against this contract from Stage A.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (CAMELYON17 exercises it first). Caveat: check CheXpert/MIMIC feature-redistribution against PhysioNet/AIMI terms; if barred, ship CAMELYON17 end-to-end + CXR split-manifests/scripts only, stating the boundary.

**21. [Major · R5+R3] The clinician-facing audit panel — the venue centrepiece — has zero code and is scheduled behind the deepest results block.** No CORE-vs-deferrable tag among DAR checks (A)–(E); if it ships as a static screenshot, explainability-via-auditability is argued from design, not demonstrated.
- *Why it blocks:* The reason the paper fits the collection is the *runnable* panel; a screenshot-only panel is the soft spot a venue reviewer probes. *(Couples to rank 2.)*
- *Fix:* Designate DAR check (A) construction-faithfulness (discrepancy-rate target 0) **plus a minimal runnable panel renderer over cached logits** as CORE, built alongside CAMELYON17 Stage R in Wk1–5; scope (C)–(E) as "insurance, cut if behind"; ship the panel as a runnable script. (F) stays correctly IRB-gated.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (renderer over CAMELYON17 cached logits is credentialing-independent).

### 🟡 Minor

**22. [Minor · R4] The novelty rebuttal is distributed, never stated as one labelled "what is new here" paragraph adjacent to the "no new guarantee" concession.** A skim reader meets the concession more prominently than the defence.
- *Why it weakens:* Insufficient-novelty is the most likely desk/review rejection for a deliberately-composed pipeline; if the defence must be reconstructed, the honest framing reads as a weakness conceded.
- *Fix:* Add one titled **"What is new here"** paragraph (measurement/auditability discipline; the previously-unquantified OOD gap turned into one stated-exposure-set leakage number; the routing-faithful DAR with executable §3.4(A) invariant; the pre-registered honest-negative failure-mode catalog) adjacent to the concession, in both memo and intro.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓ (pure prose).

**23. [Minor · R4] "Faithful-by-construction" compresses to bare "faithful" in abstract/title usage** where the "faithful to the routing, not the diagnosis" clause doesn't survive — so "faithful" reads as "correct explanation," which the (A)/(B) checks show it is not.
- *Why it weakens:* An XAI-venue reviewer (the Rudin/Adebayo lineage the doc cites) is primed to challenge any "faithful by construction" claim; a dropped clause reads as a guarantee the paper disclaims.
- *Fix:* Adopt a **self-limiting head term** ("routing-faithful"/"gate-faithful" DAR) in abstract and headings; keep the long clause in body text; apply across all docs. Bundle the panel-rendering nits (never co-locate per-case error with `α_acc`; drop "best-controlled" on the badge surface; demote the `α_acc+α_ood` convenience sum; global "best-controlled"→"lowest measured").
- *Fix verified:* in-scope ✓ (removing the convenience sum *removes* rather than revives a union bound) · closes ✓ · feasible ✓. Caveat: apply the rename across all docs to avoid a term mismatch.

**24. [Minor · R1+R3] Residual honesty/disclosure hardening (cheap, convergent).** Print realized per-subgroup and subgroup×class `n` with an "unpowered" tag on `m_lab≈50–200` strata (R1-6); state an explicit HITL operating model per benchmark conceded as a limitation (R1-4); promote the auditability-as-explanation framing into the **intro** and add an **explainability column** to `competitor_matrix.csv` (R3-1/3/4).
- *Why it weakens:* None blocks; each is a disclosed-limitation hardening or placement move that strengthens the honesty brand and venue fit at near-zero cost.
- *Fix:* Land as a single documentation pass.
- *Fix verified:* in-scope ✓ · closes ✓ · feasible ✓. Caveat: do **not** let the HITL note escalate into a demand for prospective clinician-throughput data (out-of-scope); do **not** manufacture a venue precedent — keep the Grad-CAM/SHAP contrast as the recognizable-attribution insurance.

---

## 2. The first-strike weakness

**The matched-coverage confound (rank 1).** A hostile, conformal-literate reviewer attacks here *first* because the pipeline's three routing mechanisms each remove hard cases the baselines must answer — so the entire headline accepted-risk advantage is readable as **bought by abstaining more, not by shift correction** — and nothing in the protocol forces the comparison at equal answer-rate (`coverage_min` is only a degeneracy floor; coverage-beside-risk is only co-reporting). It is the single cheapest line that discredits *every* headline number at once.

**Smallest defusing change:** Add **one pre-registered sentence** committing to report every headline risk at **matched total answer-rate** between pipeline and each baseline, with the 3-way routing decomposition printed beside each number and **AURC** as the threshold-free companion. It's a re-run on already-cached logits, lands well before the deadline, introduces no guarantee, and converts the most damaging attack into a demonstrated control.

---

## 3. Submission-readiness verdict — **REVISE**

This is a fundamentally sound and unusually honest measure-and-report paper. The panel found **no fatal design flaw and no smuggled guarantee** — the disciplined "name the method and its assumption" framing holds up, and the three out-of-scope attacks were correctly discarded. The risks are concentrated in **two confound-control gaps** that a conformal/XAI-literate referee will hit immediately, plus **execution/scheduling** discipline. All are closable before the deadline.

**Three must-fix before submission:**
1. **Pre-register the matched-total-answer-rate comparison** (rank 1) with the 3-way routing decomposition + AURC + raw-base-rate anchor beside every headline cell, so no headline risk number can be read as abstention-bought.
2. **Add the no-IRB accept-side faithfulness check** to prereg §5.5 (rank 2): bin *answered* cases by chip/`n_eff`/shift-regime-badge and report measured accepted error per bin with intervals — so the trust interface that *is* the contribution gets quantitative validation rather than design-only assertion.
3. **Lock the two load-bearing measurement claims into the design** (ranks 3 & 4): the AUROC-tagged prevalence-**reweighting** real-data panel with a pre-committed downgrade, and the **two-tier (far- AND near-OOD)** exposure set with near-OOD leakage reported-as-worse-never-certified.

**Feasibility vs 2026-10-05:** Feasible **if** the dated credentialing GO/NO-GO (rank 18) is set now and the CAMELYON17 harness + audit-panel renderer are built credentialing-independent by Wk5 (ranks 19–21). Every must-fix is a re-run on cached logits or a prose/pre-registration edit, and the heaviest validity check (rank 5 interval-coverage) is purely synthetic and ungated — **the binding risk is schedule discipline, not the science.**

---

## Discarded as out-of-scope / already-handled (findings against the panel)

1. **R3-5 — "faithful" scope-clause-in-abstract.** Penalizes an unwritten abstract; the clause is already paired everywhere it appears (method_note §6.6, explainability_framing §3.1, prereg §5.5/§8, positioning_memo), and check (A) already supplies the measured construction-faithfulness evidence. The live abbreviation-survival risk is carried by rank 23.
2. **R4-6 — "TRUECAM over-claim."** False against the current artifact: `competitor_matrix.csv` row 2 already carries the sourced, softened characterization, and the prose applies "unquantified" to the *gap this paper measures*, not to TRUECAM's own admission. Residual is a cosmetic page-pin only.
3. **R6-6 — "per-case panel shows `σ̃` numbers without an interval."** Demanding a per-case interval would violate the **no-per-case-guarantee invariant** (prereg §8); §6.1/§6.5 already label these cohort-not-per-case. Residual is an optional editorial cross-reference.
