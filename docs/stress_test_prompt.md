# Applied Adversarial Stress-Test — reusable `ultracode` meta-prompt

**What this is.** A paste-and-run meta-prompt that spawns a panel of adversarial reviewers to stress-test the *applied* trustworthy-selective-prediction paper (the CAMELYON17-WILDS + CheXpert↔MIMIC-CXR pipeline targeting Springer *Discover Computing* "Intelligent Medicine"). It attacks the paper **as the applied, measure-and-report, no-new-guarantee artifact it actually is** — clinical realism, baseline fairness, leakage/confounds, statistical rigor, the explainability/venue bar, honesty leaks, and deadline feasibility — and returns a prioritized, fix-attached weakness list plus a submission-readiness verdict. It is the **applied-scope successor to the deleted theory-era D9 stress-test template**: where D9 attacked a certificate that no longer exists, this attacks the measurement discipline, the auditability layer, and the clinician interface that replaced it.

**Relationship to the Appendix-G prompts.** This panel subsumes and extends the single-axis red-team prompts in `flagship-playbook.md` Appendix G (G-1 experiment-design critic, G-2 results red-team) — run this for a full-paper readiness pass across all six lenses; the Appendix-G prompts remain useful as quick single-axis checks against the prereg or a specific results table. Do not run them expecting different scope: G-1/G-2 are narrow search-assistant prompts, this is the multi-lens panel.

**How to run it.**
1. Paste the fenced block below into a fresh session, **prefixed with the word `ultracode`** on its own line. `ultracode` is the trigger that fans the panel out into parallel reviewer agents (decompose → attack → bar-check → fix → verify-fix) and merges their findings. Without the `ultracode` prefix the same block still runs as **one thorough single-agent pass** over the identical phases — lower throughput, same coverage.
2. Make sure the five live docs below are readable at the paths given (edit the paths if your tree differs). The panel reads them first; do not paste their contents inline.
3. Read the OUTPUT: a severity-ranked weakness list, each with one concrete in-scope fix; the single weakness a reviewer attacks first; and a GO / REVISE / NO-GO verdict.

> Re-run it after each major revision. It is deliberately skeptical by default and is allowed to conclude "submit as-is on this axis" only when an attack genuinely fails to land.

---

```
ultracode

# ADVERSARIAL STRESS-TEST — Applied Trustworthy Selective-Prediction Paper

You are a PANEL of adversarial peer reviewers for an APPLIED clinical-ML paper. Default posture: SKEPTICAL — assume every claim is over-stated until the docs prove otherwise, but attack the paper THAT EXISTS, not a paper you wish existed. Your job is to make this paper's REAL contribution (a measurement / auditability discipline + a clinician-facing trust interface) survive Discover Computing review — by finding where it is weak and proposing the best IN-SCOPE fix for each weakness. You do NOT get to demand a different paper.

## THE PANEL (adopt all lenses; let them disagree)
- R1 — Senior ML-for-health / clinical-AI reviewer ("Reviewer 2" energy). Attacks clinical realism, operational credibility, point-of-care usefulness of abstention/audit/route-to-human, fairness-audit adequacy, displayed-confidence safety under shift. NOT the statistics, NOT breadth.
- R2 — Reproducibility & experiment-design critic. Attacks fold leakage, baseline fairness, the cheap confounds that fake a "win" (backbone, abstention-inflation, base-model-just-worse-on-target, cherry-picked alpha, lucky seeds), whether the plots literally show the two load-bearing claims, and statistical-reporting rigor (#splits, which interval, marginal-over-what, n_eff).
- R3 — Venue-fit & explainability-bar reviewer. Attacks the "explainability via auditability" framing against a collection whose name centers Explainable AI: the XAI-label mismatch, the un-evaluated interface, the "conformal paper wearing a healthcare hat" risk, whether the audit layer is load-bearing or deletable, and whether THIS collection accepts auditability-as-explanation.
- R4 — Honesty & novelty auditor. Hunts smuggled guarantees (a deployed-pipeline property re-entering via mood/adjacency, especially alpha_acc shown next to a per-case number), and audits whether "we composed existing methods, no new guarantee" reads as insufficient novelty — and how to defend it.
- R5 — Scope & feasibility critic. Attacks timeline realism to the 2026-10-05 deadline, the SHARPNESS of the existing credentialing GO/NO-GO (the playbook already names a Wk-4 trigger + a CAMELYON17-only fallback — is the trigger a checkable dated artifact, and is the fallback paper publishable alone?), and whether the scope is finishable cleanly vs half-done across everything.
- R6 — Conformal / UQ-methodology critic. Attacks whether the MEASUREMENTS THEMSELVES are statistically sound, independent of any guarantee: bias and interval-validity of the self-normalized (Hájek) weighted-risk estimator with ESTIMATED weights; whether the betting interval — derived for a bounded MEAN — is valid on a self-normalized RATIO; the Ẑ(x) softmax-plug-in denominator bias (non-vanishing, amplified on rare classes); clip-induced bias; and what "marginal over the calibration draw" actually integrates over for a measured (not certified) number. NOT clinical, NOT venue, NOT a demand for a guarantee — a demand that the reported number means what it says.

## INPUTS (read these live files FIRST; do not infer their contents)
- docs/method_note.md        — method spine + single-source notation; selective prediction; shift-aware calibration (Ẑ-combine, NOT product); OOD routing; integrated decision rule; audit layer §6; assumptions/limitations §7
- docs/positioning_memo.md   — one-sentence applied contribution; what is dropped vs the certified era; Discover Computing venue fit; "explainability via auditability"
- docs/preregistration.md    — evaluation protocol: 2 benchmarks, shift-type diagnostic, 4 baselines, metrics, subgroups, what is FIXED before results, minimum-coverage floor
- flagship-playbook.md       — applied build/experiment milestone ladder (A→B→C→D→R→S) + timeline + risk register + attribution discipline
- analysis/competitor_matrix.csv — competitor landscape (note: every row is a conformal/selective-prediction method; there is no explainability axis in the matrix)

CONTRIBUTION (one sentence): an applied, trustworthy, auditable selective-prediction pipeline for clinical image classification under realistic covariate + label shift, assembled ENTIRELY from existing citable methods (weighted conformal / RCPS, BBSE and MLLS+BCTS label-shift estimation, post-hoc OOD routing), whose contribution is a MEASUREMENT / AUDITABILITY discipline and a clinician-facing TRUST INTERFACE that make residual shift-induced degradation visible at the point of care — NOT a new statistical guarantee.
VENUE: Springer *Discover Computing*, "Intelligent Medicine: ML and Explainable AI for Next-Generation Healthcare" (explainability + trustworthy/auditable ML). DEADLINE: 2026-10-05.
BENCHMARKS: CAMELYON17-WILDS (covariate-dominant, cross-hospital); CheXpert↔MIMIC-CXR (mixed covariate+label, cross-site, both directions).

## INVARIANTS (NEVER undermine these — an attack that violates one is INVALID; discard it)
1. NO NEW GUARANTEE. The contribution is the measure-and-report + auditability discipline + clinician interface. Every fix must keep the paper APPLIED. Never "add a guarantee / certificate / bound" as a fix.
2. Methods are existing and citable, used as published; each cited guarantee is a property OF THAT METHOD UNDER ITS OWN ASSUMPTION, never of the deployed pipeline.
3. Depth over breadth: two imaging benchmarks taken deep; multi-modality is honest future work, not a deficiency to be fixed by adding datasets.
4. Every guarantee sentence names its method + its assumption. The words certify/certified/guarantee are banned for the deployed pipeline.
5. The labeled target slice D_tar^lab (m_lab ≈ 50–200) exists ONLY to MEASURE residual degradation and fit a scalar correction — it identifies nothing and certifies nothing.

## OUT OF SCOPE (the stress test MUST NOT demand any of these; if a lens wants one, kill it as out-of-scope)
- A guarantee / certificate / finite-sample distribution-free bound on R_T^accept, coverage, or OOD leakage for the deployed pipeline (partly PROVABLY IMPOSSIBLE: Fang et al. 2022; Yang–Kuchibhotla–Tchetgen 2024).
- A certified additive α = α_acc + α_ood union-bound (these are two SEPARATELY-measured operating budgets by design).
- The temporal / online / DtACI adaptive-conformal track (permanently scope-cut; in none of the docs).
- Exotic / additional label-shift estimators (RLLS, GS-B³SE, FMAPLS, Wang & Qiao) or baselines beyond the four pre-registered — MLLS+BCTS primary, BBSE baseline/diagnostic are FIXED.
- Reviving the abandoned 5-condition "guarantee that survives shift" / GLS-anticausal-identifiability / FJS-as-identification / ε_M-certificate framing.
- Multi-modality breadth (ECG, retina, 2nd pathology, dermatology) as a BLOCKER. (Naming the single cheapest credible extra modality as a rebuttal aside is allowed; demanding it is not.)
- Saliency / SHAP / attention / concept maps as THE required explainability contribution (the explainability-via-auditability stance is a deliberate venue-fit choice; attribution is at most optional insurance, never a required pivot).
- Per-case or per-subgroup coverage/risk guarantees from the audit layer or RLCP; prospective / RCT / deployed-clinician data (a lightweight think-aloud or faithfulness check is the in-scope substitute, never a clinical trial).

## PHASES (run in order; if `ultracode`, fan the lenses across parallel agents and merge)

### PHASE 1 — DECOMPOSE into atomic units
List the framework's atomic units so each can be attacked in isolation AND at its seams:
- CLAIMS: the one-sentence contribution; the two load-bearing MEASUREMENT claims — (C-label) label-aware Ẑ-combine beats covariate-only UNDER PREVALENCE SHIFT; (C-ood) the OOD screen measurably reduces far-OOD leakage UNDER A DELIBERATELY IMPERFECT detector; the "explainability via auditability" claim; the "no new guarantee is still a contribution" novelty claim.
- METHODS: frozen f; selective gate g/τ; weighted RCPS + WSR betting UCB; ŵ_cov (discriminator); ŵ_lab (MLLS+BCTS primary, BBSE baseline); the Ẑ-combine; OOD score o + t_ood routing; w_max routing; LTT multiple-looks.
- BASELINES: naive split-conformal (ŵ≡1); temperature-scaling-only; TRUECAM-style detect-and-remove; BBSE-label-only.
- METRICS: realized selective risk (Hájek + betting CI); coverage + 3-way routing decomposition; AURC; target ECE of σ̃; OOD AUROC/FPR95 + leakage vs α_ood + exposure-set sensitivity; subgroup risk/coverage/routing; reliability diagnostics (n_eff, κ(Ĉ_S), q̂_T-vs-Ĉ_S p̂_T, clip-rate).
- AUDIT LAYER: per-case panel (σ̃, per-case risk band, weight-context chip ŵ, n_eff flag, shift-regime badge, OOD flag); subgroup/fairness audit; per-case decline-attribution (which of g/o/ŵ_cov fired + margin to each threshold).
- EXPERIMENTAL PLAN: 7+ disjoint folds; stress axes (small m, hardest pair, both directions, prevalence sweep, w_max sweep); attribution discipline (standard backbone is the base; DDU/SNGP/ensemble are ablations only).
- TIMELINE: ~3.5 months; CheXpert/MIMIC credentialing as the schedule-critical long pole.

### PHASE 2 — ADVERSARIALLY ATTACK each unit AND the seams (use these concrete vectors)
For every attack: name the unit/seam, the failure, and WHY IT BITES THIS PAPER (the contribution is measurement + interface). Discard any attack that violates an INVARIANT or lands in OUT OF SCOPE.

CLINICAL REALISM & POINT-OF-CARE (R1):
- Decision granularity: does a measured selective-risk change on CAMELYON17 PATCHES / CXR NLP-mined finding-labels map to a decision a clinician acts on (slide/patient-level metastasis; per-study read)? If only patch/token metrics exist, force slide/study-level aggregation OR an explicit claim scoped to "algorithmic component reliability," not "point-of-care trust."
- Workflow absorbability: at the headline operating point, what concrete human workload does abstention create, at the unit a human actually triages (a WSI is millions of patches; "defer 30% of studies" may be net-negative)? Force coverage at the human triage unit + a named integration point (pre-read triage / second-reader / flag-on-report).
- Route-to-human realism: the routed pile is the hard/OOD/no-AI-assist cases; state the human-in-the-loop model (availability, expertise, turnaround) and force an ASYMMETRIC-COST / clinical-consequence analysis of the LEAKAGE cases (a missed tumor ≠ a misread normal) — report error severity, not only 0-1 selective risk.
- Displayed-confidence trust contract: σ̃ is recalibrated on SOURCE and drifts under exactly the appearance shift the paper targets, so the displayed number is least trustworthy when it matters most. Force target-domain ECE / reliability diagrams of σ̃ PER benchmark, PER subgroup, at the deployed operating point — and a concrete panel behavior when ECE is poor (widen / grey / annotate), not a footnote.
- Panel comprehension: σ̃, the representativeness chip (a density-ratio quantity), n_eff, the shift-regime badge are statistician-facing. Force either a lightweight clinician think-aloud / documented design rationale, OR an explicit limitation that the interface is measured-for-fidelity but NOT validated for clinician decision impact, plus a failure-mode discussion (automation bias from a green panel; alarm fatigue from frequent up-weighted chips; misreading a case-level number as a per-patient guarantee).
- Fairness-audit adequacy: subgroup risks are on D_tar^lab (m_lab ≈ 50–200 total) → per-subgroup-per-class intervals are uselessly wide; intersectional cells empty. Force realized labeled-target n per subgroup (and per subgroup×class), honest interval widths, naming which strata are UNPOWERED, leading with the ABSTENTION/coverage disparity (silent under-service) not just accepted-risk disparity — and a decision on whether m_lab supports ANY subgroup claim vs honest "demonstrated-on-pooled, subgroups underpowered."

REPRODUCIBILITY, BASELINE FAIRNESS, CONFOUNDS (R2):
- Fold-disjointness at the RIGHT unit: prove disjointness at PATIENT/SLIDE/STUDY ID (not patch/image row) across all 7+ folds, INCLUDING the σ̃-recalibration fold, O, and D_tar^lab vs D_cal/D_bbse^src. Force a leakage probe (deliberately leak a patient into D_cal and test; report how much realized risk moves) and printed group-ID set-intersection = 0 in the artifact.
- Baseline fairness: identical frozen f, folds, target test AND equal operating-point search budget for all four baselines (does the naive baseline pick its τ on the same grid with the same multiple-looks correction?). Is any baseline crippled by being denied abstention the full pipeline has? Force MATCHED-COVERAGE comparison.
- Abstention-inflation: three routing mechanisms (w_max, t_ood, τ) can each inflate abstention. Force co-reported coverage + 3-way decomposition for every headline risk number; confirm no winning cell sits below the pre-registered coverage_min; force the matched-coverage win.
- Backbone / base-model confound: report frozen f's raw target-vs-source error (no abstention, no weighting) next to every headline so "base model just worse on target" is ruled in/out; force the efficiency-frontier ablation to show the pipeline>baseline ordering holds on the STANDARD backbone (not only on DDU/SNGP/ensemble) — if the win needs a fancy net, the win IS the backbone.
- Does a plot literally show C-label? Force the figure: covariate-only vs Ẑ-combined realized risk under the pre-registered prevalence sweep, with separation exceeding the betting intervals, PLUS the double-count panel (naïve product = oracle×Z; Ẑ-divide = oracle). If only synthetic isolates prevalence, force downgrading the real-data claim to "shown on synthetic only."
- Is the label-shift win just covariate weighting in disguise? Combined shift is non-identifiable; force an ablation toggling covariate-weight OFF / label-weight ON (and vice versa) on the SAME real pairs, tagged by discriminator-AUROC regime, with the q̂_T-vs-Ĉ_S p̂_T check displayed so anti-causal violation is visible.
- Does C-ood hold under a DELIBERATELY imperfect detector / hard near-OOD exposure set (not just easy off-modality probes)? Force exposure-set sensitivity, per-benchmark AUROC/FPR95, and a head-to-head vs the TRUECAM-style detect-and-remove baseline on the SAME O.
- Statistical rigor: per headline cell, force #splits, #seeds, what the interval is marginal over (calibration draw / splits / seeds), whether the WSR betting bound is actually used in the WEIGHTED path (not a silent range-bound fallback), and whether pipeline-vs-baseline intervals actually SEPARATE or overlap. Force n_eff next to every weighted number; low-n_eff cells are low-reliability, not averaged in.
- Cherry-picking: force the win shown across the FULL pre-registered grid/stress axes (small m, hardest pair, both directions, prevalence sweep, w_max sweep) + seed spread; non-winning cells reported as the failure-mode catalog, not hidden. Force commit/timestamp evidence that the grid, budgets, subgroups, τ_disc, coverage_min were fixed BEFORE any target metric.
- Clinical-imaging leakage traps: NLP-derived "uncertain" CXR labels handled identically across all baselines/folds; probe whether the domain discriminator keys on a scanner/site SHORTCUT (overlay text, padding, intensity) vs clinical appearance — if it is a shortcut detector, the per-case representativeness chip is auditing the wrong thing and must be caveated.

VENUE FIT & EXPLAINABILITY BAR (R3):
- XAI-label mismatch: does the draft EXPLICITLY argue (with trustworthy-ML / uncertainty-as-explanation / FUTURE-AI-style citations, early in the intro) that auditable reliability disclosure IS explanation, and pre-empt the "this is just UQ, not explainability" reviewer? Force that one-paragraph rebuttal to actually exist in the draft, not only in the memo.
- Interface un-evaluated: every pre-registered artifact is a model-side metric; the interface is only specified. Force a deadline-feasible MINIMUM: (a) a faithfulness check that the chip / badge / n_eff flag actually co-vary with realized accepted-risk degradation ("heavily up-weighted cases show X higher measured error"), and/or (b) a small structured expert walkthrough / think-aloud on K worked cases. Reject "we describe the panel" as sufficient.
- Load-bearing test: if §6 (audit layer) were deleted, what claim fails? If "none — §§1–5 stand alone," the explainability framing is decoration. Force at least one HEADLINE result that ONLY the audit layer produces — the cleaner primary instance is a subgroup-attributable disparate-abstention finding (an unambiguous pure-measurement result with no tag-to-guarantee slippage risk); a secondary instance is the shift-regime badge demonstrably PREVENTING a silent substitution that would otherwise occur (show, on a real pair, a case where pooling/substitution would have mis-stated the regime and the badge catches it) — as a measured auditability win, NOT as a regime-conditional guarantee. Evaluated, not merely described.
- Elevate per-case DECLINE-ATTRIBUTION (which of g/o/ŵ_cov fired + margin to each threshold — already latent in the routing decomposition) into a named, faithful-BY-CONSTRUCTION explainability contribution, contrasted with post-hoc saliency. This is the cheapest, most venue-aligned win; force the reframing and a check that the attribution is correct/stable.
- Trustworthy-but-not-explainable: list side-by-side what the paper offers as TRUSTWORTHINESS (measured risk, leakage budget, diagnostics) vs EXPLAINABILITY (a per-case account of WHY it declined/answered + how to read it). If the two lists are identical, the explainability leg is hollow — force a separable explanatory component or an honestly narrowed venue claim.
- Reviewer-expectation calibration: force quoting the collection's call-for-papers scope language and mapping each claim to it, + at least one precedent (this collection or a sibling Springer XAI-health venue) where UQ/audit-as-explanation was accepted — or a defensive framing paragraph + considering lightweight attribution as insurance. "The memo says it fits" is NOT evidence.

HONESTY & NOVELTY (R4):
- Smuggled-guarantee hunt: sentence-by-sentence, does any claim let a clinician infer a deployed-pipeline property that is only true of a cited method under its assumptions (especially the per-case panel showing α_acc next to a per-case error number)? List each leak with a rewrite into a measured statement with an interval.
- Novelty defense: write the one-paragraph rebuttal to "you just composed existing methods, no new guarantee — this is insufficiently novel," grounded in the measurement discipline + the auditable OOD-leakage number + the faithful decline-attribution interface + the honest-negative failure-mode catalog. Confirm that defense exists in the draft, not only here.

SCOPE & FEASIBILITY (R5):
- Credentialing GO/NO-GO sharpness: the playbook §8 already names a Wk-4 trigger and a CAMELYON17-only fallback. Pressure-test it: is "slips past Wk 4" a checkable event (what concrete credentialing artifact must be in hand by which calendar date — CITI completion? PhysioNet approval? AIMI registration?), and is the CAMELYON17-only paper actually a publishable Discover-Computing artifact on its own (single benchmark, single modality — does the depth-over-breadth defense survive losing the mixed covariate+label benchmark entirely)? Force a calendar date tied to a named artifact, not a week-number tied to a vague "land".
- MUST-HAVE vs NICE-TO-HAVE cut: force an explicit list of which benchmark/pairs/directions/subgroups are core to the headline (finished cleanly) vs honestly-scoped-future-work if time runs short — so the paper is not half-done across everything.
- Reproducibility-from-logs: force the claim that one script (analysis/figures.py --from-logs) regenerates every table/figure, that frozen split manifests at the patient/slide unit + credentialing steps ship (credentialed-data-safe: code + manifests + cached frozen-f logits/features even when raw images cannot be released), the feature cache + σ̃ recalibrator are versioned with the splits, and the audit panel exists as a RUNNABLE artifact, not just a paper figure.

MEASUREMENT-SOUNDNESS — IS THE REPORTED NUMBER WHAT IT CLAIMS? (R6):
- Hájek-estimator interval validity: the headline realized selective risk is a self-normalized `(Σŵℓ)/(Σŵ)` ratio with ESTIMATED weights, but the betting / hedged-capital UCB is derived for a bounded MEAN. Force either a citation/derivation that the interval transfers to the self-normalized ratio, OR an EMPIRICAL interval-coverage check on synthetic data with known ground-truth risk, OR an honest downgrade to "nominal interval, coverage validated empirically." A measure-and-report paper still owes a VALID interval even with no PAC claim — this is in scope, not a guarantee demand.
- `Ẑ(x)` plug-in bias: the combine divides by `Ẑ(x)=Σ ŵ_lab σ̃(f(x))`, a plug-in for `E_{p_S(·|x)}[w_lab]`. This denominator bias does NOT vanish with sample size and is amplified on high-weight / rare classes. Force the measured effect of the σ̃ recalibration on the risk estimate and a sensitivity to the recalibration choice — a biased denominator must not silently shift the headline number.
- Clip bias: routing/clipping at `w_max` removes the tail that would dominate the estimate but also BIASES the retained weighted risk. Force the clip fraction logged next to every number AND a clip-bias sensitivity (sweep `w_max`) so a low reported risk is not an artifact of aggressive clipping.
- "Marginal over what": the prereg calls the interval "the confidence of a reported interval, not a PAC promise." Force precision on what randomness it integrates over (the calibration/weight-estimation draw? the test sample? both?) and that the reported #splits actually RESAMPLES that randomness — an interval that holds the estimated weights fixed across splits understates uncertainty.
- Estimated-weight uncertainty not in the interval: `ŵ_cov` and `ŵ_lab` are estimated and their error is reported via diagnostics (`n_eff`, `κ(Ĉ_S)`) but is NOT inside the betting interval. Force an explicit statement that the displayed interval is CONDITIONAL on the estimated weights, with weight-estimation error carried as the named diagnostics — so the interval is not over-read as total uncertainty. (Honest scoping, not a guarantee.)

### PHASE 3 — HOLD IT TO THE REVIEWER + VENUE BAR
For the surviving attacks, judge as a Discover Computing reviewer would: would they accept? Specifically — (a) is the explainability bar met or does the audit layer read as relabeled UQ; (b) is the novelty ("composition + discipline + interface, no new guarantee") defensible or does it read as thin; (c) is the clinical bar met (does a measured number attach to a decision a clinician makes); (d) are the measurements clean enough that the two load-bearing claims are actually demonstrated, not asserted. Mark each attack as ACCEPT-BLOCKING, ACCEPT-WEAKENING, or COSMETIC.

### PHASE 4 — PROPOSE THE BEST IN-SCOPE FIX per real weakness
For each surviving weakness, give ONE concrete fix that keeps the paper applied (measure-and-report + auditability + interface). Fixes must be of these kinds (never "add a guarantee"):
- report at MATCHED coverage + aggregate to the clinical decision unit (slide/patient; per-study read) alongside patch/label-token level;
- add severity-weighted / asymmetric loss + report the clinical consequence of leakage cases, not just the rate;
- report realized labeled-target n per subgroup (+ per subgroup×class), honest interval widths, lead with abstention/coverage disparity, name unpowered strata;
- report target-domain ECE / reliability diagrams of σ̃ per benchmark/subgroup at the operating point + a concrete panel behavior on drift;
- add a faithfulness check (chip/badge/n_eff co-vary with realized degradation) and/or a small expert think-aloud; OR state plainly the interface is fidelity-measured-not-decision-validated + discuss automation-bias/alarm-fatigue/misread failure modes;
- state the explicit human-in-the-loop model + concrete routed workload at the human's triage unit;
- document patient/slide-level fold disjointness (printed set-intersection = 0), identical NLP-label handling across baselines, and a discriminator-shortcut probe/ablation;
- publish the complete per-pair / per-direction / per-stress-cell table with no aggregation hiding a losing pair; justify α_acc/α_ood as clinically defensible budgets;
- elevate per-case decline-attribution to a named faithful-by-construction explainability result + an early "auditability IS explanation" framing paragraph with citations + a "why not saliency" subsection (faithfulness grounds);
- make ≥1 headline result depend on the audit layer (shift-regime badge / subgroup-attributable disparate abstention);
- rewrite implied-guarantee sentences into measured statements with intervals;
- ship code + frozen split manifests + cached frozen-f logits/features + a runnable audit panel; dated credentialing GO/NO-GO + a CAMELYON17-only fallback; explicit MUST-HAVE vs NICE-TO-HAVE cut;
- validate the Hájek-risk betting interval's coverage EMPIRICALLY on synthetic ground-truth risk; report clip-bias (sweep `w_max`) and `Ẑ`-plug-in sensitivity; state plainly the displayed interval is conditional-on-the-estimated-weights, with weight-estimation error carried as the named diagnostics (`n_eff`, `κ(Ĉ_S)`) — never as a guarantee.

### PHASE 5 — ADVERSARIALLY VERIFY each proposed fix
For every fix, attack it: (a) is it actually IN SCOPE (no INVARIANT broken, nothing from OUT OF SCOPE)? (b) does it actually CLOSE the weakness, or just relocate it? (c) is it feasible before 2026-10-05 given the credentialing long pole? (d) does it introduce a NEW weakness (e.g. a clinician walkthrough that, if under-powered, becomes its own attack surface)? Discard or rewrite any fix that fails. A fix that requires a guarantee, a new estimator, a new modality, the temporal track, or the revived gap framing is INVALID by construction.

## OUTPUT FORMAT (exactly this, concise)
1. PRIORITIZED WEAKNESS LIST — ranked by likelihood of BLOCKING acceptance. Each row:
   - [SEVERITY: Blocking / Major / Minor] Weakness (1 sentence, name the unit/seam + the lens)
   - Why it blocks/weakens acceptance at Discover Computing (1 sentence)
   - IN-SCOPE FIX (1–2 sentences, concrete, applied — never a guarantee)
   - Fix verified? (in-scope ✓ / closes-weakness ✓ / feasible-by-deadline ✓ — or the caveat)
2. THE FIRST-STRIKE WEAKNESS — the single thing a hostile reviewer attacks first, and the smallest change that defuses it.
3. SUBMISSION-READINESS VERDICT — GO / REVISE / NO-GO, with the 3 must-fix-before-submission items and a one-line feasibility call against the 2026-10-05 deadline.

## DISCIPLINE
- Attack the paper that exists. Every fix keeps it applied; "add a guarantee" is never a fix.
- An attack that demands anything in OUT OF SCOPE, or violates an INVARIANT, is itself a finding AGAINST the reviewer — discard it and note why.
- A "honest negative" (a characterized failure regime, an unpowered subgroup said to be unpowered, a losing pair reported) is a FEATURE of this paper, not a defect — do not penalize honesty; penalize HIDDEN weakness.
- Prefer the cheapest fix that genuinely closes the weakness. Flag any fix that cannot land before the deadline.
```

---

### Notes on scope (why this prompt is shaped the way it is)

- It folds the six reviewer lenses' concrete attack vectors directly into Phases 1–2 (decision-granularity, workflow absorbability, leakage-at-the-right-unit, matched-coverage, the two load-bearing measurement plots, the load-bearing-audit-layer test, the smuggled-guarantee hunt, the dated credentialing GO/NO-GO), so the run is specific to *this* paper rather than a generic "find problems" pass.
- The INVARIANTS and OUT OF SCOPE sections are load-bearing: they stop the panel from drifting back into the abandoned theory paper (a guarantee, the additive α-budget, the DtACI track, exotic estimators, the 5-condition gap, multi-modality-as-blocker, saliency-as-required-XAI). Every fix the panel can propose is constrained to the applied kinds in Phase 4, and Phase 5 explicitly discards any fix that needs a guarantee/estimator/modality/temporal-track/gap-revival.
- It deliberately protects honest negatives (characterized failure regimes, unpowered subgroups, losing pairs) so the stress test sharpens the measure-and-report contribution instead of pressuring the author to hide weakness.
