# Flagship Playbook — Trustworthy, Auditable Selective Prediction for Clinical Imaging Under Distribution Shift
### An *applied* build/experiment plan for the Springer *Discover Computing* "Intelligent Medicine" collection — *supersedes the theoretical certificate-driven build plan*

> **What changed (read first).** This playbook was rewritten from a theoretical-contribution plan (a distribution-free risk *certificate* that survives covariate + label shift with OOD routing folded into one error budget) into an **applied, empirical build plan** for a single journal paper. The contribution is now a **measurement / auditability discipline + a clinician-facing trust interface** built *entirely from existing, citable methods*, claiming **no new guarantee**. The abandoned theory — the 5-condition "guarantee that survives shift" novelty core, the formal theorem with explicit constants, the Generalized-Label-Shift / anticausal-identifiability "tension" machinery, the certified additive `α = α_acc + α_ood` budget, the temporal DtACI track, and the exotic label-shift estimators — is **cut**. The old theory version is recoverable in git history; do not restate it.
>
> **This playbook does not restate the method, the positioning, or the protocol — it points to them.** Three documents are authoritative and are the single sources of truth. Where this playbook would otherwise repeat them, it links instead:
> - [`docs/method_note.md`](docs/method_note.md) — the method spine and **the single source of truth for all notation, symbols, weights, thresholds, and budgets** (§1 notation + §1.8 quick-reference; §2 selective prediction; §3 shift-aware calibration; §4 OOD routing; §5 integrated decision rule; §6 the clinician-facing audit layer; §7 assumptions and honest limitations).
> - [`docs/positioning_memo.md`](docs/positioning_memo.md) — the applied one-sentence contribution, the scope decision, and the *Discover Computing* venue fit.
> - [`docs/preregistration.md`](docs/preregistration.md) — the applied evaluation protocol (datasets, shift-type diagnostic, the four baselines, metrics, what is fixed before results).

## Conventions

- **The three docs win every conflict.** This playbook is a build/experiment plan layered *on top of* the three authoritative docs above. Anything here that appears to touch notation, the method, the contribution, or the protocol defers to them; if this file and a doc ever disagree, the doc is right.
- **Notation lives in one place.** All symbols, weights, thresholds, and budgets are defined once in [method_note.md §1](docs/method_note.md) (with the §1.8 quick-reference table). This playbook redefines nothing; every symbol here carries exactly its method-note meaning. `monospace` = a file / symbol / variable.
- **"Selective / conformal layer"** = the risk-control machinery (RCPS / LTT), *not* the neural net (that's the "base model").
- **The honesty discipline (the #1 rule, on every page).** We claim **no new guarantee**. Each method's guarantee is cited strictly as a property *of the method that owns it, under that method's own assumptions*. Under realistic clinical shift we **measure and report** the resulting degradation; we never certify it. The words *certify / certified / guarantee* are **banned for the deployed pipeline**.
- **Effort key.** **S** ≤ a day · **M** a few days · **L** a week+ · **XL** multi-week.

---

## 0. The contribution on one page (applied)

**One-sentence contribution** (verbatim authority: [positioning_memo.md](docs/positioning_memo.md)). An applied, trustworthy, **auditable selective-prediction pipeline** for clinical image classification under realistic covariate + label shift — assembled **entirely from existing, citable methods** (weighted conformal / RCPS, BBSE and MLLS+BCTS label-shift estimation, post-hoc OOD routing) — whose contribution is a **measurement / auditability discipline and a clinician-facing trust interface** that make the residual shift-induced degradation **visible at the point of care**, **not a new statistical guarantee.**

**The honesty discipline (the #1 rule, on every page).** We **claim no new guarantee.** Each component guarantee is cited strictly as a property *of the method that owns it, under that method's own assumptions* — RCPS `(α,δ)` control only under cal/test exchangeability; weighted-conformal `1−α` *marginal* coverage only under pure covariate shift with the *oracle* likelihood ratio; BBSE/MLLS consistency only under label shift with an invertible `Ĉ_S`; LTT as a plain multiple-looks correction. None of these survives the source→target shift as a deployment certificate, and we **say so**. Under realistic clinical shift we **measure and report** the resulting degradation; we never certify it. The words *certify / certified / guarantee* are **banned for the deployed pipeline** (see [positioning_memo.md "What this memo drops"](docs/positioning_memo.md)).

Two published impossibility results make measure-and-report the *only* honest stance, not a concession — they are cited as the **reason to measure, not as scaffolding for a new bound**: OOD detection is not distribution-free learnable in the unrestricted setting (Fang et al., NeurIPS 2022), and finite-sample distribution-free *conditional* coverage under covariate shift is not attainable once the density-ratio weight is estimated (Yang, Kuchibhotla & Tchetgen Tchetgen, JRSS-B 2024). A third structural fact compounds them: combined covariate-and-label shift is **not identifiable** from unlabeled target covariates alone.

**What we measure (the object), not certify.** The quantity we care about is the answered-case target risk `R_T^accept := E_{P_T}[ ℓ(Y, ŷ(X)) | A(X) ]` over the in-scope/answered event `A(x)` ([method_note.md §1.1, §1.5.1](docs/method_note.md)). We tune operating thresholds on calibration data, apply the weighted correction, and **measure** realized selective risk, coverage, and OOD-leakage on held-out target — reporting the degradation overall and by subgroup. The two operating budgets `α_acc` (accepted in-scope risk) and `α_ood` (far-OOD leakage on a stated exposure set `O`) are **separately measured** operating budgets; their sum is a *reporting convenience*, **not** a union-bound certificate ([method_note.md §1.6](docs/method_note.md); [preregistration.md §6.2, §8](docs/preregistration.md)).

**The explainability hook (why this is the right venue).** The contribution is *explainability via auditability*: rather than a saliency map, the [audit layer (method_note.md §6)](docs/method_note.md) surfaces, per case, the routing decision over `A(x)`, the recalibrated posterior `σ̃(f(x))`, a per-case weight-context chip (`ŵ(x,ŷ)` as a *representativeness* flag), and the cohort reliability flag `n_eff`; and per cohort, a subgroup audit of selective risk, coverage, and routing rates with finite-sample intervals. Its explicit job is to make the residual degradation the method *warns about* **visible at the point of care** — an interpretable account of the model's own reliability. This is the honest counterpart to detect-and-remove pipelines (e.g. **TRUECAM**): we do not beat them with a stronger guarantee; we **measure and report the residual leakage** they leave unquantified, and route the flagged tail to a human.

**Venue & deadline.** Springer *Discover Computing*, collection **"Intelligent Medicine: Machine Learning and Explainable AI for Next-Generation Healthcare"** (open access; central emphasis = explainability + trustworthy/auditable ML). **Submission deadline 2026-10-05.** Today is **2026-06-26** — about 3.5 months out. Venue mapping is argued in full in [positioning_memo.md "Venue fit"](docs/positioning_memo.md).

**Done =** a journal-grade applied paper + a reproducible release that, on **two imaging benchmarks** (CAMELYON17-WILDS; CheXpert ↔ MIMIC-CXR, both directions) under realistic covariate + label shift, **measures whether the shift-aware pipeline maintains/improves realized selective risk and coverage relative to the naive baseline** where **naive split-conformal degrades** — together with a **measured far-OOD leakage screen** against a named `α_ood` budget on a stated, swappable exposure set, and a **subgroup audit** with finite-sample intervals — **all MEASURED, none certified.** We *want* the shift-aware pipeline to maintain useful selective risk at useful coverage and *expect* the naive baseline to degrade, but the deliverable is the measurement, not an assured outcome. Concretely: every headline number is either a finite-sample *estimate* (`x̂`) with its interval or a *measured* diagnostic, with its assumption and its measured degradation disclosed.

---

## 1. Strategy: one applied paper, built in staged engineering milestones

**The shape.** *Publishing* = one *Discover Computing* submission. *Engineering* = a staged ladder where each layer is built and **validated on synthetic data as an empirical engineering milestone** before the next layer is stacked. You are not proving a bound at each gate; you are **de-risking one applied paper** by checking that each added layer behaves as intended on data where you control the ground truth. (The detailed file-by-file build sequence and the synthetic milestone for each layer are §5; this section fixes the *principle*.)

**Empirical milestone ladder (engineering checks, NOT proofs that a bound holds).** Each milestone is a *measured comparison against the naive baseline* on synthetic data with known shift — it answers "does this layer empirically do its job?", never "does `R_T^accept ≤ α`?":

```
M-1  exchangeable selective RCPS   → on exchangeable synthetic data, CHECK that the RCPS-selected
                                      operating point reproduces the known risk–coverage trade-off
                                      (a wiring sanity-check under exchangeability, not a
                                      budget-satisfaction claim)
M-2  + covariate weights           → under known synthetic covariate shift, MEASURE whether the
                                      weighted path recovers realized selective risk that the
                                      unweighted (naive) path loses
M-3  + label-shift weights         → under known synthetic prevalence shift, TEST whether covariate-
                                      only weighting is insufficient and whether the label-aware
                                      (MLLS+BCTS) correction measurably improves realized risk;
                                      sweep a growing class-conditional residual to MAP the point where
                                      the correction degrades (a measured failure boundary, not a proof)
M-4  + OOD screen                   → with a deliberately imperfect detector, MEASURE whether the screen
                                      reduces far-OOD leakage vs no screen, and report the residual
                                      leakage on O against α_ood
M-5  real-data headline            → on both benchmarks: MEASURE realized selective risk and coverage
                                      for the shift-aware pipeline vs naive conformal and the
                                      TRUECAM-style baseline, reporting the degradation of each
M-6  subgroup audit                → per-subgroup realized selective risk, coverage, and routing
                                      rates with finite-sample intervals (reporting, not a guarantee)
```

These milestones gate **engineering progress** (if M-2's weighted path does not empirically beat the unweighted path on data with *known* shift, the real-data result cannot exist — fix the layer first). They are explicitly **not** certificate gates: no milestone claims `≥ α`, an "integrated budget bound," or an identifiability result. The OOD step is a **measured screen** with reported leakage on a stated, swappable `O` — distribution-free OOD detection is provably not learnable (Fang et al. 2022), so there is nothing to certify there. The combine step combines the two weights via the per-`x` corrector `Ẑ(x)`: `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` ([method_note.md §3.3](docs/method_note.md)) — **not** the naive product `ŵ_cov·ŵ_lab`, which double-counts. The protocol that fixes what each real-data milestone measures is [preregistration.md](docs/preregistration.md); this ladder defers all metric and cutoff definitions to it.

**Attribution discipline (the rule that keeps the win attributable).** The headline is produced with a **standard backbone** + the standard uncertainty score `u(x)` + the **Mahalanobis++** L2-normalized tied-covariance OOD score (Müller et al. 2025; the [method_note.md §4.1](docs/method_note.md) primary detector — **keeping** its in-distribution-only L2-normalization and dropping only the OOD-coupled FGSM/ensemble add-ons; the plain Lee et al. 2018 tied-covariance score, normalization removed, is the canonical-baseline ablation). Fancier base models (DDU, SNGP, deep ensemble) appear **only as ablations layered on top**, never as the foundation. Reason: the measured selective-risk / auditability result is **base-model-agnostic** — a fancier net only moves *efficiency* (less abstention at equal measured risk), not the auditability discipline that is the contribution. Building the core on a fancier net would hand a reviewer the obvious question, *"is the win just the backbone?"*. Reframed honestly: **the realized-risk / auditability result is base-model-agnostic; the net only moves efficiency** ([method_note.md §1.3](docs/method_note.md); [positioning_memo.md](docs/positioning_memo.md)).

**Depth over breadth, stated honestly.** Two imaging benchmarks taken deep — **CAMELYON17-WILDS** (covariate-dominant, cross-hospital scanner/stain) and **CheXpert ↔ MIMIC-CXR** (mixed covariate + label shift, cross-site, *both directions* as two deployments). Multi-modality breadth (ECG, retina, a second pathology task, dermatology) is **explicit, honestly-scoped future work**, *not* a v1 obligation, and the discussion says so plainly. Dataset authority is [preregistration.md §2](docs/preregistration.md); the depth-over-breadth rationale is [positioning_memo.md "What this memo keeps"](docs/positioning_memo.md).

**Long pole — start now.** CAMELYON17-WILDS loaders (`p-lambda/wilds`) unblock the synthetic-through-real build immediately. **CheXpert ↔ MIMIC-CXR access is the schedule-critical long pole**: begin Stanford AIMI + PhysioNet/CITI credentialing **today** (human approval, not engineering, is the gate). The full ~3.5-month milestone timeline to the 2026-10-05 deadline is §8; the single decision baked in here is *credentialing starts immediately, in parallel with the CAMELYON17 build.*

---

## 2. Foundations status — what is already DONE

The foundations for this applied paper are complete. This section is a status board, not a work plan: everything below is written and authoritative, and the rest of this playbook builds on it rather than re-deriving it.

| Foundation | Artifact | Status |
|---|---|---|
| Competitor matrix | [`analysis/competitor_matrix.csv`](analysis/competitor_matrix.csv) — one row per audited work, with which trustworthy-prediction ingredient it does/doesn't supply and the reusable asset | **Done** — the honest contrast (esp. TRUECAM) is sourced from here, not re-litigated as a "gap" |
| Method spine (single source of truth for notation) | [`docs/method_note.md`](docs/method_note.md) — notation; selective prediction / abstention; shift-aware calibration; OOD routing; integrated decision rule; auditability layer; assumptions & limitations | **Done** |
| Positioning (applied contribution + venue fit) | [`docs/positioning_memo.md`](docs/positioning_memo.md) — the one-sentence contribution, what is deliberately *not* claimed, Discover Computing fit | **Done** |
| Pre-registration (evaluation protocol) | [`docs/preregistration.md`](docs/preregistration.md) — datasets, shift-type diagnostic, baselines, metrics, what is fixed before results | **Done** |
| Phase-0 learning notebook | `phase0_learning.ipynb` (steps 0.1.1–0.1.6: split conformal, CRC, selective + selection-bias trap, RCPS/CRC/LTT, weighted conformal, OOD/distance-awareness), cross-checked against canonical sources | **Done** |

**Honesty discipline carried from the docs (the #1 rule for everything below).** We claim **no new guarantee**. Each method's guarantee is cited only as a property **of that method under its own assumptions** (RCPS `(α,δ)` control under cal/test exchangeability; weighted conformal `1−α` *marginal* coverage under pure covariate shift with the *oracle* likelihood ratio; BBSE *consistency* under label shift with an invertible confusion matrix; MLLS *consistency* additionally requiring the base classifier be calibrated — which is exactly why BCTS is in the pipeline). Under realistic clinical shift those assumptions break, and we **measure and report** the resulting degradation — never certify it. The words *certify / certified / guarantee* are not used for the deployed pipeline.

Because the foundations are settled, the remainder of this playbook is purely a **build/experiment plan**: it points to the three docs as the authority and never restates or contradicts them.

---

## 3. Method — fully specified in the method note (do not restate here)

The method is **fully specified in [`docs/method_note.md`](docs/method_note.md)**, which is also the **single source of truth for all notation, weights, thresholds, and budgets**. This playbook does not restate it and introduces no theory. One-paragraph recap, only so the build stages below are legible:

> A **frozen** base model `f` produces logits, a recalibrated posterior `σ̃(f(x))`, and penultimate features `φ(x)`. A **selective gate** `g(x)=1[u(x)≤τ]` answers confident cases and defers the rest to a clinician ([§2](docs/method_note.md)). **Shift-aware weighting** corrects the source-calibrated risk for deployment: a domain-discriminator covariate weight `ŵ_cov(x)` and an MLLS+BCTS label-prevalence correction `ŵ_lab(y)` (with BBSE as the baseline/diagnostic), **combined via the per-`x` corrector `Ẑ(x)`** as `ŵ(x,y)=ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` — **not the naïve product**, which double-counts ([§3](docs/method_note.md)). An **OOD routing** screen sends suspected out-of-distribution scans (`o(x)>t̂_ood`) and untrustworthy-weight cases (`ŵ_cov>w_max`) to a human ([§4](docs/method_note.md)). An **integrated decision rule** assembles these into one accept event `A(x)` ([§5](docs/method_note.md)), and an **auditability layer** surfaces per-case and per-subgroup trust signals ([§6](docs/method_note.md)). The object of interest is the answered-case target risk `R_T^accept`; we calibrate thresholds on source, apply the weighted correction, and then **measure** realized selective risk, coverage, and OOD leakage on held-out target — reporting the degradation, never bounding it.

**What lives in the method note — go there, don't duplicate:**

- **Notation, symbols, weights, thresholds, budgets** — [`method_note.md §1`](docs/method_note.md) (and its §1.8 quick-reference table). This is the *only* place notation is defined; the old playbook's notation and symbol-decoder appendices are replaced by this pointer.
- **Selective prediction / risk-controlled abstention** — [§2](docs/method_note.md).
- **Shift-aware calibration**, including the `Ẑ(x)`-combine derivation and worked 2-class double-count example, and the reported reliability diagnostics — [§3](docs/method_note.md).
- **OOD routing** (Mahalanobis++ L2-normalized tied-covariance primary; plain Lee et al. tied-covariance, energy, kNN ablations; measured leakage on the exposure set `O`) — [§4](docs/method_note.md).
- **Integrated decision rule** and the two **separately-measured** operating budgets `α_acc`, `α_ood` — [§5](docs/method_note.md).
- **Clinician-facing auditability layer** — [§6](docs/method_note.md).
- **Assumptions and honest limitations**, anchored on the two impossibility results (Fang et al. 2022; Yang, Kuchibhotla & Tchetgen Tchetgen 2024) — [§7](docs/method_note.md).

**Explicitly dropped from the old §3 (the abandoned theory).** There is **no** formal theorem with explicit constants, **no** finite-sample selective+weighted-RCPS bound or label-shift corollary, and **no** Generalized-Label-Shift / Generalized-Target-Shift "identifiability tension" with a residual `r(x,y)=p_T(x|y)/p_S(x|y)` term. The applied combine is `Ẑ(x)`-division under a factorizable-shift premise **adopted as a modeling choice** (not identifiable from unlabeled target); its residual is **measured** on the small labeled slice `D_tar^lab`, never identified or certified. The two budgets `α_acc` and `α_ood` are reporting knobs, **not** a certified additive `α = α_acc + α_ood` union-bound. (See [`positioning_memo.md`](docs/positioning_memo.md) "What this memo drops" for the full list of withdrawn claims.)

---

## 4. Experimental plan — pre-registered in the protocol (do not duplicate)

The evaluation protocol is the authority in **[`docs/preregistration.md`](docs/preregistration.md)** — datasets, splits, shift construction, the shift-type diagnostic, baselines, metrics, subgroups, and everything fixed before results. This playbook recaps it only enough to drive the build sequence; the prereg governs any conflict.

**Two benchmarks, depth over breadth** ([prereg §2](docs/preregistration.md)):
- **CAMELYON17-WILDS** — binary tumor-vs-normal, cross-hospital scanner/stain shift; the **covariate-dominant** benchmark. Each source→target hospital pair is reported separately, **never pooled**. Loaders via `p-lambda/wilds`. *Unblocks the build now.*
- **CheXpert ↔ MIMIC-CXR** — shared-label chest-radiograph finding classification, cross-site **mixed covariate + label** shift; **both directions** run as two separate deployments. *Credentialing (Stanford AIMI; PhysioNet + CITI) is the long pole — start immediately.*
- **Far-OOD exposure set `O`** — off-modality / wrong-anatomy / non-medical probes; *stated and swappable*, used to set `t̂_ood` and measure leakage ([method_note §4.3](docs/method_note.md), [prereg §5.2](docs/preregistration.md)).

**The shift-type diagnostic** ([prereg §3](docs/preregistration.md)): per source→target pair, the **domain-discriminator AUROC** on `φ(x)` is the shift-severity signal. A fixed-before-results cutoff `τ_disc` only **tags the reporting regime** — *pure-label-shift* (AUROC near chance) vs *combined covariate+label* — so the cleaner label-shift result is never silently substituted for the harder combined case. It is a regime **TAG, not a guarantee gate**.

**Four baselines, exactly** ([prereg §4](docs/preregistration.md)) — same folds, same frozen `f`, same target test set:
1. naive split-conformal / split-RCPS (`ŵ≡1`, the "ignore the shift" reference);
2. temperature-scaling-only (post-hoc calibration, no weighting, no OOD routing);
3. a **TRUECAM-style empirical pipeline** (OOD detect-and-remove) — the honest head-to-head; we beat it by **measuring and reporting** the residual leakage it leaves unquantified, *not* by a stronger guarantee;
4. BBSE label-weighting-only (the label-shift reference and consistency diagnostic).

*(Wang & Qiao 2025 is **not** a baseline; the exotic label-shift estimators RLLS/GS-B³SE/FMAPLS are **not** estimator ablations; the temporal DtACI track is **scope-cut**. The full pipeline of method_note §§3–5 is the method under test against the four above.)*

**Metric families** ([prereg §5](docs/preregistration.md)) — all measured on held-out target, reported with sampling uncertainty, none a certificate:
- **realized selective (accepted) risk** `R̂_T^accept` via the self-normalized (Hájek) estimator with a betting/hedged-capital interval at confidence `δ`;
- **coverage**, decomposed into the three routing outcomes (abstain-on-weight, route-on-OOD, defer-on-uncertainty);
- **AURC**; target **ECE** of `σ̃`;
- **OOD AUROC / FPR95**, measured far-OOD **leakage** vs `α_ood`, and **exposure-set sensitivity**;
- **subgroup audit** — per-subgroup selective risk, abstention/defer rate, and routing rates with finite-sample intervals (reporting, **not** a fairness guarantee);
- **reliability diagnostics** (not gates): `n_eff` (Kish), `κ(Ĉ_S)`, the `q̂_T`-vs-`Ĉ_S p̂_T` consistency check, per-pair discriminator AUROC, clip-induced abstention rate.

**What the prereg fixes before results** ([prereg §6](docs/preregistration.md)): the `(τ,λ,t_ood)` grid `G` and its LTT multiple-looks wrapper (a plain multiplicity correction, **no** new validity claim); the operating budgets `α_acc`, `α_ood`, `δ`; the subgroups; the minimum-coverage floor; routing/weight controls (`w_max`, `ĉ`, `[w_lab,min,w_lab,max]`, Mahalanobis++ (L2-normalized) primary, MLLS+BCTS primary, `σ̃` recalibrator); and the stress axes (small `m`, source→target pair/direction, target-prevalence sweep). The labeled slice `D_tar^lab` is **measurement-only** — it identifies nothing and certifies nothing ([prereg §7](docs/preregistration.md), [method_note §1.7](docs/method_note.md)); if unavailable for a pair, the combined-shift result is reported **uncorrected**.

**What is explicitly not claimed** ([prereg §8](docs/preregistration.md)): no certificate of `R_T^accept`; no certified `α = α_acc + α_ood` split; no identification from `D_tar^lab`; no distribution-free OOD guarantee; no new guarantee from LTT under shift; no per-case guarantee. The success criterion is a **measurement** statement — does the shift-aware pipeline empirically maintain/improve realized selective risk versus the naive baseline, and does the OOD screen measurably reduce leakage — **not** "accepted risk holds ≤ α."

*The fixed object is a measurement-and-reporting protocol, not a claim. Anything in this playbook that touches datasets, baselines, metrics, or success criteria defers to [`docs/preregistration.md`](docs/preregistration.md).*

---

## 5. Build sequence — engineering milestones

The pipeline is built one layer at a time. Each layer is validated on **synthetic data with a known shift** before the next is stacked — but the synthetic checks are **empirical engineering milestones, not certificate gates**. A milestone asks a *measurement* question — *does the shift-aware layer empirically maintain or improve the realized selective risk where the naive baseline degrades? does the OOD screen measurably reduce far-OOD leakage?* — and **never** "does a bound hold (`≤ α`)." Consistent with the honesty discipline of [method_note.md](docs/method_note.md), no synthetic check proves a guarantee; each method's own guarantee is a property of *that method under its assumptions* (cited in [method_note.md §3.5](docs/method_note.md) and [§7](docs/method_note.md)), and under shift we **measure and report** degradation.

The value of building staged is purely de-risking: if a synthetic milestone shows the layer does *not* do what it should on data where the truth is known, the corresponding real-data result cannot be trusted — stop and fix the layer before stacking the next. This is debugging infrastructure for one paper, not a ladder of lesser papers.

> **Why these milestones are measurements, not gates.** "Restores/maintains realized selective risk" below always means: *the realized weighted accepted-in-scope risk `R̂_w` (the self-normalized Hájek estimator of [method_note.md §1.7](docs/method_note.md)) measured on a held-out target split, reported with its betting interval, moves in the expected direction relative to the unweighted naive baseline.* It is never a claim that the risk is certified to sit below a budget. On synthetic data we additionally know the oracle weights, so the milestone also checks that the *estimated* weights track the oracle — a debugging convenience the real benchmarks do not afford.

**Repo scaffold.** Build into the tree in [§5.4](#54-repo-tree). Pin the environment ([Appendix E](#appendix-e--environment--command-cheat-sheet)), wire experiment tracking, Hydra configs + `set_seed()`, and `pytest`. The fold-disjointness discipline of [method_note.md §1.7](docs/method_note.md) (`D_cal`, `D_bbse^src`, `D_disc`, `D_tar`, `D_tar^lab`, `O`, held-out test all mutually disjoint) is **asserted in code** from Stage A onward — leakage between folds is the single most common way a milestone passes for the wrong reason.

Effort key: **S** ≤ a day · **M** a few days · **L** a week+.

---

### Stage A — selective layer *(M)*

**Files.** `conformal/selective.py` (selection gate + accepted-region calibration), `conformal/rcps.py` (RCPS threshold selection with the WSR / hedged-capital betting upper bound from `gostevehoward/confseq`, `betting.py`). Reference impl: `aangelopoulos/rcps`. The selective-risk functional, the RCPS `inf`-rule, and the betting UCB are specified in [method_note.md §2.2](docs/method_note.md); the WSR bound is chosen because importance weights (Stage B onward) inflate the risk-estimate variance.

**Milestone A.** On a held-out **source** split (exchangeable, no shift), the selective layer *runs end-to-end and reproduces the risk–coverage trade-off*: sweeping `τ` traces a monotone risk–coverage curve, and the RCPS-selected `λ̂` lands at the intended operating point on that curve across many calibration/test resamples. This confirms the selection-on-the-accepted-region calibration and the betting UCB are wired correctly — it is a *reproduction-of-the-known-trade-off* milestone, not a claim that accepted risk is certified. (Under exchangeability RCPS does carry its own `(α,δ)` property, cited as a property of RCPS per [method_note.md §2.2](docs/method_note.md); we do not re-assert it once shift enters.)

---

### Stage B — covariate weights *(M)*

**Files.** `conformal/weights.py` (domain discriminator `d(x)` on the frozen embedding `φ(x)` + clipped covariate ratio `ŵ_cov`), weighted path in `rcps.py`. The discriminator density-ratio estimator, the base-rate constant `ĉ = n_S/n_T`, and the clip/route cap `w_max` are in [method_note.md §1.5](docs/method_note.md); the weighted-quantile convention follows `ryantibs/conformal`.

**Milestone B.** On a held-out **target** split with a *known synthetic covariate shift*, the **weighted** pipeline *measurably restores realized selective risk where the unweighted naive baseline degrades*: `R̂_w` under `ŵ_cov` moves measurably back toward its source level while the `ŵ ≡ 1` baseline's measured target risk visibly climbs. Because the shift is synthetic the oracle `w_cov` is known, so the milestone also checks `ŵ_cov` tracks the oracle. This is a measured restoration of risk, not a recovered guarantee — consistent with the impossibility of a finite-sample certificate once the weight is estimated ([method_note.md §7](docs/method_note.md); Yang, Kuchibhotla & Tchetgen Tchetgen 2024). Report the clip-induced abstention rate.

---

### Stage C — label-shift weights *(M)*

**Files.** `conformal/label_shift.py`. **Primary** prevalence estimator: **MLLS + BCTS** (recalibrate `f` with bias-corrected temperature scaling, then EM), per Alexandari et al. 2020 via the `abstention` package in `kundajelab/labelshiftexperiments`. **Baseline / diagnostic:** vanilla BBSE `ŵ_lab = Ĉ_S⁻¹ q̂_T` (Lipton et al. 2018). Combine the covariate and label weights via the per-`x` corrector `Ẑ(x)` — `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)`, **not** the naive product `ŵ_cov·ŵ_lab`, which double-counts (worked 2-class example in [method_note.md §3.3](docs/method_note.md)). Floor/ceiling `ŵ_lab ∈ [w_lab,min, w_lab,max]` after simplex projection.

**Milestone C.** On a held-out target split with a *known synthetic prevalence shift* (class-conditionals held fixed), **label-aware weighting measurably improves realized risk over the covariate-only Stage-B pipeline**: `R̂_w` under the `Ẑ`-combined weight beats covariate-only, and the `Ẑ`-divide path matches the oracle `w_lab` while the naive-product path inflates by exactly `Z(x)` (the milestone reproduces the double-count and confirms `Ẑ` removes it). Report the **BBSE conditioning diagnostic** `κ(Ĉ_S)` and the **consistency check** `q̂_T` vs `Ĉ_S p̂_T` ([method_note.md §3.5](docs/method_note.md)) so the regime where the label estimate is least trustworthy is visible. This is a measured improvement over covariate-only — not a certificate, and not an identification of the combined case (the factorizable-shift premise is a modeling choice, [method_note.md §3.3](docs/method_note.md)/[§7](docs/method_note.md)).

---

### Stage D — OOD routing *(M)*

**Files.** `ood/detector.py`. **Primary** detector: **Mahalanobis++** — the L2-normalized tied-covariance Mahalanobis score (Müller et al. 2025) on the frozen embedding, built from `mueller-mp/maha-norm`, **keeping** its *in-distribution-only, parameter-free* L2-normalization (it sees no OOD, so the exposure set `O` stays disjoint/swappable) and deliberately dropping only the OOD-coupled add-ons — the FGSM input-perturbation and per-layer logistic-ensemble (they would couple the detector to the exposure set; [method_note.md §4.1](docs/method_note.md)). **Ablations:** the **plain Lee et al. 2018 tied-covariance** score (L2-normalization removed) as the canonical baseline, plus kNN (`deeplearning-wisc/knn-ood`) and energy (Liu et al. 2020). Routing rule and the decoupling of `t_ood` (far-OOD guard) from `w_max` (near-OOD/variance guard) are in [method_note.md §4.2](docs/method_note.md).

**Milestone D.** On the OOD-exposure set `O` and a *deliberately imperfect detector setting*, the screen **measurably reduces far-OOD leakage into the answered path**: report **AUROC / FPR95** of `o(x)` separating in-scope from `O`, and the **measured leakage** — the fraction of `O` scoring `≤ t̂_ood` — at the operating `t̂_ood` set to spend the `α_ood` budget on `O`. The milestone is that routing *reduces* the measured leakage relative to no screen, with `α_ood` reported as a **separately-measured operating budget**, never a certified additive term in any `α = α_acc + α_ood` union-bound. Distribution-free OOD detection is provably not learnable in the unrestricted setting (Fang et al. 2022; [method_note.md §4.4](docs/method_note.md)), which is exactly why this is a measured screen and the leakage is audited, not promised.

---

### Stage R — real-data headline *(M each dataset)*

Run the **full pipeline** of [method_note.md §§3–5](docs/method_note.md) (MLLS+BCTS label weight, discriminator covariate weight, `Ẑ`-combine, OOD routing, LTT-wrapped `(τ,λ,t_ood)` grid, optional `D_tar^lab` scalar correction) against the **four pre-registered baselines** — naive split-conformal (`ŵ ≡ 1`), temperature-scaling-only, TRUECAM-style OOD detect-and-remove (`iamownt/TRUECAM`, run as the head-to-head empirical baseline), and BBSE label-weighting-only — on **both benchmarks**, across many splits and the pre-registered stress axes. Datasets, folds, baselines, and metrics are the authority of [preregistration.md §§2–6](docs/preregistration.md); recap only:

- **CAMELYON17-WILDS** (`p-lambda/wilds`) — covariate-dominant cross-hospital shift; each hospital pair reported separately, never pooled ([preregistration.md §2.1](docs/preregistration.md)).
- **CheXpert ↔ MIMIC-CXR** — mixed covariate + label cross-site shift; **both directions** as two deployments ([preregistration.md §2.2](docs/preregistration.md)). **Credentialing (Stanford AIMI; PhysioNet + CITI) is the long pole — start now** (see [§8](#8-milestone-timeline-2026-06-26--2026-10-05--35-months)).
- **Far-OOD exposure set `O`** — off-modality / wrong-anatomy / non-medical probes, stated and swappable ([method_note.md §4.3](docs/method_note.md)).

**Milestone R.** On **both** datasets, the shift-aware pipeline **maintains/improves realized selective risk on answered cases relative to the naive and TRUECAM-style baselines where they measurably break — measured, not certified**: report `R̂_T^accept` (Hájek, with its betting interval) and the coverage decomposition (abstain-on-weight / route-on-OOD / defer-on-uncertainty) on held-out target, alongside AURC and target ECE of `σ̃` ([preregistration.md §5.1](docs/preregistration.md)). The headline is a *measured comparison* — our measured residual risk/leakage is smaller than the baselines' under the same folds and frozen `f` — not a claim that our risk is held below a budget. Per [preregistration.md §6.4](docs/preregistration.md), operating points below the minimum-coverage floor are reported as **degenerate** (risk driven down only by answering almost nothing), so a favorable number cannot be manufactured by collapsing coverage. *If a result is unconvincing, iterate the method, not the plot;* a characterized failure regime is a feature of the failure-mode catalog ([method_note.md §3.6](docs/method_note.md)), not an embarrassment.

> **Attribution discipline (keep the win attributable).** Milestone R is produced with the **standard backbone** as the base. Fancier nets (DDU / SNGP / deep ensemble) appear **only as ablations layered on top** — the realized-risk and auditability result is **base-model-agnostic**; a fancier net only moves *efficiency* (less abstention at equal measured risk), never validity. Building the core on a fancier net invites the *"is the win just the backbone?"* objection ([method_note.md §1.3](docs/method_note.md); [positioning_memo.md](docs/positioning_memo.md)).

---

### Stage S — subgroup / fairness audit *(M)*

Re-compute the selective-prediction quantities **within each pre-registered subgroup** ([preregistration.md §5.3, §6.3](docs/preregistration.md); [method_note.md §6.2](docs/method_note.md)): per-site / scanner / hospital, and the clinically salient strata available per corpus (sex, age band, self-reported race where ethically collected, case-type strata). For each subgroup, on labeled target (`D_tar^lab` where available), report **selective risk** `R̂^accept_s` with its finite-sample (betting) interval, **abstention / defer rate**, and **OOD-routing and weight-routing rates**, plus the per-subgroup discriminator AUROC and `κ(Ĉ_S)`. Selective prediction can *magnify* disparities (Jones et al. 2021), and a subgroup that is disproportionately abstained on is being silently under-served even when its accepted risk looks low — the audit surfaces that coverage gap explicitly.

**Milestone S.** Disparate selective risk and disparate abstention are made **visible and attributable** — to a scanner, a prevalence gap, or an ill-conditioned `Ĉ_S` — across subgroups on both benchmarks. This is **reporting with finite-sample intervals, not a fairness guarantee**: the subgroup audit is *not* wrapped in a new multiplicity-corrected certificate, and the LTT multiple-looks correction applies only to operating-threshold selection ([method_note.md §6.2](docs/method_note.md)), never to the subgroup audit. **`rohanhore/RLCP`** (randomly-localized conformal) is **optional** — if used, it strengthens toward *approximate* conditional coverage as a reported diagnostic, still not a per-subgroup guarantee.

---

### Code-review step *(S)*

After Stage S, run a conformal-aware code review focused on: fold leakage between `D_cal`, `D_disc`, `D_bbse^src`, `D_tar`, `D_tar^lab`, `O`, and test — **assert group-id (patient/slide) fold-disjointness, not just row-disjointness, and run a leakage probe** (e.g. a discriminator that should be at chance across folds that share no group); off-by-one in the conformal quantile / RCPS `λ̂`; **a unit-test that the weighted-path WSR betting UCB is actually exercised (not silently falling back to a range-bound default), plus the betting-interval non-overlap win-rule** (the reported improvement holds only where the intervals do not overlap); whether `ŵ_cov` and `ŵ_lab` are normalized and **combined via `Ẑ` rather than multiplied**; **a `w_max` sweep and a `Ẑ`-sensitivity report** (how realized risk / coverage / clip-rate move as the cap and the corrector are perturbed); **the discriminator-shortcut audit** (the covariate discriminator must key on genuine `φ`-shift, not a batch/scanner-id artifact); whether clipping bias is logged (clip fraction); whether selection is calibrated on the accepted region; and whether LTT wraps the joint `(τ, λ, t_ood)` grid. **Every headline number is reported at a matched answer-rate** so a coverage difference cannot masquerade as a risk difference. **All synthetic milestones (A–D) must still reproduce after fixes.** No milestone is a proof; each is a measured behavior the fix must not break.

---

### 5.1 Consolidated milestone checklist (honor in order)

Build the layers **in order** as engineering milestones, each validated on synthetic data first, then on the two benchmarks. Each milestone is an *empirical* check on realized selective risk and measured leakage — it proves no bound; the synthetic checks are debugging infrastructure (localize which layer is wrong before stacking the next), not certificate gates.

- [ ] **A** — selective layer runs and reproduces the risk–coverage trade-off on a held-out source split. (Exercises RCPS *under its own exchangeability assumption* — the one regime where its `(α,δ)` property applies.)
- [ ] **B** — weighted pipeline measurably restores realized selective risk under known covariate shift where the unweighted naive baseline degrades. Check is *relative improvement*, not "restores ≤ α"; report the clip-induced abstention rate.
- [ ] **C** — label-aware (`Ẑ`-combined) weighting measurably improves realized risk over covariate-only under prevalence shift; the `Ẑ`-divide path matches the oracle while the naive-product path inflates by `Z(x)`; BBSE conditioning `κ(Ĉ_S)` + the `q̂_T`-vs-`Ĉ_S p̂_T` consistency check reported. (No certificate; the combined weight is **not** identifiable from unlabeled target.)
- [ ] **D** — OOD screen measurably reduces far-OOD leakage on `O` (AUROC / FPR95 + measured leakage at `t̂_ood` against `α_ood`). `α_acc` and `α_ood` are **two separately measured operating budgets**, never a certified additive split.
- [ ] **R** — full pipeline maintains realized selective risk where naive + TRUECAM-style baselines measurably break, **measured on both datasets**, per source→target pair (never pooled), with the coverage decomposition and the minimum-coverage floor honored. Realized accepted risk is the Hájek `R̂_w` with its betting interval at `δ`.
- [ ] **S** — subgroup selective risk / abstention / routing rates reported with finite-sample intervals; disparities made visible and attributable. Reporting, **not** a fairness guarantee.
- [ ] **Attribution** — the headline is produced on the standard backbone; DDU/SNGP/deep-ensemble appear **only as ablations**; the measurement/auditability result is base-model-agnostic, the net moves only efficiency.
- [ ] **Citation locators** — every cited guarantee resolves to the exact theorem/assumption it is attributed to; no guarantee is attributed to the deployed pipeline.
- [ ] **Matched answer-rate** — every headline number reported at a matched answer-rate (a coverage gap cannot masquerade as a risk gap).
- [ ] **Leakage probe** — group-id (patient/slide) fold-disjointness asserted, and a leakage probe across folds returns chance.
- [ ] **WSR UCB exercised** — unit-test confirms the weighted path uses the WSR betting UCB (no range-bound fallback); betting-interval **non-overlap win-rule** applied to every claimed improvement.
- [ ] **`w_max` sweep + `Ẑ`-sensitivity** report logged; **discriminator-shortcut audit** confirms the covariate discriminator keys on genuine `φ`-shift, not a scanner/batch-id artifact.
- [ ] Code review complete; A–D still reproduce; fold-disjointness asserted.

---

### 5.2 Applied risk register

Every row is a build/measurement risk for the applied pipeline. The theory risks of the old playbook (combined-weight identifiability *as a validity risk*, vacuous Hájek bound *as new-math-needed*, multiple-testing-invalidity *as a validity risk*) are **dropped** — they presupposed a certificate this paper does not claim.

| Risk | Symptom | Mitigation | Where |
|---|---|---|---|
| **Data leakage across folds** | Suspiciously good target risk/coverage | Assert mutual disjointness of `D_cal`, `D_bbse^src`, `D_disc`, `D_tar`, `D_tar^lab`, `O`, test in code; code review | [method_note §1.7](docs/method_note.md); code-review step |
| **Self-deception via abstention** | "Low risk" bought by answering almost nothing | Report coverage *with* risk; pre-registered minimum-coverage floor flags degenerate operating points | [preregistration §5.1, §6.4](docs/preregistration.md) |
| **Dataset-access latency** | Stuck waiting on MIMIC/CheXpert approval | Credential on day one; lead the build with CAMELYON17 | [§8](#8-milestone-timeline-2026-06-26--2026-10-05--35-months); [preregistration §2](docs/preregistration.md) |
| **Overclaiming (no new guarantee)** | "Valid under any shift" / "holds ≤ α" / "certified" creeps into prose | Every guarantee sentence names its method *and* its assumption; the words *certify/certified/guarantee* are banned for the deployed pipeline; reviewer-2 pass | [positioning_memo](docs/positioning_memo.md); [preregistration §8](docs/preregistration.md) |
| **Breadth creep** | Chasing ECG/retina/2nd-pathology; nothing submitted | Two imaging benchmarks taken deep; multi-modality is explicit future work | [positioning_memo](docs/positioning_memo.md); [preregistration §2](docs/preregistration.md) |
| **Calibration drift under shift** | Displayed posterior `σ̃(f(x))` miscalibrated on target; clinician-facing confidence stale; `Ẑ(x)` denominator biased | Recalibrate (temperature/Platt) on a held-out source fold; **measure** target ECE rather than assume it away; report drift with its sampling uncertainty | [method_note §3.5, §6.5, §7.6](docs/method_note.md); [preregistration §5.1](docs/preregistration.md) |
| **OOD exposure-set non-representativeness** | `t̂_ood` / leakage estimate are artifacts of the particular `O` chosen | `O` is a *stated, swappable* exposure model; report exposure-set sensitivity (how `t̂_ood` and leakage move when `O` changes) | [method_note §4.3](docs/method_note.md); [preregistration §5.2](docs/preregistration.md) |
| **Weight / label-shift estimation error** | Synthetic pass, real-data degradation | Clip + route the high-weight tail; MLLS+BCTS over vanilla BBSE under imbalance; report `n_eff`, `κ(Ĉ_S)`, clip-induced abstention rate; measure residual on `D_tar^lab` | [method_note §3.5, §1.7](docs/method_note.md); Milestones C, R |
| **Backbone confound** | Gains attributed to a fancier net, not the pipeline | Standard backbone is the base; DDU/SNGP/ensemble are ablations only; the auditability result is base-model-agnostic | [method_note §1.3](docs/method_note.md); attribution discipline |
| **Matched-coverage confound** | "Lower risk" is really a lower answer-rate, not a better answered set | Report every headline at a **matched answer-rate**; the accept-side co-variation figure shows risk and coverage move together | [preregistration §5.1](docs/preregistration.md); §7.2; code-review step |
| **Accept-side not validated** | Improvement shown only on the routed/declined tail, not on the *answered* cases | Validate the **accept-side** explicitly — realized `R̂_T^accept` on answered cases at matched coverage, not just the routing rates | [method_note §5.2, §6.2](docs/method_note.md); [preregistration §5.1](docs/preregistration.md) |
| **Group-leakage** | Patient/slide appears in two folds → optimistic risk | Assert **group-id** fold-disjointness (not row-level); leakage probe in code review | [method_note §1.7](docs/method_note.md); code-review step |
| **Hájek-interval validity** | The self-normalized (Hájek) betting interval mis-covers under heavy weight tails / small `n_eff` | Report `n_eff` beside every Hájek `R̂_w`; flag weight-dominated regimes; use the variance-adaptive WSR betting UCB; do **not** read the interval as a guarantee | [method_note §1.6, §1.7, §2.2](docs/method_note.md); [preregistration §5.1](docs/preregistration.md) |

---

### 5.3 Repo map — applied methods only

Each repo is used *as published* for the method it owns; its guarantee is a property of that method under its own assumptions (per [method_note §3.5](docs/method_note.md)). The theory-only repos (microsoft GLS, stanford-futuredata/SparseJointShift, isgibbs/DtACI, RLLS, GS-B³SE, FMAPLS) are **dropped** — they belonged to the abandoned temporal / exotic-estimator / GLS-identification tracks.

| Component | Repo | Role | Defer to |
|---|---|---|---|
| Risk-control spine (RCPS) | [aangelopoulos/rcps](https://github.com/aangelopoulos/rcps) | concentration bounds + `λ̂` reference | [method_note §2.2, §3.5](docs/method_note.md); [prereg §5.1](docs/preregistration.md) |
| Variance-adaptive UCB (WSR / betting) | [gostevehoward/confseq](https://github.com/gostevehoward/confseq) | `betting.py` hedged-capital bound (weights inflate risk-estimate variance) | [method_note §1.6, §2.2](docs/method_note.md) |
| Multiple-looks correction (LTT) | [aangelopoulos/ltt](https://github.com/aangelopoulos/ltt) | plain multiplicity correction over the `(τ,λ,t_ood)` grid — **not** a new validity guarantee | [method_note §3.4](docs/method_note.md); [prereg §6.1](docs/preregistration.md) |
| Less-conservative comparison (CRC) | [aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk) | expectation-control comparison row (no `δ` tail) | CRC kept as the less-conservative comparison row, [method_note §2.2, §3.5](docs/method_note.md); the repo is the canonical conformal-risk-control implementation and is an addition not enumerated in method_note §3.5's repo list |
| Covariate weights | [ryantibs/conformal](https://github.com/ryantibs/conformal) | `weighted.quantile` convention | [method_note §3.5](docs/method_note.md) |
| Label-shift estimator (primary) | [kundajelab/labelshiftexperiments](https://github.com/kundajelab/labelshiftexperiments) | the `abstention` package — MLLS+BCTS (Alexandari 2020); BBSE as baseline/diagnostic | [method_note §1.5, §3.5](docs/method_note.md); [prereg §4, §6.5](docs/preregistration.md) |
| OOD detector (primary) | [mueller-mp/maha-norm](https://github.com/mueller-mp/maha-norm) | **Mahalanobis++ as-is — keep the in-distribution-only L2-normalization**, drop only the OOD-coupled add-ons (FGSM perturbation, per-layer logistic ensemble); the plain Lee et al. 2018 tied-covariance score (normalization removed) is the canonical-baseline ablation | [method_note §4.1](docs/method_note.md) |
| OOD detector (ablation) | [deeplearning-wisc/knn-ood](https://github.com/deeplearning-wisc/knn-ood) | kNN OOD score | [method_note §4.1](docs/method_note.md) |
| Data loaders | [p-lambda/wilds](https://github.com/p-lambda/wilds) | CAMELYON17-WILDS | [prereg §2.1](docs/preregistration.md) |
| Head-to-head baseline | [iamownt/TRUECAM](https://github.com/iamownt/TRUECAM) | TRUECAM-style OOD detect-and-remove — the honest contrast | [prereg §4](docs/preregistration.md); [positioning_memo](docs/positioning_memo.md) |
| Subgroup / local coverage (optional) | [rohanhore/RLCP](https://github.com/rohanhore/RLCP) | **optional** approx. conditional coverage; the subgroup audit is *reporting*, not a conditional-coverage guarantee | [method_note §6.2](docs/method_note.md); [prereg §5.3](docs/preregistration.md) |
| Your repo | [Toepatella/calibrated-selective-prediction-clinical-shift](https://github.com/Toepatella/calibrated-selective-prediction-clinical-shift) | the build | — |

---

### 5.4 Repo tree

> Reframed from the abandoned theory build: **`conformal/aci.py` is removed** (the temporal DtACI track is scope-cut for the applied paper — it appears in none of the three docs), and **`conformal/budget.py` is reframed** away from a certified `α = α_acc + α_ood` split toward bookkeeping for **two separately-measured operating budgets** ([method_note §1.6, §4.3](docs/method_note.md), [preregistration §6.2](docs/preregistration.md)). The `tests/` are **empirical-milestone tests** (the synthetic A–D checks of §5.1), not certificate gates.

```
calibrated-selective-prediction-clinical-shift/
├── README.md                 # overview + exact data-access steps
├── requirements.txt / env    # pinned deps + lockfile (incl. confseq, wilds)
├── configs/                  # one YAML per experiment (Hydra)
├── data/
│   ├── camelyon17.py         # WILDS loaders + frozen split manifest
│   ├── cxr.py                # CheXpert/MIMIC loaders + label harmonization
│   └── splits/               # frozen index files (all folds disjoint)
├── models/                   # frozen backbones; cached logits + features (φ); recalibrator σ̃
├── ood/
│   └── detector.py           # Maha++ L2-normalized tied-cov (primary); plain Lee-2018 + kNN/energy ablations
├── conformal/
│   ├── selective.py          # selection gate g(x)=1[u≤τ]; accepted-region calibration   [Stage A]
│   ├── rcps.py               # RCPS λ̂ + WSR (betting) UCB + weighted (Hájek) path        [Stage A/B]
│   ├── crc.py                # CRC comparison row (expectation control, no δ)
│   ├── ltt.py                # Learn-then-Test multiple-looks correction over (τ,λ,t_ood)
│   ├── weights.py            # domain discriminator d(x); clipped ŵ_cov; w_max routing    [Stage B]
│   ├── label_shift.py        # MLLS+BCTS (primary) + BBSE (diagnostic); Ẑ combine — NOT product [Stage C]
│   └── budget.py             # α_acc / α_ood: two SEPARATELY-MEASURED operating budgets
│                             #   (NOT a certified additive split)
├── experiments/              # run scripts (method × dataset × seed); pre-registered stress axes
├── analysis/
│   ├── metrics.py            # realized selective risk + betting CI; coverage decomposition; AURC;
│   │                         #   ECE; OOD AUROC/FPR95 + leakage vs α_ood; subgroup audit;
│   │                         #   reliability diagnostics (n_eff, κ(Ĉ_S), q̂_T-vs-Ĉ_S p̂_T, clip rate)
│   ├── figures.py            # one script regenerates every figure from logs
│   └── competitor_matrix.csv
├── tests/                    # EMPIRICAL milestone tests A–D (pytest) — measured behaviors,
│                             #   NOT bound-proving gates; no changepoint test
├── docs/                     # method_note · positioning_memo · preregistration (authoritative)
└── paper/                    # manuscript + figures
```

*Notation, weights, thresholds, and budgets are the single source of truth in [method_note.md §1](docs/method_note.md) (and its §1.8 quick-reference table); positioning is [positioning_memo.md](docs/positioning_memo.md); the evaluation protocol is [preregistration.md](docs/preregistration.md). This build section defers to them rather than restating.*

**Reproducibility release manifest (pre-registered, frozen under one version tag).** The release is a single versioned artifact bundle, tagged by the **split hash** so the tag, the splits, and every downstream cached object are pinned together:

- frozen **patient/slide-level split index files** (all folds disjoint per [method_note §1.7](docs/method_note.md));
- cached **per-image logits + `φ` features**;
- the fitted **`σ̃` recalibrator** and the **`Ẑ` corrector**;
- **run logs** (one logged run per method × dataset × seed).

`analysis/metrics.py` and `analysis/figures.py` are built against this **from-logs contract from Stage A onward**: every metric, table, and figure regenerates from the manifest, never from a live model pass. **Credentialed-data-safe boundary:** ship **CAMELYON17 artifacts end-to-end** (open access); for **CheXpert/MIMIC ship the split-manifests + regeneration scripts only**, after checking PhysioNet/AIMI redistribution terms — never the cached images/logits if the terms forbid redistribution. (See also [Appendix E](#appendix-e--environment--command-cheat-sheet) for the from-logs commands.)

---

## 6. Run, ablate, report (deep, on the two benchmarks)

Everything in this section is **measured on held-out target data and reported with its sampling uncertainty** — no result here is a certificate. The datasets, source→target pairs, baselines, metrics, subgroups, and stress axes are fixed in advance in the [preregistration](docs/preregistration.md); this section is the run/ablate/report discipline, not a redefinition of them. The honesty rule carries through every table: a guarantee sentence names the method it belongs to and the assumption that method needs (RCPS `(α,δ)` under exchangeability; weighted-conformal `1−α` *marginal* coverage under pure covariate shift with the oracle ratio; BBSE/MLLS *consistency* under label shift), and under shift we report the realized number, never a recovered guarantee. The two operating budgets `α_acc` and `α_ood` are **separately measured** — their sum is a reporting convenience, never a union-bound certificate ([method_note §1.6](docs/method_note.md), [prereg §6.2](docs/preregistration.md)).

The pipeline under test is the full method of [method_note §§3–5](docs/method_note.md) (MLLS+BCTS label weight, discriminator covariate weight, `Ẑ`-combine, OOD routing, LTT-wrapped `(τ,λ,t_ood)` grid, optional `D_tar^lab` scalar correction), run against the four pre-registered baselines ([prereg §4](docs/preregistration.md)): naive split-conformal (`ŵ≡1`), temperature-scaling-only, the TRUECAM-style detect-and-remove pipeline, and BBSE-label-weighting-only.

### 6.1 The ablation set (one fixed grid, regenerated from logs)

Each ablation changes **one** component and re-measures the [prereg §5](docs/preregistration.md) metrics on the same folds, same frozen `f`, same target test set, so any movement is attributable to that component. Tabulate all of these from logged runs (one script → every table/figure):

- **OOD detector `{Mahalanobis++ (primary), plain Lee-2018, energy, kNN}`** ([method_note §4.1](docs/method_note.md)). Report OOD AUROC / FPR95 separating in-scope from the exposure set `O`, and the measured far-OOD leakage at the operating `t̂_ood`, for each detector. The routing logic is identical across detectors; this isolates the screen's discriminative power, not a validity claim. Deploy the **Mahalanobis++** L2-normalized tied-covariance score as the primary (the `maha-norm` repo, **keeping** its in-distribution-only L2-normalization), with the plain Lee et al. 2018 tied-covariance score (normalization removed) as the canonical-baseline ablation, and deliberately omit the FGSM-perturbation and per-layer logistic-ensemble add-ons so the detector stays OOD-agnostic and `O` stays swappable.
- **Label-shift estimator `{MLLS+BCTS (primary), BBSE (baseline/diagnostic)}`** ([method_note §3.5](docs/method_note.md), [prereg §6.5](docs/preregistration.md)). This is an estimator comparison, **not** a "validity-flat / efficiency-moves" certificate ablation: report the realized selective risk, the prevalence-correction error on `D_tar^lab`, and the BBSE conditioning `κ(Ĉ_S)`. The docs fix exactly these two estimators — do not reach for a fancier label-shift estimator from the literature as a substitute.
- **Covariate-only vs combined weight.** Covariate-only (`ŵ_cov`) versus the combined `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` ([method_note §3.3](docs/method_note.md)). This is where the `Ẑ`-divide (not the naïve product) earns its keep on the mixed benchmark; report realized risk and coverage under each, with the per-pair discriminator AUROC as the shift-severity context.
- **`w_max` and target-size `m` sensitivity.** Sweep the routing cap `w_max` and the unlabeled-target size `m` over their fixed grids ([prereg §6.6](docs/preregistration.md)); report realized risk, coverage, clip-induced abstention rate, and `n_eff` at each cell. Small `m` and aggressive clipping are where the weighted estimate gets unreliable — the point is to **show** where, not to hide it.
- **Base-model efficiency frontier `{standard, DDU/SNGP, deep ensemble}`** — an **efficiency comparison, not a validity claim** ([method_note §1.3](docs/method_note.md); attribution discipline below). The standard backbone is the deployed base; DDU/SNGP and deep ensembles appear **only** here. The frontier plot shows that a fancier net buys *efficiency* (less abstention at the same realized risk), while the measured-risk/auditability behavior is base-model-agnostic — so the headline win is attributable to the pipeline, not the net. State it as exactly that, never as "validity flat."

**Attribution discipline (keep).** Building the headline on SNGP would hand a reviewer the kill — *"is the win just the backbone?"* The standard backbone is the foundation; every fancier net is an ablation layered on top. The reported reliability/auditability result does not depend on the net; the net only moves the efficiency frontier.

### 6.2 Confound checks (rule out the cheap explanations)

For each positive result, run the analysis that rules out the boring explanation, per [prereg §5–6](docs/preregistration.md):

- **Coverage *with* risk — low risk must not be bought by near-total abstention.** Always report realized selective risk *alongside* coverage and its three-way routing decomposition (abstain-on-weight, route-on-OOD, defer-on-uncertainty; [method_note §5.2](docs/method_note.md)). Enforce the pre-registered **minimum-coverage floor** `coverage_min` ([prereg §6.4](docs/preregistration.md)): any operating point below it is reported as **degenerate**, not as a favorable risk number.
- **Weight-error stress.** Deliberately degrade the domain discriminator and re-measure; report how realized risk and `n_eff` move as the covariate weights get noisier.
- **Small-`m` breakdown.** Drive `m` down its grid until the weighted estimate becomes unreliable; report the regime where `n_eff` collapses and the realized risk estimate destabilizes (flagged via `n_eff`, never silently averaged in).
- **BBSE-error stress.** Push `Ĉ_S` toward ill-conditioning (weak base model, rare/hard classes) and re-measure; report `κ(Ĉ_S)`, worst-class error, and the `q̂_T`-vs-`Ĉ_S p̂_T` consistency check ([method_note §3.5](docs/method_note.md)) as the label-shift correction degrades.

### 6.3 Find — and report — a shift where it degrades

A **characterized failure boundary is a feature of a measure-and-report paper, not an embarrassment.** Use the pre-registered stress axes ([prereg §6.6](docs/preregistration.md)) — small `m`, the hardest CAMELYON17 hospital pair, both CheXpert↔MIMIC directions, the target-prevalence sweep — to locate where the realized selective risk degrades or the combined-shift residual on `D_tar^lab` grows large. Then report it as the empirical failure-mode catalog ([method_note §3.6](docs/method_note.md)): the discriminator-AUROC level at which anti-causal `p(x∣y)`-invariance visibly breaks, the `κ(Ĉ_S)` at which the label correction misfires on minority classes, and the exposure-set hardness at which OOD leakage climbs. This catalog *is* the auditability contribution — it shows where the correction is least trustworthy rather than papering over it.

**Done when** every table and figure regenerates from logged runs by one script, the minimum-coverage floor and the reliability diagnostics (`n_eff`, `κ(Ĉ_S)`, per-pair discriminator AUROC, clip-induced abstention rate) accompany every number, and the top alternative explanations from §6.2 are ruled out with specific plots.

---

## 7. Write the paper

**Venue.** Springer *Discover Computing*, the **"Intelligent Medicine: Machine Learning and Explainable AI for Next-Generation Healthcare"** collection (open access; **deadline 2026-10-05**). The collection's centre of gravity is **explainability and trustworthy / auditable ML** for clinical decision support — match the template now and write to that emphasis. The fit is direct and is argued in [positioning_memo "Venue fit"](docs/positioning_memo.md): the deliverable is a measured, subgroup-auditable account of trustworthiness under realistic clinical shift, with no overclaimed guarantee.

**Lead the paper with trustworthy, auditable selective prediction under shift — *not* "guarantee xor shift-robustness."** There is no maintained distribution-free guarantee to claim and no gap-finder framing to revive. The one-sentence contribution is the [positioning_memo](docs/positioning_memo.md) sentence: an applied, auditable selective-prediction pipeline assembled from existing methods whose contribution is a **measurement / auditability discipline and a clinician-facing trust interface**, not a new statistical guarantee.

### 7.1 Manuscript structure (no theory section)

1. **Introduction** — the clinical trust gap (models that answer every case, silently, under deployment shift); the one-sentence contribution; and the **honesty stance up front** — we compose published methods, claim no new guarantee, and *measure and report* degradation under shift, anchored by the two impossibility results ([method_note §7.1](docs/method_note.md): Fang et al. 2022 on OOD non-learnability; Yang, Kuchibhotla & Tchetgen Tchetgen 2024 on no finite-sample conditional coverage under estimated covariate weights) that make measure-and-report the only honest stance.
2. **Related work + competitor matrix** — one row per trustworthy-selective-prediction work, columns: abstains? · distribution-free property and its assumption · covariate / label shift handled? · OOD handling · real medical shift data? The honest contrast with **TRUECAM** ([positioning_memo](docs/positioning_memo.md)): it restores validity by detecting and removing OOD inputs and does not claim a maintained distribution-free guarantee in deployment; we beat it only by **measuring and reporting** the residual leakage it leaves unquantified, not with a stronger guarantee.
3. **Methods** — point to [method_note.md](docs/method_note.md); do **not** restate the spine or re-derive notation. Summarize the pipeline (selective gate → shift-aware weighted calibration → OOD screen → integrated decision rule → audit layer) and the one estimation step worth showing inline: the combine-by-`Ẑ` correction (`ŵ(x,y)=ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)`, not the product) with the worked 2-class double-count example ([method_note §3.3](docs/method_note.md)).
4. **Experimental setup** — point to [preregistration.md](docs/preregistration.md); briefly recap the two benchmarks (CAMELYON17-WILDS covariate-dominant, per hospital pair; CheXpert↔MIMIC mixed, both directions), the four baselines, the fold-disjointness discipline, the shift-type diagnostic (per-pair discriminator AUROC with the fixed-before-results cutoff `τ_disc`, a **regime tag, not a guarantee gate**), and the fixed operating-point grid + budgets + stress axes.
5. **Results** — the empirical spine, in order:
   - **Naive breaks.** The `ŵ≡1` split-conformal baseline's realized target risk degrades under shift (it carries RCPS's property only under exchangeability, which the deployment breaks).
   - **Shift-aware holds up under measurement.** The full pipeline's realized selective risk is the **lowest measured** among the baselines **at a matched answer-rate** — stated as *measured improvement*, never "risk held at ≤ α." The accept-side co-variation figure (§7.2) shows the answered-case risk and coverage moving together so the gain is not a hidden coverage drop.
   - **OOD screen measurably reduces leakage.** Measured far-OOD leakage on `O` vs the TRUECAM-style detect-and-remove baseline, with exposure-set sensitivity **and a near-OOD tier** (inputs close to the in-scope manifold, where the screen is hardest).
   - **Subgroup / fairness audit.** Per-subgroup selective risk, abstention, and routing rates with finite-sample intervals ([method_note §6.2](docs/method_note.md)) — reporting, not a fairness guarantee.
   - **Operating-envelope / failure boundary.** The §6.3 characterized boundary: where, on the stress axes, the correction degrades.
6. **Clinician-facing auditability / interpretability** — the explainability hook of the venue, drawn from [method_note §6](docs/method_note.md): the per-case calibrated risk panel (recalibrated posterior `σ̃`, weight-context representativeness chip `ŵ(x,ŷ)`, cohort `n_eff` flag), the subgroup audit, and the explicit OOD + shift-regime flags. Frame it as *explainability via auditability* — the interface explains where and why the model declines and where the shift correction is least trustworthy, an interpretable account of the model's own reliability at the point of care.
7. **Honest limitations / discussion** — point to [method_note §7](docs/method_note.md): per-component assumptions and where each breaks; the cleanest regime (pure label shift, fewest assumptions) named as the cleanest *evaluation* regime, not a corollary; and the genuinely open edges — combined-shift non-identifiability from unlabeled target, the untestable exposure-set transfer assumption behind OOD leakage, and source-fit recalibration drift — plus **multi-modality breadth as honestly-scoped future work** ([positioning_memo](docs/positioning_memo.md): two imaging benchmarks, depth over breadth). Do **not** frame the open edges as "closing a 5-condition gap."
8. **Reproducibility** — pinned env, configs, seeds, one script per figure; a data-access README (CheXpert/MIMIC cannot be redistributed — ship splits + credentialing steps); Zenodo DOI in the paper.

### 7.2 Figures

Headline realized selective-risk vs coverage under covariate+label shift (full pipeline vs the four baselines, **with every headline number reported at a matched answer-rate** and the minimum-coverage floor marked); the **accept-side co-variation figure** (realized `R̂_T^accept` and coverage plotted together on the *answered* cases, so a risk drop bought by answering less is visible as a coverage drop, never hidden); measured OOD leakage vs the TRUECAM-style baseline with exposure-set sensitivity, **including a near-OOD tier** (inputs close to the in-scope manifold, not only far-OOD probes); risk–coverage / AURC curves; the **base-model efficiency frontier** (`{standard, DDU/SNGP, ensemble}` — labeled "efficiency moves, auditability is base-model-agnostic," never "validity flat"); the **`w_max` and `Ẑ`-sensitivity panels** (how realized risk, coverage, clip-rate, and `n_eff` move as the routing cap and the corrector are perturbed); the subgroup audit table with finite-sample intervals; the shift-regime / per-pair discriminator-AUROC diagnostic; and an annotated screenshot of the running clinician-facing per-case panel (the explainability figure, rendered by the §8.1 CORE renderer). **Cut** the DtACI change-point figure and any figure that would assert a combined or additive budget bound "holds" — both belong to dropped scope.

### 7.3 Prose discipline (the top priority)

Every guarantee sentence names its **method and its assumption**; no guarantee is attributed to the pipeline as a whole. Never write "risk is held at ≤ α" or "certified / guaranteed" of the deployed pipeline — we *want* `R_T^accept` low, *measure* it, and *report* the degradation ([positioning_memo](docs/positioning_memo.md) bans "certify/certified/guarantee" for the deployed system). `δ` is the confidence of a *reported interval*, not a PAC promise about the pipeline. Lead with auditability; quote TRUECAM's own "cannot be guaranteed in deployment" as the gap you *measure and report*, not one you out-guarantee.

### 7.4 Reviewer pass and release

- **Reviewer simulation** — three reviewers (a conformal/UQ-literate reader, a clinical-ML practitioner, a skeptical generalist). The UQ reviewer checks that every guarantee is correctly attributed to its method under its assumption and that no certificate is smuggled in; the clinician checks the abstention/deferral behavior and the audit panel are usable and the shifts clinically realistic; the generalist hunts for overclaiming and whether two imaging benchmarks suffice (the rebuttal: depth over breadth, with the cheapest credible extra modality named as future work). Apply the top revisions.
- **Honest "what remains open"** — write the one-paragraph open-edges statement from [method_note §7](docs/method_note.md): combined-shift non-identifiability, OOD-exposure-set transfer, recalibration drift, and multi-modality breadth — as named open edges, not bugs.
- **Release** — pinned env + lockfile, Hydra configs, seeds, one script regenerating every figure from logged runs; data-access README; archived Zenodo DOI cited in the manuscript.

---

## 8. Milestone timeline (2026-06-26 → 2026-10-05, ~3.5 months)

The **long pole is data credentialing** — PhysioNet/CITI for MIMIC-CXR and Stanford AIMI registration for CheXpert need human approval. **Start credentialing on day one.** CAMELYON17-WILDS is open-access and unblocks the build immediately, so the engineering ladder runs in parallel with credentialing.

### 8.0 CORE vs NICE-TO-HAVE (cut line for the deadline)

What a complete, submittable paper requires (**CORE**) versus what strengthens it if time allows (**NICE**). If the schedule slips, NICE is cut first and the CORE list is what ships. Everything CORE is built on the **primary benchmark** (CAMELYON17 under the NO-GO branch, see §8 dated GO/NO-GO) and is **credentialing-independent**.

| CORE (must ship) | NICE-to-have (cut first under slippage) |
|---|---|
| Full pipeline (selective gate → `Ẑ`-combined weighting → OOD screen → integrated decision rule → audit layer) on the primary benchmark | The second CheXpert/MIMIC direction (both deployments) |
| The **four pre-registered baselines** ([prereg §4](docs/preregistration.md)) on the primary benchmark | DDU/SNGP/deep-ensemble **efficiency frontier** ablation |
| Headline realized selective **risk–coverage at matched answer-rate** ([prereg §5.1](docs/preregistration.md)) | Energy / kNN **OOD ablations** (Mahalanobis primary is CORE) |
| The **OOD-leakage screen** with measured leakage on `O` vs `α_ood` | DAR stability checks **(C)–(E)** ([method_note §6](docs/method_note.md)) |
| The **subgroup audit** with finite-sample intervals ([prereg §5.3](docs/preregistration.md)) | `w_max` / `Ẑ`-sensitivity sweeps beyond the single reported panel |
| The **DAR construction-faithfulness check** (the record is faithful to the routing decision, not the diagnosis) | The near-OOD exposure tier beyond the headline `O` |
| **Standard backbone** as the deployed base | — |

> **Rebalance rule.** Because the CORE list is credentialing-independent, the **CAMELYON17 Stage R/S harness AND the runnable audit-panel renderer** are built and debugged **by ~Wk 5** (see §8.1 below), so when the mixed benchmark lands it is a **re-run on new data, not a fresh build**. Backbone and any deep-ensemble training are pulled into an **early window and cached once** ([§5.4 `models/`](#54-repo-tree)), so the later experimental blocks run on cached logits/features.

| Window | Milestone | Notes |
|---|---|---|
| **Wk 0–1** (Jun 26 – Jul 9) | **Unblock data + scaffold.** Submit MIMIC/PhysioNet+CITI and CheXpert/AIMI credentialing *immediately*. Pull CAMELYON17-WILDS. Scaffold the repo (§5.4), pin the env, wire configs/seeds + fold-disjointness asserts. **Train the standard backbone (and any deep-ensemble members) in this early window and cache logits + `φ` features once** ([§5.4 `models/`](#54-repo-tree)) so all later calibration/weight/diagnostic reruns are CPU-seconds. **Stage A** + **Milestone A**. | Credentialing clock starts now; CAMELYON17 carries the build while it runs. Backbone caching is pulled early so nothing downstream is GPU-blocked. |
| **Wk 1–4** (Jul 9 – Aug 6) | **Build the pipeline as engineering milestones.** In order: selective RCPS on the accepted region → covariate weights → label-shift weights (MLLS+BCTS, BBSE baseline) → `Ẑ`-combine → OOD screen. **Stages B, C, D** + Milestones B/C/D on synthetic shift, incl. the unreliability diagnostics. Lock the fixed-before-results grid/budgets/subgroups per [preregistration §6](docs/preregistration.md). **Schedule the runnable clinician-panel renderer over cached logits as a CORE artifact** (a real renderer, not a paper screenshot — see §8.1). | Each layer validated on synthetic data as an **empirical check** — *does the shift-aware pipeline maintain/improve realized selective risk vs naive; does the OOD screen measurably reduce leakage* — **not** a proof a bound holds. Never stack past a layer that fails its check. |
| **Wk 4–5** (Aug 6 – Aug 20) | **CAMELYON17 headline + code review + harness/renderer complete.** **Stage R** on CAMELYON17 (full pipeline + 4 baselines per hospital pair, many splits, stress axes); confound checks (§6.2); fix leakage / off-by-one / clip-bias. Attribution ablation frontier. **By the end of this window the CAMELYON17 Stage R/S harness AND the audit-panel renderer are built and debugged** (credentialing-independent), so the mixed-benchmark block becomes a re-run, not a build. By now CXR credentials should be landing. | First real-data result; covariate-dominant benchmark. The dataset-agnostic harness + renderer are now frozen. |
| **Wk 5–7** (Aug 20 – Sep 3) | **CheXpert↔MIMIC headline (both directions) — a RE-RUN, not a build.** Point the already-debugged Stage R/S harness at the mixed benchmark; full ablation set (§6.1): OOD `{Maha, energy, kNN}`, estimator `{MLLS+BCTS, BBSE}`, covariate-only vs combined, `w_max`/`m` sweeps, base-model efficiency frontier (on the early-cached backbones/ensembles). Run the prevalence sweep and locate the §6.3 failure boundary. **Stage S** subgroup audit. | Mixed-shift benchmark; the deepest experimental block. Because the harness/renderer are frozen at Wk 5, this window is execution + analysis, not engineering. CXR access must land by early-Sep. |
| **Wk 7–9** (Sep 3 – Sep 17) | **Auditability layer + all figures.** Build the clinician-facing panel ([method_note §6](docs/method_note.md)); subgroup/fairness audit with intervals; failure-mode catalog; regenerate every figure/table from logs by one script. | The explainability hook for the venue. |
| **Wk 9–10** (Sep 17 – Sep 24) | **Write the full draft** (§7 structure). Methods/setup point to the three docs; results + auditability + honest limitations are the new prose. | Prose discipline: every guarantee sentence names its method + assumption. |
| **Wk 10–11** (Sep 24 – Oct 1) | **Reviewer pass + revisions.** Three-reviewer simulation; apply top revisions; write the open-edges paragraph; finalize the reproducible release + Zenodo DOI. | |
| **Wk 11** (Oct 1 – **Oct 5**) | **Buffer + submit.** Final formatting to the Discover Computing template, reference check, submit. | ~4-day buffer absorbs slippage. |

> **Compute note (single GPU).** Cache `f`'s logits + features once (Stage R prep); the conformal / weight / BBSE / budget math then reruns on the cached numbers on CPU in **seconds**. Deep-ensemble *ablations* are `K` sequential trainings — schedule overnight. One script regenerates every figure from logged runs.

> **Dated GO/NO-GO on the second benchmark (artifact-keyed, not week-keyed).** By **2026-07-31** (illustrative — confirm against real PhysioNet/AIMI turnaround), **GO two-benchmark iff** the PhysioNet credentialed-access approval **and** the Stanford AIMI CheXpert access are **both in hand**; **else COMMIT to CAMELYON17-primary** with CheXpert/MIMIC as a **drop-in-if-it-arrives** depth-second benchmark (the build and ablation harness are dataset-agnostic, so the second benchmark drops in whenever access lands). The trigger is the **artifact** (the two approvals), not the calendar week. Track the credentialing artifacts as **run-config checkboxes** (record the date each clears):
>
> - [ ] CITI human-subjects training complete — date: ____
> - [ ] PhysioNet credentialed-access approval (MIMIC-CXR) granted — date: ____
> - [ ] Stanford AIMI registration / CheXpert access granted — date: ____
>
> **Risk-managed contingencies.** Under the NO-GO branch, lead the submission with the **CAMELYON17** headline (Stage R/S complete there) and present CheXpert↔MIMIC as the depth-second benchmark as soon as access lands. If the combined-shift residual on `D_tar^lab` proves too noisy on a pair, report that pair **uncorrected** ([method_note §3.4 step 3](docs/method_note.md)) rather than delaying. Multi-modality breadth (ECG / retina / 2nd-pathology / derm) remains explicit, honestly-scoped **future work** ([positioning_memo](docs/positioning_memo.md), [preregistration §2](docs/preregistration.md)), never a v1 obligation; chasing a third modality is the classic way to arrive at the deadline with nothing submitted.

### 8.1 The clinician-panel renderer is a CORE artifact, not a screenshot

Scheduled in the **Wk 1–5** window over cached logits (not deferred to the figures pass): build a **runnable** clinician-facing audit-panel renderer that consumes the cached per-image logits + `φ` features and emits the per-case trust panel of [method_note §6](docs/method_note.md) — the routing decision over `A(x)`, the recalibrated posterior `σ̃(f(x))`, the per-case weight-context representativeness chip `ŵ(x,ŷ)`, the cohort reliability flag `n_eff`, and the OOD / shift-regime flags — plus the routing-faithful **Decline-Attribution Record** (the record is *faithful to the routing decision, not the diagnosis*). It is a CORE deliverable in the §8.0 table: the paper's explainability figure is then a screenshot **of the running renderer**, and the renderer ships in the release. The panel surfaces the two operating budgets `α_acc` and `α_ood` as **separately-measured** values; their sum is shown only as an explicitly-labelled reporting convenience, never beside a per-case number and never as a single "total" figure on the panel surface.

---

# Appendices

> Reference material for the applied build. The three docs are authoritative and these appendices only point to them, never restate or override them: the method spine and **all notation** live in [docs/method_note.md](docs/method_note.md) (§1 is the single source of truth for every symbol, weight, threshold, and budget); the one-sentence contribution and venue fit in [docs/positioning_memo.md](docs/positioning_memo.md); the evaluation protocol in [docs/preregistration.md](docs/preregistration.md). **Honesty discipline throughout:** we claim *no new guarantee*; each method's guarantee is cited as a property *of that method under its own assumptions*; under realistic clinical shift we **measure and report** degradation, never certify it.

---

## Appendix A — Background: the ideas this project composes

> *You have already built these as toy notebooks (`phase0_learning.ipynb`, steps 0.1.1–0.1.6), so treat this as a corrected reference, not a blocker. The one framing correction versus older drafts: the **distribution-free guarantee is not the contribution**. Each guarantee below belongs to its own method under its own assumptions; clinical shift breaks those assumptions, and **the contribution is the auditable composition + the clinician-facing trust interface + the measure-and-report discipline**, not a guarantee that survives shift.*

### A.1 The math and ML you actually need
- **Function `f(x)`** — put an image in, get logits out; softmax squashes them to probabilities; the penultimate vector `φ(x)` (features) is what the OOD detector and domain discriminator act on.
- **Conditional probability `P(A∣B)`** — "how often `A`, *among* the `B` cases." The whole pipeline is about risk *among the cases the model chose to answer*.
- **`P_S` vs `P_T`** — the data pattern at the **s**ource (calibration) vs **t**arget (deployment) hospital. The premise calibrated on `P_S` may not hold on `P_T`; we measure how much it degrades.
- **Train / calibration / test, no leakage** — train fits `f`; *calibration* sets thresholds; test is untouched. No leakage between folds is sacred and is asserted in code ([method_note §1.7](docs/method_note.md)).
- **`α`** = tolerated error rate · **`δ`** = confidence of a *reported interval* (not a PAC promise about the pipeline) · **"distribution-free"** = a method-level property that holds regardless of data shape *under that method's assumptions*.

### A.2 The five ideas, with corrected framing
1. **Conformal / risk control (CRC, RCPS, LTT) — the honesty thermostat.** Convert a shaky score into an operating threshold with a calibration-time property. **CRC** controls the *expected* loss; **RCPS** gives `(α,δ)` PAC control; **LTT** is a multiple-looks correction for tuning several knobs. *Each property holds only under cal/test **exchangeability**.* It is **not** maintained under shift — we measure realized risk on target instead.
2. **Selective prediction — knowing when to say "I don't know."** A threshold `τ` on uncertainty answers the confident cases and defers the rest to a clinician. Selection biases the answered subset, so the controlling threshold is calibrated *on the accepted region* ([method_note §2](docs/method_note.md)).
3. **Distribution shift (two flavors) — the new-hospital problem.** *Covariate shift:* `p(x)` moves → reweight by a density ratio `ŵ_cov(x)` from a source-vs-target discriminator. *Label/prevalence shift:* `p(y)` moves → estimate target proportions with **MLLS+BCTS** (calibrate first, then EM) and reweight by `ŵ_lab(y)`. Combining them is **not** a product: divide by the per-`x` corrector `Ẑ(x)` ([method_note §3.3](docs/method_note.md)). These corrections are *reported, not guaranteed*: once the weight is estimated only asymptotic / doubly-robust validity is available (Yang–Kuchibhotla–Tchetgen Tchetgen 2024), and combined shift is not identifiable from unlabeled target alone.
4. **OOD detection + distance awareness — "not even the right kind of input."** A feature-space distance (Mahalanobis++ primary — the L2-normalized tied-covariance score; plain Lee et al. tied-covariance and kNN as ablations) routes inputs too strange to score to a human. Optional spectral normalization (an efficiency-side ablation, not a deployed requirement) can make feature-distance more meaningful; the primary detector is the L2-normalized **Mahalanobis++** score, keeping its in-distribution-only normalization and dropping only the OOD-coupled add-ons ([method_note §4.1](docs/method_note.md)). This is a **measured screen** with a reported leakage rate, **not** an in-the-guarantee step — distribution-free OOD detection is provably *not learnable* in the unrestricted setting (Fang et al. 2022).
5. **Auditability + the trust interface — the actual contribution.** Composing the above with strict fold-disjointness, self-normalized (Hájek) weighted risk, a variance-adaptive bound, and a fixed set of reliability diagnostics (`n_eff`, `κ(Ĉ_S)`, the `q̂_T`-vs-`Ĉ_S p̂_T` check, per-pair discriminator AUROC) — and surfacing the residual degradation per case and per subgroup at the point of care. *This composition + interface + measure-and-report discipline is the contribution* ([positioning_memo](docs/positioning_memo.md)).

### A.3 The project in one paragraph (corrected)
A frozen classifier gives scores and features. **Selective prediction** decides answer-vs-defer. **Conformal/RCPS** set the operating threshold on calibration. **Covariate + label weighting** correct the case-mix for a new hospital; **OOD routing** sends inputs too strange to score to a human. The contribution is doing all of this *as one auditable pipeline whose residual shift-induced degradation is measured and reported* — overall, by subgroup, and per case — **not** a guarantee that provably survives shift. The two named regimes are stated honestly: pure label shift is the cleanest *evaluation* regime (fewest assumptions), the combined covariate+label case is the stress extension whose residual is measured on the small labeled slice `D_tar^lab`.

### A.4 Optional skill-ramp (search for current free versions)
3Blue1Brown / StatQuest (math intuition); Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (the on-ramp); Tibshirani et al. 2019 (weighted conformal); Lipton et al. 2018 (BBSE); Bates et al. 2021 (RCPS); Waudby-Smith & Ramdas 2024 (betting bound).

---

## Appendix B — Applied reading list (methods actually used)

> *Restricted to what the applied pipeline composes, plus the two impossibility results that anchor the measure-and-report stance and the trustworthy-ML refs that fit the venue. Theory-only / exotic / temporal references that the applied pipeline does **not** use are collected in one note at the end (B.7) so they do not crowd the working list.*

**B.1 Conformal & risk control (the operating-threshold layer)**
- Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* (2005) — foundations.
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* — the on-ramp.
- Bates, Angelopoulos, Lei, Malik, Jordan — *Distribution-Free, Risk-Controlling Prediction Sets (RCPS)* (JACM 2021, [arXiv:2101.02703](https://arxiv.org/abs/2101.02703)) — `(α,δ)` control **under cal/test exchangeability**; the cited selective-risk property.
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control (CRC)* (ICLR 2024, [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)) — expectation control; the less-conservative comparison row.
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test (LTT)* (AOAS 2025, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)) — used purely as a **multiple-looks correction** over the `(τ,λ,t_ood)` grid; no new validity claim.
- Waudby-Smith & Ramdas — *Estimating Means of Bounded Random Variables by Betting* (JRSS-B 2024, [arXiv:2010.09686](https://arxiv.org/abs/2010.09686)) — the variance-adaptive WSR/hedged-capital UCB used for the reported interval.

**B.2 Conformal under shift (the covariate/label corrections)**
- Tibshirani, Foygel Barber, Candès, Ramdas — *Conformal Prediction Under Covariate Shift* (NeurIPS 2019, [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)) — weighted conformal; `1−α` **marginal** coverage **only under pure covariate shift with the oracle likelihood ratio**.
- Podkopaev & Ramdas — *Distribution-Free UQ for Classification Under Label Shift* (UAI 2021, [arXiv:2103.03323](https://arxiv.org/abs/2103.03323)) — the conformal-native label-shift treatment.

**B.3 Label-shift estimation (the `ŵ_lab(y)` module)**
- Alexandari, Kundaje, Shrikumar — *MLLS with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation* (ICML 2020, [arXiv:1901.06852](https://arxiv.org/abs/1901.06852)) — **the primary prevalence estimator**; calibrate first (BCTS), then EM.
- Lipton, Wang, Smola — *Detecting and Correcting for Label Shift (BBSE)* (ICML 2018, [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)) — the **baseline / consistency diagnostic** estimator; consistent under label shift with invertible `Ĉ_S`.
- Garg, Wu, Balakrishnan, Lipton — *A Unified View of Label Shift Estimation* (NeurIPS 2020, [arXiv:2003.07554](https://arxiv.org/abs/2003.07554)) — MLLS ≡ BBSE under different calibration; the shared invertibility identifiability condition.

**B.4 Selective prediction & OOD detection**
- Geifman & El-Yaniv — *Selective Classification for DNNs* (NeurIPS 2017) and *SelectiveNet* (ICML 2019, [arXiv:1901.09192](https://arxiv.org/abs/1901.09192)) — the selection head; risk control comes from the calibration layer, not the architecture.
- Jones et al. — *Selective Classification Can Magnify Disparities Across Groups* (ICLR 2021, [arXiv:2102.11203](https://arxiv.org/abs/2102.11203)) — the subgroup-audit motivation.
- Müller et al. — *Mahalanobis++* (2025; repo [mueller-mp/maha-norm](https://github.com/mueller-mp/maha-norm)) — the **L2-normalized tied-covariance** Mahalanobis score deployed as the **primary** OOD detector; the in-distribution-only normalization keeps it OOD-agnostic (only the FGSM/ensemble add-ons are dropped).
- Lee et al. — *Mahalanobis OOD* (NeurIPS 2018, [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)) — the **plain tied-covariance** score, retained as the **canonical-baseline ablation**.
- Sun et al. — *kNN OOD* (ICML 2022, [arXiv:2204.06507](https://arxiv.org/abs/2204.06507)) — the ablation detector.
- Liu et al. — *Energy-based OOD Detection* (NeurIPS 2020, [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)) — energy-score ablation.

**B.5 The two impossibility results (why we measure, not promise)**
- Fang, Li, Lu, Dong, Han, Liu — *Is Out-of-Distribution Detection Learnable?* (NeurIPS 2022, [arXiv:2210.14707](https://arxiv.org/abs/2210.14707)) — **OOD detection is not distribution-free learnable** in the unrestricted setting; the reason OOD leakage is audited, not certified.
- Yang, Kuchibhotla, Tchetgen Tchetgen — *Doubly Robust Calibration of Prediction Sets Under Covariate Shift* (JRSS-B 2024, [arXiv:2203.01761](https://arxiv.org/abs/2203.01761)) — **finite-sample distribution-free conditional coverage under covariate shift is not attainable** once the weight is estimated; only asymptotic / doubly-robust validity survives. The reason the covariate correction is reported, not guaranteed.

**B.6 Datasets, calibration, trustworthy-ML (venue-relevant)**
- Koh et al. — *WILDS* (ICML 2021, [arXiv:2012.07421](https://arxiv.org/abs/2012.07421)); Bandi et al. — *CAMELYON17* (IEEE TMI 2018).
- Irvin et al. — *CheXpert* (AAAI 2019, [arXiv:1901.07031](https://arxiv.org/abs/1901.07031)); Johnson et al. — *MIMIC-CXR* (Sci Data 2019).
- Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) — ECE.
- TRUECAM ([arXiv:2501.00053](https://arxiv.org/abs/2501.00053)) — the head-to-head empirical baseline (OOD detect-and-remove); the honest contrast in [positioning_memo](docs/positioning_memo.md) — our contrast with it is **not** a stronger guarantee but *measuring and reporting* the residual leakage its detect-and-remove framing leaves unquantified.
- Hore & Barber — *RLCP* ([arXiv:2310.07850](https://arxiv.org/abs/2310.07850)) — *optional*; the subgroup audit is reporting with finite-sample intervals, **not** a conditional-coverage guarantee.

**B.7 Background, not used in the applied pipeline.** The following anchored earlier theory tracks that are **out of scope** for this paper and are listed only so the lineage is traceable, not as working references: the Generalized-Label-Shift / Generalized-Target-Shift identification line (Zhang et al. 2013; Gong et al. 2016; Tachet des Combes et al. 2020) and Sparse Joint Shift (Chen, Zaharia & Zou 2022) — the applied combine is a **modeling-choice** `Ẑ(x)`-division with the residual *measured*, not an identification result; the temporal/online-drift line (Gibbs & Candès, ACI/DtACI) — the temporal track is **cut**; and the exotic label-shift estimators (RLLS, GS-B³SE, FMAPLS) and Wang & Qiao 2025 (generalized covariate shift with posterior drift) — **not** estimators or baselines here. The prereg fixes MLLS+BCTS (primary) and BBSE (baseline) as the only label-shift estimators and exactly four baselines ([preregistration §4](docs/preregistration.md)).

---

## Appendix C — Dataset cheat-sheet

> *Authority for splits, folds, source→target directions, and the shift-type diagnostic is [preregistration §2–3](docs/preregistration.md). This is the access/friction quick-reference.*

| Dataset | Modality | Shift mechanism | Access | Friction |
|---|---|---|---|---|
| **CAMELYON17-WILDS** | Histopathology | Cross-hospital scanner/stain (**covariate-dominant**) | `wilds` package, open | **Low — start here; unblocks the build now** |
| **CheXpert** | Chest X-ray | Cross-site institution/protocol + prevalence (**mixed**) | Stanford AIMI registration | Medium — **start credentialing now** |
| **MIMIC-CXR** | Chest X-ray | Cross-site institution/protocol + prevalence (**mixed**) | PhysioNet credentialed + CITI training | Medium–High — **long pole; start now** |

- **CAMELYON17-WILDS:** binary tumor-vs-normal patches; each source→target **hospital pair evaluated separately, never pooled** ([preregistration §2.1](docs/preregistration.md)). The covariate-dominant benchmark; appearance drift `p(x∣y)` is present at harder pairs, which is exactly why residual degradation is measured rather than assumed away.
- **CheXpert ↔ MIMIC-CXR:** shared-label chest-radiograph finding classification; **both directions** (CheXpert→MIMIC and MIMIC→CheXpert) as two separate deployments ([preregistration §2.2](docs/preregistration.md)). Mixed covariate + label shift.
- **Credentialing is the long pole.** CheXpert (Stanford AIMI) and MIMIC-CXR (PhysioNet + CITI human-subjects training) both require human approval that takes weeks. Begin both **immediately** in parallel with the CAMELYON17 build, which needs no credentialing.
- **Far-OOD exposure set `O`** — off-modality / wrong-anatomy / non-medical probes used to set `t_ood` and **measure leakage** against `α_ood`. `O` is a *stated, swappable* modeling choice, not ground truth; exposure-set sensitivity is reported ([method_note §4.3](docs/method_note.md), [preregistration §5.2](docs/preregistration.md)).
- **Future-work breadth (honest, not a v1 obligation).** Multi-modality extension — PTB-XL (ECG), EyePACS↔APTOS (retina), MIDOG (2nd-pathology scanner shift), ISIC↔Fitzpatrick17k (dermatology) — is stated as explicit future work. The paper is two imaging benchmarks taken deep ([positioning_memo](docs/positioning_memo.md): depth over breadth).

---

## Appendix D — Applied glossary (one line each)

> *Corrected framing: every line names a property of a method, never a promise about the deployed pipeline.*

- **Exchangeability** — joint distribution invariant to ordering; the assumption split-conformal / RCPS need. Clinical shift breaks it — every failure here is an exchangeability violation.
- **Marginal coverage** — holds on average over the random cal+test draw, not conditional on a given `x`.
- **CRC** — calibrate a threshold so the *expected* bounded loss ≤ `α_acc`; no `δ` tail. The less-conservative comparison row.
- **RCPS** — `(α_acc, δ)` PAC risk control on the realized calibration set/model, **only under cal/test exchangeability**; we use `λ̂` as an operating threshold and *measure* its realized target risk.
- **LTT** — a multiple-hypothesis-testing correction for jointly tuning `(τ,λ,t_ood)`; used here purely as a multiple-looks correction, no new validity guarantee.
- **WSR / betting (hedged-capital) bound** — a variance-adaptive upper confidence bound; preferred because importance weights inflate risk-estimate variance. Sets the width of the *reported interval*.
- **Selective risk / coverage** — error among answered / fraction answered; the risk–coverage curve / AURC summarizes the trade-off.
- **Weighted conformal** — reweight calibration by a density ratio; transports `1−α` marginal coverage **only under pure covariate shift with the oracle ratio**.
- **Covariate vs label shift** — `p(x)` moves (label rule fixed) vs `p(y)` moves (case-mix); different importance weights, not interchangeable.
- **Density ratio `ŵ_cov(x)`** — `p_T(x)/p_S(x)` from a source-vs-target discriminator; unreliable in the far tail → clip at `w_max` and route the overflow to abstain.
- **Label weight `ŵ_lab(y)`** — `p_T(y)/p_S(y)` from BBSE / MLLS+BCTS target proportions.
- **BBSE** — estimates target label proportions from the source confusion matrix `Ĉ_S` + target predictions, no target labels; *consistent* under label shift with invertible `Ĉ_S`. Baseline / diagnostic estimator.
- **MLLS+BCTS** — the **primary** prevalence estimator; maximum-likelihood label shift with bias-corrected temperature-scaled (calibrated) outputs.
- **`Ẑ(x)` double-count corrector** — the per-`x` normalizer that combines the two weights via `ŵ(x,y)=ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` **instead of the product** ([method_note §3.3](docs/method_note.md)).
- **Two operating budgets `α_acc`, `α_ood`** — separately measured budgets for accepted in-distribution error and far-OOD leakage. Their sum is a **reporting convenience only**, never a certified additive bound.
- **Far-OOD** — inputs unlike training; routed to a human as a **measured screen** with reported leakage on a stated exposure set `O` (distribution-free OOD detection is provably not learnable, Fang et al. 2022).
- **`n_eff` (Kish)** — effective sample size `(Σŵ)²/Σŵ²`; a **reported reliability diagnostic** flagging weight-dominated regimes, not a certification gate.
- **`κ(Ĉ_S)`** — conditioning of the source confusion matrix; a diagnostic flagging where the label-shift estimate is least trustworthy.
- **Distance-aware UQ (SNGP/DDU) & spectral normalization** — single-pass models / Lipschitz constraints that make feature-distance meaningful; ablations that buy *efficiency*, not validity.
- **RLCP (randomly localized conformal)** — optional local-weight randomization toward approximately conditional coverage; the subgroup audit remains *reporting with intervals*, not a guarantee.

---

## Appendix E — Environment & command cheat-sheet

> *Pin the env, run a config, regenerate every figure from logs. The internal tests are **empirical-milestone checks** (does the shift-aware pipeline maintain/improve realized selective risk vs the naive baseline; does the OOD screen measurably reduce leakage) — **not** proofs that a bound holds.*

```bash
# env (pin + lockfile)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision wilds numpy scipy scikit-learn pandas matplotlib wandb pytest confseq
pip freeze > requirements.lock

# data (confirm current WILDS API; CAMELYON17 needs no credentialing)
python -c "from wilds import get_dataset; get_dataset('camelyon17', download=True)"

# run an experiment from a config
python -m experiments.run --config configs/camelyon17_ours.yaml seed=0

# empirical-milestone checks (engineering gates, NOT bound proofs):
#  shift-aware weighting maintains/improves realized selective risk vs the naive baseline;
#  the OOD screen measurably reduces far-OOD leakage on the exposure set O.
pytest tests/ -k "milestone_selective or milestone_covariate or milestone_label or milestone_ood" -v

# regenerate every figure/table from logged runs (one script)
python -m analysis.figures --from-logs runs/ --out paper/figures/
```

> **Compute note (single GPU).** Cache `f`'s logits + features once, so the conformal calibration, the weighting, and the diagnostics rerun in **seconds on CPU**. Deep-ensemble baselines = `K` sequential trainings — schedule overnight. The WSR / MLLS+BCTS / weighting math runs on the cached numbers.

> **Repo note.** The engineering layers are `conformal/{selective,rcps,crc,ltt,weights,label_shift}.py`, `ood/detector.py`, and `analysis/{metrics,figures}.py`. The temporal `conformal/aci.py` is **dropped** (scope cut). `conformal/budget.py` tracks `α_acc` and `α_ood` as **two separately-measured operating budgets**, not a certified additive split.

---

## Appendix F — Notation pointer

**Notation, symbols, weights, thresholds, and budgets are the single source of truth in [docs/method_note.md §1](docs/method_note.md)** — see the §1.8 quick-reference table for every symbol. Nothing in this playbook redefines them; where a symbol appears here it carries exactly its method-note meaning. (The old standalone "notation & formulas" and "symbol decoder" appendices are retired in favor of this pointer to avoid a second, drifting source of truth.)

---

## Appendix G — Applied red-team prompts (optional)

> *Two prompts only, both reframed for the applied measure-and-report paper. The old theory proof-check / gap-coverage prompts are **dropped** — there is no bound to prove and no 5-condition gap to close. Drop either prompt into a search-enabled assistant against your actual artifact.*

**G-1 · Experiment-design critic** (run against [preregistration.md](docs/preregistration.md) before results)
```
You are an experienced ML-for-health reviewer for the Springer Discover Computing
"Intelligent Medicine" collection (explainability + trustworthy/auditable ML). Critique this
APPLIED evaluation plan for a trustworthy selective-prediction pipeline under clinical shift
[PASTE: two imaging benchmarks taken deep; covariate+label shift; the four pre-registered
baselines (naive split-conformal, temperature scaling, TRUECAM-style detect-and-remove, BBSE
label-weighting); MLLS+BCTS primary estimator; the Ẑ-combine; the measured metrics + reliability
diagnostics; the per-pair discriminator-AUROC shift-type diagnostic with a fixed τ_disc cutoff].
The paper claims NO new guarantee — it MEASURES and REPORTS realized selective risk, coverage,
and far-OOD leakage under shift, overall and by subgroup.
Tell me: (a) which baseline a reviewer would say is missing, given exactly four are pre-registered;
(b) whether the metrics actually demonstrate the MEASUREMENT claims — does a plot show label-aware
weighting reducing realized selective risk vs covariate-only under prevalence shift, and the OOD
screen measurably reducing leakage under a DELIBERATELY imperfect detector?; (c) the right way to
report a calibration-draw-marginal MEASURED risk empirically (how many splits, what plot, what
interval); (d) the most likely confound (base model just worse on target; the win coming from the
backbone not the pipeline) and how the {standard, DDU, SNGP, ensemble} ablation rules it out —
remembering the auditability result is base-model-agnostic, the net only moves efficiency;
(e) whether two imaging benchmarks deep is defensible for this venue, and which single extra
modality is the cheapest credible rebuttal if a reviewer demands breadth.
Do not propose adding a guarantee — critique the measurement protocol.
```

**G-2 · Results red-team** (run against your results before writing the central claim)
```
You are a skeptical senior co-author. Here are my results [PASTE tables/figures/setup]. The claim
is a MEASUREMENT one: under realistic covariate+label shift the pipeline's realized selective risk,
coverage, and far-OOD leakage are reported with intervals, and the residual degradation is made
visible per subgroup and per case — NOT that any risk is "held at ≤ α" or certified.
Try to explain my positive result WITHOUT the pipeline being the cause: confounds, abstention
inflation (low risk bought by near-total abstention — check against the pre-registered minimum-
coverage floor), fold leakage, lucky splits, weak baselines, cherry-picked α, dataset quirks, the
base model simply worse on target, or the win coming from the backbone rather than the composition.
Specifically attack: (i) is the "label-shift" improvement just covariate weighting in disguise?
(ii) does the measured OOD leakage hold up when the detector is genuinely imperfect, or only with a
near-perfect detector? (iii) are the reliability diagnostics (n_eff, κ(Ĉ_S), the q̂_T-vs-Ĉ_S p̂_T
check) flagging the regimes where my numbers should be distrusted, and did I report them honestly?
For each alternative, give the exact additional analysis or plot that rules it out. Then tell me
which MEASUREMENT claims the data supports vs. which I am overstating, and rewrite my central claim
sentence so it asserts a measured, interval-bounded degradation — never a guarantee.
```
