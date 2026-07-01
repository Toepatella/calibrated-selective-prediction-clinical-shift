# Build Gates — the staged synthetic build ladder (G-A → G-E)

*Single source of truth for the staged engineering build of the trustworthy selective-prediction pipeline. This document operationalizes the empirical-milestone ladder of [`flagship-playbook.md` §5 / §5.1](../flagship-playbook.md) (Stages A→B→C→D, then the real-data headline R and subgroup audit S) as a sequence of **synthetic gates checked against known ground truth**. It introduces no method, no notation, and no claim: every symbol carries its [`method_note.md §1`](method_note.md) meaning, and every criterion is grounded in a specific doc section. Where a criterion is not stated in the docs it is flagged `(proposed — not yet in docs)`.*

---

## 1. Ladder philosophy

Each rung **adds exactly one shift mechanism** on top of the previous rung, and ends in a **synthetic gate** run on data whose ground truth we construct and therefore know exactly. A gate is an **engineering checkpoint**, never a clinical guarantee. Concretely, a gate verifies that the implementation **reproduces a cited method's stated property, under that method's own assumptions, on synthetic data with known ground truth**. It is *not* evidence that the property survives real clinical shift, and it is *not* a certificate for the deployed pipeline.

```
G-A  exchangeable selective RCPS          (no shift; the one regime where RCPS's (α_acc,δ) PAC property holds)
G-B  + covariate weights                  (known synthetic covariate shift; weighted path recovers what naive loses)
G-C  + label / prevalence-shift weights   (known synthetic prevalence shift; BBSE/MLLS + the Ẑ-divide combine)
G-D  + integrated OOD budget & routing    (known in-scope vs far-OOD; the decoupled three-gate accept rule A(x))
G-E  + temporal / online adaptation       (PROPOSED STRETCH — SCOPE-CUT in all three docs; future-work appendix only)
─────────────────────────────────────────
(then) Stage R  real-data headline   ·   Stage S  subgroup audit       (MEASURED degradation, never a certificate)
```

The honesty discipline of [`flagship-playbook.md` §0](../flagship-playbook.md) governs every gate: we **claim no new guarantee**; each component guarantee is cited only as a property *of the method that owns it, under that method's own assumptions*; the words *certify / certified / guarantee* are **banned for the deployed pipeline**. The two impossibility results the project rests on — OOD detection is not distribution-free learnable (Fang et al. 2022), and finite-sample distribution-free conditional coverage under covariate shift is not attainable once the weight is estimated (Yang, Kuchibhotla & Tchetgen Tchetgen 2024) — are the *reason* these gates verify reproduction of a cited property rather than prove a new bound (`method_note §7.1`).

### 1.1 The global distinction: HARD GATE vs REPORTED DIAGNOSTIC

Every completion criterion in this document is exactly one of two kinds, and they are never conflated:

- **HARD GATE** — must pass to proceed up the ladder. Checkable against **synthetic ground truth** we constructed. A failed hard gate means the layer is wired wrong; stop and fix it before stacking the next rung (`flagship-playbook.md §5`, §5.1).
- **REPORTED DIAGNOSTIC** — **measured and reported, never gated**. These are reliability signals: the Kish effective sample size `n_eff = (Σŵ)²/Σŵ²`, the condition numbers `κ(Ĉ_S)` / `κ(Σ̂)`, the empirical interval-coverage check, AURC, exposure-set sensitivity, and the consistency checks. A diagnostic may be *self-tested* (e.g. assert `n_eff` falls as a synthetic shift is made more extreme over a fixed grid) so the plumbing is verified, but it is **never** used to pass or fail a rung (`method_note §1.7`, §7.5; `prereg §5.4`). One nuance: `n_eff` is never a gate, yet at the real-data headline its **pre-registered floor drives an explicit exclusion** — cells below the `n_eff` floor are dropped from headline aggregation (`prereg §6.7`, "Below-`n_eff`-floor exclusion"; see §9). The diagnostic is still never a pass/fail rung gate; it is a reporting protocol on the headline average.

> Mapping a synthetic-gate pass to a clinical promise is the cardinal error this document forbids. A gate confirms a method's own property holds where the method's assumptions hold (synthetic ground truth). Real shift breaks those assumptions; the residual is **measured** at Stage R, never recovered as a certificate.

---

## 2. Summary table

| Gate | Adds (one shift mechanism) | Hard-gate pass condition (one line) | Key artifacts | Status |
|---|---|---|---|---|
| **G-A** | Exchangeable selective RCPS (no shift) | Sweeping `τ` traces the **known** monotone risk-coverage curve, the RCPS inf-rule returns the loosest gate with accepted-region UCB `< α_acc`, and the `(α_acc,δ)` PAC property holds empirically across resamples | `conformal/rcps.py`, `conformal/selective.py`, `conformal/folds.py` (implemented); `tests/test_gate_a_exchangeable.py` + `tests/_synth.py` generator (implemented); `configs/gate_a.toml` run-config | ✅ **COMPLETE — 15/15 pass** |
| **G-B** | Covariate weights `ŵ_cov(x)` | Under **known** synthetic covariate shift the weighted Hájek risk `R̂_w` measurably recovers toward source while the `ŵ≡1` naive path degrades, and `ŵ_cov` tracks the oracle ratio | `conformal/weights.py` (1-line stub); weighted path in `rcps.py`; `tests/test_gate_b_covariate.py` (empty) | **Stub + empty test** |
| **G-C** | Label / prevalence-shift weights + `Ẑ`-combine | BBSE/MLLS recover the **known** `p_T(y)`; the `Ẑ`-divide combine recovers the oracle weight while the naïve product double-counts; label-aware path beats covariate-only | `conformal/label_shift.py` (1-line stub); `conformal/weights.py` (G-B prereq); `tests/test_gate_c_label_shift.py` (empty) | **Stub + empty test** |
| **G-D** | Integrated OOD budget & routing | `o(x)` separates in-scope from far-OOD `O`; `t_ood` spends `α_ood` on `O`; the screen measurably reduces leakage; the three-gate `A(x)` and decoupled routing compose correctly | `ood/detector.py` (1-line stub), `conformal/budget.py` (1-line stub); `tests/test_gate_d_integrated_ood.py` (empty) | **Stub + empty test** |
| **G-E** | Temporal / online adaptation (ACI / DtACI) | *(every criterion is proposed — not yet in docs)* ACI long-run coverage → `1−α` on synthetic drift streams; doc-amendment precondition unmet | `conformal/aci.py` (removed from tree); `tests/test_gate_e_temporal.py` (does not exist) | **OUT OF SCOPE / future-work appendix** |

Status legend reflects the live repo: **G-A is complete** — Stage A code (`conformal/rcps.py`, `conformal/selective.py`, `conformal/folds.py`, exported by `conformal/__init__.py`), the synthetic generator (`tests/_synth.py`), the committed run-config (`configs/gate_a.toml`), and the full gate suite (`tests/test_gate_a_exchangeable.py`, **15/15 passing** under `pytest` configured by `pyproject.toml`). `weights.py`, `label_shift.py`, `budget.py`, `crc.py`, `ltt.py`, `ood/detector.py` remain one-line stubs; `tests/test_gate_{b,c,d}*.py` remain empty; there is no `test_gate_e`.

---

## 3. Shared conventions (X-CONVENTIONS)

These apply to **every** gate and are not repeated per rung.

- **Seeds & resamples.** All synthetic draws use `numpy`'s `np.random.default_rng(seed)`, matching the phase-0 conventions. Each gate runs many Monte-Carlo cal/test resamples; the resample and seed counts are fixed in the run config (`prereg §6.7` fixes resamples/seeds for the real headline; the synthetic-gate counts here — e.g. `T ≥ 1000` for G-A PAC checks, `≥ 500` for G-C recovery, `≥ 200` for G-D leakage — are **proposed** engineering choices, not doc-pinned).
- **The interval is the WSR self-normalized (Hájek) betting bound.** Every reported risk uses the self-normalized (Hájek) weighted risk `R̂_w = Σ(ŵ·ℓ)/Σŵ` so the unknown `E[w]=1` cancels (`method_note §1.7`), with the variance-adaptive Waudby-Smith & Ramdas hedged-capital UCB (`method_note §1.6`, §2.2; implemented as `wsr_ucb` in `conformal/rcps.py`). The weighted-path WSR UCB is **unit-tested to confirm it is actually exercised** — no silent range-bound `[0,1]` fallback (`prereg §6.7`).
- **Matched operating point.** Any pipeline-vs-baseline comparison (covariate-only vs `Ẑ`-combined, screen-on vs screen-off) is read at a **matched total answer-rate**, so a lower risk bought by answering less is never compared against a higher-coverage point as if equal (`prereg §5.1`, §6.7). A "win" follows the **non-overlapping-interval rule** of `prereg §6.7` (intervals must not overlap; a conservative test, not a p-value).
- **Fold-disjointness at the group-id level.** From Stage A onward, the folds used by a gate (`D_cal`, `D_bbse^src`, `D_disc`, `D_tar`, `D_tar^lab`, `O`, held-out test) are mutually **disjoint**, asserted in code at the **group-id** level (slide / patient on real data; synthetic group ids on synthetic data), emitting a printed **set-intersection = 0 artifact** over every fold pair (`method_note §1.7`; `prereg §2.3`). This is a HARD GATE wherever a gate constructs a split.
- **The prereg §6.8 synthetic interval-coverage check.** A pre-registered empirical coverage check of the WSR/Hájek interval is run **on synthetic data with known ground-truth weighted risk and known weights**, reporting realized coverage vs nominal `1−δ` across the `n_eff` regimes. If the interval under-covers in a regime it is relabelled **"nominal interval, coverage validated empirically"** there — this relabel action is the operative, load-bearing part of §6.8 and rides along on every per-gate coverage-check bullet below. This is a **REPORTED DIAGNOSTIC**, synthetic, and **not gated on data credentialing** (`prereg §6.8`).
- **Numeric budgets are run-config values, not doc constants.** `α_acc`, `α_ood`, `δ`, `coverage_min`, `w_max`, `[w_lab,min, w_lab,max]` are declared "fixed in advance" (`prereg §6.2`, §6.4, §6.5) but the docs do **not** pin their numeric values. Any concrete number used in a gate check below (e.g. `α_acc = 0.1`) is a **proposed** gate-config value to be committed before the gate is authoritative.

---

## 4. G-A — Exchangeable selective RCPS (foundation)

### Objective
Demonstrate the calibration-layer spine — the selection gate `g(x)=1[u(x)≤τ]` (`method_note §2.1`), the accepted-region RCPS inf-rule `λ̂` (`§2.2`), and the WSR hedged-capital betting UCB (`§1.6`, §2.2) — is wired correctly **under exchangeability (no shift)**, the one regime where RCPS's own `(α_acc, δ)` PAC property holds (Bates et al. 2021, Def. 1 + Thm. 1). This is the wiring/sanity gate (Stage A / Milestone A = M-1), **not** a budget-satisfaction or deployment claim. The trap it must guard is the **selection-bias trap**: calibrate ON the accepted region (`g=1`), never on the full marginal (`method_note §1.4`, §2.2).

### Introduces
- `selection_gate`, `selective_risk`, `risk_coverage_curve`, `aurc`, `select_threshold`, `SelectiveResult` (`conformal/selective.py`).
- `rcps_lhat` inf-rule, `wsr_ucb`, `hb_ucb`, `binom_cdf` (`conformal/rcps.py`).
- Fold-disjointness assertion discipline wired in from Stage A onward (`method_note §1.7`).

### Builds on
Phase-0 notebook steps 0.1.1–0.1.4 (split conformal, CRC, selective prediction + selection-bias trap, RCPS/CRC/LTT backbone), ported into `conformal/rcps.py` and `conformal/selective.py` (from phase0 cells 38/40).

### Synthetic construction (the known ground truth)
Fully synthetic, no clinical data; the gate runs without dataset credentialing (`prereg §6.8`). Draw `D_cal` and a test set i.i.d. from **one** exchangeable population `P` — identical distribution, so cal/test are exchangeable (the precondition of `method_note §2.2`). For each point draw a scalar uncertainty `u` and a binary loss `ℓ ∈ {0,1}` (default 0–1 loss, `§1.4`) from a **known** conditional error model `r(u)=P(ℓ=1|u)` monotone non-decreasing in `u`. Because `r(u)` and the marginal of `u` are analytic, the **true accepted risk** `R(τ)=E[ℓ|u≤τ]` and **true coverage** `P(u≤τ)` are known in closed form (or to arbitrary precision by large-sample integration) for every `τ` — the ground truth each check compares against. The nested RCPS family is the gates indexed by `τ`. A second **"naïve"** construction calibrates `τ` on the **full-marginal** risk to exhibit the selection-bias trap.

### Definition of Done

> **STATUS: COMPLETE (2026-06-30).** All 8 hard gates and 4 diagnostics below are implemented in `tests/test_gate_a_exchangeable.py` and **pass 15/15** (`pytest`, ~65 s) against the synthetic ground truth in `tests/_synth.py` and the committed run-config `configs/gate_a.toml`; boxes are checked accordingly. An adversarial 5-lens review (DoD-completeness, ground-truth math, honesty rails, bugs/flakiness, canonical faithfulness) was run and its three confirmed findings fixed: an ASCII-only fold-disjointness artifact (was a `∩` glyph that crashed cp1252 stdout), `rc_points` raised 200000→1e6 so the ≤0.01 risk tolerance holds across seeds, and the min_accept test rewritten to genuinely exercise break-on-first-failure.

**HARD GATES (must pass):**

- [x] **Monotone risk-coverage trade-off matches the analytic curve.** On `N=200000` points, for a grid of `≥50` `τ`-quantiles, assert coverage is non-decreasing in `τ` (`np.diff ≥ −1e-9`) and `|empirical_accepted_risk(τ) − analytic_R(τ)| ≤ 0.01` at every grid point. *Source: `method_note §2.1`–§2.2; flagship Milestone A.*
- [x] **`selective_risk` / coverage correctness.** Unit test on a hand-built array: returned risk = mean loss over exactly the `g=1` subset and coverage = accepted fraction (atol `1e-12`); `np.isnan` risk when `τ` accepts 0 points. *Source: `method_note §1.4` (`R^accept := E[ℓ|g=1]`), §2.1.*
- [x] **Accepted-region calibration vs the selection-bias trap.** Over `≥1000` resamples (`n_cal=2000`): accepted-region `select_threshold` controls true-accepted-risk at `tau_hat ≤ α_acc` in `≥(1−δ)` of trials, while a deliberately-wrong **full-marginal** calibration violates it in a materially larger fraction (assert wrong-path violation rate `> 2δ`). Ground truth = analytic accepted risk at each `tau_hat`. *Source: `method_note §1.4`, §2.2; phase0 0.1.3.*
- [x] **RCPS inf-rule correctness (`rcps_lhat` / `select_threshold`).** On a deterministic `cal_table` with hand-known UCB ordering, assert the returned `τ` is exactly the boundary (last column with UCB `< α` before the first UCB `≥ α`), and `SelectiveResult(controlled=False, tau_hat=None)` when even the tightest certifiable gate has UCB `≥ α` (then abstain on every case). *Source: `method_note §2.2`; `conformal/rcps.py::rcps_lhat`, `conformal/selective.py::select_threshold`.*
- [x] **`(α_acc, δ)` PAC property holds empirically under exchangeability.** `T≥1000` trials, `n_cal=2000`, proposed `α_acc=0.1, δ=0.1`: empirical miscoverage `mean(true_accepted_risk(tau_hat) > α_acc) ≤ δ + 1.96·sqrt(δ(1−δ)/T)` **(proposed)**. Run with both `wsr_ucb` and `hb_ucb`. *Source: the underlying `(α_acc, δ)` property is `method_note §2.2` (Bates 2021 Def.1+Thm.1, under exchangeability) — doc-backed; the Monte-Carlo miscoverage-tolerance encoding (the Wald normal-approx band on the empirical miscoverage rate) is a **proposed** engineering check, stated in no doc.*
- [x] **WSR UCB is a valid `(1−δ)` upper bound and variance-adaptive.** On samples with known mean `μ`, over `≥1000` trials assert `P(wsr_ucb ≥ μ) ≥ 1−δ` (the validity half — doc-anchored) and `median(wsr_ucb) ≤ median(hb_ucb)` in a low-variance regime **(proposed)**. *Source: the `(1−δ)` validity of `wsr_ucb` is `method_note §1.6`, §2.2 (Waudby-Smith & Ramdas 2024); the docs justify WSR as variance-adaptive/materially tighter (`method_note §1.6`, §2.2, §5.4; `rcps.py` docstring) but state **no** median-ordering claim, so the `median(wsr_ucb) ≤ median(hb_ucb)` inequality is a **proposed** engineering check.*
- [x] **`binom_cdf` agrees with a brute-force binomial.** For several `(k,n,p)`, `|binom_cdf(k,n,p) − Σ_{j≤⌊k⌋} C(n,j)p^j(1−p)^(n−j)| ≤ 1e-9`. *Source: `conformal/rcps.py` docstring ("cross-checked against a brute-force binomial in the Gate-A tests").*
- [x] **Fold-disjointness at the group-id level.** `len(set(cal_group_ids) & set(test_group_ids)) == 0` for the synthetic split. *Source: `method_note §1.7` (set-intersection = 0 artifact).*

**REPORTED DIAGNOSTICS (measured, not gated):**

- [x] **Minimum-coverage / degeneracy guard.** A `tau_hat` whose realized coverage is below `coverage_min` is flagged degenerate, not reported as a favorable low risk. Proposed `coverage_min=0.1` (not numerically pinned in docs). *Source: `prereg §6.4`.*
- [x] **Empirical interval-coverage report.** Realized coverage `mean(true_accepted_risk ≤ wsr_ucb)` vs nominal `1−δ`; relabel "nominal, coverage validated empirically" if it under-covers. *Source: `prereg §6.8`.*
- [x] **AURC sanity.** `aurc(u, losses) < mean(losses)` (accepting confident cases first beats answering everything). *Source: `method_note §2.1`; `conformal/selective.py::aurc`.*
- [x] **`n_eff` plumbing.** At G-A weights are trivially 1 so `n_eff = n`; the diagnostic is reported, never gated. *Source: `method_note §1.7`, §3.5.*

### Required artifacts
- `tests/test_gate_a_exchangeable.py` — **implemented; 15 tests (8 hard gates + 4 diagnostics), all passing.**
- `conformal/rcps.py :: wsr_ucb, hb_ucb, rcps_lhat, binom_cdf` — **implemented.**
- `conformal/selective.py :: selection_gate, selective_risk, risk_coverage_curve, aurc, select_threshold, SelectiveResult` — **implemented.**
- `conformal/__init__.py` exporting the Stage-A surface — **implemented (now also lists `folds`).**
- A synthetic generator with known `r(u)=P(ℓ=1|u)` and analytic `R(τ)`/coverage — **implemented as `tests/_synth.py::PolyErrorModel`** (`u~Uniform(0,1)`, `r(u)=Σ c_k u^k`; closed-form `R(τ)=Σ c_k τ^k/(k+1)`, `coverage(τ)=τ`; a 5-model family swept to avoid a single-construction pass).
- A group-id fold-disjointness helper — **implemented as `conformal/folds.py::assert_group_disjoint`** (prints an ASCII set-intersection = 0 artifact per fold pair; `method_note §1.7`).
- Run-config + test wiring — **`configs/gate_a.toml`** (the committed proposed budgets/counts) and **`pyproject.toml`** (`pytest` `pythonpath`).

### Honesty rails (must-not-claim)
- NOT a guarantee about deployment or real clinical data — a synthetic wiring check of RCPS **under exchangeability only** (`method_note §2.2`).
- Does NOT claim `R_T^accept ≤ α_acc` on target/shifted data; no shift exists at this rung (shift enters G-B onward).
- Does NOT certify the combined/deployed pipeline; the `(α_acc, δ)` property is cited only as a property of RCPS under exchangeability (Bates 2021), re-verified, never re-asserted once shift enters.
- `n_eff` and any conditioning numbers are REPORTED DIAGNOSTICS, never pass/fail gates (`method_note §1.7`, §3.5).
- No new theorem, no keystone lemma, no impossibility result, no integrated-budget bound — the abandoned theory stays abandoned (`flagship-playbook.md §0`, §3).

### Open questions — RESOLVED at G-A completion (2026-06-30)
- **Run-config committed.** `α_acc=0.1, δ=0.1, coverage_min=0.1, T=1000, n_cal=2000` are now committed to `configs/gate_a.toml`; they remain **proposed** engineering values (not doc-pinned), now auditable in one place. `rc_points` was raised from the doc's illustrative `200000` to `1e6` there so the `≤0.01` risk tolerance holds across seeds at the tightest τ-quantile (at 200000 the tightest cell's Monte-Carlo noise failed the hard gate on ~5% of seeds; verified 0/40 at 1e6).
- **Test + generator + fold helper written.** `tests/test_gate_a_exchangeable.py`, `tests/_synth.py::PolyErrorModel`, and `conformal/folds.py::assert_group_disjoint` all exist and pass (15/15).
- **min_accept × break-on-first-failure tested.** `test_inf_rule_stops_at_first_failure_no_wander` uses a non-monotone "bad-band" construction (risk passes on tight τ, fails inside the band, and — because the band mass is small — a looser gate near τ=1 genuinely passes its UCB); it asserts the inf-rule stops at the first failure and refuses the looser passing gate, and notes why min_accept (which skips only sub-threshold *tightest* cells) can never bypass a failing certifiable column.
- **Single-construction pass avoided.** The risk-coverage gate sweeps a 5-model family (`configs/gate_a.toml` `families`): four linear and one quadratic monotone `r(u)`, varying base rate, slope, and curvature.
- *(Remaining, minor)* the numeric budgets stay "proposed" until the paper's preregistration pins them; nothing else in G-A is open.

---

## 5. G-B — + Covariate-shift weights

> The G-B per-gate spec was not separately supplied; this rung is reconstructed from [`flagship-playbook.md` Stage B / Milestone B](../flagship-playbook.md) (lines under "Stage B — covariate weights" and the §5.1 checklist item B) and `method_note §1.5`, §3.2. Numeric thresholds here are **proposed** unless a doc ref is given.

### Objective
Second rung: add the domain-discriminator covariate weight `ŵ_cov(x) = clip((d/(1−d))·ĉ, 0, w_max)` (`method_note §1.5`). On a held-out **target** split with a **known synthetic covariate shift**, demonstrate that the **weighted** Hájek path measurably restores realized selective risk where the unweighted naive path (`ŵ≡1`) degrades, and that `ŵ_cov` tracks the known oracle ratio. This is a **measured restoration**, not a recovered guarantee — consistent with the impossibility of a finite-sample certificate once the weight is estimated (`method_note §7.1`; Yang, Kuchibhotla & Tchetgen Tchetgen 2024).

### Introduces
- `conformal/weights.py` — domain discriminator `d(x)` on the frozen embedding `φ(x)`, the base-rate constant `ĉ = n_S/n_T`, the clipped covariate ratio `ŵ_cov`, and `w_max` routing (`method_note §1.5`, §3.2).
- The weighted (Hájek) risk path `R̂_w = Σ(ŵ·ℓ)/Σŵ` in `rcps.py`.
- K-fold cross-fitting of `d` so weights are not fit on the points they reweight (`method_note §1.5`, §3.4 step 1).

### Builds on
G-A (`conformal/rcps.py`, `conformal/selective.py` — implemented).

### Synthetic construction (the known ground truth)
Draw source and target covariates from explicit marginals `p_S(x)`, `p_T(x)` with **invariant** `p(y|x)` (covariate shift by construction), so the **oracle covariate ratio** `w_cov*(x) = p_T(x)/p_S(x)` is known in closed form per point. The accepted target risk has a known population value, so the Hájek weighted estimate over source-drawn calibration points can be checked against it. All folds (`D_disc`, `D_cal`, target test) are drawn group-disjoint.

### Definition of Done

**HARD GATES (must pass):**

- [ ] **Weighted path restores risk where naive degrades** (as a signed non-overlap check). Under the known covariate shift, at a matched total answer-rate over `≥200` resamples, the weighted `R̂_w` under `ŵ_cov` is **lower** than the `ŵ≡1` naive target risk with **non-overlapping** WSR intervals (`prereg §6.7` win-rule). The codeable claim is this signed non-overlap only (weighted `<` naive, intervals disjoint); the narrative "moves back toward the source level" is reported context, **not** a pinned effect size, since no doc fixes how far toward source counts as a pass. *Source: `flagship-playbook.md` Milestone B; `method_note §1.7`, §3.2; `prereg §6.7`.*
- [ ] **`ŵ_cov` tracks the oracle ratio.** Since the shift is synthetic, `w_cov*(x)=p_T(x)/p_S(x)` is known per point. Over `≥200` MC trials at the large-sample setting (`n_S, n_T ≥ 5000`), proposed pass line — mirroring G-C's BBSE-recovery gate — `mean_x |log ŵ_cov(x) − log w_cov*(x)| ≤ 0.1` (mean absolute log-ratio, restricted to the unclipped mass `ŵ_cov < w_max`), with the error shrinking monotonically (within MC noise) as `n_T` grows over a fixed grid — consistency, not a finite-sample certificate. **All numbers `(proposed)`.** *Source: `flagship-playbook.md` Milestone B ("checks `ŵ_cov` tracks the oracle") states the direction only; the tolerance, sample size, trial count, and error metric are **proposed** (the docs pin no per-gate threshold — `prereg §6.7`, §6.8 fix the win-rule and coverage check, not this).*
- [ ] **Base-rate constant `ĉ` enters only through clipping (Remark-3).** With `ĉ = n_S/n_T` known synthetically, assert `ŵ_cov` uses `ĉ` and that `ĉ` cancels in the self-normalized (Hájek) weighted quantile — i.e. it changes the estimate ONLY via the `w_max` clip boundary, not via the normalized weights. *Source: `method_note §1.5` (Tibshirani et al. 2019 eq.12 + Remark 3), §7.3.*
- [ ] **Group-id fold-disjointness** between `D_disc`, `D_cal`, and target test (set-intersection = 0). *Source: `method_note §1.7`.*
- [ ] **`d` is cross-fitted (out-of-fold scoring).** `d(x)` is fit by K-fold on `D_disc` and every scored point receives its `ŵ_cov` from a fold it was NOT used to fit; assert no point is scored in-fold, so weights are not optimistically fit on the points they reweight. *Source: `method_note §1.5`, §3.4 step 1.*
- [ ] **WSR weighted-path UCB is exercised** on the weighted estimator (no silent range-bound fallback). *Source: `prereg §6.7`.*

**REPORTED DIAGNOSTICS (measured, not gated):**

- [ ] **Clip-induced abstention rate** — the mass routed by `ŵ_cov > w_max`, since clipping biases the weights. *Source: `method_note §3.2`, §3.6; `flagship-playbook.md` Milestone B.*
- [ ] **Clip-induced bias sensitivity.** Beside the clip-abstention rate, report how the Hájek weighted risk moves across a small `w_max` grid (clipping biases the weights, so a non-flat cell means the reported risk is partly a clip artifact); self-test on synthetic data with known `w_cov*`. *Source: `method_note §1.5`, §3.2, §3.6; `prereg §5.4`, §6.6.*
- [ ] **`w_max` routed mass costs only coverage (decoupling note).** At G-B, `w_max` is a variance/boundedness control on the weighted estimator whose routed tail costs only coverage — it is NOT an OOD guard (far-OOD with `d(x)≈0` is not flagged by `ŵ_cov`; that guard is `t_ood`, introduced at G-D). No far-OOD claim is attached to `w_max` here. *Source: `method_note §3.2`, §4.2, §1.5.1.*
- [ ] **`n_eff = (Σŵ)²/Σŵ²`** beside every weighted risk; flag low-`n_eff` regimes. *Source: `method_note §1.7`.*
- [ ] **Discriminator AUROC / overlap diagnostic** as a covariate-shift-severity signal; weight-distribution summary (median/95/99/max). *Source: `method_note §1.5`, §3.4 step 1; `prereg §5.4`.*
- [ ] **Discriminator-shortcut audit.** Report `d`-AUROC before vs after masking/perturbing an injected synthetic "acquisition shortcut" feature; on synthetic data a self-test confirms the AUROC collapses when the planted shortcut is removed (the plumbing that, on real data, caveats the representativeness chip and the §3.2 regime tag when a discriminator keys on an artifact rather than clinical appearance). REPORTED, never gated. *Source: `method_note §3.5`; `prereg §5.4`, §3.2.*
- [ ] **Doubly-robust (DR / AIPW) covariate correction — key secondary analysis** (`prereg §4A` item 2). Augment the estimated `ŵ_cov` with a **cross-fitted** source outcome model `m̂(x) ≈ E_S[ℓ | φ(x)]`: `R̂_DR ∝ Σ_A [ m̂(x) + ŵ(ℓ − m̂(x)) ]` (self-normalized over the accepted in-scope set). On synthetic G-B data with a **deliberately misspecified** `ŵ_cov` (or a misspecified `m̂`), self-test that DR shrinks the residual toward the known target risk when **either** the weight **or** `m̂` is correct — the AIPW double-robustness (Yang, Kuchibhotla & Tchetgen Tchetgen 2024). The **Hájek single-model estimate stays the auditable primary**; DR is the residual-shrinkage diagnostic (DR-vs-Hájek delta + both WSR intervals), **asymptotic** not a finite-sample certificate, and under *joint* covariate+label shift it is an exploratory, caveated arm. REPORTED, never gated. *Source: `prereg §4A` item 2; `method_note §7.1`.*
- [ ] **Synthetic interval-coverage check (`prereg §6.8`)** of the weighted Hájek interval, run on synthetic data with **known ground-truth weighted risk and known weights**, reporting realized coverage vs nominal `1−δ` across the `n_eff` regimes; where it under-covers in a regime, relabel it **"nominal interval, coverage validated empirically"** there (the operative §6.8 action). Synthetic, not gated on data credentialing. *Source: `prereg §6.8`.*

### Required artifacts
- `conformal/weights.py` — **currently a 1-line stub; must implement** `d(x)` (cross-fitted), `ŵ_cov` with `ĉ` and `w_max` clip/route.
- Weighted (Hájek) path in `conformal/rcps.py` / `conformal/selective.py`.
- `tests/test_gate_b_covariate.py` — **currently empty.**
- A synthetic generator with known `p_S(x)`, `p_T(x)`, invariant `p(y|x)`, and closed-form `w_cov*(x)` — **not present.**

### Honesty rails (must-not-claim)
- MUST NOT attach a finite-sample / distribution-free coverage certificate to `ŵ_cov`: weighted conformal's `1−α` marginal coverage holds only under **pure covariate shift with the oracle likelihood ratio** (Tibshirani et al. 2019, Cor. 1); our weight is estimated (`method_note §1.5`, §7.1).
- MUST NOT treat the measured restoration as evidence the correction survives real cross-site shift — it verifies a cited property on known synthetic ground truth only.
- MUST NOT present `w_max` routing as an OOD guard: `w_max` catches the huge-weight / near-OOD variance tail only; the far-OOD guard `t_ood` does not exist until G-D (`method_note §3.2`, §4.2).
- `n_eff`, clip rate, discriminator AUROC are REPORTED DIAGNOSTICS, never gates (`method_note §1.7`; `prereg §5.4`).

### Open questions
- The per-gate numeric thresholds (oracle-tracking tolerance, resample count, the non-overlap margin) are **proposed** — the docs fix the win-rule (`prereg §6.7`) and the synthetic interval-coverage check (`prereg §6.8`) but not per-gate pass thresholds.
- `conformal/weights.py` and `tests/test_gate_b_covariate.py` are stub/empty; the synthetic covariate-shift generator is not yet present. G-B must be implemented before G-C's combined-weight checks and G-D's routing can run.

---

## 6. G-C — + Label / prevalence-shift correction (BBSE, MLLS+BCTS, Z-combine)

### Objective
Third rung (G-A exchangeable → G-B covariate → G-C label/prevalence). On synthetic data with **known** class-conditionals and a **known** target prevalence `p_T(y)`, demonstrate that (1) BBSE `ŵ_lab = Ĉ_S⁻¹ q̂_T` recovers the known ratio `p_T(y)/p_S(y)` given a well-conditioned confusion matrix; (2) MLLS+BCTS recovers the same prevalence and is BCTS-calibration-dependent; (3) the two weights combine via the per-`x` normalizer `Z(x)=E_{p_S(·|x)}[w_lab]` and are **NOT multiplied** — reproducing the double-count of the naïve product and confirming `Ẑ`-divide removes it, matching the worked 2-class example (`p_S=(.5,.5)→p_T=(.9,.1)`: truth `w=1.8`, naïve product `3.24`, `Ẑ`-divide recovers `1.8`). It must also show the label-aware `Ẑ`-combined path measurably lowers the Hájek realized selective risk versus the covariate-only G-B path under known prevalence shift. `κ(Ĉ_S)` and the `q̂_T` vs `Ĉ_S p̂_T` consistency check are REPORTED, never gated.

### Introduces
- BBSE `ŵ_lab = Ĉ_S⁻¹ q̂_T` (Lipton et al. 2018) with simplex projection + floor/ceiling `[w_lab,min, w_lab,max]`.
- MLLS + BCTS prevalence estimator (Alexandari et al. 2020): BCTS recalibration then EM.
- Per-`x` double-count corrector `Ẑ(x) = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}` with recalibrated softmax `σ̃`.
- Combine identity `ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)` (**NOT** the product).
- `κ(Ĉ_S)` / σ_min / effective-rank conditioning diagnostic; `q̂_T` vs `Ĉ_S·p̂_T` consistency check.
- Proposed ablations: MAPLS Dirichlet prior (`prereg §4A.1`), MLLS-without-BCTS (`prereg §4A.4`), the 5-arm `Ẑ`-combine decomposition (`prereg §4A.3`).

### Builds on
G-A (implemented) and **G-B** (`conformal/weights.py` clipped `ŵ_cov` and the Hájek path — **currently a stub, so G-B must be implemented before G-C's combined-weight checks can run**).

### Synthetic construction (the known ground truth)
A K-class generator (start `K=2` to mirror the `method_note §3.3`/§7.4 worked example, then `K≥3`) with **explicit, invariant** class-conditionals `p(x|y)` (e.g. well-separated Gaussians), so label shift holds by construction. Fix `p_S(y)` (e.g. `(.5,.5)`) and a known `p_T(y)` (e.g. `(.9,.1)`); the oracle ratio `w_lab*(y)=p_T(y)/p_S(y)` is then known (e.g. `(1.8,0.2)`). A frozen classifier `f` with a known confusion matrix `C_S` (well-conditioned; small `κ(C_S)`) yields `q_T` whose population value `C_S @ p_T` is known. Because `p(x|y)` is fixed, `p_S(y|x)` is closed-form, so the oracle `Z*(x)=Σ_{y'} w_lab*(y')·p_S(y'|x)` and oracle combined weight `w*(x,y)` are known per point. A **second** regime overlays a known covariate shift (G-B oracle `w_cov` known) on the prevalence shift; a **third** makes `C_S` progressively ill-conditioned to confirm `κ(C_S)` rises (reported) and the simplex floor keeps `w_lab` finite. All folds (`D_bbse^src`, `D_tar`, `D_cal`, `D_tar^lab`, test) drawn group-disjoint.

### Definition of Done

**HARD GATES (must pass):**

- [ ] **BBSE recovers the known label ratio.** Over `≥500` MC trials (fresh `D_bbse^src`, `D_tar` each), `max_y |ŵ_lab(y) − w_lab*(y)| ≤ 0.05` at the large-sample setting (`n_src, n_tar ≥ 5000`), with mean abs error shrinking monotonically (within MC noise) as `n_tar` grows over a fixed grid (250/1000/5000) — consistency, not a finite-sample certificate. *Source: `method_note §1.5`, §3.4 step 2; flagship Milestone C.*
- [ ] **MLLS+BCTS recovers the known prevalence.** Over `≥500` trials, `max_y |p̂_T(y) − p_T(y)| ≤ 0.05` at large sample, and `max_y |ŵ_lab,MLLS − ŵ_lab,BBSE| ≤ 0.10` (same identifiability target). *Source: `method_note §1.5`; `prereg §6.5`.*
- [ ] **`Ẑ`-divide recovers the oracle while the naïve product double-counts** (the load-bearing exactness check). On `K=2` pure-label-shift (`w_lab*=(1.8,0.2)`) at a class-1-dominant `x` with `p_S(1|x)≈1`: (a) naïve product `≈ 1.8·1.8 = 3.24` (within `1e-2`, oracle posteriors); (b) `Ẑ`-divide recovers `1.8` (`|ŵ − 1.8| ≤ 1e-6`, oracle `Ẑ`); (c) on the combined covariate+label synthetic, `mean|w_Zdivide − w*| ≤ 0.02` (oracle inputs) while the naïve product's mean error is `≥ 5×` larger. *Source: `method_note §3.3`, §7.4 (worked 2-class example).*
- [ ] **`Z` collapses correctly in the two pure regimes.** Pure label shift: `w_cov ≡ Z` (so `w = w_lab`) to `≤ 1e-6` per point. Pure covariate shift (`w_lab ≡ 1`): `Z ≡ 1` and `w = w_cov` to `≤ 1e-6`. *Source: `method_note §3.3`.*
- [ ] **Label-aware path lowers the Hájek risk vs covariate-only.** On a held-out target split with known prevalence shift, `R̂_w` under the `Ẑ`-combined weight is lower than covariate-only over `≥200` resamples with **non-overlapping** WSR intervals at the pre-registered `δ` (`prereg §6.7` win-rule), and tracks the known population accepted risk `E_{p_T}[ℓ|A]` to within the interval half-width. *Source: flagship Milestone C; `method_note §1.7`; `prereg §5.1`, §6.7.*
- [ ] **Simplex projection + floor/ceiling keep `w_lab` finite under ill-conditioning.** On the ill-conditioned regime, projected `p̂_T` lies on the simplex (nonneg, sums to 1 within `1e-8`) and every `ŵ_lab(y) ∈ [w_lab,min, w_lab,max]` with `w_lab,min > 0`, so `min_x Ẑ(x) > 0` (an invariant, not an accuracy gate). *Source: `method_note §1.7`, §3.4 step 2; `prereg §6.5`.*
- [ ] **Group-id fold-disjointness** across `D_bbse^src`, `D_tar`, `D_cal`, `D_tar^lab`, test. *Source: `method_note §1.7`.*

**REPORTED DIAGNOSTICS (measured, not gated):**

- [ ] **`κ(Ĉ_S)` conditioning** (and σ_min / effective rank) computed across the well-conditioned vs progressively-merged-class regimes; self-test that it is monotone increasing across that fixed degradation grid. Never gates the rung. *Source: `method_note §1.5`, §3.4 step 2, §7.5; `prereg §5.4`.*
- [ ] **Anti-causal consistency `q̂_T` vs `Ĉ_S·p̂_T`** near zero on a label-shift-satisfying synthetic, materially larger when `p(x|y)` is deliberately perturbed; reported as a continuous diagnostic. *Source: `method_note §3.4` step 2, §3.5, §7.5; `prereg §5.4`.*
- [ ] **`n_eff` (Kish)** beside every Hájek risk; self-test that it falls as the prevalence shift is made more extreme (heavier `w_lab` tail). *Source: `method_note §1.7`, §7.5; `prereg §5.4`.*
- [ ] **Synthetic interval-coverage check** of the `Ẑ`-combined weighted Hájek interval vs nominal `1−δ` across the `n_eff` regimes; where it under-covers in a regime, relabel it **"nominal interval, coverage validated empirically"** there (the operative §6.8 action). *Source: `prereg §6.8`.*
- [ ] **MLLS-without-BCTS** degrades the prevalence estimate when `f` is miscalibrated (BCTS is load-bearing); gap reported with betting intervals. *Source: `prereg §4A` item 4; `method_note §1.5`.*
- [ ] **MAPLS small-`m` ablation** reduces label-weight error vs MLLS+BCTS in the small-`m` regime, or reports "no measured small-`m` advantage". *Source: `prereg §4A` item 1.*
- [ ] **`Ẑ`-combine 5-arm decomposition** (no-reweight / covariate-only / label-only / naïve product / `Ẑ`-divide) reported at matched answer-rate; the only hard sub-claim is the (e)-matches-oracle / (e)-beats-(d) exactness check gated above. *Source: `prereg §4A` item 3.*
- [ ] **Factorization-entanglement diagnostic** (`ŵ_joint` vs factorized `ŵ`). Estimate a **direct joint weight** `ŵ_joint(x,y)` on `D_tar^lab` (real data) — and, as a synthetic self-test, on a constructed **entangled** regime (where `p(x|y)` moves *with* `p(y)`) vs a **factorizable** regime — and report its divergence (mean-squared log-ratio, estimated KL) from the factorized `ŵ_lab·ŵ_cov/Ẑ`. Self-test that the divergence is ≈0 in the factorizable regime and materially larger in the entangled one, so the diagnostic **detects** when the factorizable-shift premise fails; where it is large on a real pair, the combined claim is **downgraded** (`prereg §3.1` downgrade rule). `ŵ_joint` is a slice-level diagnostic **only**, never deployed (not identifiable from unlabeled target; high-variance at `m_lab`). REPORTED, never gated. *Source: `method_note §3.4` step 3, §7.4; `prereg §5.4`.*

### Required artifacts
- `conformal/label_shift.py` — **1-line stub; must implement** `bbse_weights(C_S, q_T)` (simplex projection + floor/ceiling), `mlls_bcts(...)` (BCTS + EM), `z_corrector(w_lab, σ̃)`, `combine_weights(w_lab, w_cov, Z)` returning `w_lab·w_cov/Z` (NOT product), `kappa_cs(C_S)`, `bbse_consistency(q_T, C_S, p_T_hat)`.
- `conformal/weights.py` — **G-B prerequisite (1-line stub):** clipped `ŵ_cov(x)`, needed to exercise the combined weight.
- A recalibrated softmax `σ̃` (temperature/Platt on a held-out source fold) feeding `Ẑ` (`method_note §3.3`/§3.4 step 3).
- `tests/test_gate_c_label_shift.py` — **currently empty.**
- Reuse Stage A `wsr_ucb` and the Hájek accepted-region machinery — implemented.
- A synthetic generator with known `p(x|y)`, `p_S(y)`, `p_T(y)`, `C_S`, closed-form `Z*(x)`/`w*(x,y)` — **not present.**

### Honesty rails (must-not-claim)
- MUST NOT claim the combined covariate+label weight is identifiable from unlabeled target alone — it is NOT; the factorizable-shift premise is a modeling choice and the residual is only **measured** on `D_tar^lab` (`method_note §3.3`, §7.1, §7.4; `prereg §8`). The factorization-entanglement diagnostic (above) makes that premise **falsifiable per pair**, but it is a measured divergence, not a test yielding a certificate.
- MUST NOT claim any `(α,δ)` or distribution-free guarantee for `w_lab`, MLLS+BCTS, or the combined weight under shift — BBSE/MLLS carry only **consistency** under label shift given an invertible `Ĉ_S` (`method_note §7.3`).
- MUST NOT gate the rung on `κ(Ĉ_S)`, `n_eff`, the `q̂_T`-vs-`Ĉ_S p̂_T` check, or any reliability diagnostic — REPORTED only (`method_note §7.5`; `prereg §5.4`).
- MUST NOT claim the anti-causal `p(x|y)`-invariance premise holds on real data — the single untestable premise of the label-shift regime; it breaks under cross-scanner appearance change (`method_note §3.5`, §7.6).
- MUST NOT combine the weights by multiplication anywhere in the deployed path — that double-counts; only the `Ẑ`-divide combine is correct (`method_note §3.3`).
- MUST NOT present the label-shift regime's lower measured risk as a certified corollary — it is the cleanest empirical regime, not a guarantee (`method_note §3.5`, §7.6).
- MUST NOT assume the `Ẑ(x)` softmax plug-in bias is removed by recalibration — it is non-vanishing and amplified on rare/high-`w_lab` classes; its residual is reported via a sensitivity sweep (`method_note §3.5`, §7.4; `prereg §5.4`).

### Open questions
- `conformal/weights.py` (G-B) and the recalibrator `σ̃` are prerequisites for the `Ẑ`-divide checks; G-C cannot be fully exercised until G-B exists.
- `tests/test_gate_c_label_shift.py` is empty, as are gate A/B/D tests; no synthetic gate harness or generator fixture exists yet.
- The numeric thresholds (0.05 prevalence-recovery tolerance, `≥500` MC trials, `1e-6` oracle-combine tolerance, the non-overlap margin) are **proposed** — the docs state the methods and the win-rule (`prereg §6.7`) and the synthetic interval-coverage check (`prereg §6.8`) but do not fix per-gate pass thresholds.
- Whether MAPLS / MLLS-without-BCTS / the 5-arm decomposition run as part of the G-C **synthetic** gate or only at the real-data stage is not pinned in the docs (`prereg §4A` lists them as pipeline ablations); treated here as **proposed** reported ablations.
- The `prereg §3.1` target-prevalence sweep is defined for the mixed real benchmark; its synthetic analogue (re-weighting whole synthetic cases so only `p_T(y)` moves) is implied but not separately specified.
- MLLS+BCTS reference is `kundajelab/labelshiftexperiments` (`abstention` package); the canonical-source cross-check with `assert`s (per the project's faithfulness discipline) is not yet done.

---

## 7. G-D — + Integrated OOD budget & routing

### Objective
Add the OOD screen and the integrated decision rule on top of the G-A/B/C weighted selective pipeline. On synthetic features with **known** in-distribution vs far-OOD ground truth, demonstrate that (1) the feature-space OOD score `o(x)` (Mahalanobis++ L2-normalized tied-covariance primary; plain Lee et al. tied-covariance, energy, kNN ablations) separates in-scope from a held-out exposure set `O`; (2) the threshold `t_ood` set on `O` to spend a target leakage budget `α_ood` measurably **reduces** far-OOD leakage into the answered path vs no screen, with realized leakage on `O` reported against `α_ood`; (3) the decoupled routing rule `A(x)={g=1} ∩ {o≤t_ood} ∩ {ŵ_cov≤w_max}` composes the two scope guards correctly — `w_max` catches the huge-weight near-OOD tail while `t_ood` is the **sole** far-OOD guard (a far-OOD point can have `d(x)≈0`, so `ŵ_cov` does not flag it). The rung shows `α_acc` and `α_ood` behave as two **separately measured** operating budgets, never a certified additive union split.

### Introduces
- `ood/detector.py` — **Mahalanobis++** primary `o(x)`: the L2-normalized tied-covariance Mahalanobis score `o(x)=min_c (φ̃−μ̂_c)ᵀ Σ̂⁻¹ (φ̃−μ̂_c)` on frozen features `φ̃ = φ/‖φ‖` (built from `mueller-mp/maha-norm`, Müller et al. 2025), **keeping** the *in-distribution-only, parameter-free* L2-normalization (it sees no OOD, so `O` stays disjoint/swappable) while **dropping** only the OOD-coupled add-ons (FGSM input perturbation, per-layer logistic-regression ensemble). The **plain Lee et al. 2018 tied-covariance score** (normalization removed) is the **canonical-baseline ablation**; kNN and energy are further ablations (`method_note §4.1`).
- `κ(Σ̂)` condition number and effective rank of the tied covariance (feature-collapse diagnostic; `method_note §4.1`; `prereg §5.4`).
- `conformal/budget.py` — `t_ood` selection on `O` as the smallest cutoff whose measured far-OOD leakage `≤ α_ood` (monotone scan); bookkeeping of `α_acc` / `α_ood` as two **separately-measured** operating budgets (**NOT** a certified additive split).
- Integrated answered event `A(x)` with the decoupled routing `ŵ_cov>w_max OR o>t_ood` and the three-way coverage decomposition (abstain-on-weight / route-on-OOD / defer-on-uncertainty) (`method_note §1.5.1`, §4.2, §5.1).
- OOD metrics: AUROC, FPR95, measured leakage = fraction of `O` scoring `≤ t_ood` (`prereg §5.2`).

> Note: the live `conformal/budget.py` stub docstring still reads "alpha = alpha_acc + alpha_ood"; per `flagship-playbook.md §5.4` this file is **reframed** to two separately-measured budgets and carries **no** certified additive split (`method_note §5.3`; `prereg §6.2`, §8). The docstring must be corrected when the module is implemented.

### Builds on
- G-A (`conformal/rcps.py` + `conformal/selective.py`) — implemented.
- G-B (`conformal/weights.py` `ŵ_cov` clipped at `w_max`) — **stub; must be filled** (G-D consumes `w_cov`/`w_max` for routing).
- G-C (`conformal/label_shift.py` MLLS+BCTS/BBSE + `Ẑ`-combine) — **stub; G-D stacks the OOD screen on the combined-weight pipeline.**

### Synthetic construction (the known ground truth)
Reusing the phase-0 0.1.6 toy: `D=20`-dim Gaussian features, `K≈5` classes with per-class means `μ_c` and a shared **tied** covariance `Σ` so the Mahalanobis GDA model is **exactly correct** in-distribution. Draw (a) an in-scope pool from the class-conditionals; (b) a **far-OOD** exposure set `O` as a disjoint shell far from every class Gaussian → **known OOD label** per point, so AUROC/FPR95/leakage all have ground truth; optionally a near-OOD tier reported separately (`prereg §5.2`). Known quantities: the oracle OOD indicator (a near-perfect detector → AUROC `≈1.0` on the far tier); the analytic far-OOD leakage as a function of cutoff (so `t_ood` selection can be checked to hit `α_ood` within scan resolution); a deliberately **imperfect** detector setting (score on a feature-collapsed/low-rank embedding) where some far-OOD leaks, making "screen reduces leakage" a non-trivial measured delta; a far-OOD point engineered with `d(x)≈0` (so `ŵ_cov` does not flag it → `t_ood` is the sole far-OOD guard); and a near-OOD point with `ŵ_cov>w_max` but `o≤t_ood` (so `w_max` routes it while `t_ood` does not — decoupling). Stack on the G-A/B/C pipeline so `A(x)` and its three-way decomposition are exercised end-to-end. Seeded via `np.random.default_rng(seed)`.

### Definition of Done

**HARD GATES (must pass):**

- [ ] **Mahalanobis++ (and the plain-score ablation) cross-check a reference implementation.** On `≥5` seeded synthetic draws, the **primary** `o(x)` from `ood/detector.py` equals the L2-normalized tied-covariance Mahalanobis `min_c (φ̃−μ̂_c)ᵀ Σ̂⁻¹ (φ̃−μ̂_c)` (with `φ̃=φ/‖φ‖`, and `μ̂_c`/`Σ̂` fit on **normalized** source features) from a brute-force numpy reference within `atol=1e-8`; confirm the L2-normalization path is **ON** for the primary, and that the **plain Lee et al. ablation** (normalization OFF) matches the unnormalized reference. *Source: `method_note §4.1` — Müller et al. 2025 Maha++ is the primary (keep the in-distribution-only L2-normalization; drop only the OOD-coupled FGSM/ensemble add-ons); plain Lee et al. 2018 is the canonical-baseline ablation.*
- [ ] **OOD score separates in-scope from the known far-OOD `O`.** AUROC `≥ 0.95` averaged over `≥200` resamples on the far tier; also report FPR95. *(AUROC pass-line proposed; the docs mandate reporting AUROC/FPR95 but state no numeric pass-line — `prereg §5.2`; `method_note §4.3`.)*
- [ ] **`t_ood` is the smallest cutoff on `O` whose measured leakage `≤ α_ood`.** For `α_ood ∈ {0.01, 0.05, 0.10}`, realized leakage `≤ α_ood` and within one scan-grid step of `α_ood`, compared against the analytic leakage-vs-cutoff curve. *Source: `method_note §4.3`; `prereg §6.2`; flagship Milestone D.*
- [ ] **The screen measurably reduces far-OOD leakage vs no screen** (the core Milestone-D claim), using the imperfect detector: mean leakage into `A(x)` with `t_ood` ON strictly lower than OFF, with non-overlapping intervals over `≥200` resamples. *Source: flagship Milestone D / §5.1 item D; `method_note §4.4`.*
- [ ] **`t_ood` is the SOLE far-OOD guard.** On the engineered far-OOD-with-`d≈0` points, routed-out rate under `w_max`-only `≈ 0` while under `o>t_ood` `≈ 1.0`. *Source: `method_note §4.2`, §1.5.1.*
- [ ] **The two routing mechanisms are DECOUPLED.** On near-OOD points with `ŵ_cov>w_max` but `o≤t_ood`, they are routed by the weight gate and NOT the OOD gate; the three-way decomposition sums to `1−coverage` and each mechanism's count matches the constructed ground-truth membership exactly. *Source: `method_note §4.2`, §3.2, §5.1, §1.5.1.*
- [ ] **`A(x)` is the exact conjunction of the three gates.** For `≥1000` randomized `(u,o,ŵ_cov,τ,t_ood,w_max)` inputs, `A(x) == (u≤τ) ∧ (o≤t_ood) ∧ (ŵ_cov≤w_max)` elementwise (zero mismatches); gate evaluation order does not change the answered set. *Source: `method_note §1.5.1`, §5.1.*
- [ ] **`α_acc` and `α_ood` tracked as two separately measured budgets; no additive union split asserted.** `budget.py` exposes them as distinct reported quantities; any `α` sum is a reporting-convenience headline only. Negative check: the code/test makes NO claim that realized accepted risk + realized leakage `≤ α_acc+α_ood` with confidence `1−δ`. *Source: `method_note §1.6`, §4.3, §5.3; `prereg §6.2`, §8.*
- [ ] **Fold disjointness:** `O` disjoint from the in-scope calibration/test pools (set-intersection = 0, printed invariant). *Source: `method_note §1.7`; `prereg §2.3`, §5.2.*
- [ ] **No regression up the ladder.** `tests/test_gate_a/b/c` plus `test_gate_d` all pass in one pytest invocation. *Source: flagship code-review step ("A–D still reproduce"); §5.1.*

**REPORTED DIAGNOSTICS (measured, not gated):**

- [ ] **Exposure-set sensitivity of `t_ood`** — re-run `t_ood` selection on `≥2` distinct synthetic `O` and report how `t_ood` and realized leakage move. *Source: `method_note §4.3`; `prereg §5.2`.*
- [ ] **`κ(Σ̂)` + effective rank** of the tied covariance; verify on a deliberately low-rank feature block that effective rank drops and `κ` rises (feature-collapse signature). *Source: `method_note §4.1`; `prereg §5.4`.*
- [ ] **Routing rule is detector-agnostic** — swap `o(x)` among {Maha++, plain-Lee, energy, kNN} — with ReAct as an OOD-agnostic wrap over any of them — holding the routing logic fixed; the routing *rule* is identical (detector-agnostic, `method_note §4.1`) and each detector yields a valid AUROC/FPR95/leakage triple at its own leakage-matched `t_ood`. (The *answered set* is NOT expected to be identical across detectors — different scores at a matched-leakage `t_ood` route different cases; only the rule/logic is invariant.) *Source: `method_note §4.1`; `prereg §5.2`; flagship §6.1.*
- [ ] **near-OOD tier leakage** reported separately from far-OOD and expected worse; asserted NOT presented as a bound. *Source: `prereg §5.2`; `method_note §4.4` (Fang et al. 2022).*
- [ ] **`n_eff`** on the answered-and-in-scope cohort; **clip-induced abstention rate** (mass routed by `ŵ_cov>w_max`). *Source: `method_note §1.7`, §3.2, §4.2; `prereg §5.4`.*
- [ ] **High-precision metadata pre-screen (real-data only).** On real deployment a **DICOM/header metadata pre-screen** precedes the learned `o(x)` as a *confident-flag, not a hard reject* (wrong modality / body-part routed out; missing or ambiguous tags fall through to `o(x)`; `method_note §4.2`), with the **image-vs-metadata disagreement rate** logged as a Stage-R number. It is **not exercised in the synthetic gate** (no DICOM headers) — named here so its omission from the synthetic ladder is explicit, not an oversight. *Source: `method_note §4.2`.*
- [ ] **Synthetic interval-coverage check** of the weighted Hájek interval on the answered-and-in-scope cohort vs nominal `1−δ` across the `n_eff` regimes; where it under-covers in a regime, relabel it **"nominal interval, coverage validated empirically"** there (the operative §6.8 action). *Source: `prereg §6.8`.*

### Required artifacts
- `ood/detector.py` — **1-line stub:** **Mahalanobis++** (L2-normalized tied-covariance) as the primary `o(x)`, with the plain Lee et al. 2018 tied-covariance score (normalization removed) as the canonical-baseline ablation, plus kNN + energy ablations; `κ(Σ̂)` + effective-rank; AUROC/FPR95 helpers.
- `ood/__init__.py` — **does not yet exist** (export the detector API).
- `conformal/budget.py` — **1-line stub (and stale docstring):** `set_t_ood(O_scores, α_ood)` monotone-scan; `α_acc`/`α_ood` as two separately-measured budgets; measured-leakage reporter; **no certified additive split.**
- Integrated decision rule producing `A(x)` + the three-way coverage decomposition (wired into `selective.py` or `budget.py` atop the G-B/C weighted path).
- `tests/test_gate_d_integrated_ood.py` — **currently empty.**
- A seeded synthetic OOD-feature generator yielding in-scope, far-OOD `O`, optional near-OOD tier, and the engineered `d≈0` far-OOD and `ŵ_cov>w_max` near-OOD probe points with known ground truth.
- Reuse of G-A `conformal/rcps.py`/`selective.py`, plus G-B `weights.py` and G-C `label_shift.py` (both stubs that G-B/G-C must fill first).

### Honesty rails (must-not-claim)
- MUST NOT claim any distribution-free or finite-sample OOD guarantee: OOD detection is not distribution-free learnable in the unrestricted setting (Fang et al. 2022); `t_ood` is a measured screen with a reported leakage rate on a stated, swappable `O` (`method_note §4.4`, §7.1; `prereg §8`).
- MUST NOT assert a certified additive `α = α_acc + α_ood` union-bound; their sum is a reporting convenience and the two budgets are separately measured (`method_note §5.3`; `prereg §6.2`, §8).
- MUST NOT claim the synthetic AUROC/leakage numbers survive real clinical (near-OOD) shift — the gate verifies the Mahalanobis GDA model's behavior under its own Gaussian assumption; near-OOD overlapping ID support is exactly the regime the impossibility covers (`method_note §4.4`).
- MUST NOT treat a passing G-D as evidence the deployed pipeline controls `R_T^accept` — an engineering wiring/measured-reduction check (project stance; `method_note §1.1`).
- MUST NOT treat `κ(Σ̂)`, effective rank, or `n_eff` as pass/fail gates (`method_note §1.7`, §4.1; `prereg §5.4`).
- MUST NOT couple the detector to `O` (no FGSM-perturbation / per-layer logistic-ensemble add-ons, no OOD-containing validation tuning); the deployed score and `t_ood` selection see only in-distribution features plus the disjoint exposure scan. The Maha++ L2-normalization we **keep** is itself in-distribution-only (each `φ(x)` scaled by its own norm; `μ̂_c`/`Σ̂` fit on normalized *source* features), so it does NOT couple the detector to `O` — only the FGSM/ensemble add-ons would, which is why those alone are dropped (`method_note §4.1`).
- MUST NOT conflate the two routing mechanisms: `w_max` is a variance/boundedness control whose routed mass costs only coverage; `t_ood` is the sole far-OOD guard (`method_note §4.2`, §5.1).

### Open questions
- Numeric pass-lines for OOD AUROC (proposed `≥0.95`) and the leakage-vs-`α_ood` tolerance (proposed within one scan-grid step) are **proposed** — the docs mandate reporting AUROC/FPR95/leakage and setting `t_ood` to spend `α_ood`, but state no synthetic numeric thresholds.
- Whether the near-OOD tier is in the synthetic G-D construction or deferred to real data (`prereg §5.2` mandates two tiers on real `O`; on synthetic data the far tier is the load-bearing hard gate, the near tier an optional reported diagnostic).
- The MC resample/seed count for the leakage-reduction interval (proposed `≥200`) is an engineering choice; `prereg §6.7` fixes resamples/seeds for real headline intervals but not synthetic gate tests.
- G-B (`weights.py`) and G-C (`label_shift.py`) are still stubs; G-D's routing and three-way decomposition cannot be fully exercised until `ŵ_cov`/`w_max` (G-B) and the `Ẑ`-combined weight (G-C) exist — G-D should not be implemented before B and C are filled.
- Energy/kNN OOD ablations are NICE-to-have (cut first under slippage) in the flagship CORE-vs-NICE table; only the Mahalanobis primary screen is CORE for the gate.

---

## 8. G-E — + Temporal / online adaptation (ACI / DtACI) — PROPOSED STRETCH RUNG, CURRENTLY OUT OF SCOPE

> **SCOPE VERDICT (read from the repo + docs, not inferred from the "G-A → G-E" framing): G-E is NOT a firm rung of the live applied ladder — it is a STRETCH rung explicitly SCOPE-CUT in every authoritative doc.** The live applied ladder is A→B→C→D, then the real-data headline (Stage R) and the subgroup audit (Stage S) — `flagship-playbook.md §5` (Stages A/B/C/D/R/S) and §5.1 (milestone checklist A,B,C,D,R,S). The temporal/online track is cut in five places: `flagship-playbook.md §0` intro ("the temporal DtACI track … is cut"), §4 ("the temporal DtACI track is scope-cut"), §5.3 (isgibbs/DtACI "dropped — they belonged to the abandoned temporal … tracks"), §5.4 ("`conformal/aci.py` is removed … it appears in none of the three docs"; the tests carry "no changepoint test"), and `method_note §7.6` item (vi) ("Temporal / within-target drift … is out of scope and belongs to post-market surveillance (control charts / CUSUM on realized error)"). ACI survives ONLY as a phase-0 **learning** demo (notebook step 0.1.5, checked against `herbps10/AdaptiveConformal`), never as a build gate. **THEREFORE every completion criterion below is `(proposed — not yet in docs)`, and this rung is admissible only as a clearly-labeled optional future-work appendix — and only AFTER the three docs are amended to admit it (today they forbid it).**

### Objective *(only if ever admitted)*
On a synthetic non-stationary stream with a label-feedback loop, demonstrate that the cited online-conformal property reproduces — long-run empirical coverage converging to `1−α` with **no** assumption on the drift (Gibbs & Candès 2021 ACI; DtACI as the dynamically-tuned variant) — on synthetic data with known ground truth, claiming no new guarantee and making no deployment/clinical claim.

### Introduces *(all proposed; none in any doc)*
- `conformal/aci.py` *(was explicitly REMOVED from the tree per `flagship-playbook.md §5.4`; would have to be re-added)* — online ACI: `α_{t+1} = α_t + γ(α − err_t)`, `err_t = 1[Y_t ∉ C_t(α_t)]`, fixed step size `γ>0` (Gibbs & Candès 2021).
- DtACI: an expert ensemble over a grid of step sizes `{γ_1..γ_k}` aggregated by exponential-reweighting so `γ` need not be hand-tuned (Gibbs & Candès 2022).
- A streaming/online evaluation harness with a sliding/expanding calibration window and a label-feedback loop.
- A synthetic non-stationary score-stream generator with known controllable drift (gradual ramp, abrupt changepoint, oscillating).
- `tests/test_gate_e_temporal.py` *(does not exist; only `test_gate_{a,b,c,d}*.py` exist, all empty)*.

### Builds on
- G-D (the rung the live ladder actually ends on before the real-data headline; `flagship-playbook.md` Stage D).
- Stage A static RCPS/selective machinery (`conformal/rcps.py`, `conformal/selective.py`) — the static-coverage baseline ACI would be contrasted against.
- `phase0_learning.ipynb` step 0.1.5 — the ACI drift-stream learning demo (the only place ACI legitimately lives today), checked vs `herbps10/AdaptiveConformal`.

### Synthetic construction (the known ground truth)
A synthetic 1-D nonconformity-**score** stream with known drift. At each `t=1..T` draw `s_t ~ F_t` where `F_t` drifts in a known way; the predictor emits a one-sided conformal threshold `q_t(α_t)` from its calibration window and "covers" iff `s_{t+1} ≤ q_t`. Three regimes: (1) **stationary** control (`F_t` fixed) — ACI must not degrade vs vanilla split conformal; (2) **gradual drift** — `F_t` mean ramps linearly so vanilla fixed-threshold coverage departs from `1−α` while ACI stays at `1−α`; (3) **abrupt changepoint** — `F_t` jumps at `T/2`; ACI recovers within a bounded transient, DtACI faster than fixed-γ ACI. Known target = the marginal coverage `1−α` (e.g. `α=0.1` → `0.90`). ACI's theorem (Gibbs & Candès 2021 Prop 4.1): the time-averaged miscoverage `(1/T)Σ err_t → α` for **any** sequence, with a finite-`T` envelope `|(1/T)Σ err_t − α| ≤ (α_1 + γ)/(γT)` when `α_t` is clipped to `[0,1]`. Proposed: `M=500` replicates per regime, `T=5000`, seeded per replicate (numpy-only).

### Definition of Done

**HARD GATES — all `(proposed — not yet in docs)`:**

- [ ] **ACI long-run coverage convergence on gradual drift** `(proposed — not yet in docs)`. `M=500`, `T=5000`, `α=0.10`, `γ=0.05`: `|mean_over_replicates(1 − mean_t(err_t)) − 0.90| ≤ 0.01`; finite-`T` envelope `|(1/T)Σ err_t − 0.10| ≤ (α_1+γ)/(γT)` for `≥99%` of replicates (`α_t` clipped to `[0,1]`). *Property: Gibbs & Candès 2021 Prop 4.1 / phase0 0.1.5. (`method_note §7.6`(vi) names temporal drift OUT OF SCOPE.)*
- [ ] **Vanilla baseline visibly breaks on the same stream** `(proposed — not yet in docs)`. On regime (2), `|mean vanilla coverage − 0.90| ≥ 0.05` while `|mean ACI coverage − 0.90| ≤ 0.01` (ACI's gap `≥5×` smaller), matched seeds. *Mirrors the relative-improvement structure of Milestones B/C; not in any doc.*
- [ ] **Abrupt-changepoint recovery** `(proposed — not yet in docs)`. Recovery time = first `t` after the changepoint where trailing-500-step coverage re-enters `[0.88, 0.92]`. Median ACI recovery `≤ 1000` steps; median DtACI recovery `≤` median best-fixed-γ ACI. *DtACI property, Gibbs & Candès 2022; not in any doc.*
- [ ] **Stationary no-harm control** `(proposed — not yet in docs)`. On regime (1), `|mean ACI coverage − 0.90| ≤ 0.01` AND `|mean vanilla coverage − 0.90| ≤ 0.01`; ACI mean interval width within 5% of vanilla (no stationary penalty). *Standard no-harm control, analogue of Stage-A sanity check; not in any doc.*
- [ ] **Canonical-source cross-check** `(proposed — not yet in docs)`. On a fixed seed and shared stream, the `α_t` trajectory and running-coverage curve match a reference (`isgibbs/AdaptiveConformalInference` or the phase-0 `herbps10/AdaptiveConformal`) to `1e-6`. *Per the project's canonical-source-faithful discipline; phase0 0.1.5.*
- [ ] **DOC-GATE / PRECONDITION (not a code check)** `(proposed — not yet in docs)`. This rung MUST NOT be built or merged until `method_note`, `positioning_memo`, and `preregistration` are amended to ADD temporal/online adaptation to scope. Today all three forbid it (`method_note §7.6`(vi); `flagship-playbook.md §0/§4/§5.3/§5.4`; the applied-paper-scope memory lists "the temporal/DtACI track" under do-not-revive). The precondition: a doc diff exists that (a) removes the `method_note §7.6`(vi) out-of-scope statement, (b) adds an online-adaptation method subsection citing Gibbs & Candès 2021/2022, and (c) adds a prereg protocol entry for the synthetic stream — BEFORE any `test_gate_e` code. Until then `in_docs=false` stands and the rung is appendix/future-work only. *Source: PROJECT STANCE + applied-paper-scope memory + `method_note §7.6`(vi) + `flagship-playbook.md §0/§4/§5.3/§5.4`.*

**REPORTED DIAGNOSTICS — all `(proposed — not yet in docs)`:**

- [ ] **Conditional / windowed coverage trajectory** `(proposed — not yet in docs)`. Emit the trailing-window coverage over time for ACI, DtACI, vanilla, plus the `α_t` path; annotate that local coverage is NOT controlled — only the Cesàro/time-average is. No threshold. *Honest-limitation diagnostic, analogue of `method_note §7.5` / `prereg §5.4` discipline.*
- [ ] **Step-size / clip sensitivity sweep** `(proposed — not yet in docs)`. Sweep `γ ∈ {0.005,0.01,0.05,0.1,0.5}`; log long-run coverage (should stay near 0.90 by the envelope for all `γ`) and median changepoint recovery time (decreases then destabilizes as `γ` grows). Demonstrates the γ-tuning pain DtACI removes. No threshold.

### Required artifacts *(all proposed)*
- `tests/test_gate_e_temporal.py` — **does not exist** (only `test_gate_{a,b,c,d}*.py` exist, all empty 0-line stubs; no aci/temporal test).
- `conformal/aci.py` — **does not exist; explicitly REMOVED per `flagship-playbook.md §5.4`** and would have to be re-added.
- A synthetic drift-stream generator + online harness with a label-feedback loop and known per-step drift (gradual / changepoint / stationary).
- A canonical-source cross-check fixture (`isgibbs/AdaptiveConformalInference` or the phase-0 `herbps10/AdaptiveConformal`).
- A doc amendment across `method_note.md` + `positioning_memo.md` + `preregistration.md` admitting online adaptation to scope (**PRECONDITION; today the docs forbid this rung**).

### Honesty rails (must-not-claim)
- MUST NOT claim G-E is a firm/required rung — it is OUT OF SCOPE and scope-cut in all three authoritative docs and the applied-paper-scope memory; the live ladder ends at Stage D + R + S.
- MUST NOT revive any abandoned theory: no temporal certificate, no keystone lemma, no impossibility theorem, no "full method" guarantee. ACI/DtACI are cited methods invoked only as properties of THOSE methods under THEIR own assumptions.
- MUST NOT claim ACI's long-run-coverage property transfers to the clinical pipeline or survives real temporal drift on CheXpert/MIMIC/CAMELYON17. The synthetic gate reproduces a cited property on synthetic data only.
- MUST NOT claim instantaneous / conditional / per-case coverage. ACI controls only the time-averaged (Cesàro) miscoverage; local windowed coverage oscillates and is reported, never gated.
- MUST NOT present DtACI as eliminating all tuning or adaptive to "any" drift in a deployment sense — the no-assumption-on-drift property is a worst-case time-average on the observed sequence, not a guarantee of good local behavior or clinical validity.
- MUST NOT use the words certify/certified/guarantee for the deployed pipeline.
- MUST NOT build or merge this rung before amending `method_note` (esp. removing the §7.6(vi) out-of-scope statement), `positioning_memo`, and `preregistration`.

### Open questions
- Is G-E firmly in scope or a stretch rung? **Answer (confirmed from repo + docs): a STRETCH rung, currently OUT OF SCOPE / scope-cut. Admissible only as optional future work after a doc amendment.**
- Does synthetic long-run-coverage reproduction add anything the paper needs, given `method_note §7.6`(vi) already routes temporal drift to post-market surveillance (CUSUM/control charts on realized error) as the named, doc-sanctioned alternative? If CUSUM-style monitoring is the doc's chosen answer, ACI/DtACI may be redundant to scope.
- If admitted, should temporal drift be addressed by ACI/DtACI or by the already-named post-market control-chart/CUSUM monitor (`method_note §7.6`(vi))? The docs currently favor the latter.
- ACI's error feedback `err_t` requires a timely `Y_t`; clinical deployment rarely provides labels on time — a scope obstacle, not just engineering.
- `tests/test_gate_{a,b,c,d}*.py` are all empty 0-line stubs; the A–D gates themselves are unimplemented, so G-E is far beyond the current build frontier and should not be prioritized over filling A–D.

---

## 9. Real-data headline (post-G-E)

> The synthetic ladder ends at G-D (G-E is out of scope). Passing the synthetic gates **de-risks** the real-data run; it does **not** transfer any property to it. Stages R and S are the live ladder's terminus (`flagship-playbook.md §5`, §5.1).

**Two imaging benchmarks, taken deep** (`prereg §2`; `flagship-playbook.md §1`, §4):

- **CAMELYON17-WILDS** — binary tumor-vs-normal patch classification under cross-**hospital** scanner/stain shift; the **covariate-dominant** benchmark. Each source→target **hospital pair** is reported separately, **never pooled** (`prereg §2.1`).
- **CheXpert ↔ MIMIC-CXR** — shared-label chest-radiograph finding classification under cross-**site** **mixed covariate + label** shift; **both directions** (CheXpert→MIMIC and MIMIC→CheXpert) run as two separate deployments (`prereg §2.2`). Credentialing (Stanford AIMI; PhysioNet + CITI) is the schedule-critical long pole.
- **Far-OOD exposure set `O`** — off-modality / wrong-anatomy / non-medical probes; stated and swappable, used to set `t̂_ood` and measure leakage (`method_note §4.3`; `prereg §5.2`).

**The headline is MEASURED degradation, never a recovered certificate.** The full pipeline (`method_note §§3–5`) is run against the four pre-registered baselines (naive split-conformal `ŵ≡1`; temperature-scaling-only; TRUECAM-style OOD detect-and-remove; BBSE label-weighting-only) on the same folds and frozen `f`. We report:

- realized selective (accepted) risk `R̂_T^accept` via the self-normalized (Hájek) estimator with its WSR betting interval at `δ` (`prereg §5.1`);
- coverage, decomposed into the three routing outcomes (abstain-on-weight / route-on-OOD / defer-on-uncertainty); AURC; target **ECE, NLL, and Brier** of `σ̃` (`prereg §5.1`);
- measured far-OOD leakage on `O` against `α_ood`, with exposure-set sensitivity;
- the Stage-S subgroup audit — per-subgroup selective risk, abstention/defer rate, and routing rates with finite-sample intervals (`prereg §5.3`);
- the **Decline-Attribution Record (DAR) construction-faithfulness invariant** — for every declined case, assert the reported firing set equals the gate predicates the routing rule actually evaluated false and each margin equals the recomputed score − threshold (`m_u=τ̂−u`, `m_ood=t̂_ood−o`, `m_w=w_max−ŵ_cov`); **discrepancy target exactly 0** (an executable invariant, not a modelling residual). This is the explainability leg's one executable Stage-R check; the fuller DAR faithfulness/stability battery is in `prereg §5.5` (`method_note §6.6`).

The success criterion is a **measurement** statement (`flagship-playbook.md §4`, §5.1; `prereg §8`): does the shift-aware pipeline empirically **maintain/improve** realized selective risk versus the naive baseline where naive split-conformal **measurably breaks**, and does the OOD screen **measurably reduce** leakage — **not** "accepted risk holds `≤ α`." Every headline is reported at a **matched answer-rate** (a coverage gap cannot masquerade as a risk gap); a "win" follows the **non-overlap interval rule** (`prereg §6.7`); operating points below `coverage_min` are flagged **degenerate** (`prereg §6.4`); and cells below the pre-registered `n_eff` **floor are excluded from headline aggregation** — formalizing the low-reliability `n_eff` *flag* (§5.4) into an explicit headline *exclusion*, so `n_eff` (never a rung gate) still drives a headline exclusion without ever becoming a gate (`prereg §6.7`, "Below-`n_eff`-floor exclusion").

**What is explicitly NOT claimed at Stage R** (`prereg §8`; `method_note §7`): no certificate of `R_T^accept`; no certified `α = α_acc + α_ood` split; no identification from `D_tar^lab`; no distribution-free OOD guarantee; no new guarantee from LTT under shift; no per-case guarantee. Each component method's guarantee is cited only as a property of that method under its own assumptions; real clinical shift breaks those assumptions, and the residual is **measured and reported with its sampling uncertainty**, never claimed away.
