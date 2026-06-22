# Gap 3 — Calibrated Selective Prediction Under Clinical Distribution Shift
### A start-to-finish project playbook — *instructor's-manual edition*

> **What this is.** A micromanaged, step-by-step plan that takes you from "I have a verified open gap" to "I have a submitted (then journal-grade) paper." Every phase is broken into atomic steps you can check off. It is phased: **Phase 0** locks foundations, **Phase 1** delivers the *minimal-viable-novel* (MVN) conference paper, **Phase 2** extends to the flagship. Each phase has a **definition of done** and a **go/no-go gate**.
>
> **How to read a step.** Every step uses the same block so you always know what to do and how to know you're done:
>
> > **Step X.Y — Title**
> > - **Goal:** the one outcome this step produces.
> > - **Do:** the concrete actions, lettered (a), (b), (c).
> > - **Output:** the artifact/file/number that now exists.
> > - **Done when:** the objective check that lets you move on.
> > - **Pitfall:** the specific mistake people make here.
>
> Some steps carry a **▶ Paste-in prompt** — drop it into a search-enabled assistant for a second brain. All prompts are collected in **Appendix D**.
>
> **Calibration.** This edition assumes you're **newer to heavy math and PyTorch**, so it *teaches* the hard concepts instead of presuming them — start with the **Primer** right after this section (it builds the math notation, the coding basics, and the five core ideas from the ground up). None of the complex material has been removed; the Primer, the **▸ In plain terms** notes inside steps, and **Appendix H** (a symbol-by-symbol decoder) just make it readable. If the Primer's assumed starting point is off for you, say so and I'll recalibrate. Pacing is open-ended (depth over speed), so steps are sequenced by dependency with rough effort bands, not calendar dates.
>
> **Conventions.** `monospace` = a file, command, symbol, or variable. "Backbone" without qualifier = the risk-control layer (CRC/RCPS/LTT), not the neural net (that's "base model"). Effort bands: **S** ≤ a day, **M** a few days, **L** a week+, **XL** multi-week.

---

## 0. The contribution on one page

**The gap.** No clinically validated selective-prediction method simultaneously: **(1)** abstains/defers uncertain cases; **(2)** gives a *distribution-free* guarantee on the error rate of answered cases; **(3)** keeps that guarantee approximately valid under real clinical shift (new hospital/scanner/population); **(4)** routes far-OOD inputs to abstain; **(5/5′)** is benchmarked across one — then several — real public medical shift datasets.

**Why it's open.** Methods with the guarantee lack far-OOD routing + multi-shift validation; the one method that handles shift + far-OOD (TRUECAM) has *no* guarantee and is single-modality. Nobody has put **selection + covariate/label shift + far-OOD** under **one** distribution-free risk-control statement that provably survives a real shift across diverse public benchmarks.

**Novelty core (never trim):** conditions **(2) + (3)** — a distribution-free error guarantee on answered cases that survives a real shift.

| Objective | Difficulty | Novelty weight | Trim policy |
|---|---|---|---|
| (1) Abstain/defer | Easy | Low (table stakes) | Keep — trivial |
| (2) Distribution-free guarantee | Moderate | **Core (spine)** | Never |
| (3) Holds under real shift | **Hard → Very Hard** | **Core (the open problem)** | Never |
| (4) Far-OOD routing | Easy bolt-on → Moderate integrated | Medium | Keep simple in Phase 1; it *helps* (3) |
| (5) ≥1 public shift benchmark | Easy | Low (validation) | Minimum bar |
| (5′) Multiple diverse benchmarks | Moderate effort | Low (breadth) | **The safe trim — defer to Phase 2** |

**Phase split.**
- **Phase 1 (MVN):** (1) + (2) + scoped (3) (covariate shift only; "approximately valid given unlabeled target data") + (4) bolt-on, on **two** public imaging benchmarks (CAMELYON17-WILDS + CheXpert↔MIMIC-CXR).
- **Phase 2 (flagship):** add label shift, OOD folded into the risk budget, the multi-modality sweep, ACI temporal track, stronger theory, journal write-up.

**Done =** Phase 1: a submittable paper + reproducible release whose headline plot shows your method holding target risk under a real shift where naïve conformal breaks. Phase 2: the journal-grade superset across modalities and shift types.

---

# PRIMER — learn these ideas first (start here if you're newer to math/coding)

> **Who this is for / assumed starting point.** You can run Python and follow a tutorial; you know high-school/intro-college math (functions, basic probability, averages); maybe you've trained one model. **Not assumed:** comfort with dense math notation, proofs, or PyTorch fluency. Goal: after this Primer you can read *every* symbol and idea in the playbook. (If this is too basic or too advanced, tell me and I'll adjust.)
>
> **How to use the Primer:** read P.1–P.4 once for intuition (an afternoon), keep **Appendix H** open as a symbol decoder, and do the optional skill-ramp in P.5 *alongside* Phase 0, not before it.

## P.1 The math you'll actually need (from scratch)

- **Function `f(x)`** — a machine: put `x` in, get an output. Our model `f` takes an image `x` and outputs scores.
- **Probability `P(A)`** — how often `A` happens, between 0 and 1.
- **Conditional probability `P(A | B)`** — how often `A` happens *among the cases where `B` is true*. **This is the key one:** our whole guarantee is about the error rate *among the cases the model chose to answer*.
- **Distribution, and `P_S` vs `P_T`** — the "pattern" of the data: which inputs are common, which labels go with them. `P_S` = the source hospital's pattern; `P_T` = a different (target) hospital's pattern. The project is about a promise made on `P_S` still holding on `P_T`.
- **Expectation / average `E[Z]`** — the long-run average of `Z`. `E[ℓ]` = average loss. `E[ℓ | answered]` = average loss among answered cases.
- **Quantile / percentile** — the value below which a given fraction of the data sits (the 95th percentile = the value 95% of points fall under). Conformal uses a high quantile of "wrongness scores" as a cutoff.
- **`α` (alpha)** — the error rate you're willing to tolerate (e.g., `0.05` = 5%). **`δ` (delta)** — a small allowed chance that the *promise itself* fails (confidence = `1−δ`).
- **A "bound"** — instead of an exact number, a promise like "≤ 5%." **"Distribution-free"** means that promise holds *no matter what the data pattern is* — you didn't have to assume a bell curve or anything.
- **Symbols you'll see:** `≤` at most · `≥` at least · `∝` proportional to (equal up to a constant) · `⌈·⌉` round up · `inf`/`sup` the smallest/largest value that works · `1[condition]` equals 1 if true else 0 · `x̂` ("x-hat") an *estimate* computed from data. (All collected in **Appendix H**.)

## P.2 The ML/coding you'll actually need

- **Classifier** — a model that outputs a probability for each class; the highest is its prediction.
- **Logits → softmax** — logits are raw scores; softmax squashes them into probabilities that sum to 1.
- **Features (embeddings)** — the vector the network computes *just before* the final layer: a numeric summary of the input. The out-of-distribution detector works on these, not the raw image.
- **Train / calibration / test splits** — *train* fits the model; *calibration* is a separate labeled set used only to set thresholds; *test* is untouched, for honest evaluation. **Keeping them separate (no "leakage") is sacred** — most ways to accidentally cheat are a leak between these.
- **Why a separate calibration set at all** — conformal's promise needs fresh, labeled data the model didn't learn from; that's what makes the guarantee trustworthy.
- **PyTorch, in a paragraph** — a library for building/running neural nets. For this project you'll mostly *load a pretrained model*, *run it* to get logits and features, and *save those numbers*. Much of the actual contribution is math done on the saved numbers (runs in seconds on a laptop CPU) — you do **not** need to be a deep-learning engineer to do it.
- **GPU vs CPU** — training a net wants a GPU; the conformal/RCPS math runs on cached numbers on a CPU, fast.

## P.3 The five core ideas in plain English (then the math, gently)

**1. Conformal prediction — the honesty thermostat.**
*Analogy:* a weather forecaster who, instead of a vague "70% chance," gives answers calibrated so they're right exactly as often as promised. *What it does:* converts a model's shaky confidence into a cutoff with a *real* coverage promise. *The math, gently:* for each calibration point compute a "wrongness score" (how surprised the model is by the true answer); take a high quantile of those scores as a cutoff `q̂`; on a new input, keep every answer less surprising than `q̂`. The kept sets contain the truth about `1−α` of the time — **as long as new data resembles calibration data** (this resemblance assumption is called *exchangeability*). *Why we need it:* it delivers condition **(2)** — a promise, not a vibe.

**2. Risk control — what *kind* of promise (CRC vs RCPS vs LTT).**
*Analogy:* three ways to promise a pizza arrives in 30 minutes — "on average" (CRC), "I'm 95% sure *this* pizza will" (RCPS), or "...even though I'm juggling several timers" (LTT). *What they do:* all three pick a threshold from calibration data so the error stays ≤ `α`. **CRC:** controls the *average* error. **RCPS:** adds a statistical safety margin so it's `1−δ` confident *this specific model's* error is ≤ `α` — the stronger promise medicine wants. **LTT:** lets you tune several knobs at once (your `τ`, `λ`, `t_ood`) and keep the promise valid. *Why:* the spine of **(2)**; RCPS is your clinical-grade version.

**3. Selective prediction — knowing when to say "I don't know."**
*Analogy:* a triage nurse who handles the clear cases and escalates the unsure ones. *What:* the model answers confident cases and *abstains/defers* the rest to a clinician. *The math, gently:* a threshold `τ` on an uncertainty score; you summarize behavior with a **risk–coverage curve** (error vs how much you chose to answer). *The catch:* by answering only the easy ones you've *biased* the group, so you must set the cutoff *on the answered group itself* — a subtlety the playbook keeps returning to. *Why:* condition **(1)**, and it interacts with the guarantee in (2).

**4. Distribution shift + weighting / ACI — the new-hospital problem.**
*Analogy:* a cake recipe tuned at sea level flops in the mountains; you must adjust. *What:* when the target hospital's data differs from your calibration data, the promise can quietly break. **Weighting** re-balances your calibration set to "look like" the new hospital; **ACI** keeps nudging the threshold over time as it sees mistakes. *The math, gently:* weight each calibration case by how *target-like* it is — a "density ratio" `w(x)` estimated by a small classifier that learns to tell source from target; ACI raises/lowers the threshold based on recent realized errors. *Why:* condition **(3)** — the hard, novel heart of the project.

**5. OOD detection + distance-aware features — "this isn't even the right kind of input."**
*Analogy:* a bouncer who turns away anyone obviously not on the list. *What:* inputs wildly unlike training (wrong scan type, corrupted image, a disease never seen) get routed straight to abstain. *The math, gently:* measure how *far* an input's features sit from the cloud of training features (Mahalanobis or nearest-neighbor distance); far = out-of-distribution. **Spectral normalization** is a training tweak that keeps the feature space honest so that "far" really means far (without it, very different inputs can collapse to look nearby). *Why:* condition **(4)** — and it *protects* (3) by discarding exactly the cases weighting can't safely handle.

## P.4 How the five combine (your project in one paragraph)

A frozen classifier produces **scores** and **features** for each input. **Selective prediction** decides answer-vs-abstain. **Conformal + RCPS** turn each "answer" into a *promised* error rate. **Weighting (or ACI)** keeps that promise valid at a new hospital. **OOD routing** discards inputs too strange to promise anything about. Your contribution is doing *all of this at once*, with the promise **provably surviving the new hospital**, demonstrated on real public datasets — which nobody has done.

## P.5 Optional skill-ramp (do it *alongside* Phase 0, not as a blocker)

A light on-ramp; search for current free versions, don't pay before you've looked:
- **Python/PyTorch:** PyTorch's official "60-Minute Blitz"; fast.ai's *Practical Deep Learning*.
- **Math intuition:** 3Blue1Brown (linear algebra, calculus, probability); StatQuest (statistics & ML basics).
- **Conformal prediction:** Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (the paper *and* its companion code/tutorial) — the single best starting point for this project.
- **Tip:** you learn these fastest by doing the Phase 0 toy notebooks (Steps 0.1.1–0.1.6) with one of the above open beside you.

---

# PHASE 0 — Foundations & gap lock-in

**Goal:** own the theory cold, extract exactly what each prior work leaves unsolved, re-confirm the gap is open. **Prereqs:** a GPU you can run small jobs on; your existing audit report. **Deliverables:** a toy-conformal notebook, a competitor matrix, a dated re-verification verdict, a positioning memo. **Effort:** L–XL.

> **Mindset for Phase 0:** you are not "reading the literature," you are *building working intuition by re-implementing the primitives on toy data*. Every concept below ends in a tiny runnable artifact. If you can't make coverage appear empirically on synthetic data, you don't understand it yet.

## 0.1 Build the conceptual spine (by re-implementing it)

> **Step 0.1.1 — Split conformal, by hand.** *(S)*
> - **Goal:** see distribution-free marginal coverage appear empirically.
> - **Do:** (a) take any trained classifier on a toy set (CIFAR-10, or synthetic Gaussians). (b) Hold out `n` calibration points; compute nonconformity scores `s_i = 1 − f_{y_i}(x_i)`. (c) Set `q̂ =` the `⌈(n+1)(1−α)⌉`-th smallest score. (d) On test points, form the set `{y : 1 − f_y(x) ≤ q̂}`; measure the fraction of test points whose true label is in the set, over many random cal/test splits.
> - **Output:** `notebooks/00_conformal_demo.ipynb` with a histogram of realized coverage across splits.
> - **Done when:** mean realized coverage ≈ `1−α` and you can derive the `⌈(n+1)(1−α)⌉` quantile from memory.
> - **Pitfall:** using the `n`-quantile (no `+1`) — undercovers for small `n`. Exchangeability is the only assumption; note it explicitly.
> - **▸ In plain terms:** a "nonconformity score" is just *how surprised the model is* by the correct answer. You gather these on a labeled hold-out set, pick a high cutoff (a quantile), and keep every answer less surprising than the cutoff. Done right, the kept sets contain the truth about `1−α` of the time — a real promise. (Primer P.3 #1.)

> **Step 0.1.2 — Conformal risk control (CRC).** *(S)*
> - **Goal:** control an *expected loss*, not just miscoverage.
> - **Do:** (a) define a bounded monotone loss `ℓ(λ)` (e.g., a miscoverage or error indicator as a threshold `λ` moves). (b) Implement the CRC threshold so `E[ℓ(λ̂)] ≤ α` (Angelopoulos et al., ICLR 2024). (c) Verify on the toy set.
> - **Output:** a `crc()` function + a test showing the expected loss lands at/below `α`.
> - **Done when:** you can state precisely *what* CRC controls (expectation over the calibration draw too) and why that differs from a per-model guarantee.
> - **Pitfall:** forgetting the loss must be *monotone* in `λ` and *bounded* for the vanilla CRC bound.

> **Step 0.1.3 — Selective prediction + why naïve filtering breaks.** *(M)*
> - **Goal:** build the abstain gate and feel the selection bias.
> - **Do:** (a) add `g(x) = 1[u(x) ≤ τ]` with `u` = softmax-response or entropy. (b) Plot the **risk–coverage curve** and compute **AURC**. (c) Demonstrate the trap: calibrate conformal on *all* data, then evaluate only on accepted points — show the guarantee no longer holds on the accepted subset.
> - **Output:** `notebooks/01_selective.ipynb` with the risk–coverage curve and the broken-guarantee demo.
> - **Done when:** you can explain in one sentence why selection breaks exchangeability and why you must calibrate *on the accepted region*.
> - **Pitfall:** treating coverage and selective risk as the same axis — they're not; report both.

> **Step 0.1.4 — Choose the risk-control backbone: CRC vs RCPS vs LTT.** *(M)*
> - **Goal:** pick the layer that owns condition (2), with eyes open.
> - **Do:** (a) implement **RCPS** (Bates et al., JACM 2021): pick `λ̂` from a concentration **upper** confidence bound (Hoeffding–Bentkus; or Waudby-Smith–Ramdas betting bound to tighten) so `P(R(λ̂) ≤ α) ≥ 1−δ`. (b) On the toy set, plot RCPS's `(α,δ)` realized risk vs CRC's expectation control — see RCPS is more conservative. (c) Read enough of **LTT** (Learn-then-Test; Angelopoulos et al. 2021) to know it controls *non-monotone* losses and *multiple* thresholds via multiple testing.
> - **Output:** `conformal/rcps.py` + a comparison plot; a one-paragraph decision note.
> - **Done when:** you've written down: **RCPS/LTT is the primary backbone** (PAC `(α,δ)` is the clinical statement; LTT fits your 3-knob `τ,λ,t_ood` design), **CRC is the less-conservative comparison.**
> - **Pitfall:** thinking RCPS "fixes shift" — it does not; it's a stronger *risk-control* statement under exchangeability. Shift is handled separately (0.1.5).
> - **▸ In plain terms:** CRC promises "*on average* the error is ≤ α." RCPS promises "I'm `1−δ` sure *this* model's error is ≤ α" — stronger and more honest for medicine, at the price of being a bit more cautious. LTT lets you make that promise while tuning several dials (`τ, λ, t_ood`) at once. (Primer P.3 #2.)

> **Step 0.1.5 — Shift tools: weighted conformal + ACI.** *(M)*
> - **Goal:** see exactly how shift breaks the guarantee and how each tool repairs it.
> - **Do:** (a) simulate covariate shift (resample inputs by a known `w(x) = p_T/p_S`). (b) Show naïve conformal undercovers on the target; implement **weighted conformal** (Tibshirani et al. 2019) with the *true* weights and show coverage restored. (c) Replace true weights with weights from a source-vs-target **domain discriminator** `d(x)`, `w(x) ∝ d(x)/(1−d(x))`, clipped — observe degradation as the discriminator weakens. (d) Implement **ACI** (Gibbs & Candès 2021) on a synthetic *stream* with drift; watch `α_t` self-correct.
> - **Output:** `notebooks/02_shift.ipynb` with three panels: naïve breaks, weighted restores, weight-error degrades; plus an ACI stream demo.
> - **Done when:** you can state that weighted conformal = *batch covariate shift with a target sample*, ACI = *online with label feedback, long-run guarantee*.
> - **Pitfall:** importance-weight blow-up — unbounded `w` destroys finite-sample bounds; note this now (it motivates OOD routing).
> - **▸ In plain terms:** at a new hospital the data looks different, so the promise can break. "Weighting" counts each calibration case more or less depending on how much it resembles the new hospital, which repairs the promise — unless a case is *so* different that its weight explodes, which is your cue to abstain instead. (Primer P.3 #4–5.)

> **Step 0.1.6 — OOD + distance awareness.** *(M)*
> - **Goal:** build the far-OOD signal and see why feature geometry matters.
> - **Do:** (a) implement feature-space **Mahalanobis** (Lee et al. 2018) and **kNN** (Sun et al. 2022) OOD on the penultimate features; compute AUROC vs a held-out OOD set. (b) Train the same backbone with **spectral normalization** and recompute — observe better separation (approx. bi-Lipschitz features resist *feature collapse* where far-OOD maps near in-distribution). (c) Connect it: the far tail where `w(x)` blows up is exactly what OOD routing removes.
> - **Output:** `notebooks/03_ood.ipynb` with OOD AUROC ± spectral norm.
> - **Done when:** you can explain why "route far-OOD to abstain" both protects patients (4) *and* keeps the weighted guarantee usable (3).
> - **Pitfall:** computing Mahalanobis on collapsed features and concluding OOD detection "doesn't work" — fix the features first.

> **Step 0.1.7 — The honesty constraint (write it on a sticky note).** *(S)*
> - **Goal:** fix the exact claim you can defend.
> - **Do:** write the assumption set you will repeat everywhere: *covariate shift only* + *unlabeled target sample of size `m`* + *far-OOD routed away so residual density ratio is bounded* ⇒ "**approximately valid**." Note that exact distribution-free validity under *arbitrary* shift with *zero* target info is impossible.
> - **Output:** one line at the top of your positioning memo.
> - **Done when:** every future guarantee sentence you write carries this assumption.
> - **Pitfall:** drifting into "valid under any shift" in the abstract — instant reject.

**▶ Paste-in prompt — concept tutor / steelman** (run after 0.1.1–0.1.7; full text in Appendix D-1).

## 0.2 Read the prior art with intent

> **Step 0.2.1 — Build the competitor matrix.** *(M)*
> - **Goal:** one row per paper, columns that matter to *your* gap.
> - **Do:** make a spreadsheet `analysis/competitor_matrix.csv` with columns: `Title | Venue | Year | Preprint? | satisfies (1)? (2)? (3)? (4)? | exact missing piece | reusable asset (code/split/score)`. Fill from your audit: SCRC, SCoRE, Conformal Triage, TRUECAM, Liang & Sun, the "Pitfalls" paper, the 2026 triage-under-prevalence audit, plus seed baselines (weighted/adaptive conformal, SelectiveNet, ensembles/MC-dropout, Mahalanobis/energy/kNN).
> - **Output:** the filled matrix.
> - **Done when:** no "missing piece" cell is empty, and SCRC's proof structure is summarized in your own words (it's your closest ancestor).
> - **Pitfall:** judging from abstracts — open the methods; the "what's missing" claim must be exact.

> **Step 0.2.2 — Clone and skim the reusable code.** *(S–M)*
> - **Goal:** know what you can stand on.
> - **Do:** clone the repos for SCRC/RCPS/LTT/WILDS and any baseline; confirm they run on a toy input.
> - **Output:** `third_party/` notes on what each repo gives you.
> - **Done when:** you've identified which baseline implementations you'll reuse vs. reimplement.

> **Step 0.2.3 — Write the residual-contribution sentences.** *(S)*
> - **Goal:** the three sentences your paper's intro will compress.
> - **Do:** state exactly what remains novel after *all* competitors: the composition of selection + weighted RCPS + far-OOD routing under one guarantee that survives real shift, across ≥2 public benchmarks.
> - **Output:** three sentences pasted into the memo.
> - **Done when:** a skeptical reader can't point to one paper that already does it.

## 0.3 Re-verify the gap is still open

> **Step 0.3.1 — Run the falsification re-check.** *(S)*
> - **Goal:** confirm nothing closed the MVN since your audit.
> - **Do:** run the falsification prompt (Appendix D-2), focused on mid-2025→today camera-readies (MICCAI/MLHC/CHIL/ML4H/NeurIPS/ICML/ICLR/TMLR/MedIA/npj Digital Medicine).
> - **Output:** `analysis/reverify_<date>.md` with verdict + evidence table.
> - **Done when:** verdict is OPEN or PARTIALLY ADDRESSED, dated.
> - **Pitfall:** trusting a single search pass; cross-check the top hits manually.

> **Step 0.3.2 — Decide: proceed or pivot.** *(S)*
> - **Goal:** a clear go decision.
> - **Do:** if OPEN/PARTIAL → proceed. If a single work CLOSED the exact MVN → pivot to the nearest open variant (jump to label shift, or an uncovered modality) and update the memo.
> - **Done when:** the decision and rationale are written down.

## 0.4 Positioning memo (Phase 0 deliverable)

> **Step 0.4.1 — Write the 2–3 page memo.** *(M)*
> - **Goal:** everything the paper's intro will compress, in one place.
> - **Do:** sections — (i) gap + one-sentence contribution + assumption set (0.1.7); (ii) competitor matrix + residual sentences (0.2); (iii) re-verification verdict + date (0.3); (iv) the **falsifiable MVN claim**: *"On {A,B}, our method maintains selective risk ≤ α on accepted in-distribution-enough cases under {shift}, where naïve split-conformal selective prediction does not, given an unlabeled target sample of size m."*
> - **Output:** `docs/positioning_memo.md`.
> - **Done when:** the falsifiable claim is a single, testable sentence.

> ### ● GATE 0 → 1 (check all before coding the real method)
> - [ ] You can derive split-conformal + CRC, and state RCPS's `(α,δ)` guarantee, from memory.
> - [ ] Toy notebooks 00–03 reproduce: coverage, risk–coverage, weighted-restores-coverage, OOD AUROC.
> - [ ] Competitor matrix has no empty "missing piece" cell.
> - [ ] Re-verification verdict is OPEN/PARTIAL and dated.
> - [ ] The falsifiable MVN claim is written.
> - **If any box is empty, stay in Phase 0.**

---

# PHASE 1 — The minimal-viable-novel paper

**Goal:** the smallest experiment that fully proves the (2)+(3) core, plus (1) and a bolt-on (4), on two imaging benchmarks. **Deliverables:** a method note, a locked design doc, a tested codebase, the headline result, a submitted paper, a release. **Effort:** XL (the bulk of the project).

## 1.1 Formalize the method (paper-first; the novelty core)

> **Step 1.1.1 — Write the notation table.** *(S)*
> - **Goal:** unambiguous symbols before any prose.
> - **Do:** define `P_S, P_T` (source/target), base model `f`, point prediction `ŷ(x)`, uncertainty `u(x)`, selection `g(x)=1[u(x)≤τ]`, loss `ℓ(y,ŷ)∈[0,1]`, weights `w(x)`, OOD score `o(x)`, thresholds `τ,λ,t_ood`, target sample size `m`.
> - **Output:** a notation block in `docs/method_note.md`.
> - **Done when:** every symbol you'll use later is defined once.

> **Step 1.1.2 — State the target guarantee.** *(S)*
> - **Goal:** the exact thing you promise.
> - **Do:** write `R_T^accept := E_{(X,Y)~P_T}[ ℓ(Y,ŷ(X)) | g(X)=1, X not far-OOD ] ≤ α`.
> - **Output:** one boxed equation in the method note.
> - **Done when:** you can say in words: "expected loss among accepted, not-far-OOD target cases ≤ α."
> - **Pitfall:** conflating this with marginal coverage — it's a *conditional* (on acceptance) *risk*, on the *target*.
> - **▸ In plain terms:** read the equation right-to-left — among target-hospital cases the model *chose to answer* that aren't bizarre inputs, the *average mistake rate* stays at or below your budget `α`. The `E[… | …]` just means "average, restricted to those cases." (Primer P.1 + P.3 #2.)

> **Step 1.1.3 — Enumerate the three exchangeability-breakers + neutralizers.** *(M)*
> - **Goal:** show each threat is handled.
> - **Do:** write, for each: **Selection** → calibrate the controlling threshold *on the accepted region* (selective RCPS/CRC). **Covariate shift** → reweight calibration by `ŵ(x)` from a clipped domain discriminator (weighted form). **Far-OOD tail** → where `ŵ>w_max` or `o(x)>t_ood`, route to abstain; this removes the mass that invalidates the weighted bound — the safety mechanism *is* the math mechanism.
> - **Output:** a three-row table in the note.
> - **Done when:** none of the three is left unaddressed.

> **Step 1.1.4 — Write the calibration + inference algorithm.** *(M)*
> - **Goal:** implementable pseudocode.
> - **Do:** transcribe and adapt:
> ```
> CALIBRATION (source cal set + unlabeled target sample):
>   1. Fit domain discriminator d(x); w(x)=clip(d(x)/(1-d(x))*c, 0, w_max).
>   2. Fit OOD score o(x) on f's features; pick t_ood for a small abstain budget ρ.
>   3. Among NON-OOD calibration points, choose (τ, λ) so the WEIGHTED risk on
>      accepted points <= α via your backbone (RCPS upper bound / LTT test / CRC).
>   4. Record (τ, λ, t_ood, w_max, c).
> INFERENCE on target x:
>   if o(x)>t_ood or w(x)>w_max:  abstain (far-OOD)      # (4)
>   elif u(x)>τ:                  abstain (defer)         # (1)
>   else:                         answer ŷ(x), risk<=α   # (2)+(3)
> ```
> - **Output:** the algorithm block in the note.
> - **Done when:** a competent coder could implement it without asking you questions.

> **Step 1.1.5 — Decide CRC vs RCPS vs LTT for the deployed claim.** *(S)*
> - **Goal:** commit the backbone.
> - **Do:** default to **weighted RCPS** for the `(α,δ)` statement; use **LTT** if you tune `(τ,λ,t_ood)` jointly (multiple testing keeps it valid); keep **CRC** as the less-conservative comparison. Note the bonus: with importance weights the RCPS bound *widens with weight variance*, so OOD-routing the high-weight tail also *tightens* it.
> - **Output:** a "backbone decision" paragraph.
> - **Done when:** the choice and its justification are written.

> **Step 1.1.6 — Write assumptions, theorem, proof sketch.** *(L)*
> - **Goal:** a defensible guarantee statement.
> - **Do:** state assumptions (covariate shift; bounded density ratio after OOD routing; target sample `m`); label finite-sample vs asymptotic; write the composition (selective + weighted RCPS) as a theorem with a proof *sketch* that chains the cited results; characterize how weight-estimation error degrades the bound.
> - **Output:** the theorem + sketch in the note.
> - **Done when:** every step of the sketch cites a specific result you can point to.
> - **Pitfall:** hand-waving the composition — selection × weighting × OOD each touch exchangeability; show they compose.

> **Step 1.1.7 — Red-team the method.** *(S)*
> - **Do:** run the method red-team prompt (Appendix D-3) against the full note; fix the weakest claim.
> - **Done when:** you've addressed the single attack a reviewer hits first.

## 1.2 Lock experimental design *before* coding

> **Step 1.2.1 — Fix datasets + exact splits.** *(S)*
> - **Do:** **CAMELYON17-WILDS** (5-hospital scanner/stain shift; official ID vs OOD split). **CheXpert ↔ MIMIC-CXR** (institution/population shift; one direction source→target). Define a *target calibration* slice (unlabeled) disjoint from *target test*.
> - **Output:** `docs/design.md` §datasets with split definitions.
> - **Done when:** the calibration/test boundary is written and unambiguous.

> **Step 1.2.2 — Fix shift type + scope.** *(S)*
> - **Do:** covariate shift only for the MVN; state label/prevalence shift is Phase 2.
> - **Done when:** scope sentence is in the design doc.

> **Step 1.2.3 — Define metrics with formulas.** *(M)*
> - **Do:** *primary* — realized selective risk on accepted target vs `α`, reported as a **distribution over many cal/test splits** (histogram with the `α` line). *Operating* — risk–coverage curve + AURC (source and target). *Shift stress* — coverage/risk degradation, yours vs naïve. *OOD* — detector AUROC; accepted-risk with vs without routing. *Secondary* — base-model ECE, abstention rate, accepted accuracy.
> - **Output:** `analysis/metrics.py` signatures + the design doc §metrics.
> - **Done when:** each metric has a formula and a target.
> - **Pitfall:** reporting a single realized risk — the guarantee is marginal over the calibration draw; show the distribution.

> **Step 1.2.4 — Fix the baseline set.** *(S)*
> - **Do:** (1) naïve split-conformal selective [the foil]; (2) SelectiveNet/softmax-response; (3) SCRC (guarantee, no shift); (4) weighted conformal w/o selection; (5) deep ensembles or MC-dropout + threshold [strong UQ baseline]; (6) **yours**.
> - **Done when:** the list is frozen in the design doc.

> **Step 1.2.5 — Fix ablations.** *(S)*
> - **Do:** weighting on/off; OOD routing on/off; `w_max` sensitivity; target size `m` sensitivity; OOD detector choice; backbone RCPS vs CRC.
> - **Done when:** listed.

> **Step 1.2.6 — Pre-register the success criterion.** *(S)*
> - **Do:** write to `docs/preregistration.md`: "Yours holds risk ≤ α under shift on both datasets while ≥1 baseline visibly violates it, at comparable or better coverage." Commit it before seeing results.
> - **Done when:** committed to git with a timestamp.
> - **Pitfall:** deciding success *after* seeing results — that's how you fool yourself.

> **Step 1.2.7 — Critique the design.** *(S)*
> - **Do:** run the experiment-design critic prompt (Appendix D-4); add any missing baseline.

## 1.3 Environment & repo scaffolding

> **Step 1.3.1 — Create the repo tree.** *(S)*
> - **Do:** scaffold (full tree in Appendix E):
> ```
> data/  models/  conformal/  ood/  experiments/  analysis/  tests/  docs/  paper/
> ```
> - **Output:** the directory skeleton + `README.md`.
> - **Done when:** `git init` done, structure committed.

> **Step 1.3.2 — Pin the environment.** *(S)*
> - **Do:** `conda`/`uv` env + `requirements.txt` (torch, wilds, numpy, scipy, scikit-learn, pandas, matplotlib, wandb/mlflow, pytest); add a lockfile.
> - **Done when:** a fresh env reproduces from the lockfile.

> **Step 1.3.3 — Wire experiment tracking.** *(S)*
> - **Do:** init W&B/MLflow; log dataset, shift split, `α`, `m`, weights/clip, `t_ood`, realized risk, coverage, AURC, seed.
> - **Done when:** a dummy run logs all fields.

> **Step 1.3.4 — Config + seeding.** *(S)*
> - **Do:** Hydra/YAML configs (one file per experiment); a `set_seed()` that seeds python/numpy/torch and logs it.
> - **Done when:** two runs with the same config+seed match bit-for-bit on metrics.

> **Step 1.3.5 — Test harness.** *(S)*
> - **Do:** `pytest` skeleton; CI optional. The conformal core (1.6) gets synthetic-data tests with known ground-truth guarantees — non-negotiable.
> - **Done when:** `pytest` runs green on a trivial test.

## 1.4 Data pipelines

> **Step 1.4.1 — CAMELYON17 via WILDS.** *(M)*
> - **Do:** (a) `from wilds import get_dataset; ds = get_dataset('camelyon17', download=True)` (confirm current API). (b) Build loaders for train (source), ID-val, and OOD-test (target). (c) Carve an unlabeled *target calibration* slice from target, disjoint from target test. (d) Cache.
> - **Output:** `data/camelyon17.py` loaders + a split manifest.
> - **Done when:** shapes/counts print and the cal/test disjointness is asserted in code.
> - **Pitfall:** letting target test leak into weight/OOD/threshold fitting — assert disjoint index sets.

> **Step 1.4.2 — CheXpert ↔ MIMIC-CXR access + harmonization.** *(M–L)*
> - **Do:** (a) **start credentialing now** (Stanford AIMI for CheXpert; PhysioNet + CITI training for MIMIC-CXR). (b) Map to a shared label subset (common pathologies); document every mapping. (c) Harmonize preprocessing (resize, intensity normalization, handling of uncertain/blank labels). (d) Define source→target direction (+ optional reverse).
> - **Output:** `data/cxr.py` + `docs/label_mapping.md`.
> - **Done when:** both datasets yield the same label tensor schema.
> - **Pitfall:** silent label-schema mismatches between the two — they differ; verify counts per class.

> **Step 1.4.3 — Smoke dataset.** *(S)*
> - **Do:** a few-hundred-image subset that runs the whole pipeline in minutes.
> - **Done when:** end-to-end dry run completes on it.

## 1.5 Base predictor + base-model / UQ decision

> **Step 1.5.1 — Choose backbones.** *(S)*
> - **Do:** ImageNet-pretrained CNN/ViT per modality (WILDS reference model for CAMELYON17; a standard CXR backbone for chest X-ray).
> - **Done when:** backbone + training recipe fixed in config.

> **Step 1.5.2 — The base-model / UQ decision (read before training).** *(S)*
> - **Goal:** decide what produces `u(x)` and `o(x)` — without confounding your contribution.
> - **Key framing:** the conformal/RCPS layer owns the guarantee and is *base-model-agnostic*; a fancier UQ backbone only buys **efficiency** (less abstention at the same guaranteed risk) and **better OOD ranking**, never validity. So choose by quality vs compute vs *confound*.
> - **Decision for the MVN:** **standard backbone + softmax-response/energy for `u(x)` + spectral-norm-enabled Mahalanobis for `o(x)`.** Rationale: keeps attribution clean (the win is your guarantee layer, not a fancy net), while still using the *one* principled OOD trick — spectral normalization gives approximately bi-Lipschitz features so distance-based OOD is meaningful (no feature collapse).
> - **Optional MVN add:** one **deep-ensemble** baseline row as the quality ceiling (Ovadia et al. 2019), if single-GPU compute allows K sequential trainings.
> - **Deferred to Phase 2 (§2.2):** **SNGP** and **DDU** as base-model *ablations* showing the guarantee layer is backbone-agnostic and gets more efficient with distance-aware features. (DDU — spectral norm + GMM density on features — is often the cleaner fit than SNGP because it decouples the OOD density from the softmax head you conformalize; SNGP's GP head adds tuning for a benefit conformal partly subsumes.)
> - **Output:** a "UQ decision" paragraph in `docs/design.md`.
> - **Done when:** the MVN uses the standard path and SNGP/DDU are explicitly parked for Phase 2.
> - **Pitfall:** building the core method on SNGP — then a reviewer can't tell if gains come from your contribution or the backbone.

> **Step 1.5.3 — Train source-only; freeze.** *(M)*
> - **Do:** train on source train split (add spectral norm to the backbone so features are OOD-friendly); checkpoint; freeze. Everything downstream treats `f` as fixed.
> - **Done when:** a frozen checkpoint + source/target val metrics logged.
> - **Pitfall:** any peeking at target labels during training/selection — forbidden.

> **Step 1.5.4 — Export + cache scores/features/labels.** *(S)*
> - **Do:** for every cal/test point, dump logits/softmax, penultimate features (for Mahalanobis/kNN + the discriminator), labels where available. The conformal layer iterates on these, not raw images.
> - **Done when:** cached tensors load in seconds.

> **Step 1.5.5 — Sanity-check base calibration.** *(S)*
> - **Do:** reliability diagram + ECE on source and target.
> - **Done when:** you *know* the base calibration (conformal will fix coverage regardless, but you must interpret results).

## 1.6 Implement the conformal selective layer (staged, with checkpoints)

> **Step 1.6.1 — Stage A: exchangeable selective RCPS/CRC.** *(M)*
> - **Do:** implement split-conformal/CRC/RCPS with a selection threshold calibrated *on the accepted region*.
> - **Output:** `conformal/selective.py`.

> **Step 1.6.2 — Stage A test (gate).** *(S)*
> - **Do:** on synthetic *exchangeable* data, realized accepted-risk ≤ α at the nominal rate across many splits.
> - **Done when:** the histogram straddles α as theory predicts. **If it fails, your calibration is wrong — stop and fix.**

> **Step 1.6.3 — Stage B: weights.** *(M)*
> - **Do:** domain discriminator `d(x)`; `w(x)=clip(d/(1−d)*c,0,w_max)`; verify weight self-normalization.
> - **Output:** `conformal/weights.py`.
> - **Pitfall:** clipping biases the risk estimate — account for it; log the clip fraction.

> **Step 1.6.4 — Stage B: weighted RCPS calibration.** *(M)*
> - **Do:** fold weights into the RCPS upper bound (variance inflates the CI — that's expected and is the lever OOD routing pulls).
> - **Output:** weighted path in `conformal/rcps.py`.

> **Step 1.6.5 — Stage B test (gate).** *(S)*
> - **Do:** synthetic *known* covariate shift: (i) true weights restore risk ≤ α on target; (ii) no weights visibly violate it.
> - **Done when:** both appear. **This is the in-vitro headline result — if it's absent here, it won't appear on real data.**

> **Step 1.6.6 — Stage C: OOD route.** *(M)*
> - **Do:** plug Mahalanobis/kNN `o(x)`; route `o(x)>t_ood` or `w>w_max` to abstain.
> - **Output:** `ood/detector.py` + routing in the inference path.

> **Step 1.6.7 — Stage C test (gate).** *(S)*
> - **Do:** inject synthetic far-OOD; confirm they're abstained at the target rate and accepted-risk stays ≤ α *with* routing, degrades *without*.
> - **Done when:** both appear.

> **Step 1.6.8 — Code review.** *(S)*
> - **Do:** run the conformal code-review prompt (Appendix D-5); fix leakage/off-by-one/clip-bias issues.
> - **Done when:** all synthetic gates still pass after fixes.

## 1.7 Run experiments, baselines, ablations

> **Step 1.7.1 — Smoke run.** *(S)* End-to-end on the smoke set; confirm logging.
> **Step 1.7.2 — CAMELYON17, all six methods.** *(M)* Sweep `α`; repeat over many cal/test splits.
> **Step 1.7.3 — Headline figure first.** *(S)* Realized risk vs target `α` under shift — yours ≤ α, naïve above. *If unconvincing, iterate the method, not the plot.*
> **Step 1.7.4 — CheXpert↔MIMIC, all methods.** *(M)* Repeat the protocol.
> **Step 1.7.5 — Ablations.** *(M)* Run the 1.2.5 set; tabulate from logs.
> **Step 1.7.6 — Marginal-validity histograms.** *(S)* Aggregate the multi-split realized risks into the headline distribution per method/dataset.
> - **Done when:** every table/figure regenerates from logged runs by a single script.

## 1.8 Validity & red-team analysis

> **Step 1.8.1 — Confound check.** *(S)* Report coverage with risk; show validity isn't bought by near-total abstention.
> **Step 1.8.2 — Weight-error stress.** *(S)* Degrade the discriminator; show graceful degradation.
> **Step 1.8.3 — Split sensitivity.** *(S)* Vary `m`; find the small-`m` breakdown.
> **Step 1.8.4 — Find a failure.** *(S)* Locate and *report* a shift where it fails — a characterized boundary is a feature.
> **Step 1.8.5 — Results red-team.** *(S)* Run the results red-team prompt (Appendix D-6); rewrite the central claim to match the evidence.
> - **Done when:** you can rule out the top 3 alternative explanations with specific plots.

## 1.9 Write the paper

> **Step 1.9.1 — Pick venue + template.** *(S)* MICCAI/MLHC/CHIL/ML4H or a NeurIPS/ICML workshop (fast); TMLR/MedIA for the fuller version. Match length/format now.
> **Step 1.9.2 — Outline to the claim.** *(S)* Intro (gap + one-sentence contribution + assumption) → related work (your matrix) → method (1.1) → experiments (1.7) → limitations (1.8) → reproducibility. Lead with the structural finding ("guarantee xor shift-robustness — nobody has both").
> **Step 1.9.3 — Figures.** *(M)* Headline risk-vs-α-under-shift; risk–coverage curves; ablation table; OOD-routing illustration.
> **Step 1.9.4 — Draft prose.** *(L)* Every guarantee sentence carries its assumption; quote TRUECAM's "cannot be guaranteed in deployment" as motivation/contrast.
> **Step 1.9.5 — Reviewer-2 pass.** *(S)* Run the Reviewer-2 prompt (Appendix D-7); apply the top-3 revisions.
> - **Done when:** the draft survives the three-reviewer simulation without an unaddressed major weakness.

## 1.10 Reproducibility release

> **Step 1.10.1 — Clean the repo.** *(S)* Pinned env, configs, seeds, one script per figure/table.
> **Step 1.10.2 — Data-access README.** *(S)* You can't redistribute CheXpert/MIMIC — ship splits + code + exact access steps.
> **Step 1.10.3 — Archive.** *(S)* Zenodo DOI; link it in the paper.

> ### ● GATE 1 → 2
> - [ ] All synthetic gates (1.6.2/1.6.5/1.6.7) pass.
> - [ ] Headline figure: risk ≤ α under real shift where a baseline violates it, on **both** datasets.
> - [ ] Limitations honestly bound the claim.
> - [ ] Paper submitted + reproducible release archived.
> - **Do not start Phase 2 before the MVN is written up — an unwritten result is an unfinished result.**

---

# PHASE 2 — The flagship version

**Goal:** turn the MVN into the definitive paper that closes (or near-closes) the full gap — all conditions, multiple modalities, stronger theory, journal-grade. Each sub-section is an independent increment; do them roughly in order because each de-risks the next. **Effort:** a second project of comparable size — ship increments, don't fan out. Rule: each increment = one new capability + its own validity test + its own ablation showing it helps.

## 2.1 Add label / prevalence shift

> **Step 2.1.1 — Estimate target label proportions without target labels.** *(M)*
> - **Do:** implement BBSE (Lipton et al. 2018) or an EM/confusion-matrix correction to estimate `p_T(y)`.
> - **Output:** `conformal/label_shift.py`.
> - **Done when:** on synthetic prevalence shift, estimated proportions track the truth.

> **Step 2.1.2 — Fold label-shift importance weights into calibration.** *(M)*
> - **Do:** combine label-shift weights with covariate weights into the weighted RCPS calibration.
> - **Done when:** the calibration accepts combined weights.
> - **Pitfall:** double-counting when both shifts are present — derive the combined weight, don't multiply naïvely.

> **Step 2.1.3 — Validity test (gate).** *(S)*
> - **Do:** synthetic *known* prevalence shift: covariate-only weighting fails; label-aware weighting restores risk ≤ α.
> - **Done when:** both appear. This is the biggest theory step up from the MVN — budget time.

## 2.2 Integrate OOD into the risk budget (principled) + base-model/UQ ablation

> **Step 2.2.1 — Allocate one error budget across acceptance + OOD leakage.** *(L)*
> - **Goal:** answer TRUECAM's weakness by making OOD part of the *certified* pipeline, not an empirical patch.
> - **Do:** split total `α` into (a) accepted in-distribution error and (b) residual risk from imperfect OOD routing; derive the end-to-end bound that accounts for detector error.
> - **Output:** the integrated-budget theorem in the method note.
> - **Done when:** the bound holds when the OOD detector is deliberately imperfect, where the bolt-on version silently leaks risk.

> **Step 2.2.2 — Spectral-norm features as the principled OOD substrate.** *(M)*
> - **Do:** standardize the OOD detectors (Mahalanobis/kNN) on spectrally-normalized features across all base models; this is the bi-Lipschitz argument made load-bearing.
> - **Done when:** OOD AUROC + integrated-budget validity reported on spectral-norm features.

> **Step 2.2.3 — Base-model / UQ ablation: {standard, DDU, SNGP, deep ensemble}.** *(L)*
> - **Goal:** show the guarantee layer is backbone-agnostic and quantify the *efficiency* gain from better uncertainty.
> - **Do:** run your full method with each base model holding the conformal/RCPS layer fixed; compare risk–coverage efficiency (abstention at fixed guaranteed risk) and OOD AUROC. Expect: deep ensembles ≈ quality ceiling (K× cost); **DDU** a strong, cheaper single-pass option (often the cleaner fit than SNGP); **SNGP** competitive but more tuning; **standard** the attribution baseline.
> - **Output:** the base-model ablation table + an "efficiency frontier" plot.
> - **Done when:** you can state which backbone sets the frontier and confirm validity is unchanged across all (only efficiency moves).
> - **Pitfall:** reporting a backbone's *calibration* as if it were the guarantee — validity comes from the conformal layer; the backbone only moves efficiency/OOD.

## 2.2a Temporal / continual shift track (ACI)

> **Step 2.2a.1 — Build a temporal benchmark.** *(M)*
> - **Do:** order a site's data in time (or use acquisition-date metadata) to create a drifting stream with (delayed) labels.
> - **Done when:** a streaming loader yields time-ordered batches.

> **Step 2.2a.2 — Run the selective predictor with ACI.** *(M)*
> - **Do:** make the threshold time-adaptive — update `α_t`/risk level from realized error as labels arrive; keep the selection rule and OOD route unchanged. Use **DtACI** or strongly-adaptive online CP (Bhatnagar et al., ICML 2023) for abrupt drift.
> - **Output:** `conformal/aci.py`.

> **Step 2.2a.3 — Validity test (gate) + honest caveat.** *(S)*
> - **Do:** inject a change-point; show ACI restores long-run risk where a fixed weighted threshold drifts out of control. State plainly that this guarantee is **long-run/amortized**, not per-case finite-sample — a complementary promise to the RCPS bound, not the same one.
> - **Done when:** the change-point plot + caveat are written.

## 2.3 Multi-modality sweep (condition 5′)

> **Step 2.3.1 — ECG: PTB-XL cross-cohort.** *(L)*
> - **Do:** 1D backbone (ResNet1D/transformer); define a cross-cohort/site split; reuse the conformal/OOD core unchanged.
> - **Done when:** the risk-vs-α-under-shift story replicates (or you report where it doesn't).

> **Step 2.3.2 — Retina: EyePACS ↔ APTOS.** *(L)*
> - **Do:** fundus backbone; cross-population shift; same core.
> - **Done when:** replicated/characterized.

> **Step 2.3.3 — Pathology #2: MIDOG scanner shift.** *(M)*
> - **Do:** complements CAMELYON17 with a different scanner-shift; same core.
> - **Done when:** replicated/characterized.

> **Step 2.3.4 — Per-modality breakdown.** *(S)*
> - **Do:** a table of where the guarantee holds vs strains across modalities/shift types — a highlight, not an embarrassment.
> - **Done when:** the breakdown is written.

## 2.4 Strengthen the theory

> **Step 2.4.1 — Promote to a formal statement.** *(L)*
> - **Do:** state the finite-sample bound for selective + weighted RCPS under bounded density ratio with far-OOD routed out; characterize approximation error as a function of weight-estimation + detector error; give the label-shift corollary.
> - **Done when:** the statement has explicit constants/assumptions.

> **Step 2.4.2 — External theory check.** *(S)*
> - **Do:** have a conformal-literate person (advisor/collaborator/the theorist prompt) verify the composition.
> - **Done when:** no unaddressed objection remains.

## 2.5 Fairness, subgroup validity, clinical framing

> **Step 2.5.1 — Subgroup audit.** *(M)*
> - **Do:** report per-subgroup coverage *and* risk (sex, age, site); selective prediction can *magnify* disparities (Jones et al. 2021).
> - **Done when:** subgroup tables exist + disparities are discussed.

> **Step 2.5.2 — Deployment/workflow framing.** *(S)*
> - **Do:** describe who sees abstentions, what a "defer" triggers, the abstention-budget cost in practice.
> - **Done when:** a workflow paragraph is written — this is what makes it clinically credible.

## 2.6 Journal write-up

> **Step 2.6.1 — Pick venue.** *(S)* Medical Image Analysis / IEEE TMI / npj Digital Medicine / TMLR by theory-vs-clinical balance.
> **Step 2.6.2 — Assemble the superset.** *(L)* MVN + label shift + integrated OOD + multi-modality + strengthened theory + subgroup audit + an honest "what remains open."
> **Step 2.6.3 — Flagship gap-coverage audit.** *(S)* Run the flagship audit prompt (Appendix D-8); write the honest "what remains open" paragraph from it.

> ### ● GATE 2 → done
> - [ ] Covariate + label shift handled; OOD inside the certified budget.
> - [ ] Replicated across ≥3 modalities/shift types (with honest per-modality breakdown).
> - [ ] Formal statement with explicit assumptions, externally checked.
> - [ ] Subgroup-validity audit included.
> - [ ] Journal manuscript submitted + public reproducible release.

---

# Cross-cutting: decision gates & go/no-go

Re-read at every gate; they prevent the two classic failures (building on a broken core; polishing a result that won't survive review).

1. **After 0.3:** still open? If a single work closed the exact MVN, pivot *before* coding.
2. **After 1.6.5 (synthetic shift):** does weighted RCPS restore risk ≤ α in vitro? If not, the real result cannot exist — fix the method.
3. **After 1.7.3 (headline figure):** beats naïve conformal under real shift on *both* datasets? If only one, diagnose before writing.
4. **After 1.8 (red-team):** is the win a confound (abstention inflation, weak baseline, leakage)? If yes, it isn't real yet.
5. **Before each Phase 2 increment:** is the MVN submitted? Don't fan out before the core is banked.

---

# Risk register (failure modes → mitigations)

| Risk | Symptom | Mitigation | Step |
|---|---|---|---|
| Self-deception on the guarantee | Risk controlled only via near-total abstention | Report coverage with risk; pre-register success | 1.2.6, 1.8.1 |
| Data leakage | Suspiciously perfect target coverage | Assert disjoint cal/weight/OOD/test indices; code review | 1.4.1, 1.6.8 |
| Weight-estimation error | Holds in vitro, fails on real shift | Clip weights; route high-weight tail to OOD-abstain; stress-test | 1.6.3, 1.8.2 |
| Base model conflated with method | "Maintained risk" is just a worse target model abstaining | Freeze base model; report accepted-accuracy + coverage; ablate | 1.5.3, 2.2.3 |
| Backbone confound | Gains attributed to SNGP not the contribution | MVN uses standard backbone; SNGP/DDU only as Phase-2 ablation | 1.5.2, 2.2.3 |
| Gap closes mid-project | A new preprint does it all | Re-verify at 0.3 and before submission; keep pivot ready | 0.3.1 |
| Scope creep into flagship early | Many half-finished modalities, nothing submitted | Enforce GATE 1→2 | Gate 1→2 |
| Dataset access latency | Stuck waiting on MIMIC/CheXpert | Start credentialing in Phase 0; lead with CAMELYON17 | 1.4.2 |
| Overclaiming in writing | "Valid under any shift" | Every guarantee sentence carries its assumption; Reviewer-2 | 0.1.7, 1.9.5 |

---

# Appendix A — Curated reading list

*Confirm exact venue/year as you go; pull bleeding-edge competitor links (SCRC, SCoRE, TRUECAM, Conformal Triage, "Pitfalls," 2026 triage-audit) from your audit report.*

**Conformal & risk control**
- Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* (2005) — foundations.
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* — the on-ramp.
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* (ICLR 2024) — expectation control; one backbone for (2).
- Bates, Angelopoulos, Lei, Malik, Jordan — *Distribution-Free, Risk-Controlling Prediction Sets* (JACM 2021) — RCPS; PAC `(α,δ)`. **The clinical-safety backbone.**
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test* (2021) — non-monotone / multi-threshold risk control.

**Conformal under shift**
- Tibshirani, Foygel Barber, Candès, Ramdas — *Conformal Prediction Under Covariate Shift* (NeurIPS 2019) — weighted conformal.
- Gibbs & Candès — *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021) — online (Phase 2 temporal track).
- Gibbs & Candès — *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts* — DtACI.
- Bhatnagar, Wang, Xiong, Bai — *Improved Online Conformal Prediction via Strongly Adaptive Online Learning* (ICML 2023).

**Selective prediction**
- El-Yaniv & Wiener — *On the Foundations of Noise-Free Selective Classification* (JMLR 2010).
- Geifman & El-Yaniv — *Selective Classification for DNNs* (NeurIPS 2017); *SelectiveNet* (ICML 2019).
- Jones et al. — *Selective Classification Can Magnify Disparities Across Groups* (ICLR 2021) — Phase 2 fairness.

**OOD detection & deterministic UQ**
- Lee et al. — Mahalanobis OOD (NeurIPS 2018); Liu et al. — *Energy-based OOD* (NeurIPS 2020); Sun et al. — *kNN OOD* (ICML 2022).
- Liu et al. — *SNGP: Simple and Principled UQ via Distance Awareness* (NeurIPS 2020).
- Mukhoti et al. — *Deep Deterministic Uncertainty (DDU)*; van Amersfoort et al. — *DUQ* (2020).
- Ovadia et al. — *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift* (NeurIPS 2019) — deep ensembles as ceiling.

**Label/prevalence shift (Phase 2)**
- Lipton, Wang, Smola — *Detecting and Correcting for Label Shift (BBSE)* (ICML 2018).

**Datasets & calibration**
- Koh et al. — *WILDS* (ICML 2021); Bandi et al. — *CAMELYON17* (IEEE TMI 2018).
- Irvin et al. — *CheXpert* (AAAI 2019); Johnson et al. — *MIMIC-CXR* (Sci Data 2019).
- Wagner et al. — *PTB-XL* (Sci Data 2020); EyePACS / APTOS (Kaggle DR); MIDOG challenge.
- Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017) — ECE.

**Competitors (re-verify links from your audit):** SCRC, SCoRE, TRUECAM, Conformal Triage, "Pitfalls of Conformal Predictions for Medical Image Classification," 2026 triage-under-prevalence audit, Liang & Sun (TMLR 2024).

---

# Appendix B — Dataset access cheat-sheet

| Dataset | Modality | Shift | Access | Friction | Phase |
|---|---|---|---|---|---|
| CAMELYON17-WILDS | Histopathology | 5 hospitals/scanners | `wilds` package, open | **Low — start here** | 1 |
| CheXpert | Chest X-ray | Institution/population | Stanford AIMI registration | Medium | 1 |
| MIMIC-CXR | Chest X-ray | Institution/population | PhysioNet credentialed + CITI | Medium–High (start early) | 1 |
| PTB-XL | 12-lead ECG | Cross-cohort | PhysioNet, open | Low–Medium | 2 |
| EyePACS / APTOS | Retinal fundus | Population | Kaggle | Low | 2 |
| MIDOG | Histopathology | Scanner | Challenge site | Low–Medium | 2 |

> Begin MIMIC-CXR / CheXpert credentialing during Phase 0 — human approval is the long pole. CAMELYON17 unblocks Phase 1 immediately.

---

# Appendix C — Glossary (one line each)

- **Exchangeability** — joint distribution invariant to ordering; split-conformal's core assumption. Every failure here is an exchangeability violation.
- **Marginal coverage** — guarantee holds on average over the random cal+test draw, not conditional on a given `x`.
- **Conformal risk control (CRC)** — calibrate a threshold so the *expected* bounded loss ≤ α; expectation is over the calibration draw too.
- **RCPS (risk-controlling prediction sets)** — PAC risk control: `P(R ≤ α) ≥ 1−δ` for your one realized calibration set/model — the stronger, more conservative clinical statement.
- **Learn-then-Test (LTT)** — risk control for non-monotone losses and multiple thresholds via multiple hypothesis testing.
- **Selective risk / coverage** — error among answered / fraction answered; traded off via the risk–coverage curve.
- **Weighted conformal** — reweight calibration by density ratio `p_T/p_S` to transport the guarantee under covariate shift.
- **Density ratio `w(x)`** — `p_T(x)/p_S(x)`; estimated from a source-vs-target discriminator; unreliable in the far tail → route to abstain.
- **Adaptive conformal inference (ACI)** — online α adjustment for long-run coverage without exchangeability; needs label feedback, gives long-run (not per-case) control.
- **Covariate vs. label shift** — `p(x)` changes (label fixed) vs. `p(y)` changes; different importance weights.
- **Far-OOD** — inputs unlike training; extreme tail of shift where weighting fails; the abstain-route is both safety and math.
- **Distance-aware UQ (SNGP/DDU/DUQ)** — single-pass models whose uncertainty grows with feature-distance from training data; improves selection + OOD efficiency, not the guarantee.
- **Spectral normalization** — constrains layer Lipschitz constants → approx. bi-Lipschitz features → feature-distance OOD becomes meaningful (resists feature collapse).

---

# Appendix D — All paste-in prompts (full text)

**D-1 · Concept tutor / steelman** — §0.1
```
You are a conformal-prediction and selective-classification expert. I'm building a
selective medical classifier with a distribution-free risk guarantee on accepted cases
that must survive covariate shift, with far-OOD inputs routed to abstain.

Teach me, precisely and with the exact theorem assumptions, how these compose:
(1) split conformal prediction and the risk-control layer — conformal risk control (CRC)
    vs RCPS vs Learn-then-Test (what each guarantees: expectation vs PAC (alpha,delta) vs
    multi-threshold control; under what assumption; finite-sample vs asymptotic);
(2) selective prediction and WHY selection breaks exchangeability, and how selective
    risk control restores a guarantee on the accepted subset;
(3) weighted conformal (covariate shift) and adaptive conformal inference (online shift)
    — what each needs, and exactly where each fails;
(4) how far-OOD routing interacts with weighted conformal (density-ratio blow-up).

For each: the precise quantity guaranteed, the assumption it rests on, and the single most
common way practitioners violate that assumption without noticing. Then give me three
conceptual exam questions and grade my answers when I reply. Cite real papers with links;
flag anything you're unsure exists.
```

**D-2 · Falsification re-check** — §0.3
```
ROLE: Meticulous research-gap auditor for trustworthy ML in healthcare. Use live
literature search. Default to skepticism; actively try to prove the gap below is ALREADY
SOLVED. Today's date is your current date.

GAP: A selective-prediction method for medical models that simultaneously (1) abstains/
defers, (2) gives a distribution-free guarantee on the error rate of ACCEPTED cases,
(3) keeps that guarantee approximately valid under REAL clinical distribution shift,
(4) routes far-OOD inputs to abstain, evaluated on public medical shift benchmarks.

CLOSED only if ONE work demonstrably does ALL of: abstains; distribution-free error/
coverage guarantee on accepted cases; that guarantee holds under a real (not synthetic)
medical shift; far-OOD routed to abstain; evaluated on >=1 public medical shift dataset.

FOCUS: 2025-06 to today. Prioritize MICCAI, MLHC, CHIL, ML4H, NeurIPS/ICML/ICLR, TMLR,
Medical Image Analysis, npj Digital Medicine. Check explicitly: SCRC, SCoRE, TRUECAM,
Conformal Triage, weighted/adaptive conformal in medicine, new "selective conformal under
shift" work.

OUTPUT: Verdict OPEN / PARTIALLY ADDRESSED / CLOSED + one-line justification; an evidence
table (Title|Year|Venue|Link|which of (1)-(4) satisfied|what's missing); the closest
competitor and exactly what it leaves unsolved; the precise residual contribution still
novel. Cite every claim with a working link + date; separate peer-reviewed from preprints;
if a search fails, say so rather than guessing.
```

**D-3 · Method red-team / proof check** — §1.1
```
You are a conformal-prediction theorist and Reviewer 2. Below is my proposed method and
its claimed guarantee [PASTE setup, target guarantee, algorithm, assumptions].

Do four things, harshly:
1. State the exact conditions under which R_T^accept <= alpha holds. Finite-sample or
   asymptotic? Marginal over what randomness?
2. Find every place selection, covariate shift, or far-OOD routing breaks exchangeability
   that I have NOT neutralized. Give a concrete counterexample for each hole.
3. Pressure-test the density-ratio weighting: what happens as weights grow / overlap
   fails, and does OOD routing actually bound the residual shift? Prove or refute.
4. Name the single weakest claim a reviewer attacks first, and the smallest change that
   makes it defensible. Cite the exact theorems I rely on (weighted conformal,
   CRC/RCPS/LTT, selective conformal), with links.
Do not be reassuring. If the guarantee is only "approximate," characterize the error.
```

**D-4 · Experiment-design critic** — §1.2
```
You are an experienced ML-for-health reviewer. Critique this experimental plan for a
selective-prediction-under-shift paper [PASTE datasets, shift type, metrics, baselines,
ablations, success criterion].

Tell me: (a) which baseline is missing such that a reviewer says "but did you compare to
X?"; (b) whether my metrics demonstrate the (2)+(3) guarantee or just correlate with it;
(c) the right way to show a finite-sample, calibration-draw-marginal guarantee empirically
(how many splits, what plot, what statistic); (d) the most likely confound (e.g. the base
model just being worse on target) and how to rule it out; (e) whether two datasets is
enough, or which single third one buys the most credibility for the least work.
```

**D-5 · Conformal code review** — §1.6
```
You are a careful research engineer who has shipped conformal-prediction code. Review this
implementation of selective + weighted conformal risk control + OOD routing [PASTE code].
Check specifically: off-by-one in the conformal quantile / CRC threshold; whether the
finite-sample / concentration correction is correct; data leakage between calibration,
weight-fitting, OOD-fitting, and test; whether weights are normalized correctly and
clipping biases the risk estimate; whether selection is applied before or after
calibration (should the threshold be calibrated on the accepted region?). Then propose the
minimal synthetic-data unit tests that PROVE each guarantee, with the exact expected
numbers so I know if a test passes.
```

**D-6 · Results red-team** — §1.8
```
You are a skeptical senior co-author. Here are my results [PASTE tables/figures/setup].
Try to explain my positive result WITHOUT my method being correct: confounds, abstention
inflation, leakage, lucky splits, weak baselines, cherry-picked alpha, dataset quirks. For
each alternative explanation, give the exact additional analysis or plot that rules it out.
Then tell me which claims the data actually supports vs. which I'm overstating, and rewrite
my central claim sentence to match the evidence.
```

**D-7 · Reviewer 2 simulation** — §1.9
```
You are three reviewers for [VENUE]: a conformal theorist, a clinical-ML practitioner, and
a skeptical generalist. Here is my paper draft [PASTE]. Each writes: summary, strengths,
weaknesses, questions, score, and the one change that would most raise it. The theorist
checks the guarantee's assumptions and whether the empirics demonstrate it. The clinician
asks whether the abstention behavior is usable in a real workflow and whether the shifts
are clinically realistic. The generalist hunts for overclaiming and missing baselines. End
with the top 3 revisions, ranked by impact.
```

**D-8 · Flagship gap-coverage audit** — §2.6
```
You are auditing whether my flagship results close the original gap. The gap requires a
single method that: (1) abstains; (2) gives a distribution-free guarantee on accepted
cases; (3) holds under real covariate AND label shift; (4) routes far-OOD to abstain with
that step inside the guarantee; (5') is validated across multiple diverse public medical
shift datasets. Here are my results [PASTE]. For each of (1)-(5'), state SATISFIED /
PARTIAL / NOT, with exact evidence and the exact residual gap. Then write the most honest
one-paragraph "what remains open" statement for my discussion section.
```

---

# Appendix E — Repo tree reference

```
gap3-selective-shift/
├── README.md                 # overview + exact data-access steps
├── requirements.txt / env    # pinned deps + lockfile
├── configs/                  # one YAML per experiment (Hydra)
├── data/
│   ├── camelyon17.py         # WILDS loaders + split manifest
│   ├── cxr.py                # CheXpert/MIMIC loaders + harmonization
│   └── splits/               # frozen index files (cal/test disjoint)
├── models/                   # backbones, training, spectral-norm option
├── ood/
│   └── detector.py           # Mahalanobis / kNN / energy on features
├── conformal/
│   ├── selective.py          # selection + accepted-region calibration
│   ├── rcps.py               # RCPS (Hoeffding-Bentkus / WSR) + weighted
│   ├── crc.py                # CRC comparison backbone
│   ├── weights.py            # domain discriminator + clipped density ratio
│   ├── label_shift.py        # BBSE / EM (Phase 2)
│   └── aci.py                # adaptive conformal (Phase 2 temporal)
├── experiments/              # run scripts (method × dataset × seed)
├── analysis/
│   ├── metrics.py            # risk, coverage, AURC, ECE, OOD AUROC
│   ├── figures.py            # headline + risk-coverage + ablations
│   └── competitor_matrix.csv
├── tests/                    # synthetic-data guarantee tests (pytest)
├── docs/                     # positioning_memo, design, preregistration, method_note
└── paper/                    # manuscript + figures
```

---

# Appendix F — Notation & key formulas (quick reference)

**Notation:** `P_S,P_T` source/target · `f` frozen base model · `ŷ(x)` prediction · `u(x)` uncertainty · `g(x)=1[u(x)≤τ]` selection · `ℓ∈[0,1]` loss · `w(x)=p_T/p_S` density ratio · `o(x)` OOD score · `τ,λ,t_ood` thresholds · `α` target risk · `δ` RCPS confidence · `m` target sample size.

- **Split-conformal quantile:** `q̂ = ⌈(n+1)(1−α)⌉`-th smallest calibration score; set `{y : s(x,y) ≤ q̂}`. Marginal coverage ≥ `1−α` under exchangeability.
- **CRC:** choose `λ̂` so `E[ℓ(λ̂)] ≤ α` (monotone bounded loss). Expectation includes the calibration draw.
- **RCPS:** `λ̂ = inf{λ : UCB_δ(R(λ′)) < α ∀ λ′≥λ}` ⇒ `P(R(λ̂) ≤ α) ≥ 1−δ`. UCB via Hoeffding–Bentkus or Waudby-Smith–Ramdas.
- **Weighted conformal:** reweight calibration point `i` by `w(x_i)/Σ_j w(x_j)` (+ test-point term); transports coverage under covariate shift with known/estimated `w`.
- **Density ratio from a discriminator:** `w(x) ∝ d(x)/(1−d(x))`, `d=` P(target | x); clip at `w_max`.
- **Selective risk / coverage:** `risk = E[ℓ | g=1]`, `coverage = P(g=1)`; the risk–coverage curve / AURC summarizes the tradeoff.
- **Your target guarantee:** `E_{P_T}[ ℓ(Y,ŷ(X)) | g(X)=1, not far-OOD ] ≤ α`.

---

# Appendix G — Environment & command cheat-sheet

```bash
# env (pip path; use --break-system-packages only if system Python forces it)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision wilds numpy scipy scikit-learn pandas matplotlib wandb pytest

# data (confirm current WILDS API)
python -c "from wilds import get_dataset; get_dataset('camelyon17', download=True)"

# run an experiment from a config
python -m experiments.run --config configs/camelyon17_ours.yaml seed=0

# the synthetic guarantee gates (must pass before real data)
pytest tests/ -k "exchangeable or weighted_shift or far_ood" -v

# regenerate every figure/table from logged runs
python -m analysis.figures --from-logs runs/ --out paper/figures/
```

> **Compute note (single GPU):** cache features once (1.5.4) so conformal calibration reruns in seconds. Deep-ensemble baselines = K sequential trainings — schedule overnight. Everything else is light.

---

# Appendix H — Math & notation decoder (plain meanings)

*Keep this open while you read. Every symbol the playbook uses, in plain words.*

**Core symbols**
- `x`, `y` — an input (e.g., an image) and its true label.
- `f`, `f(x)` — the model, and the scores it outputs for input `x`.
- `ŷ(x)` ("y-hat") — the model's predicted label for `x`.
- `x̂` (a "hat" on anything) — an *estimate* computed from data, not the true value.
- `P_S`, `P_T` — the data pattern (distribution) at the **s**ource vs **t**arget hospital.

**Probability & averages**
- `P(A)` — probability of `A` (a number 0–1).
- `P(A | B)` — probability of `A` **given** `B` is true ("among the `B` cases").
- `E[Z]` — expected value = long-run **average** of `Z`.
- `E[ℓ | g=1]` — average loss **among answered cases** (`g=1` means "answered").

**The knobs and budgets**
- `α` (alpha) — the error rate you allow (`0.05` = 5%).
- `δ` (delta) — the small chance the *promise* fails; RCPS confidence is `1−δ`.
- `ℓ(y, ŷ)` — the loss: how wrong a prediction is (here scaled to `[0,1]`).
- `u(x)` — uncertainty score; high = the model is unsure.
- `g(x) = 1[u(x) ≤ τ]` — selection rule: answer (1) if uncertainty is below `τ`, else abstain.
- `τ` (tau) — the selection/abstention threshold.
- `λ` (lambda) — the risk-control threshold tuned on calibration data.
- `t_ood` — the OOD cutoff; above it, route the input to abstain.
- `m` — how many (unlabeled) target points you have to calibrate the shift correction.

**Shift & OOD**
- `w(x)` — "density ratio" `p_T(x)/p_S(x)`: how *target-like* input `x` is; used as a weight.
- `o(x)` — OOD score: how *unlike training* input `x` is (a feature-space distance).

**Math operators**
- `≤` / `≥` — at most / at least.
- `∝` — "proportional to" (equal up to a constant multiplier).
- `⌈·⌉` — ceiling: round up to the next whole number.
- `inf` / `sup` — the smallest / largest value meeting a condition.
- `1[condition]` — indicator: equals 1 if the condition holds, else 0.
- `Σ` — sum.

**Concepts in two words**
- **quantile / percentile** — value below which a given fraction of data falls.
- **exchangeable** (≈ i.i.d. here) — data points are "drawn the same way," so order doesn't matter; the resemblance assumption conformal needs.
- **distribution-free** — the promise holds regardless of the data's shape.
- **finite-sample vs asymptotic** — holds with the data you actually have vs only as data → ∞.
- **PAC** — "probably approximately correct": with probability ≥ `1−δ`, error ≤ `α` (RCPS's promise).
- **coverage** — the fraction of cases the guarantee correctly covers.
- **risk–coverage curve / AURC** — error vs how much you answer; AURC summarizes it (lower = better).
- **ECE** — expected calibration error: how honest the model's raw probabilities are.
- **nonconformity score** — how "strange"/surprising a point is vs the model's expectation.
- **bi-Lipschitz** — a mapping that doesn't squash distances, so "far" stays far (why spectral norm helps OOD).

---

*End of playbook. Work top to bottom; honor the gates; run each phase's Appendix-D prompt against your actual artifact before calling the phase done.*
