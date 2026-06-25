# D-9 · Whole-pathway adversarial stress-test + SOTA literature sweep

> **What this is.** A meta-prompt that takes your *entire* project pathway (not one artifact) plus your
> stated final goal, decomposes it into atomic units, adversarially stress-tests every unit **and the seams
> between them**, then runs a real multi-venue literature sweep to propose SOTA fixes — with working
> citations, peer-reviewed-vs-preprint flags, and an adversarial check that each proposed fix is real and
> actually closes the weakness. Complements D-1…D-8 (which each red-team a single component); this one
> audits the pathway as a *system* and asks whether it reaches the final goal.
>
> **How to run it (ultracode / multi-agent).** In Claude Code, paste the block below prefixed with the word
> `ultracode` so the harness builds it as a multi-agent workflow. Point it at your live plan files. Without
> `ultracode` it still runs as a single thorough pass — just slower and less parallel.
>
> **Reusable.** Edit only the `INPUTS` block as the plan evolves. Defaults below are filled for the current
> flagship build.

---

```
ultracode

ROLE: You are a panel of adversarial research auditors — a conformal-prediction theorist, an ML-for-health
reviewer (Reviewer 2 energy), a research engineer, and a literature-sweep specialist. Your job is NOT to be
reassuring. Your job is to find every way this project pathway is wrong, weak, insufficient, internally
inconsistent, or already-superseded, and then to propose the best available (potentially SOTA) fix for each
real weakness, grounded in the live literature. Default posture: skeptical. Assume the plan is flawed until a
unit survives attack.

=== INPUTS (edit these; everything else is method) ===
PATHWAY (read in full before doing anything):
  - flagship-playbook.md        # the operative plan: §0 contribution → §7 paper, gate ladder G-A…G-R,
                                 #   risk register, Appendices A–H
  - Gap3_audit_calibrated_selective_prediction_under_shift.md   # the gap claim + competitor landscape
  - docs/  (method_note.md, design.md, positioning_memo.md, preregistration.md — whichever exist yet)
  - README.md, repo_links.md, analysis/competitor_matrix.csv (if present)

FINAL GOAL (the thing every unit must serve):
  A journal-grade paper + reproducible release whose headline shows accepted-case risk held at ≤ α under
  REAL covariate AND label/prevalence shift on TWO imaging benchmarks (CAMELYON17-WILDS, CheXpert↔MIMIC-CXR),
  where naïve conformal and a TRUECAM-style empirical pipeline visibly break, with far-OOD error inside the
  certified budget (α = α_acc + α_ood) and a subgroup audit. Venue tier: Medical Image Analysis / IEEE TMI /
  npj Digital Medicine / TMLR.

INVARIANTS (never propose a "fix" that erodes these — flag any that does):
  - Novelty core: conditions (2) distribution-free guarantee + (3) survives covariate AND label shift +
    (4-integrated) far-OOD routing folded into the SAME risk budget. This is the contribution. Protect it.
  - Honesty constraint: every guarantee sentence carries its assumptions; "approximately valid," never
    "valid under arbitrary shift" or "exact under label shift." Do not propose changes that quietly overclaim.
  - Attribution discipline: headline runs on a STANDARD backbone; DDU/SNGP/ensembles are ablations only.

OUT OF SCOPE (do NOT spend effort demanding these; they are deliberate, honestly-stated future work):
  - (5′) multi-modality breadth (ECG/retina/2nd-pathology/derm). Two imaging datasets, deep, is the v1 scope.
    If a critic demands breadth, the only allowed output is "cheapest single extra modality as a rebuttal."

=== PHASE 1 — DECOMPOSE (one pass, then publish the unit list) ===
Read the PATHWAY and enumerate every atomic UNIT as a numbered list, grouped by type:
  (A) Strategic decisions  — single-paper-staged-internally; depth-over-breadth; attribution discipline.
  (B) Method/theory claims — §3.2 integrated target guarantee; §3.3 four breakers+neutralizers;
                             §3.3b combined weight w(x,y)=w_lab·r under anticausal Y→X + the identifiability
                             tension; §3.4 algorithm; §3.5 backbone (weighted RCPS + WSR + LTT);
                             §3.6 finite-sample theorem with explicit constants.
  (C) Assumptions          — exchangeability-after-reweighting; bounded density ratio after OOD routing;
                             BBSE/MLLS identifiability (p(x|y) fixed); r(x,y) benign / y-independent;
                             estimable p_T(y); target sample size m sufficient; loss bounded in [0,1].
  (D) Experimental design  — datasets/splits (§4.1); shift scope (§4.2); metrics (§4.3); baselines (§4.4,
                             incl. TRUECAM-style pipeline + Wang & Qiao 2025); ablations (§4.5);
                             pre-registration (§4.6).
  (E) Build stages + gates — Stage A…S and gates G-A…G-R; the "don't stack past a red gate" rule.
  (F) Meta-artifacts       — the risk register (is it complete and correct?); the repo architecture
                             (Appendix E) for leakage-prone seams; the "Done =" definition itself.
For each unit record: id, one-line statement, which PATHWAY section it lives in, and which part of the
FINAL GOAL or which INVARIANT it serves.

=== PHASE 2 — STRESS-TEST EACH UNIT (parallel critics; harsh) ===
Spawn an independent critic per unit (or per cluster of tightly-coupled units). Each answers, concretely:
  1. RESTATE the unit charitably in one sentence (steelman first).
  2. WHY IT'S BAD / RISKY: the strongest specific objection. Hidden or unstated assumptions. A concrete
     counterexample or failure scenario — not a vibe.
  3. SUFFICIENCY: even if correct, does it actually deliver its part of the FINAL GOAL, or only correlate
     with it? (e.g., does a metric demonstrate the (2)+(3)+(4-integrated) guarantee or merely look consistent
     with it?)
  4. REVIEWER ATTACK SURFACE: the first thing a hostile expert reviewer kills it on.
  5. FEASIBILITY: is the effort tag (S/M/L/XL) realistic, or is this an XL hiding as an M? (Watch §3.6 theory
     with explicit constants, §3.3b identifiability, MLLS+BCTS, the joint-discriminator fallback.)
  6. SUPERSEDED?: is this choice still SOTA, or has the field moved? (Flag for Phase 3, don't answer yet.)
  7. SEVERITY × LIKELIHOOD × TRACTABILITY, each {Critical/High/Med/Low}, with a one-line justification.
Be specific to THIS project. Generic critiques ("add more data", "consider robustness") are failures.

=== PHASE 2b — SEAM / INTERACTION PASS (the part D-1…D-8 miss) ===
Separately attack the CONNECTIONS between units — where individually-fine pieces break when composed:
  - Does WSR (betting UCB) compose VALIDLY with the weighted importance-sampling path AND LTT multiple-testing
    SIMULTANEOUSLY, or does stacking all three invalidate the (α,δ) claim?
  - Does the §3.3b combined weight w(x,y)=w_lab·r actually flow into §3.6's theorem constants, or is the
    theorem silently assuming the easy w_cov·w_lab form the plan elsewhere forbids?
  - Does Stage E's long-run/amortized DtACI promise contradict or get conflated with the finite-sample RCPS
    headline? Are they kept as distinct claims everywhere they appear?
  - Gate ladder: can G-A…G-E pass on synthetic data yet G-R still fail for a reason no synthetic gate probes?
    Name the unprobed failure mode and propose the missing gate.
  - Leakage seams in the repo architecture: cal / weight-fit / BBSE / OOD-fit / test index disjointness;
    feature-caching reuse across splits; α chosen after looking at results; multiple-testing ACROSS the two
    datasets and across α-sweeps.
  - Identifiability chain: §3.3b says no label-shift estimator rescues conditional shift r — does any later
    unit (ablation, baseline, claim) implicitly assume it does?

=== PHASE 2c — COMPLETENESS CRITIC ===
One agent asks only: what is MISSING from the pathway entirely? A failure mode never named, a baseline a
reviewer will demand, an assumption never stated, a metric that should exist, a risk-register row that's
absent or wrong. Output candidate NEW units; feed any material ones back through Phase 2.

=== PHASE 3 — SOTA LITERATURE SWEEP (parallel; one research agent per confirmed High/Critical weakness) ===
For each weakness rated High or Critical, search the LIVE literature broadly and find the best fix. Search
ACROSS these venues by name (do not rely on one index):
  arXiv · IEEE Xplore (esp. IEEE TMI, T-PAMI) · OpenReview (NeurIPS/ICLR/ICML/TMLR) · Semantic Scholar ·
  ACM Digital Library · DBLP · PubMed / Europe PMC · medRxiv / bioRxiv · Google Scholar surfacing ·
  proceedings of MICCAI, MLHC, CHIL, ML4H, AISTATS, UAI, SaTML · PMLR.
For each weakness return:
  - THE CURRENT SOTA approach to this specific sub-problem (name it, cite it).
  - ANY work that already solves the weakness, or shows the plan's current choice is dominated/outdated.
  - THE single best concrete SWAP or PATCH, stated as "replace X in §N with Y because …".
  - For every cited paper: title · authors · year · venue · WORKING link · peer-reviewed vs preprint ·
    and the EXACT quoted result/assumption that makes it relevant (inspect the method, not just the abstract,
    for any paper you call a fix or a competitor).
Recency: prioritize the last ~18–24 months for "is this still SOTA", but anchor every claim on its
foundational source. Bias toward conformal/risk-control, label-shift estimation, combined covariate+label
(generalized label shift) shift, far-OOD with error control, selective prediction, and clinical-shift
evaluation. If a search is inconclusive or returns nothing better than the plan's current choice, SAY SO and
mark the plan's choice "already SOTA — keep."

=== PHASE 4 — ADVERSARIALLY VERIFY EACH PROPOSED FIX (parallel skeptics) ===
For every fix from Phase 3, an independent skeptic checks:
  1. Does it ACTUALLY address the weakness, or just sound adjacent? (distinguish "fixes" from "related work")
  2. What NEW assumptions does it import? Does adopting it break an INVARIANT, the honesty constraint, or the
     attribution discipline? Does it quietly assume away the §3.3b conditional-shift tension?
  3. Is the citation REAL and correctly characterized? If you cannot verify the paper exists or says what's
     claimed, mark UNVERIFIED — never launder a hallucinated citation into a recommendation.
  4. Integration cost (S/M/L/XL) and what it would displace in the gate ladder.
  VERDICT per fix: ADOPT (core) / ADOPT (ablation-only) / NEEDS-MORE-EVIDENCE / REJECT — with the reason.

=== PHASE 5 — SYNTHESIZE (one report) ===
Produce a single structured report:
  1. EXECUTIVE SUMMARY — the ≤7 risks most likely to prevent the FINAL GOAL, ranked, each with its
     one-line fix and verdict.
  2. PER-UNIT FINDINGS TABLE — columns: Unit | §ref | Weakness (1 line) | Severity | SOTA fix | Citation(s) |
     Verdict | Integration cost.
  3. CRITICAL-PATH NARRATIVE — which weaknesses compound, and the dependency ORDER to fix them (keyed to the
     gate ladder: what must be fixed before G-A, before G-C, before G-R).
  4. KILL-SHOTS — the single most likely reason (a) the headline result fails to materialize, and (b) the
     paper is rejected; plus the cheapest defensible mitigation for each.
  5. NOVELTY-CORE INTEGRITY CHECK — confirm explicitly that no ADOPTED change erodes (2)+(3)+(4-integrated);
     list anything that does and why it was rejected/quarantined to ablation.
  6. HONESTY-CONSTRAINT COMPLIANCE — flag any place the plan (or a proposed fix) risks overclaiming, with the
     corrected, assumption-carrying sentence.
  7. RISK-REGISTER DIFF — rows to add / correct / delete in the playbook's risk register.
  8. PRIORITIZED ACTION LIST — concrete, ordered, each keyed to a PATHWAY § and/or gate, marked
     {do-now / before-next-gate / pre-submission / optional-v2}.
  9. HONEST GAPS — where the sweep was thin or inconclusive, what could overturn a finding, what to re-check
     before staking a claim. Surface any DISAGREEMENT between agents rather than averaging it away.

=== RULES OF ENGAGEMENT ===
- Never invent citations, links, or results. "I could not verify this" beats a confident fabrication.
- Separate peer-reviewed from preprint everywhere. Date every empirical claim about the literature.
- Steelman before you attack; a critique that misreads the plan is discarded.
- Distinguish a real fix from merely-related work, and an in-scope weakness from out-of-scope (5′) breadth.
- Respect the INVARIANTS and the honesty constraint — a "SOTA" fix that overclaims is a REJECT, not an ADOPT.
- Prefer the smallest change that makes a unit defensible over a glamorous rewrite.
- Scale depth to the request: a few critics for a spot-check, the full fan-out + loop-until-no-new-findings
  for a pre-submission audit.
```

---

## Lighter single-shot variant (no `ultracode`)

For a quick pass on one section instead of the whole plan, drop the orchestration phases and use:

```
You are an adversarial conformal-prediction + ML-for-health auditor. Read [PASTE one section, e.g. §3.3b or
Stage D]. Steelman it in one sentence, then: (1) the strongest specific objection + a concrete failure case;
(2) hidden assumptions; (3) the first thing Reviewer 2 kills it on; (4) is this choice still SOTA — search
arXiv/IEEE Xplore/OpenReview/Semantic Scholar/PubMed and name the best current alternative with a working
link, PR-vs-preprint flagged, quoting the exact relevant result; (5) the smallest change that makes it
defensible WITHOUT eroding the novelty core (2)+(3)+(4-integrated) or the honesty constraint. No invented
citations; mark anything unverified.
```
