# Research-Gap Audit — Gap 3: Calibrated Selective Prediction under Clinical Distribution Shift

**Audit date:** 2026-06-20 · **Method:** live literature search (PubMed/Europe PMC, arXiv, Semantic Scholar, Google Scholar surfacing, Nature/Springer, conference proceedings) with direct inspection of methods/claims (not abstracts alone) for the closest competitors. **Default posture:** skeptical — I actively tried to prove the gap is already solved.

---

## 1. Verdict

**PARTIALLY ADDRESSED — essentially OPEN.**

No single work satisfies all five falsification criteria. The strongest existing integration — **TRUECAM** (preprint) — combines abstention, conformal risk control, and far-OOD routing on public pathology data and nearly closes the gap **within one disease/modality (NSCLC histopathology, TCGA↔CPTAC)**, but (i) its coverage validity under shift is **empirical and achieved by detecting-and-removing OOD inputs** — the authors explicitly state coverage "cannot be guaranteed in deployment" — not a distribution-free guarantee that provably survives covariate shift, and (ii) it is **single-domain**, not validated across multiple, diverse public medical shift benchmarks. Every other candidate is strictly weaker on at least one axis: the works with a real distribution-free guarantee lack far-OOD routing and multi-dataset shift validation; the work with shift+far-OOD handling (Liang & Sun) has **no** distribution-free guarantee.

---

## 2. Falsification scorecard

The gap is **CLOSED** only if one work demonstrably does all of: **(1)** abstains/defers; **(2)** distribution-free coverage/error guarantee on accepted cases; **(3)** that guarantee/calibration holds under a **real** (not synthetic) medical shift; **(4)** routes far-OOD inputs to abstain; **(5)** is evaluated on ≥1 public medical shift benchmark. The gap statement additionally requires **(5′) multiple** diverse public medical shift datasets.

| Work | (1) Defer | (2) DF guarantee | (3) Holds under real shift | (4) Far-OOD | (5) Public med. shift | Net |
|---|---|---|---|---|---|---|
| **TRUECAM** (preprint) | ✅ | ✅ CRC | ⚠️ empirical, via OOD removal; authors concede no deployment guarantee | ✅ SNGP | ✅ but single domain (TCGA↔CPTAC) | **~4 / 5 — closest** |
| Conformal Triage (Angelopoulos, preprint) | ✅ | ✅ PPV/NPV | ⚠️ via local **recalibration** (needs target labels) | ❌ | ⚠️ private MGB site | 3 / 5 |
| Kim 2026, Sci Rep (peer-reviewed) | ✅ | ✅ split/Mondrian/weighted CP | ⚠️ weighted CP degrades gracefully; single temporal split | ❌ | ⚠️ 1 EHR set (PhysioNet 2019) | 3 / 5 |
| Audited CP (preprint) | ⚠️ sets, not native abstain | ✅ marginal coverage | ✅ CAMELYON17 (needs target labels) | ❌ covariate, not far-OOD | ✅ CAMELYON17 | 3 / 5 |
| Liang & Sun, TMLR 2024 | ✅ | ❌ score-based, **no** guarantee | ✅ covariate+label shift | ✅ label-shift/OOD | ❌ ImageNet/Amazon | 3 / 5 (no guarantee) |
| SCRC (preprint) | ✅ | ✅ CRC + PAC | ❌ assumes exchangeability | ❌ | ❌ general | 2 / 5 |
| SCoRE (preprint) | ✅ | ✅ e-values | ❌ assumes exchangeability | ❌ | ❌ general | 2 / 5 |

⚠️ = partial. **No row is all-green.** Per the rubric (some works hit 3–4 of 5 → PARTIALLY ADDRESSED), the verdict is PARTIALLY ADDRESSED, leaning OPEN.

---

## 3. Evidence table (most relevant works)

Legend — **(a)** abstain/defer · **(b)** distribution-free guarantee on accepted cases · **(c)** guarantee/calibration approx. valid under **real clinical** shift · **(d)** far-OOD → abstain. ✅ satisfied · ⚠️ partial · ❌ not satisfied. *PR = peer-reviewed; PP = preprint.*

| # | Title | Year | Venue | Link | a | b | c | d | What's missing |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **TRUECAM**: Trust in NSCLC Diagnosis with a Conformalized Uncertainty-Aware Framework (WSI) | 2024–25 | medRxiv / arXiv (PP) | [arXiv:2501.00053](https://arxiv.org/abs/2501.00053) · [medRxiv](https://www.medrxiv.org/content/10.1101/2024.12.27.24319715v1.full) · [code](https://github.com/iamownt/TRUECAM) | ✅ | ✅ | ⚠️ | ✅ | Shift-validity is empirical & via OOD **removal** (not a guarantee that survives covariate shift); **single** disease/cohort-pair; not yet PR |
| 2 | Conformal Triage for Medical Imaging AI Deployment | 2024 | medRxiv (PP) | [medRxiv:2024.02.09.24302543](https://www.medrxiv.org/content/10.1101/2024.02.09.24302543v1) | ✅ | ✅ | ⚠️ | ❌ | No far-OOD routing; shift handled by **recalibration** on local labels; private single-site (MGB), not public multi-benchmark |
| 3 | Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution shift (Kim) | 2026 | Sci Reports (PR) | [10.1038/s41598-026-40637-w](https://www.nature.com/articles/s41598-026-40637-w) | ✅ | ✅ | ⚠️ | ❌ | No far-OOD; one EHR task (sepsis), **temporal** split only; weighted CP degrades, not maintained; not imaging/multi-dataset |
| 4 | Audited Conformal Prediction under Unknown Distribution Shift | 2026 | arXiv (PP) | [arXiv:2606.14909](https://arxiv.org/abs/2606.14909) | ⚠️ | ✅ | ✅ | ❌ | Produces sets (no native deferral); needs labeled target data; covariate shift only, no far-OOD; single benchmark |
| 5 | Selective Classification Under Distribution Shifts (Liang & Sun) | 2024 | TMLR (PR) | [arXiv:2405.05160](https://arxiv.org/abs/2405.05160) | ✅ | ❌ | ✅ | ✅ | **No** distribution-free guarantee (confidence-score method); non-medical data (ImageNet/Amazon-WILDS) |
| 6 | Selective Conformal Risk Control (SCRC) | 2025–26 | arXiv (PP) | [arXiv:2512.12844](https://arxiv.org/abs/2512.12844) | ✅ | ✅ | ❌ | ❌ | Assumes exchangeability; no shift, no far-OOD; two **general** public datasets (not medical shift) |
| 7 | Conformal Selective Prediction with General Risk Control (SCoRE) | 2026 | arXiv (PP) | [arXiv:2603.24704](https://arxiv.org/abs/2603.24704) | ✅ | ✅ | ❌ | ❌ | e-value risk control valid only under exchangeability; no shift/OOD/medical validation |
| 8 | Pitfalls of Conformal Predictions for Medical Image Classification | 2025 | arXiv / UNSURE-MICCAI (workshop) | [arXiv:2506.18162](https://arxiv.org/abs/2506.18162) | — | — | — | — | *Critique*: shows conformal coverage is **unreliable under medical input/label shift** and "should not be used for selective classification" → motivates the gap |
| 9 | A Deployment Audit of Release-Side Risk in Conformal Triage under Prevalence Shift | 2026 | arXiv (PP) | [arXiv:2605.20956](https://arxiv.org/abs/2605.20956) | — | — | — | — | *Critique*: conformal-triage guarantees can break under prevalence shift → motivates the gap |
| 10 | Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster) | 2024 | ICLR (PR) | [arXiv:2208.02814](https://arxiv.org/abs/2208.02814) | ❌ | ✅ | ⚠️ | ❌ | Foundational guarantee; not selective-by-default, not medical, not far-OOD (has a weighted-shift extension only) |
| 11 | Conformal Prediction Under Covariate Shift (Tibshirani, Barber, Candès, Ramdas) | 2019 | NeurIPS (PR) | [arXiv:1904.06019](https://arxiv.org/abs/1904.06019) | ❌ | ✅ | ✅ | ❌ | Foundational weighted CP; needs known/estimable likelihood ratio; not medical/selective/far-OOD |
| 12 | Adaptive Conformal Inference Under Distribution Shift (Gibbs & Candès) | 2021 | NeurIPS (PR) | [arXiv:2106.00170](https://arxiv.org/abs/2106.00170) | ❌ | ✅ | ✅ | ❌ | Foundational online shift method; long-run coverage; not medical/selective/far-OOD |
| 13 | SelectiveNet: Integrated Reject Option (Geifman & El-Yaniv) | 2019 | ICML (PR) | [arXiv:1901.09192](https://arxiv.org/abs/1901.09192) | ✅ | ❌ | ❌ | ❌ | Abstains but no distribution-free guarantee; ID assumption; no far-OOD |
| 14 | Deep Gamblers: Learning to Abstain with Portfolio Theory (Liu et al.) | 2019 | NeurIPS (PR) | [arXiv:1907.00208](https://arxiv.org/abs/1907.00208) | ✅ | ❌ | ❌ | ❌ | Same family as SelectiveNet/softmax-response — abstention only, no guarantee/shift/OOD |
| 15 | Energy-based Out-of-Distribution Detection (Liu et al.) | 2020 | NeurIPS (PR) | [arXiv:2010.03759](https://arxiv.org/abs/2010.03759) | ❌ | ❌ | ❌ | ✅ | Pure far-OOD detector (representative of Mahalanobis/energy/KNN) — no selective coverage guarantee, no risk control |

---

## 4. Named prior art — explicit mapping to (a)–(d)

**Selective Conformal Risk Control (SCRC)** — [arXiv:2512.12844](https://arxiv.org/abs/2512.12844) (PP). (a) ✅ two-stage select-then-control; (b) ✅ SCRC-T exact finite-sample, SCRC-I PAC; (c) ❌ guarantees rest on **exchangeability** (SCRC-T even pools calibration+test); (d) ❌ none. Evaluated on two general public datasets, **not** medical shift. *Has the guarantee, lacks shift + far-OOD + medical.*

**"Pitfalls of Conformal Predictions for Medical Image Classification" (2025)** — [arXiv:2506.18162](https://arxiv.org/abs/2506.18162) (workshop/PP; lineage to UNSURE-MICCAI). A critique, not a method. It demonstrates conformal coverage **fails under dermatology/histopathology input and label shift**, that re-calibration data is needed, and that conformal "should not be used for selective classification." Directly **supports the gap being open.**

**Weighted conformal (Tibshirani et al. 2019)** — [arXiv:1904.06019](https://arxiv.org/abs/1904.06019) (PR) — and **Adaptive Conformal Inference (Gibbs & Candès 2021)** — [arXiv:2106.00170](https://arxiv.org/abs/2106.00170) (PR). (a) ❌; (b) ✅; (c) ✅ for **covariate**/online shift (weighted exchangeability / long-run coverage); (d) ❌. These are the *machinery* for (c) but are not selective, not far-OOD, and not validated on medical shift benchmarks.

**SelectiveNet ([1901.09192](https://arxiv.org/abs/1901.09192)), Deep Gamblers ([1907.00208](https://arxiv.org/abs/1907.00208)), softmax-response / SGR (Geifman & El-Yaniv 2017, [1705.08500](https://arxiv.org/abs/1705.08500)).** (a) ✅; (b) ❌ — risk control is in-distribution/PAC under the **i.i.d.** assumption, not distribution-free under shift; (c) ❌; (d) ❌. Abstention only.

**Deep ensembles ([1612.01474](https://arxiv.org/abs/1612.01474)), MC-dropout (Gal & Ghahramani 2016, [1506.02142](https://arxiv.org/abs/1506.02142)), evidential DL (Sensoy et al. 2018, [1806.01768](https://arxiv.org/abs/1806.01768)).** (a) ⚠️ thresholded abstention possible; (b) ❌ no distribution-free guarantee; (c) ❌ — calibration is known to **degrade under shift** (Ovadia et al., NeurIPS 2019, [1906.02530](https://arxiv.org/abs/1906.02530)); (d) ⚠️ heuristic. Useful score sources, not guarantees.

**OOD detection — Mahalanobis (Lee et al. 2018, [1807.03888](https://arxiv.org/abs/1807.03888)), Energy ([2010.03759](https://arxiv.org/abs/2010.03759)), KNN (Sun et al. 2022, [2204.06507](https://arxiv.org/abs/2204.06507)).** (a) ⚠️ flag/route only; (b) ❌; (c) ❌; (d) ✅. These cover **only** component (d). Note a documented caveat: best OOD-detection performance does **not** imply best abstention/error performance in clinical tasks.

---

## 5. Closest competitor(s) and exactly what they leave unsolved

**TRUECAM (the single closest work).** Three components: SNGP (uncertainty + out-of-scope detection), elimination of ambiguous tiles, and conformal risk control (CRC). It abstains/defers to pathologists, gives a distribution-free coverage target via CRC, and routes far-OOD (e.g., the TCGA-OOD set of non-lung cancers, OOD AUROC ≈ 0.95). It is tested on a **real** cross-cohort shift (TCGA 941 WSIs ↔ CPTAC 1,306 WSIs). **What it leaves unsolved:**
- **Guarantee under shift is empirical, not maintained.** TRUECAM "restores exchangeability" by *detecting and removing* OOD inputs, then leans on CRC to inflate sets for undetected OOD. The paper states coverage "**cannot be guaranteed in deployment, where unexpected OOD data may extend beyond the threshold**." So (c) holds only as far as the OOD detector does — it is not a distribution-free guarantee that provably survives genuine *in-scope covariate shift*.
- **Single domain.** One disease, one modality, one cohort-pair. The gap's "**multiple real, public medical shift datasets**" (imaging + EHR/ECG, multiple shift types) is unmet.
- **Preprint**, not yet peer-reviewed.

**Conformal Triage (Angelopoulos et al.)** — first distribution-free selective triage in medicine; abstains, guarantees PPV/NPV. But it transfers to a new population by **re-calibrating on local labeled data**, has **no far-OOD routing**, and is demonstrated on a private single site — not public, multi-dataset, or far-OOD-aware.

**Kim 2026 (Sci Rep, peer-reviewed)** — abstains + split/Mondrian/importance-weighted conformal; coverage degrades only slightly under a **temporal** sepsis shift. But it is one EHR task, no imaging, **no far-OOD module**, and coverage is empirically graceful rather than guaranteed under shift.

**Liang & Sun (TMLR 2024)** — the only candidate that natively handles abstention + covariate shift + label-shift/OOD together, but provides **no distribution-free guarantee** (it ships better confidence scores, evaluated by risk–coverage curves) and uses non-medical data.

---

## 6. The precise residual contribution that remains genuinely novel

A selective-prediction method for clinical models that **simultaneously**:

1. **Guarantees the error/risk of accepted cases distribution-free in a way that provably remains valid (or is provably corrected) under covariate/label shift** — e.g., **weighted or adaptive conformal *risk* control composed with the selection rule**, explicitly accounting for the fact that *selection itself breaks exchangeability* (selective issuance is non-exchangeable) *on top of* the shift-induced break. This is stronger than TRUECAM's "remove-OOD-to-restore-exchangeability + empirical coverage."
2. **Routes far-OOD inputs to abstain via a calibrated OOD test with its own error control**, integrated into the **same** risk budget as the selective coverage guarantee (rather than a separate heuristic detector).
3. **Holds approximately across *multiple, heterogeneous* real public medical shift benchmarks** — different modalities and shift types — with empirical evidence that **both** the accepted-case guarantee **and** the abstention calibration survive each real shift.

The novel core is the **joint, single-umbrella treatment of three exchangeability-breaking forces — selection, covariate/label shift, and far-OOD — under one distribution-free risk-control guarantee, validated broadly on real medical shift.** No current work does all three at once with a maintained guarantee and multi-domain medical validation.

---

## 7. Candidate public datasets / benchmarks to test it

- **Histopathology (scanner/hospital shift):** CAMELYON17-WILDS; MIDOG 2021/2022 (scanner domains); TCGA↔CPTAC (as TRUECAM); PANDA.
- **Chest X-ray (cross-institution):** CheXpert ↔ MIMIC-CXR ↔ NIH ChestX-ray14 ↔ PadChest.
- **ECG (cross-cohort):** PTB-XL ↔ Chapman-Shaoxing ↔ Georgia ↔ CPSC2018 (PhysioNet 2020/2021).
- **Diabetic retinopathy:** EyePACS ↔ APTOS-2019 ↔ Messidor-2.
- **Dermatology (incl. skin-tone shift):** ISIC ↔ HAM10000 ↔ Fitzpatrick17k.
- **EHR (cross-hospital / temporal):** MIMIC-IV ↔ eICU; PhysioNet Sepsis Challenge 2019 (temporal).
- **Far-OOD probes:** inject off-modality / wrong-anatomy / non-medical images, or a TCGA-OOD-style "other-disease" set, to test (d) and the abstention-calibration guarantee jointly.

A convincing study would calibrate on one source per benchmark, deploy on the shifted target, and report **accepted-case risk vs. nominal**, **abstention/deferral rate**, and **far-OOD recall** — showing the guarantee approximately holds on every benchmark.

---

## 8. Confidence and caveats

- **Confidence: HIGH** that no single **peer-reviewed** work closes the gap as specified, and **MODERATE-HIGH** on the overall landscape. The pattern is robust and was reached by an active falsification attempt: guarantee-bearing methods lack far-OOD + multi-shift validation; the shift+OOD method (Liang & Sun) lacks the guarantee; the one near-complete integration (TRUECAM) is single-domain, empirical-on-shift, and a preprint.
- **Fast-moving area / preprints.** TRUECAM, Conformal Triage, SCRC, SCoRE, Audited CP, and the Deployment Audit are **preprints** (labeled above); peer-reviewed anchors are Kim 2026 (Sci Rep), Liang & Sun (TMLR), and the foundational CP/selective/OOD papers. Treat preprint claims as unverified by review.
- **Definitional tension.** The 5-point falsification test requires only ≥1 public medical shift benchmark; the gap's headline requires **multiple**. Under the looser reading, TRUECAM is borderline "closed for NSCLC pathology specifically." Under the gap's actual wording (multiple datasets + a *maintained* guarantee under shift), it is not — hence **PARTIALLY ADDRESSED**.
- **Tool note.** The web-search tool returned transient "unavailable" errors on one batch mid-audit; I re-ran and recovered those queries. I did not exhaustively crawl the very latest (mid-2026) MICCAI/ML4H/CHIL camera-ready proceedings, so a newly posted paper could shift the picture — re-check before staking a contribution claim.
- **Depth.** Methods/results were inspected directly (not abstract-only) for TRUECAM, Kim 2026, and Liang & Sun; SCRC, SCoRE, Audited CP, and Conformal Triage were assessed from abstracts plus author summaries and should be confirmed against their full methods before final positioning.
