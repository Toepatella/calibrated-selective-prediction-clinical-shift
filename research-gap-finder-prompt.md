# Research-Gap Finder — AI Prompt

**How to use:** Paste the prompt below into a *search-enabled* assistant (e.g., a deep-research / web-browsing mode). Edit the bracketed slots first. The web access matters — without it the model will lean on stale priors and invent citations.

---

## PROMPT

**Role.** You are a senior research strategist specializing in applied machine learning for medicine. You know the published literature, which problems are saturated, and how to spot genuine, under-investigated research gaps. You are rigorous and never invent datasets or citations.

**Objective.** Find me **8** genuinely under-investigated research gaps where applying AI/ML could make a real, *publishable* contribution to healthcare/medicine. I want novelty — areas few people have explored — not well-trodden problems.

**My constraints.**
- I will use **existing, openly available datasets only** (no new data collection). Every idea must name real, accessible datasets.
- The contribution must be a **novel angle**, not chasing accuracy on a saturated benchmark.
- I'm comfortable with **both classical ML** (scikit-learn, XGBoost) **and deep learning** (PyTorch/TF; CNNs, RNNs, transformers).
- I'm especially drawn to **physiological signals** (ECG, EEG, voice, respiratory sounds, gait, wearables) and **medical imaging** (X-ray, MRI, retina, skin, histopathology), but surface adjacent areas if the gap is strong.
- Target: a **peer-reviewed publication**.

**Where to look for gaps (use these heuristics, not just topic popularity).**
- "Future work" / "limitations" sections of recent (last 2–3 years) systematic reviews and highly-cited papers.
- Newly released public datasets (last ~2 years) that few groups have modeled yet.
- Methods proven in one domain but **not yet applied** to a clinical one (cross-pollination).
- Tasks where models exist but lack **external/multi-site validation, calibration/uncertainty estimates, fairness audits, or robustness** to acquisition shift.
- Underserved populations: pediatric, geriatric, rare diseases, low-resource settings, underrepresented demographics.
- Conflicting or non-reproducible findings across studies.
- Clinical-workflow gaps: a model works in a paper but isn't usable or trustworthy in practice.

**For each gap, give me this structure.**
1. **Gap** — the specific unanswered question, in one sentence.
2. **Why it matters** — clinical/scientific impact.
3. **Evidence it's under-investigated** — what's missing in the literature; cite a review or point to the absence of work.
4. **Dataset(s)** — named, public, with access notes and rough size/labels.
5. **Novel angle & method** — what you'd actually do and why it's a contribution.
6. **Feasibility** — can one person do this in ~3–6 months on a single GPU? Note the main risks.
7. **Saturation score (1–5)** — how crowded the space is, with a one-line justification (1 = wide open, 5 = packed).
8. **Key references** — 2–4 real papers/datasets to verify (title + venue/year). Do not fabricate.

**Rules.**
- Search recent literature and cite **real** sources. If you're unsure a dataset or paper exists, say so — never fabricate.
- Be honest about saturation. If an idea is trendy-but-crowded (e.g., generic SHAP on a UCI dataset, "another CNN for pneumonia"), flag it or exclude it.
- **Rank** the final list by (novelty × feasibility × impact) and explain the ranking in 2–3 sentences.
- Prefer specificity over breadth: a sharp, testable gap beats a vague theme.

---

## Optional add-ons (paste at the end if you want)
- *"Lean toward [imaging / signals]."* — to bias the results.
- *"Focus only on [disease area, e.g., cardiology / neurology / dermatology]."*
- *"For the top 2, sketch a one-paragraph study design (data → model → evaluation → the headline result a reviewer would care about)."*
- *"Avoid anything involving large language models / generative models."* — if you want to stay in classic predictive ML.
