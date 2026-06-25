# Positioning memo

*One-sentence contribution, and the scope decision behind it. See [method_note.md](method_note.md) §3.2–§3.7 for the full derivation, [flagship-playbook.md](../flagship-playbook.md) for the build plan.*

## Contribution (one sentence)

**Calibrated selective prediction with a finite-sample risk certificate that holds under prevalence (label) shift between clinical sites without further assumptions, and extends — conditional on a small labeled target-site slice — to the realistic case where imaging acquisition (covariate) shift is also present, with the residual shift-model error reported rather than assumed away.**

## Scope decision (ratified)

Two candidate headlines were on the table:

1. **Label-shift-primary (adopted).** The certified, assumption-minimal claim is for prevalence shift alone (`p(y)` changes, `p(x∣y)` invariant) — needs only the anti-causal assumption BBSE already requires, no labeled target data, `ε_w = ε_M = 0`. Combined covariate+label shift is a secondary, conditionally-certified extension: identified via **Factorizable Joint Shift (FJS)** (Tasche 2022/2026) plus a small labeled target slice (`D_tar^lab`, ~50–200 examples) that anchors FJS's otherwise-unidentifiable scale constant and yields a *measured* `ε_M` instead of an assumed one.
2. **Combined-shift-co-primary (rejected).** Would have required committing up front to a narrow `f`-invariant restriction of the shift model — more restrictive than the FJS + labeled-slice route, for no corresponding gain in certificate strength.

(1) was ratified: it is the only fully assumption-clean version, and the combined case still appears in the paper with an honest, named residual rather than being dropped.

## On the realism of "`p(x∣y)` invariant" (important honesty note)

This assumption — that disease appearance doesn't change between sites — is **not realistic across arbitrary hospital pairs** (different scanners, protocols, and case-mix-correlated demographics all violate it to some degree). It is treated as a **testable regime, not a blanket premise**:

- Report a diagnostic (domain-discriminator AUROC / feature-space distance between source and target) per hospital pair, showing how close that pair actually is to "no covariate shift."
- The pure-label-shift certificate is the validated headline on pairs where the diagnostic confirms covariate shift is small (e.g., same-scanner, different-catchment site pairs within CAMELYON17-WILDS).
- On pairs where the diagnostic shows real covariate shift, the FJS + labeled-slice secondary certificate is the one actually being claimed, and is reported as such — not silently substituted for the label-shift headline.

This makes the headline a measured, checkable claim rather than an assumed-true one.
