# Pre-registration

*Locks the headline/secondary claim split decided in [positioning_memo.md](positioning_memo.md), ahead of running the gate ladder (G-A → G-R) in [flagship-playbook.md](../flagship-playbook.md).*

## Primary certified claim

Pure label (prevalence) shift: for hospital pairs where a pre-registered covariate-shift diagnostic (domain-discriminator AUROC between source/target features, threshold fixed before any result is seen) falls below a stated cutoff, the weighted-RCPS selective-risk certificate `R_T^accept ≤ α_acc + α_ood` holds with `ε_w = ε_M = 0`, per [method_note.md](method_note.md) §3.6 label-shift corollary.

## Secondary certified claim

Combined covariate+label shift, on the remaining hospital pairs (diagnostic above the cutoff): certified conditional on (a) Factorizable Joint Shift independence (Tasche 2022/2026) and (b) a small labeled target-site slice `D_tar^lab` (~50–200 examples, drawn before calibration, disjoint from all other folds per method_note.md §3.1). `ε_M` is reported as a measured quantity with its own slice-size uncertainty, not assumed zero.

## What is fixed before results are seen

- The covariate-shift diagnostic and its cutoff (decides which hospital pairs get the primary vs. secondary claim).
- `D_tar^lab` size and disjointness from `D_cal`, `D_tar`, `D_bbse^src`, `D_disc`, `O`.
- The grid `(τ, λ, t_ood)` and the LTT family wrapping it.
- `α = α_acc + α_ood`, `δ = δ_acc + δ_ood` and their split.

## What is explicitly NOT claimed

- A single unconditional guarantee across all hospital pairs regardless of shift type (rejected scope — see positioning memo).
- Combined-shift certification without a labeled target slice (falls back to "no certificate," per method_note.md §3.4 step 3.5 fallback).
- A direct (un-paired) confidence bound on the self-normalized risk ratio — v1 uses the conservative paired UCB/LCB + `n_eff` gate (flagship-playbook.md risk register); the tighter direct bound is an optional, not-yet-derived v2 item.
