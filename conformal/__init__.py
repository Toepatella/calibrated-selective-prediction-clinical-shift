"""Conformal / risk-control calibration layer for the selective-prediction pipeline.

Stage A (exchangeable foundation) is implemented:
  * ``rcps``      -- WSR / HB upper confidence bounds and the RCPS inf-rule, plus
                     the weighted (Hájek) risk path and its WSR betting interval
                     (``hajek_risk``, ``kish_n_eff``, ``wsr_ucb_weighted`` /
                     ``wsr_lcb_weighted``) added at Stage B.
  * ``selective`` -- selection gate, selective risk, accepted-region RCPS threshold.
  * ``folds``     -- group-id fold-disjointness discipline (method_note §1.7),
                     wired in from Stage A onward.
Stage B (covariate-shift weights) is implemented:
  * ``weights``   -- cross-fitted domain-discriminator covariate weight
                     ``ŵ_cov = clip((d/(1−d))·ĉ, 0, w_max)`` (method_note §1.5),
                     weight-distribution / AUROC diagnostics, and the doubly-robust
                     (AIPW) covariate correction.
Stage C (label / prevalence-shift weights) is implemented:
  * ``label_shift`` -- BBSE ``ŵ_lab = Ĉ_S⁻¹ q̂_T`` (simplex projection + floor/ceiling),
                     MLLS+BCTS and MAPLS prevalence estimation, the per-``x`` corrector
                     ``Ẑ(x)`` and the combine identity ``ŵ = ŵ_lab·ŵ_cov/Ẑ`` (NOT the
                     product), plus the ``κ(Ĉ_S)`` conditioning and ``q̂_T`` vs ``Ĉ_S p̂_T``
                     consistency diagnostics (method_note §1.5, §3.3, §3.4 step 2-3).
Later stages (budget, crc, ltt) remain stubs.
"""
