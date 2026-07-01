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
Later stages (label_shift, budget, crc, ltt) remain stubs.
"""
