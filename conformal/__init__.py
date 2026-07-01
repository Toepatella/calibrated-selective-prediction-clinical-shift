"""Conformal / risk-control calibration layer for the selective-prediction pipeline.

Stage A (exchangeable foundation) is implemented:
  * ``rcps``      -- WSR / HB upper confidence bounds and the RCPS inf-rule.
  * ``selective`` -- selection gate, selective risk, accepted-region RCPS threshold.
  * ``folds``     -- group-id fold-disjointness discipline (method_note §1.7),
                     wired in from Stage A onward.
Later stages (weights, label_shift, budget, crc, ltt) remain stubs.
"""
