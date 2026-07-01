"""Conformal / risk-control calibration layer for the selective-prediction pipeline.

Stage A (exchangeable foundation) is implemented:
  * ``rcps``      -- WSR / HB upper confidence bounds and the RCPS inf-rule.
  * ``selective`` -- selection gate, selective risk, accepted-region RCPS threshold.
Later stages (weights, label_shift, budget, crc, ltt) remain stubs.
"""
