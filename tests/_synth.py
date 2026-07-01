"""Synthetic ground-truth generators for the Gate-A exchangeable RCPS gate.

Everything here is FULLY SYNTHETIC (no clinical data); the Gate-A checks run with
no dataset credentialing (``prereg §6.8``). The construction gives a scalar
uncertainty ``u`` with a KNOWN marginal and a KNOWN monotone conditional error
model ``r(u) = P(l = 1 | u)``, so both the true accepted risk
``R(tau) = E[l | u <= tau]`` and true coverage ``P(u <= tau)`` are available in
closed form. This is the ground truth every Gate-A check compares against
(``build_gates.md §4``: "Synthetic construction (the known ground truth)").

This is a *wiring / sanity* fixture for RCPS **under exchangeability only**. It
proves nothing about deployment or real clinical data, claims no guarantee, and
introduces no method (``build_gates.md §4`` honesty rails).

Design
------
``u ~ Uniform(0, 1)`` and ``r(u) = sum_k c_k u^k`` a polynomial that is monotone
non-decreasing and lies in ``[0, 1]`` on ``[0, 1]``. Because ``u`` is uniform the
ground truth is elementary and exact (no quadrature needed):

    coverage(tau)      = P(u <= tau)        = tau
    accepted_risk(tau) = E[r(u) | u <= tau] = (1/tau) * int_0^tau sum_k c_k v^k dv
                                            = sum_k c_k tau^k / (k + 1)
    marginal_risk      = E[r(u)]            = sum_k c_k / (k + 1)

The linear case ``c = [a, b]`` gives ``R(tau) = a + b*tau/2`` and marginal
``a + b/2``; a quadratic term lets the gate sweep a small *family* of monotone
error models (varying base rate and slope/curvature) so a pass cannot be an
artifact of one hand-picked construction (``build_gates.md §4`` open question).
"""

import numpy as np

__all__ = ["PolyErrorModel"]


class PolyErrorModel:
    """A synthetic exchangeable population with a known monotone error model.

    Parameters
    ----------
    coeffs : sequence of float
        Polynomial coefficients ``c_0, c_1, ...`` of ``r(u) = sum_k c_k u^k``.
        Must yield ``r(u) in [0, 1]`` and ``r`` monotone non-decreasing on
        ``[0, 1]`` (both are asserted on a fine grid at construction).
    """

    def __init__(self, coeffs):
        self.c = np.asarray(coeffs, dtype=float)
        if self.c.ndim != 1 or self.c.size == 0:
            raise ValueError("coeffs must be a non-empty 1-D sequence")
        grid = np.linspace(0.0, 1.0, 100_001)
        rg = self._r(grid)
        if not np.all((rg >= -1e-12) & (rg <= 1.0 + 1e-12)):
            raise ValueError("r(u) must lie in [0, 1] on u in [0, 1]")
        if not np.all(np.diff(rg) >= -1e-12):
            raise ValueError("r(u) must be monotone non-decreasing on [0, 1]")

    # -- error model -------------------------------------------------------
    def _r(self, u):
        u = np.asarray(u, dtype=float)
        powers = u[..., None] ** np.arange(self.c.size)   # (..., K)
        return powers @ self.c

    def r(self, u):
        """Conditional error probability r(u) = P(l = 1 | u), clipped to [0, 1]."""
        return np.clip(self._r(u), 0.0, 1.0)

    # -- analytic ground truth (u ~ Uniform(0, 1)) -------------------------
    def coverage(self, tau):
        """True coverage P(u <= tau) = tau on the exchangeable population."""
        return float(np.clip(tau, 0.0, 1.0))

    def accepted_risk(self, tau):
        """True accepted-region risk E[l | u <= tau] = sum_k c_k tau^k/(k+1).

        NaN when ``tau <= 0`` (nothing accepted), matching ``selective_risk``.
        """
        tau = float(tau)
        if tau <= 0.0:
            return float("nan")
        t = min(tau, 1.0)
        k = np.arange(self.c.size)
        return float(np.sum(self.c * t ** k / (k + 1)))

    @property
    def marginal_risk(self):
        """True marginal (full-population) risk E[l] = sum_k c_k/(k+1)."""
        k = np.arange(self.c.size)
        return float(np.sum(self.c / (k + 1)))

    # -- sampling ----------------------------------------------------------
    def sample(self, n, rng):
        """Draw ``n`` i.i.d. points ``(u, losses)`` from the population.

        ``u ~ Uniform(0, 1)``; ``losses ~ Bernoulli(r(u))`` in ``{0, 1}`` (the
        default 0-1 loss of ``method_note §1.4``). ``rng`` is a
        ``numpy.random.Generator`` (``np.random.default_rng(seed)``), matching the
        phase-0 / ``build_gates.md §3`` seeding convention.
        """
        n = int(n)
        u = rng.uniform(0.0, 1.0, size=n)
        losses = (rng.uniform(size=n) < self.r(u)).astype(float)
        return u, losses
