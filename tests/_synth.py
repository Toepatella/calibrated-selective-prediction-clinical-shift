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
from scipy.stats import norm

__all__ = ["PolyErrorModel", "GaussianCovariateShift"]


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


class GaussianCovariateShift:
    """Synthetic covariate shift with an INVARIANT ``p(y|x)`` (Gate-B ground truth).

    Fully synthetic (no clinical data). A scalar covariate ``x`` is drawn from a
    Gaussian whose MEAN differs by domain while the conditional error model
    ``r(x) = P(l = 1 | x)`` is IDENTICAL across domains -- covariate shift by
    construction (``build_gates.md §5``: "explicit marginals ``p_S(x)``,
    ``p_T(x)`` with invariant ``p(y|x)``"). Everything a Gate-B check needs is then
    known in closed form or to arbitrary quadrature precision:

      * the **oracle covariate ratio** ``w_cov*(x) = p_T(x)/p_S(x)`` (closed form),
      * the true accepted-region risk ``E_d[r(x) | x <= z]`` and coverage
        ``P_d(x <= z)`` under each domain ``d`` (Gaussian quadrature),

    so the discriminator weight can be checked against the oracle and the Hájek
    weighted risk estimate against the known target accepted risk.

    Design
    ------
    ``x ~ N(mu_d, 1)`` with ``mu_S = 0`` and ``mu_T = mu_tar`` (default
    ``mu_tar < 0`` so the target concentrates in the low-``x`` / low-error tail:
    the target is *easier* in the accepted region, hence the covariate-corrected
    Hájek risk lands BELOW the naive source-level estimate -- the "weighted < naive"
    signed win of ``build_gates.md §5``). Because ``x`` is Gaussian and the shift is
    a pure location shift, ``log w_cov*(x) = mu_tar·x - mu_tar²/2`` is LINEAR in
    ``x``; a logistic domain discriminator is therefore well-specified and recovers
    the oracle ratio at large sample (the ``ŵ_cov`` tracking gate).

      * uncertainty  ``u(x) = Phi(x)`` (monotone), so ``{u <= tau} = {x <= z}`` with
        ``z = Phi^{-1}(tau)``;
      * error model  ``r(x) = clip(base + slope·sigmoid(k·x), 0, 1)`` -- a fixed
        (domain-invariant) function of ``x``, monotone non-decreasing, so higher
        ``x`` (= higher ``u``) is riskier, matching the Gate-A selection direction.
    """

    def __init__(self, mu_tar=-0.6, base=0.05, slope=0.6, k=2.0, mu_src=0.0):
        self.mu_src = float(mu_src)
        self.mu_tar = float(mu_tar)
        self.base = float(base)
        self.slope = float(slope)
        self.k = float(k)
        # r(x) must lie in [0, 1] and be monotone non-decreasing (asserted on a grid).
        grid = np.linspace(-8.0, 8.0, 200_001)
        rg = self._r(grid)
        if not np.all((rg >= -1e-12) & (rg <= 1.0 + 1e-12)):
            raise ValueError("r(x) must lie in [0, 1]")
        if not np.all(np.diff(rg) >= -1e-12):
            raise ValueError("r(x) must be monotone non-decreasing in x")

    # -- error model (domain-invariant) ------------------------------------
    def _r(self, x):
        x = np.asarray(x, dtype=float)
        return self.base + self.slope / (1.0 + np.exp(-self.k * x))

    def r(self, x):
        """Conditional error probability r(x) = P(l = 1 | x), clipped to [0, 1]."""
        return np.clip(self._r(x), 0.0, 1.0)

    def u(self, x):
        """Uncertainty score u(x) = Phi(x) in (0, 1), monotone increasing in x."""
        return norm.cdf(np.asarray(x, dtype=float))

    # -- oracle covariate ratio (closed form) ------------------------------
    def oracle_w_cov(self, x):
        """Oracle density ratio w_cov*(x) = p_T(x)/p_S(x) for the location shift.

        For ``N(mu_tar, 1) / N(mu_src, 1)`` this is ``exp((mu_tar-mu_src)·x -
        (mu_tar²-mu_src²)/2)`` -- log-linear in ``x`` (why the logistic
        discriminator is well-specified and recovers it).
        """
        x = np.asarray(x, dtype=float)
        return np.exp((self.mu_tar - self.mu_src) * x
                      - 0.5 * (self.mu_tar ** 2 - self.mu_src ** 2))

    # -- analytic ground truth (Gaussian quadrature) -----------------------
    def z_of_tau(self, tau):
        """Accept-region boundary z with {u <= tau} = {x <= z}; z = Phi^{-1}(tau)."""
        return float(norm.ppf(float(tau)))

    def _mu(self, domain):
        if domain == "S":
            return self.mu_src
        if domain == "T":
            return self.mu_tar
        raise ValueError("domain must be 'S' or 'T'")

    def coverage(self, tau, domain):
        """True coverage P_d(x <= z) under the named domain (z from ``z_of_tau``)."""
        z = self.z_of_tau(tau)
        return float(norm.cdf(z - self._mu(domain)))

    def accepted_risk(self, tau, domain):
        """True accepted-region risk E_d[r(x) | x <= z] to quadrature precision.

        ``build_gates.md §4`` explicitly sanctions "arbitrary precision by
        large-sample integration". NaN when the accept region has ~zero mass.
        """
        z = self.z_of_tau(tau)
        mu = self._mu(domain)
        lo = mu - 12.0
        if z <= lo:
            return float("nan")
        xs = np.linspace(lo, z, 400_001)
        pdf = norm.pdf(xs - mu)
        denom = np.trapezoid(pdf, xs)
        if denom <= 0.0:
            return float("nan")
        num = np.trapezoid(self.r(xs) * pdf, xs)
        return float(num / denom)

    # -- sampling ----------------------------------------------------------
    def sample(self, n, rng, domain):
        """Draw ``n`` i.i.d. points ``(x, u, losses)`` from the named domain.

        ``x ~ N(mu_d, 1)``; ``u = Phi(x)``; ``losses ~ Bernoulli(r(x))`` in
        ``{0, 1}`` (0-1 loss of ``method_note §1.4``). ``rng`` is a
        ``numpy.random.Generator`` (``build_gates.md §3`` seeding convention).
        """
        n = int(n)
        mu = self._mu(domain)
        x = rng.normal(mu, 1.0, size=n)
        u = self.u(x)
        losses = (rng.uniform(size=n) < self.r(x)).astype(float)
        return x, u, losses
