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

__all__ = ["PolyErrorModel", "GaussianCovariateShift", "LabelShiftMixture"]


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


class LabelShiftMixture:
    """Synthetic K-class label-/prevalence-shift ground truth (Gate-C fixture).

    Fully synthetic (no clinical data). Invariant class-conditionals ``p(x|y)``, a
    known source prevalence ``p_S(y)`` and target prevalence ``p_T(y)``, a frozen
    Bayes classifier ``f`` with a well-conditioned confusion matrix, a
    class-structured 0-1 loss, and an OPTIONAL covariate tilt that is either
    ORTHOGONAL to the label signal (a *factorizable* combined shift) or ALONG it
    (an *entangled* shift). Everything a Gate-C check needs is then closed form
    (``build_gates.md §6``: "explicit, invariant class-conditionals ... so label
    shift holds by construction"):

      * oracle label ratio        ``w_lab*(y) = p_T(y)/p_S(y)``,
      * closed-form source posterior ``p_S(y|x)``  ⇒  oracle
        ``Z*(x) = Σ_{y'} w_lab*(y')·p_S(y'|x)``,
      * closed-form marginal densities  ⇒  oracle ``w_cov*(x) = p_T(x)/p_S(x)``,
      * closed-form joint ratio  ``w_joint*(x,y) = p_T(x,y)/p_S(x,y)``,
      * the combine identity ``w*(x,y) = w_lab*(y)·w_cov*(x)/Z*(x)`` is EXACT in the
        factorizable regime (``tilt='x2'``) and BROKEN in the entangled regime
        (``tilt='x1'``) -- the two regimes the entanglement diagnostic must tell
        apart (``method_note §3.3``, §3.4 step 3; ``build_gates.md §6``).

    Geometry
    --------
    ``x = (x1, x2) ∈ ℝ²``. Classes are separated ALONG ``x1`` only:
    ``p(x1|y) = N(means[y], 1)``; ``x2 ~ N(0, 1)`` is class-independent noise. The
    frozen classifier is the Bayes rule under ``p_S`` using ``x1`` alone, so ``x2``
    is a covariate direction ORTHOGONAL to the label signal. A tilt on ``x2`` is
    therefore a factorizable covariate shift (its likelihood ratio ``a(x2)`` is
    class-free, so ``E_{p_S(·|y)}[a]`` is constant in ``y`` and the factorization is
    exact); a tilt on ``x1`` moves the class-discriminative coordinate and entangles
    the two mechanisms (the class-conditional ``p(x1|y)`` itself shifts, violating
    label-shift invariance).

    Because ``x2`` cancels in the Bayes posterior and ``ℓ ⟂ x | y``, the covariate-
    only reweighting (a function of ``x``) cannot recover a *label-marginal* change
    in the class mix -- which is exactly why the label-aware ``Ẑ``-combined path
    lowers the Hájek risk vs covariate-only (``build_gates.md §6`` gate 5;
    ``method_note §3`` label-shift row: "covariate-only reweighting cannot capture a
    label-marginal change").

    Honesty rails (``build_gates.md §6``): a SYNTHETIC wiring fixture with known
    ground truth. It proves nothing about deployment, claims no ``(α,δ)`` or
    distribution-free guarantee for ``w_lab`` / MLLS / the combined weight, and
    introduces no method -- BBSE/MLLS carry only *consistency* under label shift.
    """

    def __init__(self, means, p_s, p_t, rho, tilt=None, theta=0.0):
        self.means = np.asarray(means, dtype=float)
        self.p_s = np.asarray(p_s, dtype=float)
        self.p_t = np.asarray(p_t, dtype=float)
        self.rho = np.asarray(rho, dtype=float)
        self.K = self.means.size
        if not (self.p_s.size == self.p_t.size == self.rho.size == self.K):
            raise ValueError("means, p_s, p_t, rho must share length K")
        for name, p in (("p_s", self.p_s), ("p_t", self.p_t)):
            if abs(p.sum() - 1.0) > 1e-9 or np.any(p < 0):
                raise ValueError(f"{name} must be a probability vector")
        if np.any((self.rho < 0) | (self.rho > 1)):
            raise ValueError("rho must lie in [0, 1]")
        if tilt not in (None, "x1", "x2"):
            raise ValueError("tilt must be None, 'x1', or 'x2'")
        self.tilt = tilt
        self.theta = float(theta)

    # -- frozen Bayes classifier (uses x1 only) ----------------------------
    def logits(self, X):
        """Bayes logits under ``p_S``: ``-(x1-means_k)²/2 + log p_S(k)`` (x2 ignored).

        These are the frozen classifier ``f``'s scores. ``x2`` cancels in the
        softmax (it is class-independent), so the softmax equals the exact source
        posterior ``p_S(y|x)`` -- i.e. ``f`` is perfectly CALIBRATED under ``p_S``.
        Gate-C's BCTS checks deliberately MIS-calibrate these logits and confirm
        BCTS restores calibration.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x1 = X[:, 0]
        return -0.5 * (x1[:, None] - self.means[None, :]) ** 2 + np.log(self.p_s)[None, :]

    def posterior_S(self, X):
        """Closed-form source posterior ``p_S(y|x)`` = softmax of the Bayes logits."""
        lg = self.logits(X)
        lg = lg - lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Frozen hard prediction ``ŷ(x) = argmax_k p_S(y=k|x)``."""
        return np.argmax(self.logits(X), axis=1)

    def uncertainty(self, X):
        """Selection score ``u(x) = 1 - max_k p_S(k|x)`` (higher = less confident)."""
        return 1.0 - self.posterior_S(X).max(axis=1)

    # -- oracle label / covariate / combined weights (closed form) ---------
    def oracle_w_lab(self):
        """Oracle label ratio ``w_lab*(y) = p_T(y)/p_S(y)`` (shape (K,))."""
        return self.p_t / self.p_s

    def oracle_Z(self, X):
        """Oracle double-count corrector ``Z*(x) = Σ_{y'} w_lab*(y')·p_S(y'|x)``."""
        return self.posterior_S(X) @ self.oracle_w_lab()

    def _class_cond(self, X, domain):
        """``p_d(x|y)`` for every class, shape (n, K), for domain ``d`` in {'S','T'}."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x1, x2 = X[:, 0], X[:, 1]
        m = self.means.copy()
        mu2 = 0.0
        if domain == "T" and self.tilt == "x1":
            m = m + self.theta                      # entangled: shifts class coord
        if domain == "T" and self.tilt == "x2":
            mu2 = self.theta                        # factorizable: shifts noise coord
        pdf1 = norm.pdf(x1[:, None] - m[None, :])   # (n, K)
        pdf2 = norm.pdf(x2 - mu2)[:, None]          # (n, 1)
        return pdf1 * pdf2

    def p_x(self, X, domain):
        """Marginal density ``p_d(x) = Σ_y p_d(x|y)·p_d(y)`` (closed form)."""
        prev = self.p_s if domain == "S" else self.p_t
        return self._class_cond(X, domain) @ prev

    def oracle_w_cov(self, X):
        """Oracle covariate ratio ``w_cov*(x) = p_T(x)/p_S(x)`` (closed form).

        In the pure-label-shift regime (no tilt) this EQUALS ``Z*(x)`` exactly (the
        prevalence change is the only thing moving ``p(x)``); with a factorizable
        ``x2`` tilt it is ``a(x2)·Z*(x)``; with an entangled ``x1`` tilt it is a
        genuine Gaussian-mixture ratio that no longer factors through ``Z``.
        """
        return self.p_x(X, "T") / self.p_x(X, "S")

    def oracle_w_joint(self, X, y):
        """Oracle joint ratio ``w_joint*(x,y) = p_T(x,y)/p_S(x,y)`` (closed form).

        The GROUND-TRUTH combined weight. Equals ``w_lab*(y)·w_cov*(x)/Z*(x)`` iff
        the shift is factorizable -- the identity the entanglement diagnostic tests.
        """
        y = np.asarray(y)
        cs = self._class_cond(X, "S")
        ct = self._class_cond(X, "T")
        idx = np.arange(len(y))
        ps_xy = cs[idx, y] * self.p_s[y]
        pt_xy = ct[idx, y] * self.p_t[y]
        return pt_xy / ps_xy

    # -- population confusion matrix (frozen f, source) --------------------
    def confusion_S(self, rng, n=400_000):
        """Large-sample SOURCE confusion matrix ``C_S[i,k]=P_S(ŷ=i, y=k)`` (joint).

        A precise MC estimate of the population joint confusion of the frozen
        classifier under ``p_S`` (``build_gates.md §6`` sanctions large-sample
        integration for ground truth). Rows/cols index predicted/true class.
        """
        X, y, yhat, _, _ = self.sample(n, rng, "S")
        C = np.zeros((self.K, self.K))
        np.add.at(C, (yhat, y), 1.0)
        return C / n

    # -- analytic-ish accepted risk / coverage (MC oracle) -----------------
    def accepted_risk(self, tau0, domain, rng, n=2_000_000):
        """True accepted-region risk ``E_d[ℓ | u(x) ≤ tau0]`` (large-sample MC).

        ``ℓ | y ~ Bernoulli(rho[y])`` independent of ``x`` given ``y``, so the
        accepted risk is a class-mix average that DIFFERS between source and target
        whenever the prevalence differs -- the signal the label-aware path recovers.
        NaN if nothing is accepted. ``build_gates.md §6`` sanctions large-sample
        integration for the known ground truth.
        """
        X, y, yhat, loss, u = self.sample(n, rng, domain)
        acc = u <= tau0
        return float(loss[acc].mean()) if acc.any() else float("nan")

    def accepted_coverage(self, tau0, domain, rng, n=2_000_000):
        """True accepted fraction ``P_d(u(x) ≤ tau0)`` (large-sample MC)."""
        X, y, yhat, loss, u = self.sample(n, rng, domain)
        return float(np.mean(u <= tau0))

    # -- sampling ----------------------------------------------------------
    def sample(self, n, rng, domain):
        """Draw ``n`` i.i.d. points from the named domain.

        Returns ``(X, y, yhat, loss, u)``: features ``X`` shape (n, 2), true class
        ``y``, frozen prediction ``ŷ``, 0-1 loss ``ℓ ~ Bernoulli(rho[y])``, and
        selection score ``u(x)``. ``rng`` is a ``numpy.random.Generator``
        (``build_gates.md §3`` seeding convention).
        """
        n = int(n)
        prev = self.p_s if domain == "S" else self.p_t
        y = rng.choice(self.K, size=n, p=prev)
        x1 = self.means[y] + rng.normal(0.0, 1.0, size=n)
        x2 = rng.normal(0.0, 1.0, size=n)
        if domain == "T" and self.tilt == "x1":
            x1 = x1 + self.theta
        if domain == "T" and self.tilt == "x2":
            x2 = x2 + self.theta
        X = np.column_stack([x1, x2])
        yhat = self.predict(X)
        loss = (rng.uniform(size=n) < self.rho[y]).astype(float)
        u = self.uncertainty(X)
        return X, y, yhat, loss, u
