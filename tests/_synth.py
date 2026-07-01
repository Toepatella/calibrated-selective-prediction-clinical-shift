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

__all__ = [
    "PolyErrorModel",
    "GaussianCovariateShift",
    "LabelShiftMixture",
    "GaussianOODFeatures",
]


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


class GaussianOODFeatures:
    """Synthetic ``D``-dim Gaussian feature OOD ground truth (Gate-D fixture).

    Fully synthetic (no clinical data). Reuses the phase-0 0.1.6 toy: ``D``-dim
    Gaussian features, ``K`` classes with per-class means ``μ_c`` and a shared
    **tied** covariance ``Σ`` so the Mahalanobis GDA model is EXACTLY correct
    in-distribution (``build_gates.md §7``: "the Mahalanobis GDA model is exactly
    correct in-distribution"). Everything a Gate-D check needs then has known ground
    truth: the OOD label per point (in-scope vs a disjoint far-OOD shell), the
    analytic far-OOD leakage as a function of the screen cutoff (large-sample MC), the
    oracle covariate ratio ``w_cov*(x)`` for the routing gates, and engineered probe
    points that isolate the two scope guards.

    Geometry
    --------
    ``x ∈ ℝ^D`` splits into three coordinate blocks:

      * **class-discriminative** dims ``0..K-1``: ``μ_c`` has ``class_sep`` in coord
        ``c`` and 0 elsewhere, so the ``K`` classes sit at well-separated centroids.
        The frozen classifier ``f`` (``logits`` / ``uncertainty``) reads ONLY these
        dims -- a confidently-classified point can therefore still be far-OOD in the
        remaining dims (the confident-but-OOD case OOD detection targets).
      * **noise** dims ``K..D-2``: class-independent ``N(0, sd²)`` -- the subspace the
        far-OOD shell is offset into, so far-OOD is ORTHOGONAL to the class signal
        (invisible to ``f``'s class dims, visible to the full-feature detector).
      * the **covariate-shift** dim ``D-1``: class-independent, ``N(0, sd²)`` under
        source and ``N(shift_delta, sd²)`` under target, so the oracle covariate ratio
        ``w_cov*(x) = p_T(x)/p_S(x) = exp(shift_delta·x_s/sd_s² − shift_delta²/2sd_s²)``
        is closed form per point (a factorizable, class-free covariate shift).

    This decoupling is what lets the two routing mechanisms be exercised
    independently (``build_gates.md §7``): the far-OOD guard ``t_ood`` keys on the
    orthogonal noise offset (the FULL detector), while the huge-weight guard ``w_max``
    keys on the covariate-shift coordinate (``ŵ_cov``); a far-OOD point can be built
    with ``w_cov ≈ 0`` (so ``w_max`` does NOT flag it) and a near-OOD point with
    ``w_cov > w_max`` but ``o ≤ t_ood``.

    Honesty rails (``build_gates.md §7``): a SYNTHETIC wiring fixture with known
    ground truth. It proves nothing about deployment, claims no distribution-free /
    finite-sample OOD guarantee (OOD detection is not distribution-free learnable --
    Fang et al. 2022), and asserts no certified ``α = α_acc + α_ood`` union split. It
    verifies the Mahalanobis GDA model's behaviour under its OWN Gaussian assumption
    on synthetic data only; near-OOD overlapping ID support is exactly the regime the
    impossibility covers (``method_note §4.4``).
    """

    def __init__(self, D=20, K=5, class_sep=4.0, shift_delta=2.0,
                 far_radius=12.0, near_radius=5.0, id_std=1.0, sd=None, rho=None):
        if not (0 < K < D):
            raise ValueError("need 0 < K < D (K class dims + noise dims + shift dim)")
        self.D = int(D)
        self.K = int(K)
        self.class_sep = float(class_sep)
        self.shift_delta = float(shift_delta)
        self.far_radius = float(far_radius)
        self.near_radius = float(near_radius)
        self.id_std = float(id_std)
        self.shift_dim = self.D - 1
        # class-structured 0-1 loss  ℓ | y ~ Bernoulli(rho[y]) (independent of x given y),
        # so the accepted-region risk is a class-mix average (the task loss the weighted
        # Hájek path estimates on the answered-and-in-scope cohort).
        self.rho = (np.linspace(0.02, 0.30, self.K) if rho is None
                    else np.asarray(rho, dtype=float))
        if self.rho.shape != (self.K,) or np.any((self.rho < 0) | (self.rho > 1)):
            raise ValueError("rho must be a length-K vector of class loss probabilities in [0, 1]")
        # per-dim standard deviation of the tied covariance Σ = diag(sd²).
        self.sd = np.ones(self.D) if sd is None else np.asarray(sd, dtype=float)
        if self.sd.shape != (self.D,):
            raise ValueError("sd must have length D")
        if np.any(self.sd <= 0):
            raise ValueError("sd must be strictly positive")
        # class means: class c centred at class_sep on axis c (0 elsewhere).
        M = np.zeros((self.K, self.D))
        M[np.arange(self.K), np.arange(self.K)] = self.class_sep
        self.means = M
        self._noise_dims = np.arange(self.K, self.D - 1)   # dims K..D-2 (far-OOD subspace)

    # -- frozen classifier f (reads the class-discriminative dims only) ---------
    def logits(self, phi):
        """GDA logits on the class dims ``0..K-1`` only: ``-0.5‖(x−μ_c)/sd‖²`` (+const).

        ``f`` deliberately ignores the noise / shift dims, so a point sitting near a
        class centroid in class-space is CONFIDENT even when it is far-OOD in the
        orthogonal noise dims (the confident-but-OOD failure OOD detection targets).
        Uniform class prior ``1/K`` (an additive constant, irrelevant to the softmax).
        """
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        xc = phi[:, :self.K]
        mc = self.means[:, :self.K]
        sc = self.sd[:self.K]
        diff = (xc[:, None, :] - mc[None, :, :]) / sc[None, None, :]
        return -0.5 * np.sum(diff ** 2, axis=2)

    def posterior(self, phi):
        """Closed-form class posterior = softmax of the class-dim GDA logits."""
        lg = self.logits(phi)
        lg = lg - lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, phi):
        """Frozen hard prediction ``ŷ(x) = argmax_c p(c|x)`` (class dims only)."""
        return np.argmax(self.logits(phi), axis=1)

    def uncertainty(self, phi):
        """Selection score ``u(x) = 1 − max_c p(c|x)`` (higher = less confident)."""
        return 1.0 - self.posterior(phi).max(axis=1)

    # -- oracle covariate ratio (closed form; factorizable shift dim) -----------
    def oracle_w_cov(self, phi):
        """Oracle covariate ratio ``w_cov*(x) = p_T(x)/p_S(x)`` (closed form).

        The shift is a pure location shift of the class-free coordinate ``x_s`` (dim
        ``D-1``), ``N(0, sd_s²) → N(shift_delta, sd_s²)``, so the whole-``x`` ratio
        collapses to the one-dim Gaussian ratio
        ``exp(shift_delta·x_s/sd_s² − shift_delta²/2sd_s²)`` -- log-linear in ``x_s``.
        This is the population analogue of the Gate-B discriminator odds ``(d/(1−d))``;
        a far-OOD point placed at very negative ``x_s`` has ``w_cov ≈ 0`` (``d ≈ 0``),
        so ``ŵ_cov`` does NOT flag it and ``t_ood`` is the sole far-OOD guard.
        """
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        xs = phi[:, self.shift_dim]
        s2 = self.sd[self.shift_dim] ** 2
        d = self.shift_delta
        return np.exp(d * xs / s2 - 0.5 * d * d / s2)

    # -- sampling: in-scope (in-distribution) -----------------------------------
    def sample_id(self, n, rng, domain="S"):
        """Draw ``n`` in-scope points ``(phi, y, loss)`` from the class-conditionals.

        ``y ~ Uniform{0..K-1}``; ``phi = μ_y + sd·z``, ``z ~ N(0, I)``; under
        ``domain='T'`` the shift coordinate is displaced by ``shift_delta`` (the
        covariate shift, a class-free coordinate). ``loss ~ Bernoulli(rho[y])`` is the
        class-structured 0-1 task loss (``method_note §1.4``), the object the weighted
        Hájek path estimates on the answered cohort. ``rng`` is a
        ``numpy.random.Generator`` (``build_gates.md §3`` seeding convention).
        """
        n = int(n)
        y = rng.integers(0, self.K, size=n)
        phi = self.means[y] + self.sd * rng.normal(size=(n, self.D))
        if domain == "T":
            phi[:, self.shift_dim] += self.shift_delta
        elif domain != "S":
            raise ValueError("domain must be 'S' or 'T'")
        loss = (rng.uniform(size=n) < self.rho[y]).astype(float)
        return phi, y, loss

    def accepted_risk(self, tau, domain, rng, n=2_000_000):
        """True accepted-region task risk ``E_d[ℓ | u(x) ≤ tau]`` (large-sample MC).

        The class-structured loss is independent of ``x`` given ``y``; the covariate
        shift moves only a class-free coordinate, so the accept-region class mix (and
        hence this risk) is essentially domain-invariant -- the interval-coverage
        diagnostic checks WSR-interval COVERAGE of this known weighted risk, not risk
        restoration. NaN if nothing is accepted. ``build_gates.md §7`` sanctions
        large-sample integration for the known ground truth.
        """
        phi, y, loss = self.sample_id(n, rng, domain)
        acc = self.uncertainty(phi) <= tau
        return float(loss[acc].mean()) if acc.any() else float("nan")

    # -- sampling: far-OOD exposure set O (known OOD label) ---------------------
    def _orthogonal_shell(self, n, radius, rng):
        """A shell of radius ``radius`` in the NOISE subspace (dims ``K..D-2``).

        Uniform directions on the sphere of that orthogonal subspace, scaled by
        ``radius·(1 + small jitter)`` -- so the offset is invisible to ``f`` (which
        reads class dims only) but far from every class centroid for the full-feature
        detector. Class dims and the shift dim are filled by the caller.
        """
        m = self._noise_dims.size
        g = rng.normal(size=(n, m))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        scale = radius * (1.0 + 0.05 * rng.uniform(size=(n, 1)))
        block = np.zeros((n, self.D))
        block[:, self._noise_dims] = g * scale
        return block

    def sample_far_ood(self, n, rng):
        """Draw ``n`` far-OOD points ``(phi, y)`` -- CONFIDENT but out-of-scope.

        Class dims sit near a (random) class centroid with a small ``id_std`` spread,
        so ``f`` is CONFIDENT (low ``u``) and the point would be answered if the OOD
        screen were off. The orthogonal noise offset (radius ``far_radius``) makes it
        far from every class Gaussian for the FULL-feature detector (``o`` huge → the
        known-OOD label), while a class-dims-only ("imperfect") detector cannot see the
        offset and leaks it. The shift coordinate is set to 0 (``w_cov < 1 ≤ w_max``),
        so these far-OOD points pass the weight gate -- the OOD screen is the only
        guard that can route them out.
        """
        n = int(n)
        y = rng.integers(0, self.K, size=n)
        phi = self._orthogonal_shell(n, self.far_radius, rng)
        phi[:, :self.K] = self.means[y][:, :self.K] + self.id_std * rng.normal(size=(n, self.K))
        phi[:, self.shift_dim] = 0.0
        return phi, y

    def sample_near_ood(self, n, rng):
        """Draw ``n`` near-OOD points ``(phi, y)`` -- a CLOSER shell (partial ID overlap).

        Same construction as :meth:`sample_far_ood` but at the smaller ``near_radius``,
        so the offset overlaps the in-distribution noise tail: a far-tier ``t_ood``
        misses most of it. REPORTED separately and expected worse than far-OOD; it is
        NOT presented as a bound -- near-OOD overlapping ID support is exactly the
        regime the impossibility covers (Fang et al. 2022; ``method_note §4.4``).
        """
        n = int(n)
        y = rng.integers(0, self.K, size=n)
        phi = self._orthogonal_shell(n, self.near_radius, rng)
        phi[:, :self.K] = self.means[y][:, :self.K] + self.id_std * rng.normal(size=(n, self.K))
        phi[:, self.shift_dim] = 0.0
        return phi, y

    def far_ood_d0(self, n, rng, x_shift=-2.0):
        """Far-OOD probe points engineered with ``w_cov ≈ 0`` (``d ≈ 0``).

        Identical to :meth:`sample_far_ood` but the shift coordinate is pinned
        negative, so the oracle covariate ratio ``w_cov*(x) = exp(shift_delta·x_s − …)``
        is tiny (``≈ 2.5e-3`` at the default) -- the discriminator would call these
        definitely-source. Then the weight gate (``ŵ_cov > w_max``) does NOT route them
        (``w_cov ≈ 0 ≤ w_max``) while the OOD gate (``o > t_ood``) does: ``t_ood`` is the
        SOLE far-OOD guard (``build_gates.md §7`` hard gate 5; ``method_note §4.2``,
        §1.5.1). The shift is kept modest (``-2`` rather than deeply negative) so, under
        the Maha++ L2-normalization, the probe's OOD score stays clearly above the
        in-scope range -- a deeply-negative shift would rotate the normalized direction
        enough to depress ``o`` toward the in-scope tail.
        """
        phi, y = self.sample_far_ood(n, rng)
        phi[:, self.shift_dim] = float(x_shift)
        return phi, y
