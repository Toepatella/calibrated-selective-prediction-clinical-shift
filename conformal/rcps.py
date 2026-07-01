"""RCPS risk-controlling threshold search with WSR betting UCB (Stage A spine).

This is the calibration-layer guarantee of method_note.md §2.2: given a *nested*
family of decision rules indexed by a scalar ``lambda`` with risk ``R(lambda)``
monotone non-increasing in ``lambda``, pick the operating point by the RCPS
inf-rule

    lambda_hat = inf{ lambda : UCB_delta(lambda') < alpha  for all lambda' >= lambda }.

Under exchangeability of calibration and test, a bounded loss, and monotonicity,
RCPS gives the finite-sample PAC guarantee  P(R(lambda_hat) <= alpha) >= 1 - delta
(Bates et al. 2021, Def. 1 + Thm. 1). Stage A establishes this exchangeable
foundation; the weighted path for covariate/label shift is added in Stages B-C.

Upper confidence bounds:
  * ``wsr_ucb`` -- Waudby-Smith & Ramdas (2024) hedged-capital betting UCB; the
    variance-adaptive bound recommended as the RCPS UCB (Bates et al. 2021,
    §3.1.3). Preferred here because the importance weights of later stages inflate
    risk-estimate variance, so a variance-adaptive bound is materially tighter
    than a range-only one.
  * ``hb_ucb`` -- Hoeffding-Bentkus range bound (Bates et al. 2021, §3.1.1),
    kept as the conservative comparison.

Ported from the phase0 learning notebook (cells 38/40), whose betting/KL logic was
checked against the authors' code; the binomial tail here delegates to SciPy
(cross-checked against a brute-force binomial in the Gate-A tests).
"""

import math

import numpy as np
from scipy.stats import binom

__all__ = [
    "wsr_ucb",
    "hb_ucb",
    "rcps_lhat",
    "binom_cdf",
    "hajek_risk",
    "kish_n_eff",
    "wsr_ucb_weighted",
    "wsr_lcb_weighted",
]


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p). Thin SciPy wrapper (floor on k)."""
    return float(binom.cdf(math.floor(k), n, p))


def _h1(a, b):
    """Bernoulli relative entropy KL(a || b), clipped away from the boundary."""
    a = min(max(a, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def _bisect(f, lo, hi, iters=60):
    """Smallest root of a decreasing f on [lo, hi] (assumes f(lo) > 0 >= f(hi))."""
    if f(lo) <= 0:
        return lo
    if f(hi) > 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hb_ucb(samples, delta):
    """Hoeffding-Bentkus upper (1 - delta) confidence bound on the mean.

    samples: array-like of losses in [0, 1]. Returns the largest plausible mean R
    such that the min of the Hoeffding and Bentkus tails still exceeds delta.
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    muhat = min(max(float(x.mean()), 1e-9), 1 - 1e-9)
    logd = math.log(delta)

    def tail(R):
        hoeff = -n * _h1(muhat, R)                                            # log Hoeffding tail
        bent = 1.0 + math.log(max(binom_cdf(math.ceil(n * muhat), n, R), 1e-12))  # log(e * Bentkus)
        return min(hoeff, bent) - logd                                       # > 0 => R still plausible

    return 1.0 if tail(1 - 1e-9) > 0 else _bisect(tail, muhat, 1 - 1e-9)


def wsr_ucb(samples, delta):
    """Waudby-Smith & Ramdas hedged-capital betting upper (1 - delta) bound on the mean.

    samples: array-like of losses in [0, 1]. Builds the predictable-mixture betting
    capital process K_n(R) and returns the largest R not yet rejected at level delta.
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    idx = np.arange(1, n + 1)
    muhat = (np.cumsum(x) + 0.5) / (1 + idx)                  # running mean, 1/2 prior
    sig2 = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + idx)   # running variance, 1/4 prior
    sig2 = np.concatenate([[0.25], sig2[:-1]])                # make the bet predictable (lag by one)
    nu = np.minimum(np.sqrt(2 * math.log(1 / delta) / n / sig2), 1)

    def Kn(R):
        return np.max(np.cumsum(np.log(1 - nu * (x - R)))) + math.log(delta)

    return 1.0 if Kn(1 - 1e-9) < 0 else _bisect(lambda R: -Kn(R), 1e-9, 1 - 1e-9)


def rcps_lhat(cal_table, lambdas, alpha, delta, ucb_fn=wsr_ucb):
    """RCPS inf-rule on a loss table.

    cal_table: (n_cal, n_lambda) per-(point, lambda) losses in [0, 1], with columns
        ordered so the loss runs SMALL -> LARGE with column index (i.e. the rule
        shrinks / risk grows as the column index increases). ``lambdas`` is the
        matching 1-D array.
    Returns the tightest lambda whose UCB is still below alpha -- equivalently the
    inf-rule's lambda_hat, scanning from the low-risk end and stopping at the first
    column whose UCB reaches alpha (so the whole controlled tail is below alpha).
    """
    cal_table = np.asarray(cal_table, dtype=float)
    for j in range(cal_table.shape[1]):
        if ucb_fn(cal_table[:, j], delta) >= alpha:
            return lambdas[max(j - 1, 0)]
    return lambdas[-1]


# =========================================================================== #
# Weighted (Hájek) path -- Stage B onward (covariate/label shift).             #
# =========================================================================== #
# From Stage B the calibration points are drawn from the source but the target
# risk is the estimand, so every contribution is reweighted by an estimated
# importance weight ``w`` (``ŵ_cov`` at Gate B; the ``Ẑ``-combined weight at
# Gate C). The reported risk is the SELF-NORMALIZED (Hájek) weighted risk
#
#     R̂_w = Σ (w · ℓ) / Σ w                                   (method_note §1.7)
#
# so the unknown ``E[w] = 1`` cancels. The confidence bound is the same
# Waudby-Smith & Ramdas hedged-capital betting construction as ``wsr_ucb``,
# generalized to weighted observations: with the weights scaled by their clip
# cap ``w_max`` into ``W = w/w_max ∈ [0, 1]``, the per-step betting increment is
# ``W_i·(ℓ_i − R)`` (whose conditional mean is zero exactly at the population
# Hájek risk ``R = E[wℓ]/E[w]``), so the capital process is a non-negative
# martingale at the truth and Ville's inequality gives a valid ``(1−δ)`` bound
# on the weighted risk of the *plug-in* weights. This is a bound on the
# functional of the ESTIMATED weights; no distribution-free coverage certificate
# is claimed for the deployed weight (``method_note §1.5``, §7.1 -- the weight is
# estimated and clinical shift is not pure covariate shift; Gate B measures the
# residual, it does not certify it).


def hajek_risk(losses, weights):
    """Self-normalized (Hájek) weighted risk ``Σ(w·ℓ)/Σw`` (``method_note §1.7``).

    losses, weights : array-like aligned per point; losses in [0, 1], weights >= 0.
    Returns NaN when the weights sum to zero (nothing to average).
    """
    l = np.asarray(losses, dtype=float)
    w = np.asarray(weights, dtype=float)
    sw = float(w.sum())
    return float(np.sum(w * l) / sw) if sw > 0.0 else float("nan")


def kish_n_eff(weights):
    """Kish effective sample size ``(Σw)²/Σw²`` (``method_note §1.7``; diagnostic).

    A REPORTED reliability diagnostic, never a gate: it collapses toward 1 when a
    single weight dominates and equals ``n`` for uniform weights.
    """
    w = np.asarray(weights, dtype=float)
    s = float(w.sum())
    s2 = float(np.sum(w * w))
    return float(s * s / s2) if s2 > 0.0 else 0.0


def _wsr_weighted_nu(losses, W, delta):
    """Predictable hedged-capital betting fractions for the weighted risk.

    Mirrors ``wsr_ucb``'s predictable mixture but on the weighted residual
    ``W·(ℓ − μ̂)`` with ``μ̂`` the running Hájek mean; the variance estimate is
    lagged by one step so each ``ν_i`` depends only on points ``1..i−1`` (hence is
    predictable, the martingale requirement). ``W ∈ [0, 1]`` and ``ν ∈ [0, 1]`` so
    every betting factor ``1 ± ν·W·(ℓ − R)`` stays strictly positive on the open
    grid ``R ∈ (0, 1)``.
    """
    n = len(losses)
    cw = np.cumsum(W)
    cwl = np.cumsum(W * losses)
    mhat = (cwl + 0.5) / (cw + 1.0)                        # running Hájek mean, mild prior
    sig2 = (np.cumsum(W * (losses - mhat) ** 2) + 0.25) / (cw + 1.0)
    sig2 = np.concatenate([[0.25], sig2[:-1]])             # predictable (lag by one)
    return np.minimum(np.sqrt(2 * math.log(1 / delta) / n / sig2), 1.0)


def wsr_ucb_weighted(losses, weights, delta, w_max):
    """WSR hedged-capital upper ``(1−δ)`` bound on the Hájek weighted risk.

    losses, weights : per-point losses in [0, 1] and importance weights in
        ``[0, w_max]`` (already clipped at the routing cap ``w_max``).
    Returns the largest weighted risk ``R`` not yet rejected at level ``delta`` --
    the weighted analogue of ``wsr_ucb`` (which is the ``w ≡ 1`` special case up to
    the predictable-mixture bookkeeping). Genuinely variance-adaptive: it is the
    betting bound on the weighted estimator, never a range-only ``[0, 1]`` fallback.
    """
    x = np.asarray(losses, dtype=float)
    w = np.asarray(weights, dtype=float)
    W = np.clip(w / w_max, 0.0, 1.0)
    nu = _wsr_weighted_nu(x, W, delta)

    def cap(R):                                            # increasing in R
        return np.max(np.cumsum(np.log1p(nu * W * (R - x)))) + math.log(delta)

    if cap(1 - 1e-9) < 0:
        return 1.0
    return _bisect(lambda R: -cap(R), 1e-9, 1 - 1e-9)


def wsr_lcb_weighted(losses, weights, delta, w_max):
    """WSR hedged-capital lower ``(1−δ)`` bound on the Hájek weighted risk.

    The mirror of ``wsr_ucb_weighted``: the capital process bets that the true
    weighted risk exceeds ``R``, so the returned value is the smallest risk not
    rejected -- a valid ``(1−δ)`` lower confidence bound. Paired with the UCB it
    forms the two-sided betting interval whose non-overlap is the ``prereg §6.7``
    win-rule used at Gate B (weighted interval strictly below the naive interval).
    """
    x = np.asarray(losses, dtype=float)
    w = np.asarray(weights, dtype=float)
    W = np.clip(w / w_max, 0.0, 1.0)
    nu = _wsr_weighted_nu(x, W, delta)

    def cap(R):                                            # decreasing in R
        return np.max(np.cumsum(np.log1p(nu * W * (x - R)))) + math.log(delta)

    if cap(1e-9) < 0:
        return 0.0
    return _bisect(cap, 1e-9, 1 - 1e-9)
