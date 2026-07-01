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

__all__ = ["wsr_ucb", "hb_ucb", "rcps_lhat", "binom_cdf"]


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
