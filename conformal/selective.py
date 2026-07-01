"""Selection gate g(x)=1[u(x)<=tau] + accepted-region RCPS calibration.

Realizes method_note.md §2.1-§2.2 for the exchangeable foundation (Stage A):

  * The selection gate ``g(x) = 1[u(x) <= tau]`` answers low-uncertainty cases and
    abstains on the rest (§2.1).
  * The controlled object is the *accepted-region* risk  E[ l(Y, yhat) | g=1 ].
    Selection makes the answered subset non-exchangeable with the full calibration
    set, so the threshold must be calibrated ON THE ACCEPTED REGION -- the
    selection-bias trap (§1.4, §2.2). Because the gate depends only on x, the
    accepted calibration points and accepted test points are exchangeable with each
    other, so restricting calibration to points that pass the same gate restores
    the guarantee on the accepted region.

``select_threshold`` composes the gate with the RCPS inf-rule (``rcps.py``): it
scans candidate thresholds from the tightest gate (lowest accepted risk) toward the
loosest, evaluating the UCB on the *accepted* losses, and returns the loosest tau
(maximum coverage) whose accepted-region UCB is still below alpha. This is the
selective analogue of ``rcps_lhat``; under exchangeability it inherits the RCPS
PAC guarantee  P(accepted risk <= alpha) >= 1 - delta  (Gate A verifies this).
"""

from collections import namedtuple

import numpy as np

from .rcps import wsr_ucb

__all__ = [
    "selection_gate",
    "selective_risk",
    "risk_coverage_curve",
    "aurc",
    "select_threshold",
    "SelectiveResult",
]

# tau_hat: chosen threshold (None if no gate controls); coverage: accepted fraction
# on the calibration set; ucb: accepted-region UCB at tau_hat; risk_hat: empirical
# accepted risk at tau_hat; controlled: whether any gate met the (alpha, delta) target.
SelectiveResult = namedtuple(
    "SelectiveResult", ["tau_hat", "coverage", "ucb", "risk_hat", "controlled"]
)


def selection_gate(u, tau):
    """Boolean accept mask g(x) = 1[u(x) <= tau] (higher u = less certain)."""
    return np.asarray(u, dtype=float) <= tau


def selective_risk(losses, u, tau):
    """Accepted-region empirical risk and coverage at threshold ``tau``.

    Returns (risk, coverage). risk is NaN when nothing is accepted.
    """
    losses = np.asarray(losses, dtype=float)
    accept = selection_gate(u, tau)
    coverage = float(accept.mean())
    risk = float(losses[accept].mean()) if accept.any() else float("nan")
    return risk, coverage


def risk_coverage_curve(u, losses):
    """Empirical risk-coverage curve: accept lowest-uncertainty cases first.

    Returns (coverage, risk) arrays of length n, where coverage[k] = (k+1)/n and
    risk[k] is the error rate over the k+1 most-confident accepted cases.
    """
    u = np.asarray(u, dtype=float)
    losses = np.asarray(losses, dtype=float)
    order = np.argsort(u, kind="mergesort")          # stable: ties keep input order
    ls = losses[order]
    k = np.arange(1, len(u) + 1)
    coverage = k / len(u)
    risk = np.cumsum(ls) / k
    return coverage, risk


def aurc(u, losses):
    """Area under the (empirical) risk-coverage curve -- lower is better."""
    return float(risk_coverage_curve(u, losses)[1].mean())


def select_threshold(
    u,
    losses,
    alpha,
    delta,
    ucb_fn=wsr_ucb,
    taus=None,
    n_grid=100,
    min_accept=30,
):
    """Accepted-region RCPS threshold selection (the Stage-A headline).

    Scans candidate thresholds from the tightest gate (lowest accepted risk)
    toward the loosest, and returns the loosest tau whose accepted-region UCB is
    still below ``alpha`` -- the RCPS inf-rule applied to the selective risk.

    u, losses : array-like aligned per calibration point; losses in [0, 1].
    alpha     : target accepted-region risk.
    delta     : confidence of the UCB (PAC level 1 - delta).
    ucb_fn    : upper-confidence-bound function (wsr_ucb default, hb_ucb available).
    taus      : explicit candidate thresholds; default = ``n_grid`` quantiles of u.
    min_accept: thresholds accepting fewer points than this are treated as
        non-certifiable and skipped; the controlled run begins at the first
        certifiable threshold (documented departure from a strict all-tighter scan,
        because the very tightest gates have too few accepted points to bound).

    Returns a ``SelectiveResult``. If no gate controls, ``tau_hat`` is None,
    ``controlled`` is False, and the pipeline should abstain on every case.
    """
    u = np.asarray(u, dtype=float)
    losses = np.asarray(losses, dtype=float)

    if taus is None:
        qs = np.linspace(1.0 / n_grid, 1.0, n_grid)
        taus = np.quantile(u, qs)
    taus = np.sort(np.asarray(taus, dtype=float))     # ascending: tight gate -> loose gate

    best = SelectiveResult(None, 0.0, float("nan"), float("nan"), False)
    for tau in taus:
        accept = u <= tau
        n_acc = int(accept.sum())
        if n_acc < min_accept:
            continue                                  # too few accepted to certify
        ucb = ucb_fn(losses[accept], delta)
        if ucb < alpha:
            risk = float(losses[accept].mean())
            best = SelectiveResult(float(tau), n_acc / len(u), float(ucb), risk, True)
        else:
            break                                     # inf-rule: looser gates also fail
    return best
