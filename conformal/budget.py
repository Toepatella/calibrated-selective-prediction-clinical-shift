"""Integrated OOD budget selection & the decoupled three-gate routing rule (Stage D).

Realizes ``method_note §1.5.1`` / §4.2 / §4.3 / §5.1. Stage D stacks the OOD screen on
the Gate-A/B/C weighted selective pipeline and composes the integrated answered event

    A(x) = { g(x)=1 } ∩ { o(x) ≤ t_ood } ∩ { ŵ_cov(x) ≤ w_max }
         = (u ≤ τ)   ∩ ( o ≤ t_ood )    ∩ ( ŵ_cov ≤ w_max ),

with the two scope guards DECOUPLED: ``t_ood`` is the SOLE far-OOD guard (a far-OOD
point can have ``d(x) ≈ 0`` ⇒ ``ŵ_cov ≈ 0``, so the weight gate does not flag it),
while ``w_max`` is a variance / boundedness control whose routed tail (the huge-weight
near-OOD variance tail) costs only coverage (``method_note §4.2``, §1.5.1, §3.2). The
non-answered mass decomposes three ways -- route-on-OOD / abstain-on-weight /
defer-on-uncertainty -- summing exactly to ``1 − coverage``.

The screen threshold ``t_ood`` is set on the exposure set ``O`` to SPEND a target
leakage budget ``α_ood`` (the smallest measured leakage still spending the budget;
:func:`set_t_ood`). ``α_acc`` (the accepted-region risk budget of Gate A) and
``α_ood`` (this leakage budget) are two **separately-measured** operating budgets:
:func:`report_budgets` exposes them as distinct reported quantities and any ``α``
sum is a reporting-convenience headline ONLY.

Honesty rails (``build_gates.md §7``; ``method_note §5.3``, §7.1; ``prereg §6.2``, §8):
  * NO distribution-free / finite-sample OOD guarantee -- OOD detection is not
    distribution-free learnable (Fang et al. 2022); ``t_ood`` is a MEASURED screen with
    a reported leakage rate on a stated, swappable ``O``.
  * NO certified additive ``α = α_acc + α_ood`` union bound; the two budgets are
    tracked and reported SEPARATELY (``certified_additive_split = False``).
  * A passing Stage-D gate is an engineering wiring / measured-reduction check -- NOT
    evidence the deployed pipeline controls ``R_T^accept`` (``method_note §1.1``).
"""

from collections import namedtuple

import numpy as np

__all__ = [
    "leakage_rate",
    "set_t_ood",
    "accept_mask",
    "RouteDecomposition",
    "route_decomposition",
    "OperatingBudgets",
    "report_budgets",
]


def leakage_rate(o_scores, t_ood):
    """Measured leakage: fraction of ``o_scores`` that PASS the screen (``o ≤ t_ood``).

    On the exposure set ``O`` this is the far-OOD leakage into the answered path.
    Because higher ``o`` = more OOD and a point passes iff ``o ≤ t_ood``, leakage is
    NON-DECREASING in ``t_ood`` (``method_note §4.3``).
    """
    o = np.asarray(o_scores, dtype=float)
    return float(np.mean(o <= t_ood)) if o.size else float("nan")


def set_t_ood(o_scores_O, alpha_ood, grid=None):
    """OOD screen ``t_ood`` spending the leakage budget ``α_ood`` on the exposure set ``O``.

    ``o`` higher = more OOD; a point passes the screen (answered / in-scope) iff
    ``o ≤ t_ood``, so the far-OOD leakage ``leakage(t) = mean_{x∈O}[o(x) ≤ t]`` is
    NON-DECREASING in ``t``. The budget rule SPENDS ``α_ood``: over a monotone scan of
    candidate cutoffs (each distinct ``O`` score by default), return the LOOSEST cutoff
    whose measured leakage is still ``≤ α_ood`` -- equivalently the empirical
    ``α_ood``-quantile of the ``O`` scores. This maximizes in-scope coverage subject to
    the leakage budget; its realized leakage is ``≤ α_ood`` and within one scan step
    (``1/|O|``) of it. If even the tightest cutoff already leaks ``> α_ood`` (i.e.
    ``α_ood < 1/|O|``), the tightest possible screen is returned (leakage 0), never a
    cutoff that exceeds the budget.

    A MEASURED screen with a reported leakage rate on a stated, swappable ``O`` -- NO
    distribution-free OOD guarantee (Fang et al. 2022; ``method_note §4.3``, §4.4,
    §7.1; ``prereg §6.2``, §8).
    """
    o = np.sort(np.asarray(o_scores_O, dtype=float))
    n = o.size
    if n == 0:
        raise ValueError("O must be non-empty to set t_ood")
    cand = o if grid is None else np.sort(np.asarray(grid, dtype=float))
    leak = np.searchsorted(o, cand, side="right") / n          # leakage(t) per candidate cutoff
    within = np.nonzero(leak <= float(alpha_ood))[0]
    if within.size == 0:
        return float(np.nextafter(cand[0], -np.inf))           # can't spend even 1 point: route all
    return float(cand[within[-1]])                             # loosest cutoff within the budget


def accept_mask(u, o, w_cov, tau, t_ood, w_max):
    """Integrated answered event ``A(x) = (u ≤ τ) ∧ (o ≤ t_ood) ∧ (ŵ_cov ≤ w_max)``.

    All inputs are per-point arrays (or scalars). ``ŵ_cov`` here is the RAW covariate
    weight ``(d/(1−d))·ĉ`` (pre-clip): a point is routed on weight when the raw weight
    EXCEEDS ``w_max`` (an accepted point has raw weight ``≤ w_max``, so the Hájek clip
    is inactive for it). Boolean, elementwise (``method_note §1.5.1``, §5.1).
    """
    u = np.asarray(u, dtype=float)
    o = np.asarray(o, dtype=float)
    w = np.asarray(w_cov, dtype=float)
    return (u <= tau) & (o <= t_ood) & (w <= w_max)


# answered / three not-answered mechanism fractions + their boolean masks.
RouteDecomposition = namedtuple(
    "RouteDecomposition",
    ["answered", "defer_uncertainty", "route_ood", "abstain_weight",
     "mask_answered", "mask_defer", "mask_route_ood", "mask_abstain_weight"],
)


def route_decomposition(u, o, w_cov, tau, t_ood, w_max):
    """Decompose the non-answered mass into the three routing mechanisms.

    Returns a :class:`RouteDecomposition` whose four FRACTIONS
    (``answered``, ``route_ood``, ``abstain_weight``, ``defer_uncertainty``) partition
    the points and sum to exactly 1, so the three not-answered fractions sum to
    ``1 − coverage`` (``method_note §4.2``, §5.1, §1.5.1). Non-answered points are
    attributed by a fixed PRECEDENCE -- route-on-OOD (out of scope) ≻ abstain-on-weight
    (variance/boundedness control) ≻ defer-on-uncertainty (model defers) -- so a point
    failing several gates is booked to its most fundamental reason; the raw per-gate
    masks are also exposed for the decoupling checks (a point with ``ŵ_cov > w_max`` but
    ``o ≤ t_ood`` is booked to ``abstain_weight``, NOT ``route_ood``). This is a
    reporting decomposition, not a certified budget split.
    """
    u = np.asarray(u, dtype=float)
    o = np.asarray(o, dtype=float)
    w = np.asarray(w_cov, dtype=float)
    answered = accept_mask(u, o, w, tau, t_ood, w_max)
    not_ans = ~answered
    route_ood = not_ans & (o > t_ood)
    abstain_weight = not_ans & ~(o > t_ood) & (w > w_max)
    defer = not_ans & ~(o > t_ood) & ~(w > w_max) & (u > tau)
    n = float(answered.size)
    return RouteDecomposition(
        answered=float(answered.mean()),
        defer_uncertainty=float(defer.sum() / n),
        route_ood=float(route_ood.sum() / n),
        abstain_weight=float(abstain_weight.sum() / n),
        mask_answered=answered,
        mask_defer=defer,
        mask_route_ood=route_ood,
        mask_abstain_weight=abstain_weight,
    )


# The two operating budgets, tracked SEPARATELY (never summed into a certificate).
OperatingBudgets = namedtuple("OperatingBudgets", ["alpha_acc", "alpha_ood"])


def report_budgets(alpha_acc, alpha_ood, realized_accept_risk=None, realized_leakage=None):
    """Report ``α_acc`` and ``α_ood`` as two SEPARATELY-measured operating budgets.

    ``α_acc`` is Gate A's accepted-region risk budget; ``α_ood`` is the far-OOD leakage
    budget spent by :func:`set_t_ood`. They are DISTINCT reported quantities: the
    returned dict lists each budget beside its realized value, and ``alpha_sum`` is a
    reporting-convenience headline ONLY -- ``certified_additive_split = False`` records
    that the code makes NO claim that realized accepted risk + realized leakage
    ``≤ α_acc + α_ood`` with confidence ``1 − δ`` (``method_note §1.6``, §4.3, §5.3;
    ``prereg §6.2``, §8; ``build_gates.md §7`` hard gate 8 + honesty rails).
    """
    return {
        "alpha_acc": float(alpha_acc),
        "alpha_ood": float(alpha_ood),
        "realized_accept_risk": None if realized_accept_risk is None else float(realized_accept_risk),
        "realized_leakage": None if realized_leakage is None else float(realized_leakage),
        "alpha_sum": float(alpha_acc) + float(alpha_ood),   # reporting convenience ONLY
        "certified_additive_split": False,                  # NOT a certified union bound
    }
