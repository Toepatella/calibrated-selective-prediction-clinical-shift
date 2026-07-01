"""Gate-A -- exchangeable selective RCPS (the foundation / wiring gate).

Realizes the Definition of Done of ``build_gates.md §4``. G-A verifies that the
calibration-layer spine -- the selection gate ``g(x)=1[u(x)<=tau]``
(``method_note §2.1``), the accepted-region RCPS inf-rule (``§2.2``), and the WSR
hedged-capital betting UCB (``§1.6``, §2.2) -- is wired correctly UNDER
EXCHANGEABILITY (no shift), the one regime where RCPS's own ``(alpha_acc, delta)``
PAC property holds (Bates et al. 2021, Def. 1 + Thm. 1).

Honesty rails enforced by construction (``build_gates.md §4``):
  * These are SYNTHETIC wiring checks on data with known ground truth. Nothing
    here is a guarantee about deployment or real clinical data; no test asserts
    ``R_T^accept <= alpha_acc`` on shifted data (there is no shift at this rung),
    and none certifies the combined/deployed pipeline.
  * The ``(alpha_acc, delta)`` property is re-VERIFIED as a property of RCPS under
    exchangeability, never re-asserted once shift enters (B/C/D).
  * ``n_eff`` / interval-coverage / AURC / degeneracy are REPORTED DIAGNOSTICS,
    never pass/fail gates.

Each test maps to one Definition-of-Done bullet; HARD GATES assert, DIAGNOSTICS
measure/print and only sanity-check plumbing. Numeric budgets come from
``configs/gate_a.toml`` (the committed "proposed" run config).
"""

import math
import pathlib
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from conformal.rcps import wsr_ucb, hb_ucb, rcps_lhat, binom_cdf
from conformal.selective import (
    selection_gate,
    selective_risk,
    risk_coverage_curve,
    aurc,
    select_threshold,
    SelectiveResult,
)
from conformal.folds import assert_group_disjoint
from _synth import PolyErrorModel


# --------------------------------------------------------------------------- #
# Run config (committed proposed numbers -- build_gates.md §4 open question).   #
# --------------------------------------------------------------------------- #
_CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gate_a.toml"


def _load_cfg():
    with open(_CFG_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    b, mc, em, sd = raw["budgets"], raw["monte_carlo"], raw["error_model"], raw["seed"]
    return SimpleNamespace(
        alpha_acc=b["alpha_acc"],
        delta=b["delta"],
        coverage_min=b["coverage_min"],
        n_cal=mc["n_cal"],
        T=mc["T"],
        rc_points=mc["rc_points"],
        rc_grid=mc["rc_grid"],
        coeffs=em["coeffs"],
        families=em["families"],
        seed_base=sd["base"],
    )


CFG = _load_cfg()


def _wald_band(delta, T):
    """1.96-sigma Wald normal-approx half-width on an empirical rate (n=T, p=delta)."""
    return 1.96 * math.sqrt(delta * (1.0 - delta) / T)


# --------------------------------------------------------------------------- #
# Diagnostics-only helpers (REPORTED, never gated). Inlined here because G-A    #
# introduces no diagnostics module.                                            #
# --------------------------------------------------------------------------- #
def _kish_n_eff(w):
    w = np.asarray(w, dtype=float)
    return (w.sum() ** 2) / np.sum(w ** 2)


def _is_degenerate(coverage, coverage_min):
    """Degeneracy flag (prereg §6.4): realized coverage below the floor."""
    return coverage < coverage_min


def _select_threshold_marginal(u, losses, alpha, delta, ucb_fn=wsr_ucb,
                               n_grid=100, min_accept=30):
    """DELIBERATELY-WRONG full-marginal calibration -- the selection-bias trap.

    Mirrors ``select_threshold`` but bounds the *marginal* masked risk
    ``E[l * 1{g=1}]`` (losses of accepted points normalised by the FULL sample
    size N) instead of the accepted-region risk ``E[l | g=1]`` (normalised by the
    accepted count). Because it divides by N it systematically UNDER-estimates the
    accepted risk, so it certifies too-loose thresholds -- exactly the trap the
    real ``select_threshold`` avoids by calibrating ON the accepted region
    (``method_note §1.4``, §2.2). Used only as a negative control in
    ``test_selection_bias_trap``; never part of the pipeline.
    """
    u = np.asarray(u, dtype=float)
    losses = np.asarray(losses, dtype=float)
    qs = np.linspace(1.0 / n_grid, 1.0, n_grid)
    taus = np.sort(np.quantile(u, qs))
    best = None
    for tau in taus:
        accept = u <= tau
        if int(accept.sum()) < min_accept:
            continue
        masked = np.where(accept, losses, 0.0)          # normalise by full N -> trap
        if ucb_fn(masked, delta) < alpha:
            best = float(tau)
        else:
            break
    return best


# =========================================================================== #
# HARD GATES                                                                    #
# =========================================================================== #
def test_risk_coverage_matches_analytic():
    """HARD: monotone risk-coverage trade-off matches the analytic curve.

    build_gates.md §4: on rc_points points (raised from the doc's illustrative
    200000 to 1e6 for seed-robustness at the tightest quantile), for >=50
    tau-quantiles, coverage is non-decreasing in tau (np.diff >= -1e-9) and
    |empirical_accepted_risk(tau) - analytic_R(tau)| <= 0.01 at every grid point.
    Swept over a small family of monotone error models so no single construction
    can carry the pass.
    """
    rng = np.random.default_rng(CFG.seed_base)
    qs = np.linspace(1.0 / CFG.rc_grid, 1.0, CFG.rc_grid)
    for coeffs in CFG.families:
        model = PolyErrorModel(coeffs)
        u, losses = model.sample(CFG.rc_points, rng)
        taus = np.quantile(u, qs)

        covs, risks = [], []
        for tau in taus:
            risk, cov = selective_risk(losses, u, tau)
            covs.append(cov)
            risks.append(risk)
            # accepted-region risk matches the analytic R(tau)
            assert abs(risk - model.accepted_risk(tau)) <= 0.01, (coeffs, tau)
            # coverage matches analytic P(u<=tau)=tau (sampling tol)
            assert abs(cov - model.coverage(tau)) <= 0.02, (coeffs, tau)

        # coverage non-decreasing in tau
        assert np.all(np.diff(np.asarray(covs)) >= -1e-9), coeffs


def test_selective_risk_correctness():
    """HARD: selective_risk / coverage / gate correctness on a hand-built array.

    build_gates.md §4: risk = mean loss over exactly the g=1 subset, coverage =
    accepted fraction (atol 1e-12); NaN risk when tau accepts 0 points.
    """
    u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    losses = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

    # gate mask is exactly 1[u <= tau]
    np.testing.assert_array_equal(selection_gate(u, 0.35), np.array([1, 1, 1, 0, 0], bool))

    # tau=0.35 accepts {0.1,0.2,0.3} -> losses {0,1,0} -> risk 1/3, coverage 3/5
    risk, cov = selective_risk(losses, u, 0.35)
    assert abs(risk - 1.0 / 3.0) <= 1e-12
    assert abs(cov - 3.0 / 5.0) <= 1e-12

    # tau below all points -> accept nothing -> NaN risk, 0 coverage
    risk0, cov0 = selective_risk(losses, u, 0.05)
    assert math.isnan(risk0)
    assert cov0 == 0.0

    # tau above all points -> accept all -> risk = mean losses, coverage 1
    risk_all, cov_all = selective_risk(losses, u, 1.0)
    assert abs(risk_all - float(losses.mean())) <= 1e-12
    assert cov_all == 1.0

    # risk_coverage_curve: accept most-confident first; last point = marginal risk
    coverage, curve = risk_coverage_curve(u, losses)
    np.testing.assert_allclose(coverage, np.arange(1, 6) / 5.0)
    assert curve[0] == 0.0                       # most-confident point is correct
    assert abs(curve[-1] - float(losses.mean())) <= 1e-12


def test_selection_bias_trap():
    """HARD: accepted-region calibration controls; full-marginal calibration does not.

    build_gates.md §4: over >=1000 resamples (n_cal=2000) the accepted-region
    ``select_threshold`` controls the *true* accepted risk at tau_hat <= alpha_acc
    in >= (1-delta) of trials, while the deliberately-wrong full-marginal path
    violates it in a materially larger fraction (> 2*delta). Ground truth = the
    analytic accepted risk at each chosen tau_hat.
    """
    model = PolyErrorModel(CFG.coeffs)
    rng = np.random.default_rng(CFG.seed_base + 101)

    correct_viol = 0
    wrong_viol = 0
    wrong_certified = 0
    for _ in range(CFG.T):
        u, losses = model.sample(CFG.n_cal, rng)

        res = select_threshold(u, losses, CFG.alpha_acc, CFG.delta)   # accepted-region
        if res.controlled and model.accepted_risk(res.tau_hat) > CFG.alpha_acc:
            correct_viol += 1

        tau_wrong = _select_threshold_marginal(u, losses, CFG.alpha_acc, CFG.delta,
                                                n_grid=50)
        if tau_wrong is not None:
            wrong_certified += 1
            if model.accepted_risk(tau_wrong) > CFG.alpha_acc:
                wrong_viol += 1

    correct_rate = correct_viol / CFG.T
    wrong_rate = wrong_viol / CFG.T
    print(f"[selection-bias] correct-path violation rate = {correct_rate:.4f} "
          f"(<= delta={CFG.delta}); wrong-path violation rate = {wrong_rate:.4f} "
          f"(> 2*delta={2*CFG.delta}); wrong-path certified {wrong_certified}/{CFG.T}")

    # accepted-region path controls in >= (1 - delta) of trials
    assert correct_rate <= CFG.delta + _wald_band(CFG.delta, CFG.T)
    # full-marginal path violates in a materially larger fraction
    assert wrong_rate > 2 * CFG.delta


def test_rcps_inf_rule_boundary():
    """HARD: RCPS inf-rule returns the loosest gate still below alpha.

    build_gates.md §4: on a deterministic table with hand-known UCB ordering, the
    returned lambda is exactly the boundary (last column with UCB < alpha before
    the first UCB >= alpha). Also the two edge cases (all-pass, all-fail).
    """
    lambdas = np.arange(5)
    mean_ucb = lambda s, d: float(s.mean())          # UCB == the column's value

    # UCBs 0.02 0.05 0.08 | 0.11 0.15 ; alpha=0.10 -> boundary is column 2
    tbl = np.tile(np.array([0.02, 0.05, 0.08, 0.11, 0.15]), (40, 1))
    assert rcps_lhat(tbl, lambdas, 0.10, 0.10, ucb_fn=mean_ucb) == lambdas[2]

    # all columns pass -> loosest lambda
    tbl_pass = np.tile(np.array([0.01, 0.02, 0.03, 0.04, 0.05]), (40, 1))
    assert rcps_lhat(tbl_pass, lambdas, 0.10, 0.10, ucb_fn=mean_ucb) == lambdas[-1]

    # even the tightest column fails -> low-level fallback returns lambdas[0]
    tbl_fail = np.tile(np.array([0.20, 0.30, 0.40, 0.50, 0.60]), (40, 1))
    assert rcps_lhat(tbl_fail, lambdas, 0.10, 0.10, ucb_fn=mean_ucb) == lambdas[0]


def test_select_threshold_abstains_when_uncontrollable():
    """HARD: SelectiveResult(controlled=False, tau_hat=None) when nothing controls.

    build_gates.md §4: when even the tightest certifiable gate has UCB >= alpha the
    selective wrapper abstains on every case. Every accepted point being an error
    (losses all 1) makes every accepted-region UCB ~ 1 >= alpha.
    """
    rng = np.random.default_rng(CFG.seed_base + 202)
    u = rng.uniform(size=CFG.n_cal)
    losses = np.ones(CFG.n_cal)
    res = select_threshold(u, losses, CFG.alpha_acc, CFG.delta)
    assert res.tau_hat is None
    assert res.controlled is False


def test_inf_rule_stops_at_first_failure_no_wander():
    """HARD (min_accept x break-on-first-failure interaction; §4 open question).

    The load-bearing safety property: the RCPS inf-rule must STOP at the first
    certifiable failing gate and must NOT wander past it to certify a looser gate
    that happens to pass. And ``min_accept`` cannot enable such a skip: because the
    accepted count is monotone in tau, min_accept skips ONLY the tightest cells
    (n_acc < min_accept), which are strictly tighter than any certifiable gate --
    so a skipped column can never be the failing certifiable column the break must
    catch.

    Construct a NON-MONOTONE-in-tau accepted risk with a synthetic "bad band":
    clean (low error) for u<0.30, a high-error band on u in [0.30,0.37), clean
    again above. The accepted risk is then low for tight tau (passes), spikes above
    alpha inside the band (fails), and -- because the band mass is small (marginal
    ~0.083 < alpha) -- the fully-diluted risk near tau=1 dips back BELOW alpha (a
    looser gate that genuinely passes its UCB). A correct inf-rule returns a tight
    tau (before the band) and refuses the looser passing gate; a scan that wandered
    past the failure would wrongly certify tau~1. This is exactly the "cannot skip
    a failing column and silently certify" property the DoD open question names.
    """
    rng = np.random.default_rng(CFG.seed_base + 303)
    n = 20000
    u = rng.uniform(size=n)
    band = (u >= 0.30) & (u < 0.37)
    p = np.where(band, 0.92, 0.02)                  # marginal ~= 0.083 < alpha_acc
    losses = (rng.uniform(size=n) < p).astype(float)

    # the trap: a looser gate near full coverage GENUINELY passes its UCB, so a
    # scan that ignored the mid-band failure would wrongly certify it.
    loose_ucb = wsr_ucb(losses[u <= 0.99], CFG.delta)
    assert loose_ucb < CFG.alpha_acc

    # the inf-rule instead stops at the first failure: it certifies a TIGHT gate
    # (well before the bad band), not the looser passing one.
    res = select_threshold(u, losses, CFG.alpha_acc, CFG.delta)
    assert res.controlled is True
    assert res.tau_hat < 0.5                        # tight region, not the tau~1 trap
    # and it genuinely controls the true accepted risk there.
    assert float(losses[u <= res.tau_hat].mean()) <= CFG.alpha_acc


@pytest.mark.parametrize("ucb_name", ["wsr", "hb"])
def test_pac_property_under_exchangeability(ucb_name):
    """HARD: the (alpha_acc, delta) PAC property holds empirically (Bates 2021).

    build_gates.md §4: T>=1000 trials, n_cal=2000, alpha_acc=0.1, delta=0.1 --
    empirical miscoverage mean(true_accepted_risk(tau_hat) > alpha_acc) <=
    delta + 1.96*sqrt(delta(1-delta)/T). Run with both wsr_ucb and hb_ucb.
    (The (alpha_acc,delta) property is doc-backed; the Wald tolerance band is the
    proposed Monte-Carlo encoding.)
    """
    ucb_fn = wsr_ucb if ucb_name == "wsr" else hb_ucb
    model = PolyErrorModel(CFG.coeffs)
    rng = np.random.default_rng(CFG.seed_base + (11 if ucb_name == "wsr" else 22))

    violations = 0
    for _ in range(CFG.T):
        u, losses = model.sample(CFG.n_cal, rng)
        res = select_threshold(u, losses, CFG.alpha_acc, CFG.delta, ucb_fn=ucb_fn)
        if res.controlled and model.accepted_risk(res.tau_hat) > CFG.alpha_acc:
            violations += 1

    miscoverage = violations / CFG.T
    bound = CFG.delta + _wald_band(CFG.delta, CFG.T)
    print(f"[PAC/{ucb_name}] empirical miscoverage = {miscoverage:.4f} <= {bound:.4f}")
    assert miscoverage <= bound


def test_wsr_ucb_valid_and_variance_adaptive():
    """HARD: WSR UCB is a valid (1-delta) upper bound and tighter than HB.

    build_gates.md §4: on samples with known mean mu, over >=1000 trials
    P(wsr_ucb >= mu) >= 1 - delta (validity, doc-anchored), and
    median(wsr_ucb) <= median(hb_ucb) in a low-variance regime (proposed).

    The low-variance regime is a high-concentration Beta on [0,1] with known mean
    mu: values cluster well inside the [0,1] range, so the actual variance is far
    below the worst-case p(1-p) that the range/Bentkus bound must assume at that
    mean. This is exactly where WSR's variance-adaptivity is materially tighter
    (method_note §1.6, §2.2; rcps.py docstring). A Bernoulli regime would NOT show
    it -- Bentkus is already tight for binary data.
    """
    rng = np.random.default_rng(CFG.seed_base + 404)
    mu, kappa, n = 0.10, 500.0, 1000            # Beta(mu*kappa, (1-mu)*kappa): mean mu
    a, b = mu * kappa, (1.0 - mu) * kappa       # var = mu(1-mu)/(kappa+1) << range
    delta = CFG.delta

    covered = 0
    wsr_vals, hb_vals = [], []
    for _ in range(CFG.T):
        x = rng.beta(a, b, size=n)              # in [0,1], known mean = mu, low variance
        w = wsr_ucb(x, delta)
        h = hb_ucb(x, delta)
        wsr_vals.append(w)
        hb_vals.append(h)
        covered += int(w >= mu)

    coverage = covered / CFG.T
    print(f"[WSR] validity coverage = {coverage:.4f} (>= {1-delta}); "
          f"median WSR={np.median(wsr_vals):.4f} <= median HB={np.median(hb_vals):.4f}")
    # validity half (doc-anchored): a valid (1-delta) upper bound
    assert coverage >= 1 - delta
    # variance-adaptive half (proposed): WSR is tighter than the range/Bentkus bound
    assert np.median(wsr_vals) <= np.median(hb_vals)


def test_binom_cdf_matches_bruteforce():
    """HARD: binom_cdf agrees with a brute-force binomial to 1e-9.

    build_gates.md §4 / rcps.py docstring. Also confirms the floor on non-integer k.
    """
    from math import comb

    for k, n, p in [(3, 10, 0.3), (0, 5, 0.5), (5, 5, 0.2), (7, 20, 0.4), (2, 15, 0.1)]:
        brute = sum(comb(n, j) * p ** j * (1 - p) ** (n - j)
                    for j in range(0, int(math.floor(k)) + 1))
        assert abs(binom_cdf(k, n, p) - brute) <= 1e-9, (k, n, p)

    # non-integer k floors to k=3
    assert abs(binom_cdf(3.9, 10, 0.3) - binom_cdf(3, 10, 0.3)) <= 1e-12


def test_fold_disjointness_group_level():
    """HARD: group-id fold-disjointness (set-intersection = 0 artifact).

    build_gates.md §4 / method_note §1.7. A disjoint synthetic split passes and
    emits the printed artifact; an overlapping split must raise.
    """
    rng = np.random.default_rng(CFG.seed_base + 505)
    ids = np.arange(4000)
    rng.shuffle(ids)
    cal_ids, test_ids = ids[:2000], ids[2000:]

    assert assert_group_disjoint(D_cal=cal_ids, test=test_ids, verbose=True)
    assert len(set(cal_ids.tolist()) & set(test_ids.tolist())) == 0

    # negative control: overlapping folds must raise
    with pytest.raises(AssertionError):
        assert_group_disjoint(a=[1, 2, 3], b=[3, 4, 5])


# =========================================================================== #
# REPORTED DIAGNOSTICS (measured/printed, never a pass/fail gate)              #
# =========================================================================== #
def test_diagnostic_degeneracy_guard():
    """DIAGNOSTIC (prereg §6.4): a below-floor-coverage tau_hat is flagged, not
    reported as a favorable low risk. Plumbing self-test both ways; then report
    the realized coverage of a controlled run (never gated)."""
    assert _is_degenerate(0.05, CFG.coverage_min) is True
    assert _is_degenerate(0.50, CFG.coverage_min) is False

    model = PolyErrorModel(CFG.coeffs)
    rng = np.random.default_rng(CFG.seed_base + 606)
    u, losses = model.sample(CFG.n_cal, rng)
    res = select_threshold(u, losses, CFG.alpha_acc, CFG.delta)
    if res.controlled:
        flag = _is_degenerate(res.coverage, CFG.coverage_min)
        print(f"[degeneracy] tau_hat={res.tau_hat:.4f} coverage={res.coverage:.4f} "
              f"degenerate={flag} (floor={CFG.coverage_min}, REPORTED not gated)")


def test_diagnostic_interval_coverage_report():
    """DIAGNOSTIC (prereg §6.8): empirical interval-coverage of the WSR UCB vs
    nominal 1-delta on synthetic data with known ground-truth accepted risk. The
    operative §6.8 action is the relabel, emitted here if it under-covers; never a
    gate (only plumbing sanity is asserted)."""
    model = PolyErrorModel(CFG.coeffs)
    rng = np.random.default_rng(CFG.seed_base + 707)
    T = 500
    covered = controlled = 0
    for _ in range(T):
        u, losses = model.sample(CFG.n_cal, rng)
        res = select_threshold(u, losses, CFG.alpha_acc, CFG.delta)
        if res.controlled:
            controlled += 1
            covered += int(model.accepted_risk(res.tau_hat) <= res.ucb)
    realized = covered / max(controlled, 1)
    label = "nominal interval, coverage validated empirically" if realized < 1 - CFG.delta \
        else "nominal interval"
    print(f"[interval-coverage] realized={realized:.4f} vs nominal={1-CFG.delta} "
          f"over {controlled} controlled trials -> RELABEL: '{label}'")
    assert 0.0 <= realized <= 1.0                 # plumbing only, never gated


def test_diagnostic_aurc_sanity():
    """DIAGNOSTIC (method_note §2.1): AURC < marginal risk -- accepting confident
    cases first beats answering everything.

    This is the exact self-test build_gates §4 specifies ("AURC sanity.
    aurc(u, losses) < mean(losses)"). It is a deterministic STRUCTURAL property of
    an informative u (a plumbing self-test that §1.1 explicitly sanctions), NOT a
    reliability gate on AURC's realized value and NOT used to pass/fail the rung.
    """
    model = PolyErrorModel(CFG.coeffs)
    rng = np.random.default_rng(CFG.seed_base + 808)
    u, losses = model.sample(50_000, rng)
    a = aurc(u, losses)
    m = float(losses.mean())
    print(f"[AURC] aurc={a:.4f} < marginal risk={m:.4f}")
    assert a < m


def test_diagnostic_n_eff_plumbing():
    """DIAGNOSTIC (method_note §1.7, §3.5): at G-A weights are trivially 1 so the
    Kish n_eff equals n. Reported, never gated."""
    for n in (30, 500, CFG.n_cal):
        w = np.ones(n)
        assert _kish_n_eff(w) == float(n)
    # sanity: non-uniform weights strictly reduce n_eff below n
    w = np.array([1.0, 1.0, 5.0, 0.2])
    assert _kish_n_eff(w) < w.size
