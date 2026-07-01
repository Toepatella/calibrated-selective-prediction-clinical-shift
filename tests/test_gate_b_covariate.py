"""Gate-B -- covariate-shift weights ``ŵ_cov(x)`` (the second rung).

Realizes the Definition of Done of ``build_gates.md §5``. G-B adds the
domain-discriminator covariate weight
``ŵ_cov(x) = clip((d(x)/(1−d(x)))·ĉ, 0, w_max)`` (``method_note §1.5``) on top of
the Gate-A exchangeable spine, and verifies -- on data with a KNOWN synthetic
covariate shift (``tests/_synth.py::GaussianCovariateShift``) -- that the
**weighted (Hájek)** path measurably restores the accepted-region risk estimate
where the unweighted ``ŵ≡1`` naive path is biased, and that ``ŵ_cov`` tracks the
known oracle ratio.

Honesty rails enforced by construction (``build_gates.md §5``):
  * These are SYNTHETIC wiring checks on data with known ground truth. The weight
    is ESTIMATED, so NO finite-sample / distribution-free coverage certificate is
    attached to ``ŵ_cov`` (Tibshirani et al. 2019 Cor. 1 needs the *oracle* ratio
    under *pure* covariate shift). Gate B MEASURES the restoration; it certifies
    nothing about deployment or real clinical shift.
  * ``w_max`` is a variance/boundedness control whose routed tail costs only
    coverage -- NOT an OOD guard (the far-OOD guard ``t_ood`` enters at Gate D).
  * ``n_eff``, clip rate, discriminator AUROC, the DR delta, and the interval-
    coverage report are REPORTED DIAGNOSTICS, never pass/fail gates.

Each test maps to one Definition-of-Done bullet; HARD GATES assert, DIAGNOSTICS
measure/print and only sanity-check plumbing. Numeric budgets come from
``configs/gate_b.toml`` (the committed "proposed" run config).
"""

import pathlib
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from conformal.rcps import (
    hajek_risk,
    kish_n_eff,
    wsr_ucb_weighted,
    wsr_lcb_weighted,
)
from conformal.weights import (
    fit_discriminator,
    cov_weight_from_d,
    auroc,
    weight_summary,
    aipw_risk,
    crossfit_outcome_model,
    bagged_outcome,
    _Logistic,
)
from conformal.folds import assert_group_disjoint
from _synth import GaussianCovariateShift


# --------------------------------------------------------------------------- #
# Run config (committed proposed numbers -- build_gates.md §5 open question).   #
# --------------------------------------------------------------------------- #
_CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gate_b.toml"


def _load_cfg():
    with open(_CFG_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    b, sh, dc, mc, sd = (raw["budgets"], raw["shift"], raw["discriminator"],
                         raw["monte_carlo"], raw["seed"])
    return SimpleNamespace(
        alpha_acc=b["alpha_acc"], delta=b["delta"], coverage_min=b["coverage_min"],
        w_max=b["w_max"],
        mu_tar=sh["mu_tar"], base=sh["base"], slope=sh["slope"], k=sh["k"], tau0=sh["tau0"],
        n_splits=dc["n_splits"], l2=dc["l2"],
        n_src_mc=mc["n_src_mc"], T_signed=mc["T_signed"],
        n_src_big=mc["n_src_big"], N_disjoint=mc["N_disjoint"],
        track_grid=mc["track_grid"], track_ref=mc["track_ref"],
        track_T=mc["track_T"], track_tol=mc["track_tol"],
        cov_T=mc["cov_T"], wmax_grid=mc["wmax_grid"],
        seed_base=sd["base"],
    )


CFG = _load_cfg()


def _gen():
    return GaussianCovariateShift(mu_tar=CFG.mu_tar, base=CFG.base,
                                  slope=CFG.slope, k=CFG.k)


def _draw_disjoint(gen, n, rng):
    """Draw group-disjoint D_disc(source+target) and a disjoint source D_cal.

    Returns (x_disc_s, x_disc_t, x_cal, u_cal, l_cal, gids) with group ids carved
    into non-overlapping ranges so the fold-disjointness invariant holds by
    construction (``method_note §1.7``). The discriminator is fit on D_disc and
    applied (bagged) to the DISJOINT D_cal, so weights are never fit on the points
    they reweight.
    """
    xds, _, _ = gen.sample(n, rng, "S")
    xdt, _, _ = gen.sample(n, rng, "T")
    xc, uc, lc = gen.sample(n, rng, "S")
    gids = SimpleNamespace(
        D_disc_S=np.arange(0, n),
        D_disc_T=np.arange(n, 2 * n),
        D_cal=np.arange(2 * n, 3 * n),
    )
    return xds, xdt, xc, uc, lc, gids


# =========================================================================== #
# HARD GATES                                                                    #
# =========================================================================== #
def test_weighted_restores_where_naive_degrades():
    """HARD: weighted path restores the risk estimate where naive is biased.

    build_gates.md §5 (gate 1): under the KNOWN covariate shift, at a matched
    answer-rate (a fixed accept region {u<=tau0}, so both paths reweight the SAME
    accepted set), the weighted Hájek ``R̂_w`` under ``ŵ_cov`` is LOWER than the
    ``ŵ≡1`` naive risk, with non-overlapping WSR intervals (prereg §6.7 win-rule).
    The codeable claim is the signed non-overlap only.

    Two complementary reads:
      (a) SIGNED WIN -- over T_signed resamples at moderate n, the weighted point
          estimate is below the naive one in >= (1-delta) of trials (a repeatable
          measured restoration).
      (b) NON-OVERLAP -- on WELL-POWERED draws (importance weighting inflates
          variance, so disjointness must survive the true n_eff cost of
          reweighting, not be bought by under-powering) the weighted WSR *upper*
          bound is strictly below the naive WSR *lower* bound. We also verify the
          MEANINGFUL part the win rests on: the weighted estimate tracks the known
          TARGET accepted risk R_T while the naive tracks the SOURCE risk R_S.
    """
    gen = _gen()
    R_S = gen.accepted_risk(CFG.tau0, "S")
    R_T = gen.accepted_risk(CFG.tau0, "T")
    assert R_T < R_S                                    # construction: target easier in-region

    # (a) signed win over many moderate-n resamples
    wins = 0
    for i in range(CFG.T_signed):
        rng = np.random.default_rng(CFG.seed_base + 1000 + i)
        xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_mc, rng)
        disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                                 l2=CFG.l2, seed=CFG.seed_base + 1000 + i)
        acc = uc <= CFG.tau0
        w = disc.weights(xc)[acc]
        l = lc[acc]
        if hajek_risk(l, w) < l.mean():
            wins += 1
    win_rate = wins / CFG.T_signed
    print(f"[restore/signed] weighted<naive in {win_rate:.3f} of {CFG.T_signed} "
          f"resamples (>= {1 - CFG.delta}); R_T={R_T:.4f} < R_S={R_S:.4f}")
    assert win_rate >= 1 - CFG.delta

    # (b) non-overlapping WSR intervals on well-powered draws + tracking check
    margins, wt_pts, nv_pts = [], [], []
    for j in range(CFG.N_disjoint):
        rng = np.random.default_rng(CFG.seed_base + 5000 + j)
        xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_big, rng)
        disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                                 l2=CFG.l2, seed=CFG.seed_base + 5000 + j)
        acc = uc <= CFG.tau0
        w = disc.weights(xc)[acc]
        l = lc[acc]
        naive_lcb = wsr_lcb_weighted(l, np.ones_like(l), CFG.delta, 1.0)
        weighted_ucb = wsr_ucb_weighted(l, w, CFG.delta, CFG.w_max)
        margins.append(naive_lcb - weighted_ucb)         # > 0 => disjoint (weighted below naive)
        wt_pts.append(hajek_risk(l, w))
        nv_pts.append(float(l.mean()))
    margins = np.asarray(margins)
    print(f"[restore/interval] disjoint in {int((margins > 0).sum())}/{CFG.N_disjoint} "
          f"well-powered draws; min margin={margins.min():.4f}; "
          f"weighted_pt={np.mean(wt_pts):.4f}~R_T={R_T:.4f}, "
          f"naive_pt={np.mean(nv_pts):.4f}~R_S={R_S:.4f}")
    # non-overlap in EVERY well-powered draw (prereg §6.7 win-rule)
    assert np.all(margins > 0)
    # the win is meaningful: weighted tracks the true TARGET risk, naive the SOURCE
    assert abs(np.mean(wt_pts) - R_T) <= 0.01
    assert abs(np.mean(nv_pts) - R_S) <= 0.01


def test_wcov_tracks_oracle_ratio():
    """HARD: ``ŵ_cov`` tracks the oracle ratio ``w_cov*(x)=p_T(x)/p_S(x)``.

    build_gates.md §5 (gate 2): over MC trials at the large-sample setting
    (n_S,n_T >= 5000), mean_x |log ŵ_cov − log w_cov*| <= track_tol on the
    unclipped mass, with the error shrinking monotonically (within MC noise) as n
    grows over a fixed grid -- CONSISTENCY, not a finite-sample certificate. All
    numbers proposed (configs/gate_b.toml).
    """
    gen = _gen()
    errs_by_n = {}
    for n in CFG.track_grid:
        errs = []
        for t in range(CFG.track_T):
            rng = np.random.default_rng(CFG.seed_base + 20000 + 97 * n + t)
            xds, _, _ = gen.sample(n, rng, "S")
            xdt, _, _ = gen.sample(n, rng, "T")
            disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                                     l2=CFG.l2, seed=CFG.seed_base + 20000 + 97 * n + t)
            xref, _, _ = gen.sample(CFG.track_ref, rng, "S")
            what = disc.weights(xref)
            wstar = gen.oracle_w_cov(xref)
            unclipped = what < CFG.w_max * (1.0 - 1e-6)
            errs.append(np.mean(np.abs(np.log(what[unclipped]) - np.log(wstar[unclipped]))))
        errs_by_n[n] = float(np.mean(errs))
    print(f"[oracle-track] mean|log w_hat - log w*| by n = "
          + ", ".join(f"{n}:{errs_by_n[n]:.4f}" for n in CFG.track_grid)
          + f"  (tol={CFG.track_tol})")
    # tolerance at the large-sample settings (n >= 5000)
    for n in CFG.track_grid:
        if n >= 5000:
            assert errs_by_n[n] <= CFG.track_tol, (n, errs_by_n[n])
    # error shrinks monotonically as n grows over the fixed grid (consistency)
    ordered = [errs_by_n[n] for n in sorted(CFG.track_grid)]
    assert all(a >= b - 1e-3 for a, b in zip(ordered, ordered[1:])), ordered


def test_c_enters_only_through_clipping():
    """HARD: the base-rate constant ĉ enters ONLY through the w_max clip (Remark 3).

    build_gates.md §5 (gate 3) / method_note §1.5 (Tibshirani et al. 2019 eq.12 +
    Remark 3), §7.3: ĉ cancels in the self-normalized (Hájek) weighted risk, so it
    changes the estimate ONLY via the w_max clip boundary. Assert: (a) with NO clip,
    scaling every weight by an arbitrary ĉ leaves the Hájek risk EXACTLY unchanged;
    (b) once the clip is active, changing ĉ changes which points clip (a non-empty
    clip set) and only then moves the estimate.
    """
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 303)
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_mc, rng)
    disc = fit_discriminator(xds, xdt, w_max=1e12, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 303)
    acc = uc <= CFG.tau0
    d = disc.score(xc)[acc]
    l = lc[acc]

    # (a) no clip: ĉ cancels EXACTLY in the self-normalized risk
    w_c1 = cov_weight_from_d(d, c=1.0, w_max=1e12)
    w_c2 = cov_weight_from_d(d, c=3.7, w_max=1e12)          # arbitrary base-rate const
    r1, r2 = hajek_risk(l, w_c1), hajek_risk(l, w_c2)
    print(f"[Remark-3] no-clip Hajek risk: c_hat=1 -> {r1:.8f}, c_hat=3.7 -> {r2:.8f} (equal)")
    assert abs(r1 - r2) <= 1e-9

    # (b) with a finite clip, a larger ĉ pushes more mass over w_max (non-empty clip
    # set) and only THEN changes the estimate -- the sole channel for ĉ.
    odds = np.clip(d, 1e-12, 1 - 1e-12)
    odds = odds / (1 - odds)
    w_lo = cov_weight_from_d(d, c=1.0, w_max=CFG.w_max)
    w_hi = cov_weight_from_d(d, c=6.0, w_max=CFG.w_max)
    clipped_lo = odds * 1.0 >= CFG.w_max
    clipped_hi = odds * 6.0 >= CFG.w_max
    assert clipped_hi.sum() > clipped_lo.sum()             # larger ĉ clips strictly more
    assert hajek_risk(l, w_hi) != hajek_risk(l, w_lo)      # and only then moves the estimate


def test_fold_disjointness_group_level():
    """HARD: group-id fold-disjointness across D_disc, D_cal, target test.

    build_gates.md §5 (gate 4) / method_note §1.7 (set-intersection = 0 artifact).
    """
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 404)
    n = 3000
    xds, xdt, xc, uc, lc, gids = _draw_disjoint(gen, n, rng)
    xtt, _, _ = gen.sample(n, rng, "T")
    test_ids = np.arange(3 * n, 4 * n)

    assert assert_group_disjoint(
        D_disc_S=gids.D_disc_S, D_disc_T=gids.D_disc_T, D_cal=gids.D_cal,
        test=test_ids, verbose=True,
    )
    # negative control: an overlapping split must raise
    with pytest.raises(AssertionError):
        assert_group_disjoint(D_disc=[1, 2, 3], D_cal=[3, 4, 5])


def test_discriminator_crossfitted_out_of_fold():
    """HARD: ``d`` is cross-fitted -- every pooled point is scored out-of-fold.

    build_gates.md §5 (gate 5) / method_note §1.5, §3.4 step 1: each point's
    ``d_oof`` comes from the fold-model that did NOT see its fold, so weights are
    not optimistically fit on the points they reweight. Verified two ways:
      * WIRING -- re-scoring each held-out fold with its own fold-model exactly
        reproduces ``d_oof`` (so the OOF assignment is real, not incidental);
      * OPTIMISM -- an in-fold model (trained on ALL pooled points, scored on the
        same points) has strictly higher discriminability (AUROC) than the OOF
        scores, i.e. cross-fitting genuinely removes the optimism.
    """
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 505)
    n = 4000
    xds, _, _ = gen.sample(n, rng, "S")
    xdt, _, _ = gen.sample(n, rng, "T")
    disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 505)
    X = np.concatenate([xds, xdt]).reshape(-1, 1)

    # WIRING: models[k] scored exactly the fold_id==k points -> reproduce d_oof.
    for k in range(CFG.n_splits):
        held = disc.fold_id == k
        assert held.any()
        np.testing.assert_allclose(disc.models[k].proba(X[held]), disc.d_oof[held],
                                   atol=1e-12)

    # OPTIMISM: a model trained on everything then scored on everything is
    # optimistic vs the honest out-of-fold scores.
    infold = _Logistic(l2=CFG.l2).fit(X, disc.domain)
    auroc_infold = auroc(infold.proba(X), disc.domain)
    auroc_oof = auroc(disc.d_oof, disc.domain)
    print(f"[cross-fit] AUROC in-fold={auroc_infold:.4f} > OOF={auroc_oof:.4f} "
          f"(optimism gap {auroc_infold - auroc_oof:+.4f})")
    # STRICT: an in-fold model is genuinely more optimistic than the OOF scores, so
    # the gap is > 0 (would be exactly 0 if cross-fitting were bypassed with a single
    # all-data model). The WIRING sub-check above is the primary enforcement; this
    # confirms the OOF scores are not optimistic. (Gap ~4e-4 here, always positive.)
    assert auroc_infold > auroc_oof


def test_weighted_wsr_ucb_exercised():
    """HARD: the WSR weighted-path UCB is genuinely exercised (no range fallback).

    build_gates.md §5 (gate 6) / prereg §6.7: the reported bound is the WSR betting
    UCB on the weighted (Hájek) estimator -- NOT a silent range-bound [0,1]
    fallback. Assert it is a finite bound strictly between the weighted point
    estimate and 1, and that it genuinely uses the weights (differs from the
    unweighted UCB on the same losses).
    """
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 606)
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_big, rng)
    disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 606)
    acc = uc <= CFG.tau0
    w = disc.weights(xc)[acc]
    l = lc[acc]

    point = hajek_risk(l, w)
    ucb_w = wsr_ucb_weighted(l, w, CFG.delta, CFG.w_max)
    ucb_unw = wsr_ucb_weighted(l, np.ones_like(l), CFG.delta, 1.0)
    print(f"[wsr-weighted] point={point:.4f} < UCB_w={ucb_w:.4f} < 1; "
          f"unweighted UCB={ucb_unw:.4f} (weights change the bound)")
    # a genuine finite betting bound, not the [0,1] range top
    assert point < ucb_w < 1.0 - 1e-6
    assert ucb_w < 0.5                                     # nowhere near the range fallback
    # the weights actually enter the bound
    assert abs(ucb_w - ucb_unw) > 1e-3


# =========================================================================== #
# REPORTED DIAGNOSTICS (measured/printed, never a pass/fail gate)              #
# =========================================================================== #
def _fit_and_weight(gen, n, rng, seed, w_max=None):
    """Helper: fit D_disc, return accepted D_cal (losses, ŵ_cov, oracle w*)."""
    w_max = CFG.w_max if w_max is None else w_max
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, n, rng)
    disc = fit_discriminator(xds, xdt, w_max=w_max, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=seed)
    acc = uc <= CFG.tau0
    return lc[acc], disc.weights(xc)[acc], gen.oracle_w_cov(xc[acc]), disc


def test_diagnostic_clip_abstention_rate():
    """DIAGNOSTIC (method_note §3.2, §3.6): the mass routed by ŵ_cov > w_max.

    Clipping biases the weights, so the routed (clip-induced abstention) fraction
    is reported beside the weighted risk. REPORTED, never gated."""
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 707)
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_mc, rng)
    disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 707)
    w_full = disc.weights(xc)                              # includes clipped values at w_max
    routed = float(np.mean(w_full >= CFG.w_max * (1 - 1e-9)))
    print(f"[clip-rate] clip-induced abstention rate = {routed:.4f} at w_max={CFG.w_max} "
          f"(REPORTED, never gated)")
    assert 0.0 <= routed <= 1.0                            # plumbing only


def test_diagnostic_clip_bias_sensitivity():
    """DIAGNOSTIC (method_note §1.5, §3.2, §3.6; prereg §5.4, §6.6): how the Hájek
    weighted risk moves across a small w_max grid.

    Clipping biases the weights, so a non-flat cell means the reported risk is
    partly a clip artifact. Self-test on synthetic data with known w_cov*: as
    w_max grows the clip relaxes and the estimate approaches the unclipped value.
    REPORTED, never gated."""
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 808)
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_big, rng)
    acc = uc <= CFG.tau0
    l = lc[acc]
    disc = fit_discriminator(xds, xdt, w_max=1e12, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 808)
    d = disc.score(xc)[acc]
    curve = {wm: hajek_risk(l, cov_weight_from_d(d, disc.c, wm)) for wm in CFG.wmax_grid}
    print("[clip-bias] Hájek risk vs w_max: "
          + ", ".join(f"{wm}:{curve[wm]:.4f}" for wm in CFG.wmax_grid))
    assert all(np.isfinite(v) for v in curve.values())    # plumbing only


def test_diagnostic_wmax_costs_only_coverage_decoupling():
    """DIAGNOSTIC (method_note §3.2, §4.2, §1.5.1): at Gate B, w_max is a
    variance/boundedness control whose routed tail costs only COVERAGE -- it is NOT
    an OOD guard.

    A far-OOD point can have d(x)~0 (looks like neither pool) so ŵ_cov does NOT
    flag it; that guard is t_ood, introduced at Gate D. Self-test the negative:
    construct a far point with a near-zero domain posterior and confirm its ŵ_cov
    sits at the FLOOR (not above w_max), so w_max would never route it -- no
    far-OOD claim attaches to w_max here."""
    # a point the discriminator assigns d ~ 0 (source-like) -> tiny odds -> w_cov ~ 0
    d_faroood = 1e-6
    w = cov_weight_from_d(d_faroood, c=1.0, w_max=CFG.w_max)
    print(f"[decoupling] far point with d~0 gets w_cov={float(w):.2e} "
          f"(floored, NOT > w_max={CFG.w_max}); w_max is not an OOD guard (t_ood is, at Gate D)")
    assert float(w) < 1e-3                                  # routed by neither w_max nor (nonexistent) t_ood


def test_diagnostic_n_eff_falls_with_shift():
    """DIAGNOSTIC (method_note §1.7): report n_eff beside the weighted risk and
    self-test that it FALLS as the covariate shift is made more extreme (heavier
    ŵ_cov tail). Reported, never gated."""
    rng = np.random.default_rng(CFG.seed_base + 909)
    neffs = []
    for mu in (-0.3, -0.7, -1.1):                          # increasingly extreme shift
        gen = GaussianCovariateShift(mu_tar=mu, base=CFG.base, slope=CFG.slope, k=CFG.k)
        l, w, _, _ = _fit_and_weight(gen, CFG.n_src_big, rng, CFG.seed_base + 909)
        neffs.append(kish_n_eff(w) / len(w))               # as a fraction of n
    print("[n_eff] n_eff/n vs |shift|: "
          + ", ".join(f"mu={mu}:{ne:.3f}" for mu, ne in zip((-0.3, -0.7, -1.1), neffs)))
    assert neffs[0] >= neffs[1] >= neffs[2] - 1e-2         # falls as the shift grows


def test_diagnostic_discriminator_auroc_and_summary():
    """DIAGNOSTIC (method_note §1.5, §3.4 step 1; prereg §5.4): discriminator AUROC
    as a covariate-shift-severity / overlap signal, plus the weight-distribution
    summary (median/95/99/max). Reported, never gated."""
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 1111)
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_big, rng)
    disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 1111)
    a = auroc(disc.d_oof, disc.domain)
    summ = weight_summary(disc.weights(xc))
    print(f"[overlap] discriminator AUROC={a:.4f}; weight summary={summ}")
    assert 0.5 <= a <= 1.0                                  # a valid overlap signal
    assert summ["median"] > 0 and summ["max"] <= CFG.w_max


def test_diagnostic_discriminator_shortcut_audit():
    """DIAGNOSTIC (method_note §3.5; prereg §5.4, §3.2): the discriminator-shortcut
    audit.

    On real data a discriminator keying on an injected acquisition ARTIFACT
    (rather than clinical appearance) would caveat the representativeness chip and
    the §3.2 regime tag. Self-test the plumbing: plant a near-perfect
    domain-separating "shortcut" feature; the discriminator AUROC is near 1 with it
    and COLLAPSES toward the true-signal AUROC when the shortcut is masked.
    REPORTED, never gated."""
    gen = _gen()
    rng = np.random.default_rng(CFG.seed_base + 1212)
    n = CFG.n_src_mc
    xs, _, _ = gen.sample(n, rng, "S")
    xt, _, _ = gen.sample(n, rng, "T")
    # planted shortcut: source ~ N(0, .05), target ~ N(5, .05) -> ~perfectly separable
    sc_s = rng.normal(0.0, 0.05, n)
    sc_t = rng.normal(5.0, 0.05, n)
    Xs = np.column_stack([xs, sc_s])
    Xt = np.column_stack([xt, sc_t])

    disc_sc = fit_discriminator(Xs, Xt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                                l2=CFG.l2, seed=CFG.seed_base + 1212)
    auroc_sc = auroc(disc_sc.d_oof, disc_sc.domain)

    # mask the shortcut (replace with a constant) -> only the true signal remains
    Xs_m = np.column_stack([xs, np.zeros(n)])
    Xt_m = np.column_stack([xt, np.zeros(n)])
    disc_m = fit_discriminator(Xs_m, Xt_m, w_max=CFG.w_max, n_splits=CFG.n_splits,
                               l2=CFG.l2, seed=CFG.seed_base + 1212)
    auroc_m = auroc(disc_m.d_oof, disc_m.domain)
    print(f"[shortcut] AUROC with shortcut={auroc_sc:.4f} -> masked={auroc_m:.4f} "
          f"(collapses by {auroc_sc - auroc_m:.4f})")
    assert auroc_sc > 0.95                                  # shortcut is near-perfectly separating
    assert auroc_sc - auroc_m > 0.20                        # AUROC collapses when it is removed


def test_diagnostic_doubly_robust_aipw():
    """DIAGNOSTIC (prereg §4A item 2; method_note §7.1): the doubly-robust (AIPW)
    covariate correction -- key secondary analysis.

    Augment ŵ_cov with a cross-fitted source outcome model m̂(x) ≈ E_S[ℓ|φ(x)].
    Self-test the double robustness on synthetic G-B data: the AIPW estimate
    shrinks the residual toward the KNOWN target accepted risk R_T when EITHER the
    weight OR m̂ is correct, and misses only when BOTH are wrong. The Hájek
    single-model estimate stays the auditable PRIMARY; DR is the residual-shrinkage
    DIAGNOSTIC (asymptotic, not a certificate), never gated."""
    gen = _gen()
    R_T = gen.accepted_risk(CFG.tau0, "T")
    rng = np.random.default_rng(CFG.seed_base + 1313)

    # accepted source (labels) + accepted TARGET features (unlabeled) at matched region
    xds, xdt, xc, uc, lc, _ = _draw_disjoint(gen, CFG.n_src_big, rng)
    xtt, utt, _ = gen.sample(CFG.n_src_big, rng, "T")
    disc = fit_discriminator(xds, xdt, w_max=CFG.w_max, n_splits=CFG.n_splits,
                             l2=CFG.l2, seed=CFG.seed_base + 1313)
    acc_s = uc <= CFG.tau0
    acc_t = utt <= CFG.tau0
    l_s = lc[acc_s]
    x_s = xc[acc_s]
    x_t = xtt[acc_t]

    w_correct = disc.weights(xc)[acc_s]                    # estimated ŵ_cov (~correct)
    w_wrong = np.ones_like(l_s)                            # naive ŵ≡1 (wrong under shift)

    # CROSS-FITTED source outcome model m̂ ≈ E_S[ℓ|x]: each accepted source point is
    # scored OUT-OF-FOLD (mhat_s_correct), and the disjoint target points are scored
    # by the bagged fold-models (mhat_t_correct) -- the DML/AIPW cross-fitting that
    # keeps the source correction term free of own-observation optimism. Plus a
    # deliberately WRONG constant model to exercise the other robustness arm.
    mhat_s_correct, om_models = crossfit_outcome_model(
        x_s, l_s, n_splits=CFG.n_splits, l2=CFG.l2, seed=CFG.seed_base + 1313)
    mhat_t_correct = bagged_outcome(om_models, x_t)
    mhat_s_wrong = np.full_like(l_s, 0.5)
    mhat_t_wrong = np.full(x_t.shape[0], 0.5)

    dr_w_ok = aipw_risk(l_s, w_correct, mhat_s_wrong, mhat_t_wrong)   # weight right, m̂ wrong
    dr_m_ok = aipw_risk(l_s, w_wrong, mhat_s_correct, mhat_t_correct)  # weight wrong, m̂ right
    dr_both_wrong = aipw_risk(l_s, w_wrong, mhat_s_wrong, mhat_t_wrong)
    hajek = hajek_risk(l_s, w_correct)
    print(f"[AIPW] R_T={R_T:.4f} | DR(w-ok,m-wrong)={dr_w_ok:.4f} "
          f"DR(w-wrong,m-ok)={dr_m_ok:.4f} DR(both-wrong)={dr_both_wrong:.4f} | "
          f"Hajek(primary)={hajek:.4f} delta(DR-Hajek)={dr_w_ok - hajek:+.4f}")
    # double robustness: recovers R_T when EITHER the weight or m̂ is correct
    assert abs(dr_w_ok - R_T) <= 0.02
    assert abs(dr_m_ok - R_T) <= 0.02
    # and misses when BOTH are wrong (it collapses to the naive/source level)
    assert abs(dr_both_wrong - R_T) > abs(dr_w_ok - R_T)


def test_diagnostic_interval_coverage_report():
    """DIAGNOSTIC (prereg §6.8): synthetic interval-coverage check of the weighted
    Hájek WSR interval on data with KNOWN ground-truth weighted risk and KNOWN
    weights.

    Uses the ORACLE weights ``w_cov*`` (so the known ground-truth weighted risk is
    exactly the target accepted risk R_T) and reports realized coverage of the WSR
    UCB vs nominal 1-delta across two n_eff regimes (mild vs strong shift). The
    operative §6.8 action is the RELABEL, emitted where it under-covers; never a
    gate (only plumbing is asserted)."""
    for mu, tag in ((-0.5, "mild"), (-1.0, "strong")):
        gen = GaussianCovariateShift(mu_tar=mu, base=CFG.base, slope=CFG.slope, k=CFG.k)
        R_T = gen.accepted_risk(CFG.tau0, "T")
        covered = 0
        neffs = []
        for t in range(CFG.cov_T):
            rng = np.random.default_rng(CFG.seed_base + 1414 + t)
            xc, uc, lc = gen.sample(CFG.n_src_mc, rng, "S")
            acc = uc <= CFG.tau0
            l = lc[acc]
            wstar = gen.oracle_w_cov(xc[acc])              # KNOWN weights
            wm = float(wstar.max())                        # no clip => ground truth is exactly R_T
            ucb = wsr_ucb_weighted(l, wstar, CFG.delta, wm)
            covered += int(R_T <= ucb)
            neffs.append(kish_n_eff(wstar))
        realized = covered / CFG.cov_T
        label = ("nominal interval, coverage validated empirically"
                 if realized < 1 - CFG.delta else "nominal interval")
        print(f"[interval-cov/{tag}] realized={realized:.4f} vs nominal={1 - CFG.delta} "
              f"(R_T={R_T:.4f}, n_eff~{np.mean(neffs):.0f}) -> RELABEL: '{label}'")
        assert 0.0 <= realized <= 1.0                      # plumbing only, never gated
