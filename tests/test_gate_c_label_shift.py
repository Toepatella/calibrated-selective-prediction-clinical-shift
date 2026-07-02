"""Gate-C -- label / prevalence-shift weights + the ``Ẑ``-divide combine (third rung).

Realizes the Definition of Done of ``build_gates.md §6``. G-C adds, on top of the
Gate-A exchangeable spine and the Gate-B covariate weight ``ŵ_cov(x)``:

  * BBSE ``ŵ_lab = Ĉ_S⁻¹ q̂_T`` (Lipton, Wang & Smola 2018) with simplex
    projection + floor/ceiling,
  * MLLS + BCTS prevalence estimation (Alexandari, Kundaje & Shrikumar 2020),
  * the per-``x`` double-count corrector ``Ẑ(x) = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}``
    (the Saerens, Latinne & Decaestecker 2002 prior-shift normalizer) and the
    combine identity ``ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)`` -- **not** the naïve
    product. This identity is **not novel**: it is Tasche (2022, arXiv:2207.14514)
    Thm 4 / Cor 4 at ``ρ=1``, exact under the invariant-density-ratio condition
    (STRICTLY STRONGER than the He et al. 2022 FJS sense ``w(x,y)=U(x)·V(y)``),

and verifies -- on data with a KNOWN synthetic prevalence shift
(``tests/_synth.py::LabelShiftMixture``) -- that each cited method reproduces its
stated property against ground truth.

Honesty rails enforced by construction (``build_gates.md §6``):
  * SYNTHETIC wiring checks on known ground truth. BBSE/MLLS carry only
    *consistency* under label shift given an invertible ``Ĉ_S`` -- NO ``(α,δ)`` /
    distribution-free certificate for ``w_lab`` or the combined weight.
  * The combined covariate+label weight is NOT identifiable from unlabeled target
    alone; the factorizable-shift premise is a modeling choice whose residual is
    only *measured*, never certified.
  * The weights combine by the ``Ẑ``-DIVIDE, never by multiplication (that
    double-counts).
  * The ``Ẑ(x)`` softmax plug-in bias is NOT assumed removed by recalibration --
    BCTS reduces but does not eliminate it (non-vanishing, amplified on rare /
    high-``w_lab`` classes); its residual is *reported* via a sensitivity sweep
    (``test_diagnostic_z_plugin_bias_sensitivity``), never certified away.
  * ``κ(Ĉ_S)``, the ``q̂_T`` vs ``Ĉ_S p̂_T`` check, ``n_eff``, the ``Ẑ`` plug-in
    sensitivity, and the interval-coverage report are REPORTED DIAGNOSTICS, never
    pass/fail gates.

Each test maps to one Definition-of-Done bullet; HARD GATES assert, DIAGNOSTICS
measure/print and only sanity-check plumbing. Numeric budgets come from
``configs/gate_c.toml`` (the committed "proposed" run config).
"""

import pathlib
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from conformal.rcps import (
    hajek_risk,
    kish_n_eff,
    wsr_ucb_weighted,
    wsr_lcb_weighted,
)
from conformal.label_shift import (
    confusion_matrix,
    pred_hist,
    project_to_simplex,
    bbse_weights,
    fit_bcts,
    recalibrate_softmax,
    softmax,
    mlls_em,
    mlls_weights,
    mapls_em,
    z_corrector,
    combine_weights,
    kappa_cs,
    bbse_consistency,
    measure_slice_joint_weight,
    residual_on_labeled_target,
)
from conformal.weights import fit_discriminator
from conformal.folds import assert_group_disjoint
from _synth import LabelShiftMixture


# --------------------------------------------------------------------------- #
# Run config (committed proposed numbers -- build_gates.md §6 open question).   #
# --------------------------------------------------------------------------- #
_CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gate_c.toml"


def _load_cfg():
    with open(_CFG_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    return SimpleNamespace(
        b=SimpleNamespace(**raw["budgets"]),
        k2=SimpleNamespace(**raw["k2"]),
        k3=SimpleNamespace(**raw["k3"]),
        bcts=SimpleNamespace(**raw["bcts"]),
        mc=SimpleNamespace(**raw["monte_carlo"]),
        seed=raw["seed"]["base"],
    )


CFG = _load_cfg()


def _gen_k2(**over):
    """Primary K=2 generator from configs/gate_c.toml [k2], with optional overrides."""
    kw = dict(means=CFG.k2.means, p_s=CFG.k2.p_s, p_t=CFG.k2.p_t, rho=CFG.k2.rho)
    kw.update(over)
    return LabelShiftMixture(**kw)


def _gen_k3(**over):
    kw = dict(means=CFG.k3.means, p_s=CFG.k3.p_s, p_t=CFG.k3.p_t, rho=CFG.k3.rho)
    kw.update(over)
    return LabelShiftMixture(**kw)


# =========================================================================== #
# HARD GATES                                                                    #
# =========================================================================== #
def test_bbse_recovers_label_ratio():
    """HARD (gate 1): BBSE recovers the KNOWN label ratio ``w_lab*(y)=p_T/p_S``.

    build_gates.md §6 gate 1: over MC trials with fresh D_bbse^src / D_tar,
    ``max_y|ŵ_lab - w_lab*| <= bbse_tol`` at the large-sample setting (n >= 5000),
    with the mean error shrinking monotonically (within MC noise) as n grows over
    the fixed grid (250/1000/5000) -- CONSISTENCY, not a finite-sample certificate.
    A K=3 regime is checked too (build_gates.md §6: "start K=2 ... then K>=3").
    """
    gen = _gen_k2()
    wstar = gen.oracle_w_lab()
    err_by_n = {}
    for n in CFG.mc.bbse_grid:
        errs = []
        for t in range(CFG.mc.bbse_T):
            r = np.random.default_rng(CFG.seed + 1000 + 131 * n + t)
            _, ys, yhs, _, _ = gen.sample(n, r, "S")
            _, _, yht, _, _ = gen.sample(n, r, "T")
            C = confusion_matrix(ys, yhs, gen.K)
            q = pred_hist(yht, gen.K)
            w, _ = bbse_weights(C, q, gen.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
            errs.append(float(np.max(np.abs(w - wstar))))
        err_by_n[n] = float(np.mean(errs))
    print("[bbse-recover/K2] mean max|w_hat - w*| by n = "
          + ", ".join(f"{n}:{err_by_n[n]:.4f}" for n in CFG.mc.bbse_grid)
          + f"  (tol={CFG.mc.bbse_tol})")
    # tolerance at the large-sample settings (n >= 5000)
    for n in CFG.mc.bbse_grid:
        if n >= 5000:
            assert err_by_n[n] <= CFG.mc.bbse_tol, (n, err_by_n[n])
    # consistency: error shrinks monotonically as n grows over the fixed grid
    ordered = [err_by_n[n] for n in sorted(CFG.mc.bbse_grid)]
    assert all(a >= b - 1e-3 for a, b in zip(ordered, ordered[1:])), ordered

    # K=3 secondary regime at large sample
    g3 = _gen_k3()
    w3 = g3.oracle_w_lab()
    e3 = []
    for t in range(CFG.mc.k3_T):
        r = np.random.default_rng(CFG.seed + 2000 + t)
        _, ys, yhs, _, _ = g3.sample(CFG.mc.k3_n, r, "S")
        _, _, yht, _, _ = g3.sample(CFG.mc.k3_n, r, "T")
        C = confusion_matrix(ys, yhs, g3.K)
        q = pred_hist(yht, g3.K)
        w, _ = bbse_weights(C, q, g3.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
        e3.append(float(np.max(np.abs(w - w3))))
    print(f"[bbse-recover/K3] w_lab*={w3}; mean max|w_hat-w*| at n={CFG.mc.k3_n} = {np.mean(e3):.4f}")
    assert np.mean(e3) <= CFG.mc.bbse_tol


def test_mlls_bcts_recovers_prevalence():
    """HARD (gate 2): MLLS+BCTS recovers the KNOWN prevalence ``p_T(y)``.

    build_gates.md §6 gate 2: over MC trials, ``max_y|p̂_T - p_T| <= mlls_prev_tol``
    at large sample, and ``max_y|ŵ_lab,MLLS - ŵ_lab,BBSE| <= mlls_bbse_tol`` (both
    estimators agree HERE because this K=2 pure-label fixture satisfies the
    ``P(ŷ|y)`` invariance BBSE needs; they are NOT interchangeable in general --
    under a classifier-visible factorizable shift BBSE is biased while MLLS+BCTS,
    the recommended estimator under the premise, stays consistent, see
    label_shift.bbse_weights). The frozen classifier is
    DELIBERATELY miscalibrated (temperature + bias); BCTS must restore calibration
    for MLLS to be consistent (method_note §1.5).
    """
    gen = _gen_k2()
    wstar = gen.oracle_w_lab()
    b_mis = np.asarray(CFG.bcts.b_mis, dtype=float)
    perr, agree = [], []
    for t in range(CFG.mc.mlls_T):
        r = np.random.default_rng(CFG.seed + 3000 + t)
        Xs, ys, yhs, _, _ = gen.sample(CFG.mc.mlls_n, r, "S")
        Xt, _, yht, _, _ = gen.sample(CFG.mc.mlls_n, r, "T")
        # miscalibrated logits (an "uncalibrated network"); BCTS recalibrates.
        ls = gen.logits(Xs) / CFG.bcts.T_mis + b_mis
        lt = gen.logits(Xt) / CFG.bcts.T_mis + b_mis
        T, b = fit_bcts(ls, ys)
        sm_t = recalibrate_softmax(lt, T, b)
        p_hat = mlls_em(sm_t, gen.p_s)
        perr.append(float(np.max(np.abs(p_hat - gen.p_t))))
        w_mlls, _ = mlls_weights(sm_t, gen.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
        C = confusion_matrix(ys, yhs, gen.K)
        w_bbse, _ = bbse_weights(C, pred_hist(yht, gen.K), gen.p_s,
                                 CFG.b.w_lab_min, CFG.b.w_lab_max)
        agree.append(float(np.max(np.abs(w_mlls - w_bbse))))
    print(f"[mlls+bcts] mean max|pT_hat - p_T|={np.mean(perr):.4f} (tol={CFG.mc.mlls_prev_tol}); "
          f"mean max|w_MLLS - w_BBSE|={np.mean(agree):.4f} (tol={CFG.mc.mlls_bbse_tol}); "
          f"w_lab*={wstar}")
    assert np.mean(perr) <= CFG.mc.mlls_prev_tol
    assert np.mean(agree) <= CFG.mc.mlls_bbse_tol


def test_zdivide_recovers_oracle_product_doublecounts():
    """HARD (gate 3): the ``Ẑ``-divide recovers the oracle while the naïve product double-counts.

    The load-bearing exactness check (method_note §3.3, §7.4 worked 2-class example,
    ``p_S=(.5,.5)->p_T=(.9,.1)`` => ``w_lab*=(1.8,0.2)``; class 0 here is the
    method_note "class 1", the .5->.9 majority-in-target class with w_lab=1.8):
      (a) at a class-0-dominant x (``p_S(0|x)≈1``, so ``Z≈1.8``) the naïve product
          ``≈ 1.8·1.8 = 3.24`` (oracle posteriors),
      (b) the ``Ẑ``-divide recovers ``1.8`` (``|ŵ - 1.8| <= 1e-6``, oracle ``Ẑ``),
      (c) on the combined covariate+label synthetic (factorizable x2 tilt),
          ``mean|w_Zdivide - w*| <= zdiv_tol_comb`` with oracle inputs while the
          naïve product's mean error is >= zdiv_prod_mult x larger.
    """
    gen = _gen_k2()
    wlab = gen.oracle_w_lab()
    assert abs(wlab[0] - 1.8) < 1e-9 and abs(wlab[1] - 0.2) < 1e-9

    # (a)+(b): a strongly class-0-dominant x under PURE label shift (w_cov == Z).
    X0 = np.array([[-5.0, 0.0]])
    post = gen.posterior_S(X0)[0]
    Z = float(gen.oracle_Z(X0)[0])
    w_cov = float(gen.oracle_w_cov(X0)[0])           # pure label shift => equals Z
    naive = wlab[0] * w_cov                            # naïve product
    zdiv = float(combine_weights(np.array([wlab[0]]),
                                 np.array([w_cov]), np.array([Z]))[0])
    print(f"[zdivide/worked] p_S(0|x)={post[0]:.5f} Z={Z:.4f} w_cov={w_cov:.4f} | "
          f"naive_product={naive:.4f} (~3.24) -> Zdivide={zdiv:.6f} (=1.8)")
    assert abs(naive - 3.24) <= 1e-2                   # (a) product double-counts to ~3.24
    assert abs(zdiv - 1.8) <= 1e-6                     # (b) Ẑ-divide recovers the truth

    # (c): combined covariate+label (factorizable), oracle inputs, per point.
    genf = _gen_k2(tilt="x2", theta=CFG.k2.theta_cov)
    r = np.random.default_rng(CFG.seed + 4000)
    X, y, _, _, _ = genf.sample(CFG.mc.zdiv_n, r, "S")
    w_star = genf.oracle_w_joint(X, y)                 # ground-truth combined weight
    wl_y = genf.oracle_w_lab()[y]
    wc = genf.oracle_w_cov(X)
    Zc = genf.oracle_Z(X)
    w_zdiv = combine_weights(wl_y, wc, Zc)
    w_prod = wl_y * wc                                 # naïve product
    err_zdiv = float(np.mean(np.abs(w_zdiv - w_star)))
    err_prod = float(np.mean(np.abs(w_prod - w_star)))
    print(f"[zdivide/combined] mean|Zdivide - w*|={err_zdiv:.2e} <= {CFG.mc.zdiv_tol_comb}; "
          f"mean|product - w*|={err_prod:.4f} (>= {CFG.mc.zdiv_prod_mult}x larger)")
    assert err_zdiv <= CFG.mc.zdiv_tol_comb
    assert err_prod >= CFG.mc.zdiv_prod_mult * max(err_zdiv, 1e-12)


def test_z_collapses_in_pure_regimes():
    """HARD (gate 4): ``Z`` collapses correctly in the two pure regimes.

    build_gates.md §6 gate 4 / method_note §3.3:
      * PURE label shift: ``w_cov ≡ Z`` per point (so ``w = w_lab``) to <= 1e-6.
      * PURE covariate shift (``w_lab ≡ 1``): ``Z ≡ 1`` and ``w = w_cov`` to <= 1e-6.
    """
    tol = CFG.mc.collapse_tol
    n = CFG.mc.collapse_n
    # pure label shift (no covariate tilt): oracle w_cov == oracle Z, AND the MODULE
    # under test collapses to w = w_lab per point. We exercise conformal.label_shift
    # directly (z_corrector + combine_weights), not just the generator's self-
    # consistency, so a bug that dropped the /Ẑ (naïve product) or squared Ẑ would be
    # caught here rather than only incidentally in the non-pure regimes.
    genl = _gen_k2()
    r = np.random.default_rng(CFG.seed + 5000)
    Xs, ys, _, _, _ = genl.sample(n, r, "S")
    w_cov = genl.oracle_w_cov(Xs)
    dwz = float(np.max(np.abs(w_cov - genl.oracle_Z(Xs))))
    Zmod = z_corrector(genl.oracle_w_lab(), genl.posterior_S(Xs))   # module Ẑ = Σ ŵ_lab·σ
    dZ = float(np.max(np.abs(Zmod - w_cov)))                        # == oracle covariate ratio
    w_comb = combine_weights(genl.oracle_w_lab()[ys], w_cov, Zmod)  # module combine
    dwl = float(np.max(np.abs(w_comb - genl.oracle_w_lab()[ys])))   # collapses to w_lab(y)
    print(f"[collapse/pure-label] max|w_cov - Z| = {dwz:.2e}; max|Z_mod - w_cov| = {dZ:.2e}; "
          f"max|w - w_lab| = {dwl:.2e} (<= {tol})")
    assert dwz <= tol and dZ <= tol and dwl <= tol

    # pure covariate shift (p_T(y)=p_S(y) => w_lab==1): Z==1 and w==w_cov.
    genc = LabelShiftMixture(means=CFG.k2.means, p_s=CFG.k2.p_s, p_t=CFG.k2.p_s,
                             rho=CFG.k2.rho, tilt="x2", theta=CFG.k2.theta_cov)
    assert np.allclose(genc.oracle_w_lab(), 1.0)
    Xs2, _, _, _, _ = genc.sample(n, r, "S")
    dz1 = float(np.max(np.abs(genc.oracle_Z(Xs2) - 1.0)))
    # combined weight with w_lab==1, Z==1 collapses to w_cov exactly.
    w_comb = combine_weights(genc.oracle_w_lab()[np.zeros(len(Xs2), int)],
                             genc.oracle_w_cov(Xs2), genc.oracle_Z(Xs2))
    dwc = float(np.max(np.abs(w_comb - genc.oracle_w_cov(Xs2))))
    print(f"[collapse/pure-cov] max|Z - 1| = {dz1:.2e}; max|w - w_cov| = {dwc:.2e} (<= {tol})")
    assert dz1 <= tol and dwc <= tol


def test_label_aware_beats_covariate_only():
    """HARD (gate 5): the label-aware ``Ẑ``-combined path lowers the Hájek risk vs covariate-only.

    build_gates.md §6 gate 5: on a target split with KNOWN prevalence shift, ``R̂_w``
    under the ``Ẑ``-combined weight is lower than covariate-only with NON-OVERLAPPING
    WSR intervals (prereg §6.7 win-rule) and tracks the known population accepted
    risk ``E_{p_T}[ℓ|A] = R_T`` to within the interval half-width.

    The label path is genuinely ESTIMATED (BBSE ``ŵ_lab`` on a disjoint source fold +
    BCTS-recalibrated ``Ẑ``); the covariate weight is the ORACLE ``w_cov`` (validated
    at Gate B) so the comparison ISOLATES the label correction G-C adds. Because the
    loss is class-structured (``ℓ ⟂ x | y``), covariate-only reweighting -- a function
    of ``x`` -- cannot recover the label-marginal change and is biased HIGH, while the
    label-aware path recovers ``R_T`` (method_note §3 label-shift row).
    Two reads mirror Gate B: a signed win at moderate n, and interval non-overlap on
    well-powered draws (importance weighting inflates variance, so disjointness must
    survive the true n_eff cost, not be bought by under-powering).
    """
    gen = _gen_k2()
    tau0 = CFG.k2.tau0
    rr = np.random.default_rng(CFG.seed + 6000)
    R_T = gen.accepted_risk(tau0, "T", rr, n=CFG.mc.oracle_n)
    R_S = gen.accepted_risk(tau0, "S", rr, n=CFG.mc.oracle_n)
    assert R_T < R_S                                    # target easier (more easy-class mass)

    def _est_label(seed, n_bbse):
        """Estimate ŵ_lab (BBSE) and the BCTS recalibrator on a disjoint source/target fold."""
        r = np.random.default_rng(seed)
        Xb, yb, yhb, _, _ = gen.sample(n_bbse, r, "S")
        _, _, yht, _, _ = gen.sample(n_bbse, r, "T")
        C = confusion_matrix(yb, yhb, gen.K)
        wlab, _ = bbse_weights(C, pred_hist(yht, gen.K), gen.p_s,
                               CFG.b.w_lab_min, CFG.b.w_lab_max)
        T, b = fit_bcts(gen.logits(Xb), yb)
        return wlab, T, b

    # (a) signed win over many moderate-n resamples
    wins = 0
    for i in range(CFG.mc.signed_T):
        seed = CFG.seed + 6100 + i
        wlab, T, b = _est_label(seed, CFG.mc.signed_n)
        r = np.random.default_rng(seed + 777)
        Xc, yc, _, lc, uc = gen.sample(CFG.mc.signed_n, r, "S")
        acc = uc <= tau0
        wcov = gen.oracle_w_cov(Xc)[acc]
        Zc = z_corrector(wlab, recalibrate_softmax(gen.logits(Xc), T, b))[acc]
        wcomb = combine_weights(wlab[yc[acc]], wcov, Zc)
        if hajek_risk(lc[acc], wcomb) < hajek_risk(lc[acc], wcov):
            wins += 1
    win_rate = wins / CFG.mc.signed_T
    print(f"[label-vs-cov/signed] combined<cov-only in {win_rate:.3f} of {CFG.mc.signed_T} "
          f"resamples (>= {1 - CFG.b.delta}); R_T={R_T:.4f} < R_S={R_S:.4f}")
    assert win_rate >= 1 - CFG.b.delta

    # (b) non-overlapping WSR intervals on well-powered draws + tracking check
    margins, comb_pts = [], []
    for j in range(CFG.mc.N_disjoint):
        seed = CFG.seed + 6500 + j
        wlab, T, b = _est_label(seed, CFG.mc.n_big // 2)
        r = np.random.default_rng(seed + 999)
        Xc, yc, _, lc, uc = gen.sample(CFG.mc.n_big, r, "S")
        acc = uc <= tau0
        L = lc[acc]
        wcov = gen.oracle_w_cov(Xc)[acc]
        Zc = z_corrector(wlab, recalibrate_softmax(gen.logits(Xc), T, b))[acc]
        wcomb = combine_weights(wlab[yc[acc]], wcov, Zc)
        comb_ucb = wsr_ucb_weighted(L, wcomb, CFG.b.delta, float(wcomb.max()))
        cov_lcb = wsr_lcb_weighted(L, wcov, CFG.b.delta, float(wcov.max()))
        margins.append(cov_lcb - comb_ucb)              # > 0 => disjoint (combined below cov-only)
        comb_pts.append(hajek_risk(L, wcomb))
    margins = np.asarray(margins)
    print(f"[label-vs-cov/interval] disjoint in {int((margins > 0).sum())}/{CFG.mc.N_disjoint} "
          f"well-powered draws; min margin={margins.min():.4f}; "
          f"combined_pt={np.mean(comb_pts):.4f} ~ R_T={R_T:.4f}")
    assert np.all(margins > 0)                           # non-overlap in every draw (prereg §6.7)
    assert abs(np.mean(comb_pts) - R_T) <= CFG.mc.track_tol   # tracks the known target risk


def test_simplex_floor_keeps_wlab_finite():
    """HARD (gate 6): simplex projection + floor/ceiling keep ``ŵ_lab`` finite under ill-conditioning.

    build_gates.md §6 gate 6 / method_note §3.4 step 2: on an ill-conditioned regime
    (merged classes + small n => near-singular ``Ĉ_S``), the projected ``p̂_T`` lies on
    the simplex (nonneg, sums to 1 within 1e-8) and every ``ŵ_lab(y) ∈ [w_min, w_max]``
    with ``w_min > 0``, so ``min_x Ẑ(x) > 0`` (an INVARIANT, not an accuracy gate).
    """
    gi = _gen_k3(means=CFG.mc.illcond_means)             # merged K=3 class means
    r = np.random.default_rng(CFG.seed + 7000)
    _, ys, yhs, _, _ = gi.sample(CFG.mc.illcond_n, r, "S")
    Xt, _, yht, _, _ = gi.sample(CFG.mc.illcond_n, r, "T")
    C = confusion_matrix(ys, yhs, gi.K)
    kappa = kappa_cs(C)["kappa"]
    w_lab, p_hat = bbse_weights(C, pred_hist(yht, gi.K), gi.p_s,
                                CFG.b.w_lab_min, CFG.b.w_lab_max)
    Z = z_corrector(w_lab, gi.posterior_S(Xt))
    print(f"[ill-cond] kappa(C_S)~{kappa:.1f}; pT_hat={p_hat} (sum={p_hat.sum():.8f}); "
          f"w_lab={w_lab}; min Z={Z.min():.4f}")
    # projected prevalence is a valid simplex point
    assert abs(p_hat.sum() - 1.0) <= 1e-8
    assert np.all(p_hat >= -1e-12)
    # floor/ceiling hold, floor strictly positive => Z strictly positive
    assert np.all(w_lab >= CFG.b.w_lab_min - 1e-12)
    assert np.all(w_lab <= CFG.b.w_lab_max + 1e-9)
    assert CFG.b.w_lab_min > 0.0
    assert Z.min() > 0.0


def test_discriminator_single_domain_fold_raises_loud():
    """HARD (F5): a single-domain training fold makes ``fit_discriminator`` fail LOUD.

    method_note §1.5 / §1.7: group-level cross-fitting needs every fold's TRAINING split
    to contain BOTH domains. In the natural multi-site layout (disjoint source/target
    site ids) a fold can hold out an entire domain's groups, leaving the discriminator
    trained on one domain: it then converges to a constant posterior and ``ŵ_cov``
    collapses to all-equal ``w_max`` -- Kish ``n_eff`` reads ``n`` (maximally healthy) while
    the covariate correction is SILENTLY nulled to an unweighted mean. That degeneracy is
    unrecoverable, so the estimator must ASSERT rather than return degenerate weights.

    This gate pins the LOUD behaviour on two shapes and confirms a healthy multi-group /
    multi-fold split is untouched (so the guard never fires on a legitimate Gate-B-style
    call). It measures/hardens honesty -- it adds NO coverage guarantee.
    """
    w_max = 20.0
    n = 300
    rs = np.random.default_rng(CFG.seed + 17000)
    X_src = rs.normal(0.0, 1.0, size=(n, 2))
    X_tar = rs.normal(1.5, 1.0, size=(n, 2))

    # (a) one group per domain: group-level cross-fitting is impossible in principle
    # (every training split is single-domain, for ANY n_splits) -> must raise LOUD.
    gs_one = np.zeros(n, dtype=int)
    gt_one = np.ones(n, dtype=int)
    with pytest.raises(AssertionError):
        fit_discriminator(X_src, X_tar, w_max=w_max, n_splits=5, seed=CFG.seed,
                          groups_src=gs_one, groups_tar=gt_one)

    # (b) natural multi-site shape: 2 source sites + 2 target sites (disjoint ids),
    # n_splits=2. Without the guard this SILENTLY returns every source weight == w_max
    # (9/30 seeds in the review; seed 0 is degenerate) -> must raise LOUD instead.
    gs_site = np.where(np.arange(n) < n // 2, 0, 1)
    gt_site = np.where(np.arange(n) < n // 2, 2, 3)
    with pytest.raises(AssertionError):
        fit_discriminator(X_src, X_tar, w_max=w_max, n_splits=2, seed=0,
                          groups_src=gs_site, groups_tar=gt_site)

    # positive control: a healthy split (each point its own group, K folds) trains on
    # BOTH domains in every fold, so the guard is SILENT and honest weights are returned
    # (this is exactly the Gate-B default-group usage). Weights must not be degenerate.
    disc = fit_discriminator(X_src, X_tar, w_max=w_max, n_splits=5, seed=CFG.seed)
    w_src = disc.oof_weights()[disc.domain == 0]
    assert np.isfinite(w_src).all()
    assert not np.allclose(w_src, w_src[0])              # a real, non-degenerate correction
    print(f"[disc-degeneracy] one-group-per-domain and 2+2-site/n_splits=2 both raise LOUD; "
          f"healthy K-fold split returns non-degenerate w_cov "
          f"(median={np.median(w_src):.3f}, max={w_src.max():.3f}).")


def test_factorizable_means_invariant_density_ratio_not_fjs():
    """HARD (F12): "factorizable" here = the INVARIANT-DENSITY-RATIO condition, which
    is STRICTLY STRONGER than the He et al. 2022 / Tasche FJS sense ``w(x,y)=U(x)·V(y)``.

    The repo's own entangled ``x1``-tilt regime factorizes EXACTLY in the FJS sense
    (``w(x,y)=U(x)·V(y)`` to machine precision) yet BREAKS the ``Ẑ``-combine -- so the
    combine identity (Tasche 2022 Thm 4/Cor 4 at ρ=1) needs the stronger condition
    ``p_T(x|y)/p_S(x|y)`` class-free on overlapping support, and the two senses of
    "factorizable" must not be conflated (method_note §3.3, §7.4). Guards the
    terminology fix for review finding F12.
    """
    gene = _gen_k2(tilt="x1", theta=CFG.k2.theta_ent)      # entangled: p(x1|y) moves with the shift
    r = np.random.default_rng(CFG.seed + 17500)
    X, y, _, _, _ = gene.sample(40000, r, "S")
    x1, x2 = X[:, 0], X[:, 1]
    m = np.asarray(CFG.k2.means, dtype=float)
    th = float(CFG.k2.theta_ent)
    p_s = np.asarray(CFG.k2.p_s, dtype=float)
    p_t = np.asarray(CFG.k2.p_t, dtype=float)

    # FJS test w(x,y)=U(x)·V(y): log w_joint(x,y) is additive in x and y iff
    # d(x) := log w_joint(x,1) - log w_joint(x,0) is CONSTANT in x (K=2).
    def _logwj(k):
        lp_s = norm.logpdf(x1 - m[k]) + norm.logpdf(x2) + np.log(p_s[k])
        lp_t = norm.logpdf(x1 - (m[k] + th)) + norm.logpdf(x2) + np.log(p_t[k])
        return lp_t - lp_s
    fjs_defect = float(np.std(_logwj(1) - _logwj(0)))

    # The Ẑ-combine (invariant-density-ratio identity) BREAKS on the SAME regime.
    w_joint = gene.oracle_w_joint(X, y)
    fac = combine_weights(gene.oracle_w_lab()[y], gene.oracle_w_cov(X), gene.oracle_Z(X))
    combine_div = float(np.mean((np.log(fac) - np.log(w_joint)) ** 2))

    print(f"[factorizable-vs-FJS] FJS defect std(log w(x,1)-log w(x,0))={fjs_defect:.2e} "
          f"(~0 => FJS-factorizable w=U(x)V(y)); Zdivide combine divergence={combine_div:.4f} "
          f"(>0 => the STRONGER invariant-density-ratio condition FAILS)")
    assert fjs_defect < 1e-9      # w(x,y)=U(x)V(y) holds to machine precision (FJS-factorizable) ...
    assert combine_div > 1e-2     # ... yet the Ẑ-combine is materially violated (needs the stronger cond.)


def test_Z_equals_one_needs_wlab_one_not_pure_covariate():
    """HARD (F12): ``Ẑ ≡ 1`` requires ``w_lab ≡ 1``, NOT merely "pure covariate shift".

    On a prevalence-shifted mixture source (``w_lab*=(1.8,0.2)``) ``max_x|Z-1|`` is far
    from 0; it collapses to machine zero only once ``w_lab ≡ 1`` -- the qualifier the
    module header must carry (review finding F12; guards the header fix).
    """
    genl = _gen_k2()                                          # p_t=(.9,.1) => w_lab=(1.8,0.2)
    r = np.random.default_rng(CFG.seed + 18000)
    Xs, _, _, _, _ = genl.sample(40000, r, "S")
    max_dev_shift = float(np.max(np.abs(genl.oracle_Z(Xs) - 1.0)))
    genc = _gen_k2(p_t=CFG.k2.p_s, tilt="x2", theta=CFG.k2.theta_cov)   # w_lab==1
    max_dev_wlab1 = float(np.max(np.abs(genc.oracle_Z(Xs) - 1.0)))
    print(f"[Z-eq-1] max|Z-1| under prevalence shift (w_lab=(1.8,0.2))={max_dev_shift:.4f} "
          f">> when w_lab==1={max_dev_wlab1:.2e}")
    assert max_dev_shift > 0.5       # Z is NOT ≡ 1 merely because covariates are the only tilt
    assert max_dev_wlab1 <= 1e-6     # Z ≡ 1 exactly iff w_lab ≡ 1


def test_dtarlab_residual_measures_factorization_premise():
    """HARD (F17): the ``D_tar^lab`` residual MEASURES the factorizable premise from a labeled slice.

    build_gates.md §6 honesty rail / method_note §1.7, §3.4 step 3, §7; prereg §7: the
    factorizable-shift premise's residual is *measured* on a small LABELED target slice
    ``D_tar^lab`` -- never certified. This gate exercises the real shipped pathway
    (:func:`residual_on_labeled_target` + :func:`measure_slice_joint_weight`), which
    consumes labeled target pairs ``(X, y) ∼ P_T`` and REPORTS (a) the residual risk
    degradation of the factorized combined weight and (b) its divergence from a joint
    weight measured DIRECTLY on the slice (from the SAMPLE, never ``oracle_w_joint``).
    Both separate the FACTORIZABLE (x2 tilt) from the ENTANGLED (x1 tilt) regime, so
    the premise is falsifiable per pair from real labeled data. MEASURES/REPORTS only;
    it asserts SEPARATION and finiteness, never a certificate or an identification.
    """
    tau0 = CFG.k2.tau0

    def _measure(tilt, theta):
        gen = _gen_k2(tilt=tilt, theta=theta)
        r = np.random.default_rng(CFG.seed + 19000)
        # labeled SOURCE accepted cohort -> PREDICT target accepted risk under factorized w
        Xs, ys, _, ls, us = gen.sample(CFG.mc.dtarlab_src_n, r, "S")
        accs = us <= tau0
        wl_s = gen.oracle_w_lab()[ys]
        wfac_s = combine_weights(wl_s, gen.oracle_w_cov(Xs), gen.oracle_Z(Xs))
        # small labeled TARGET slice D_tar^lab (accepted region), with its factorized w
        Xt, yt, _, lt, ut = gen.sample(CFG.mc.dtarlab_m, r, "T")
        acct = ut <= tau0
        Xt_a, yt_a, lt_a = Xt[acct], yt[acct], lt[acct]
        wl_t = gen.oracle_w_lab()[yt_a]
        wfac_t = combine_weights(wl_t, gen.oracle_w_cov(Xt_a), gen.oracle_Z(Xt_a))
        rep = residual_on_labeled_target(
            wfac_s, ls, accs, Xs, ys, Xt_a, yt_a, wfac_t, gen.p_s, gen.K)
        risk_emp = float(lt_a.mean())                    # empirical target accepted risk on the slice
        risk_gap = abs(rep["risk_plugin"] - risk_emp)
        return rep, risk_gap

    rep_f, gap_f = _measure("x2", CFG.k2.theta_cov)      # factorizable (⊥ label signal)
    rep_e, gap_e = _measure("x1", CFG.k2.theta_ent)      # entangled (along label signal)
    print(f"[dtarlab/residual] factorizable: divergence={rep_f['divergence']:.4f} "
          f"risk_gap={gap_f:.4f} (m_lab={rep_f['m_lab']}, wlab_slice={np.round(rep_f['w_lab_slice'],3)}) | "
          f"entangled: divergence={rep_e['divergence']:.4f} risk_gap={gap_e:.4f}")
    # the slice-measured joint weight is built from the SAMPLE, never the oracle
    genp = _gen_k2(tilt="x2", theta=CFG.k2.theta_cov)
    Xps, yps, _, _, _ = genp.sample(200, np.random.default_rng(CFG.seed + 19001), "S")
    Xpt, ypt, _, _, _ = genp.sample(200, np.random.default_rng(CFG.seed + 19002), "T")
    w_joint_slice, _ = measure_slice_joint_weight(Xps, yps, Xpt, ypt, genp.p_s, genp.K)
    assert np.all(np.isfinite(w_joint_slice))            # plumbing: the slice estimator runs
    # (a) divergence: ~0 when factorizable, materially larger when entangled
    assert np.isfinite(rep_f["divergence"]) and np.isfinite(rep_e["divergence"])
    assert rep_f["divergence"] <= CFG.mc.dtarlab_div_fac
    assert rep_e["divergence"] >= CFG.mc.dtarlab_div_mult * max(rep_f["divergence"], 1e-12)
    # (b) residual risk degradation: small when factorizable, materially larger when entangled
    assert gap_f <= CFG.mc.dtarlab_gap_fac
    assert gap_e >= CFG.mc.dtarlab_gap_mult * max(gap_f, 1e-12)


def test_estimated_combined_pipeline_documents_gap():
    """HARD (F3): the FULLY ESTIMATED combined-regime pipeline, honest gap budget.

    Every prior combined-regime check (HG3 (c), HG4, the 5-arm / entanglement
    diagnostics) feeds ORACLE ``w_cov``/``Ẑ``/``w_lab`` into ``combine_weights`` and
    lands at ~1e-16 (``zdiv_tol_comb=0.02``). That validates the *identity*, not the
    *estimated pipeline*. This gate runs the pipeline the method actually deploys --
    ``conformal.weights.fit_discriminator`` (cross-fit ``ŵ_cov``) + BBSE ``ŵ_lab`` +
    BCTS ``Ẑ`` -- on the factorizable ``x2`` tilt, and asserts against an HONEST
    budget that DOCUMENTS the estimated-vs-oracle gap rather than hiding it
    (``build_gates.md §6``; F3):

      * the estimated mean weight error is MATERIALLY above the oracle-input budget
        (``est_err_lo <= err_est <= est_err_hi``; the gap is real, not ~oracle),
      * on the SAME slice the oracle-input error stays tiny (``<= est_oracle_max``;
        the wiring is correct -- the gap is estimation, not a bug),
      * the gap is dominated by DISCRIMINATOR misspecification: swapping the ORACLE
        ``w_cov`` back in (keeping estimated ``ŵ_lab``/``Ẑ``) removes
        ``>= est_disc_share`` of it. The linear-logit discriminator family cannot
        represent the ``log Ẑ(x1)`` term in ``log w_cov``, so the oracle ``w_cov``
        validated at Gate B does NOT transfer to the Gate-C combined regime.

    This is a MEASUREMENT, not a certificate: it asserts the gap's size/attribution,
    never that the estimated weight is close to truth.
    """
    genf = _gen_k2(tilt="x2", theta=CFG.k2.theta_cov)
    r = np.random.default_rng(CFG.seed + 20000)
    # (1) estimated ŵ_cov via the cross-fit domain discriminator (conformal.weights)
    Xs, _, _, _, _ = genf.sample(CFG.mc.est_n_disc, r, "S")
    Xt, _, _, _, _ = genf.sample(CFG.mc.est_n_disc, r, "T")
    disc = fit_discriminator(Xs, Xt, w_max=CFG.b.w_lab_max, seed=CFG.seed % 1000)
    # (2) estimated ŵ_lab via BBSE on a disjoint source fold + target predictions
    Xb, yb, yhb, _, _ = genf.sample(CFG.mc.est_n_bbse, r, "S")
    _, _, yht, _, _ = genf.sample(CFG.mc.est_n_bbse, r, "T")
    Cb = confusion_matrix(yb, yhb, genf.K)
    wlab, _ = bbse_weights(Cb, pred_hist(yht, genf.K), genf.p_s,
                           CFG.b.w_lab_min, CFG.b.w_lab_max)
    # (3) estimated Ẑ via a BCTS-recalibrated softmax
    T, b = fit_bcts(genf.logits(Xb), yb)
    # evaluation slice (disjoint draw): compare estimated vs oracle combined weight
    Xe, ye, _, _, _ = genf.sample(CFG.mc.est_n_eval, r, "S")
    w_star = genf.oracle_w_joint(Xe, ye)
    wc_est = disc.weights(Xe)
    Ze = z_corrector(wlab, recalibrate_softmax(genf.logits(Xe), T, b))
    w_est = combine_weights(wlab[ye], wc_est, Ze)
    err_est = float(np.mean(np.abs(w_est - w_star)))
    # oracle-input error on the SAME slice (identity is exact => ~0)
    w_or = combine_weights(genf.oracle_w_lab()[ye], genf.oracle_w_cov(Xe),
                           genf.oracle_Z(Xe))
    err_or = float(np.mean(np.abs(w_or - w_star)))
    # attribution: oracle ŵ_cov but estimated ŵ_lab + Ẑ (isolates the discriminator)
    w_mix = combine_weights(wlab[ye], genf.oracle_w_cov(Xe), Ze)
    err_mix = float(np.mean(np.abs(w_mix - w_star)))
    disc_share = (err_est - err_mix) / err_est if err_est > 0 else 0.0
    print(f"[estimated-combined] mean|ŵ - w*|: estimated={err_est:.4f} "
          f"(budget [{CFG.mc.est_err_lo}, {CFG.mc.est_err_hi}]) vs oracle-input={err_or:.2e} "
          f"(<= {CFG.mc.est_oracle_max}); discriminator share of gap={disc_share:.3f} "
          f"(>= {CFG.mc.est_disc_share})")
    # the estimated gap is REAL (materially above the ~oracle budget) and bounded
    assert CFG.mc.est_err_lo <= err_est <= CFG.mc.est_err_hi, err_est
    # the identity itself is exact on this slice -- the gap is estimation, not a wiring bug
    assert err_or <= CFG.mc.est_oracle_max, err_or
    # and the gap is dominated by discriminator misspecification (does NOT transfer from Gate B)
    assert disc_share >= CFG.mc.est_disc_share, disc_share


def test_fold_disjointness_group_level():
    """HARD (gate 7): group-id fold-disjointness across the five G-C folds.

    build_gates.md §6 gate 7 / method_note §1.7 (set-intersection = 0 artifact) over
    ``D_bbse^src``, ``D_tar``, ``D_cal``, ``D_tar^lab``, held-out test.
    """
    n = 2000
    gids = SimpleNamespace(
        D_bbse_src=np.arange(0, n),
        D_tar=np.arange(n, 2 * n),
        D_cal=np.arange(2 * n, 3 * n),
        D_tar_lab=np.arange(3 * n, 4 * n),
        test=np.arange(4 * n, 5 * n),
    )
    assert assert_group_disjoint(
        D_bbse_src=gids.D_bbse_src, D_tar=gids.D_tar, D_cal=gids.D_cal,
        D_tar_lab=gids.D_tar_lab, test=gids.test, verbose=True,
    )
    # negative control: an overlapping split must raise
    with pytest.raises(AssertionError):
        assert_group_disjoint(D_bbse_src=[1, 2, 3], D_tar=[3, 4, 5])


def test_bbse_singular_confusion_raises_clear_error():
    """HARD (F4): a singular Ĉ_S raises a CLEAR diagnostic, not a bare LinAlgError.

    Invertibility of Ĉ_S is a documented BBSE precondition (method_note §1.5). The
    shipped config is provably safe (well-conditioned Ĉ_S), but when a class is never
    predicted (a zero row) or never a true label (a zero column) the raw
    ``np.linalg.solve`` raises an opaque ``numpy.linalg.LinAlgError`` BEFORE the
    ``κ(Ĉ_S)`` / ``σ_min`` diagnostic can report it -- an inconsistent-hardening gap
    (``kappa_cs`` already handles ``σ_min = 0`` gracefully). This gate pins the fix:
    ``bbse_weights`` must raise a ``ValueError`` naming the non-invertibility and the
    ``σ_min`` diagnostic, WITHOUT changing the safe (invertible) path.
    """
    q = pred_hist(np.array([0, 1, 0, 1]), 2)
    p_S = np.asarray(CFG.k2.p_s, dtype=float)
    # zero row: class 1 is never predicted -> Ĉ_S has an all-zero row -> singular.
    C_zero_row = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), 2)
    assert np.allclose(C_zero_row[1], 0.0)               # the degenerate row
    with pytest.raises(ValueError, match=r"singular|invertible"):
        bbse_weights(C_zero_row, q, p_S, CFG.b.w_lab_min, CFG.b.w_lab_max)
    # zero column: class 1 is never a TRUE label -> Ĉ_S has an all-zero column.
    C_zero_col = confusion_matrix(np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1]), 2)
    assert np.allclose(C_zero_col[:, 1], 0.0)
    with pytest.raises(ValueError, match=r"singular|invertible"):
        bbse_weights(C_zero_col, q, p_S, CFG.b.w_lab_min, CFG.b.w_lab_max)
    # SAFE path unchanged: a well-conditioned Ĉ_S still returns a finite w_lab.
    C_ok = confusion_matrix(np.array([0, 0, 1, 1, 0, 1]), np.array([0, 1, 1, 1, 0, 0]), 2)
    w_ok, p_ok = bbse_weights(C_ok, q, p_S, CFG.b.w_lab_min, CFG.b.w_lab_max)
    assert np.all(np.isfinite(w_ok)) and abs(p_ok.sum() - 1.0) <= 1e-8
    print("[bbse-singular] zero-row and zero-column Ĉ_S raise clear ValueError; "
          "well-conditioned Ĉ_S unchanged")


def test_confusion_matrix_rejects_out_of_range_labels():
    """HARD (F16): ``confusion_matrix`` enforces its documented ``[0, K)`` label contract.

    ``confusion_matrix`` is documented to take integer labels in ``[0, K)``, but
    ``np.add.at`` would SILENTLY wrap a ``-1`` sentinel into class ``K-1`` (negative
    indexing), whereas ``pred_hist`` already RAISES on the same input. Before any
    real-data use, the two must agree: an out-of-range label is a ValueError, never a
    silent miscount. The in-range (safe) path is unchanged.
    """
    # a -1 sentinel in y_true must RAISE (not wrap into class K-1) ...
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1, -1]), np.array([0, 1, 0]), 2)
    # ... and pred_hist already raises on the same -1 (the behaviour we align to).
    with pytest.raises(ValueError):
        pred_hist(np.array([0, 1, -1]), 2)
    # a label == K (also out of [0, K)) must RAISE.
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 0]), 2)
    # SAFE path unchanged: in-range labels build the same joint confusion as before.
    C = confusion_matrix(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), 2)
    assert abs(C.sum() - 1.0) <= 1e-12 and C.shape == (2, 2)
    print("[cm-range] out-of-range labels (-1, K) raise ValueError; in-range unchanged")


def test_bbse_confusion_orientation_is_load_bearing():
    """HARD (F7, mutation-adequacy): BBSE recovery FAILS if ``Ĉ_S`` is transposed.

    The shipped K=2 / K=3 recovery fixtures have a (near-)SYMMETRIC population joint
    confusion ``C_S``, so solving ``Ĉ_Sᵀ⁻¹ q̂_T`` (or building the confusion with
    ``ŷ``/``y`` swapped) recovers ``w_lab*`` just as well as the correct orientation --
    gate 1 cannot tell the two apart. This gate pins the orientation down on a
    deliberately ASYMMETRIC K=3 fixture (two overlapping low classes + one far class,
    skewed source prior => ``max|C_S - C_Sᵀ| ~ 0.034``): the CORRECT orientation
    recovers ``w_lab*`` (mean ``max_y|ŵ_lab - w_lab*| <= asym_tol``), and -- asserted
    as a NEGATIVE CONTROL in the same regime -- solving on the transposed confusion
    blows past that line. A reviewer can flip ``np.add.at(C, (y_pred, y_true), …)`` to
    ``(y_true, y_pred)`` in ``confusion_matrix`` (or transpose ``C_S`` before the
    solve) and watch THIS test go red. SYNTHETIC wiring / mutation-adequacy check;
    proves nothing about deployment and adds no guarantee.
    """
    g = LabelShiftMixture(means=CFG.mc.asym_means, p_s=CFG.mc.asym_p_s,
                          p_t=CFG.mc.asym_p_t, rho=CFG.mc.asym_rho)
    wstar = g.oracle_w_lab()

    # the fixture's population C_S really is asymmetric (else the mutant is invisible).
    rC = np.random.default_rng(CFG.seed + 2700)
    C_pop = g.confusion_S(rC, n=400_000)
    confusion_asym = float(np.max(np.abs(C_pop - C_pop.T)))

    err_ok, err_transpose = [], []
    for t in range(CFG.mc.asym_T):
        r = np.random.default_rng(CFG.seed + 2700 + t)
        _, ys, yhs, _, _ = g.sample(CFG.mc.asym_n, r, "S")
        _, _, yht, _, _ = g.sample(CFG.mc.asym_n, r, "T")
        C = confusion_matrix(ys, yhs, g.K)
        q = pred_hist(yht, g.K)
        w_ok, _ = bbse_weights(C, q, g.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
        # NEGATIVE CONTROL: the same solve on the TRANSPOSED confusion (the F7 mutant).
        w_bad, _ = bbse_weights(C.T, q, g.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
        err_ok.append(float(np.max(np.abs(w_ok - wstar))))
        err_transpose.append(float(np.max(np.abs(w_bad - wstar))))
    e_ok = float(np.mean(err_ok))
    e_bad = float(np.mean(err_transpose))
    print(f"[bbse-orient/asymK3] max|C_S - C_S^T|~{confusion_asym:.4f}; "
          f"correct-orientation err={e_ok:.4f} (<= {CFG.mc.asym_tol}) vs "
          f"transposed err={e_bad:.4f} (kills the mutant); w_lab*={wstar}")
    # fixture is genuinely asymmetric (guards against a silent symmetric regression)
    assert confusion_asym >= CFG.mc.asym_min_confusion_asym
    # correct BBSE orientation recovers w_lab*
    assert e_ok <= CFG.mc.asym_tol
    # transposing C_S (the F7 orientation bug) FAILS the same line -- the whole point
    assert e_bad > CFG.mc.asym_tol


def test_mlls_prior_ratio_term_is_load_bearing():
    """HARD (F11, mutation-adequacy): the MLLS ``/p_S`` prior-ratio term is required.

    Every other MLLS call site uses ``p_S = (0.5, 0.5)``; there the fixed-point's
    prior-ratio step ``γ_j ∝ σ̃_j·(p_T/p_S)`` is renormalization-invariant to dropping
    ``/p_S``, so a mutant that deletes the division is BITWISE identical and gate 2 is
    blind to it. This gate runs the SAME MLLS+BCTS pipeline (deliberately miscalibrated
    logits, ``fit_bcts`` recalibration) on a NON-UNIFORM source prior ``p_S=(0.7,0.3)``
    with ``p_T=(0.4,0.6)``, where ``/p_S`` is load-bearing: MLLS recovers ``p_T`` and
    ``w_lab*`` to within the pass lines. A reviewer can delete the ``/ p_S`` in
    ``mlls_em`` (``ratio = p_T`` instead of ``ratio = p_T / p_S``) and watch THIS test
    go red (the mutant misses ``p_T`` by ~0.18 and ``w_lab*`` by ~0.60). SYNTHETIC
    wiring / mutation-adequacy check; consistency only, no guarantee.
    """
    g = LabelShiftMixture(means=CFG.k2.means, p_s=CFG.mc.nonunif_p_s,
                          p_t=CFG.mc.nonunif_p_t, rho=CFG.k2.rho)
    assert not np.allclose(g.p_s, g.p_s[0])   # p_S must be NON-uniform for /p_S to bite
    wstar = g.oracle_w_lab()
    b_mis = np.asarray(CFG.bcts.b_mis, dtype=float)
    perr, werr = [], []
    for t in range(CFG.mc.nonunif_T):
        r = np.random.default_rng(CFG.seed + 3500 + t)
        Xs, ys, _, _, _ = g.sample(CFG.mc.nonunif_n, r, "S")
        Xt, _, _, _, _ = g.sample(CFG.mc.nonunif_n, r, "T")
        ls = g.logits(Xs) / CFG.bcts.T_mis + b_mis
        lt = g.logits(Xt) / CFG.bcts.T_mis + b_mis
        T, b = fit_bcts(ls, ys)
        sm_t = recalibrate_softmax(lt, T, b)
        p_hat = mlls_em(sm_t, g.p_s)
        perr.append(float(np.max(np.abs(p_hat - g.p_t))))
        w_mlls, _ = mlls_weights(sm_t, g.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
        werr.append(float(np.max(np.abs(w_mlls - wstar))))
    e_p = float(np.mean(perr))
    e_w = float(np.mean(werr))
    print(f"[mlls-prior/nonunif] p_S={CFG.mc.nonunif_p_s}: prevalence err={e_p:.4f} "
          f"(<= {CFG.mc.nonunif_prev_tol}); w_lab err={e_w:.4f} "
          f"(<= {CFG.mc.nonunif_wlab_tol}); w_lab*={wstar} (deleting /p_S misses both)")
    # with a load-bearing /p_S, MLLS+BCTS recovers p_T and w_lab* (mutant fails both)
    assert e_p <= CFG.mc.nonunif_prev_tol
    assert e_w <= CFG.mc.nonunif_wlab_tol


# =========================================================================== #
# REPORTED DIAGNOSTICS (measured/printed, never a pass/fail gate)              #
# =========================================================================== #
def test_diagnostic_kappa_conditioning_monotone():
    """DIAGNOSTIC (method_note §1.5, §3.4 step 2, §7.5): ``κ(Ĉ_S)`` conditioning.

    Self-test that ``κ(Ĉ_S)`` (and ``σ_min`` / effective rank) is MONOTONE increasing
    across a fixed class-merge degradation grid (K=3 middle class merged into its
    neighbours). REPORTED, never gates the rung."""
    kappas, sigmins, eranks = [], [], []
    for sep in CFG.mc.kappa_seps:
        g = _gen_k3(means=[-sep, 0.0, sep])
        r = np.random.default_rng(CFG.seed + 8000)
        info = kappa_cs(g.confusion_S(r, n=CFG.mc.oracle_n // 10))
        kappas.append(info["kappa"]); sigmins.append(info["sigma_min"])
        eranks.append(info["eff_rank"])
    print("[kappa] sep->kappa: "
          + ", ".join(f"{s}:{k:.2f}" for s, k in zip(CFG.mc.kappa_seps, kappas))
          + " | eff_rank: " + ", ".join(f"{er:.2f}" for er in eranks))
    # monotone increasing as the classes merge (grid is decreasing separation)
    assert all(a <= b + 1e-6 for a, b in zip(kappas, kappas[1:])), kappas
    assert all(a >= b - 1e-6 for a, b in zip(sigmins, sigmins[1:])), sigmins


def test_diagnostic_anticausal_consistency():
    """HARD (self-test of a REPORTED diagnostic): ``q̂_T`` vs ``Ĉ_S p̂_T``.

    The anti-causal consistency residual is a coarse CROSS-RUN tripwire, not a
    per-pair test. It reads materially larger on a perturbed target ONLY because we
    hold ``p̂_T`` FIXED from the clean run and confront it with the perturbed
    ``q̂_T`` -- the reused-reference construction below. We ALSO measure and print
    the DEPLOYMENT-REALISTIC read (single target, ``p̂_T`` solved from that target's
    own ``q̂_T``), which is BLIND: the x1-tilt violation and a pure covariate-shift
    target both read ≈ the clean residual (see :func:`bbse_consistency` docstring).
    The residual is a REPORTED continuous diagnostic in deployment (never gates the
    rung); THIS TEST asserts only the plumbing self-test -- that the diagnostic's
    sign moves the right way on a KNOWN perturbation via the reused reference -- so
    it is a HARD wiring check, consistent with the other ``test_diagnostic_*``
    self-tests. (Scope note: within-run sign check, not detection power; the
    diagnostic can be blind to violations that preserve the target x-marginal --
    REVIEW_FINDINGS F10.)"""
    gen = _gen_k2()
    r = np.random.default_rng(CFG.seed + 9000)
    Xs, ys, yhs, _, _ = gen.sample(CFG.mc.consistency_n, r, "S")
    C = confusion_matrix(ys, yhs, gen.K)
    # clean label-shift target
    _, _, yht, _, _ = gen.sample(CFG.mc.consistency_n, r, "T")
    q_clean = pred_hist(yht, gen.K)
    _, p_hat = bbse_weights(C, q_clean, gen.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
    res_clean = bbse_consistency(q_clean, C, p_hat)["residual_l1"]
    # perturbed p(x|y): an entangled x1 tilt breaks anti-causal invariance
    gene = _gen_k2(tilt="x1", theta=CFG.k2.theta_ent)
    _, _, yhe, _, _ = gene.sample(CFG.mc.consistency_n, r, "T")
    q_pert = pred_hist(yhe, gen.K)
    res_pert = bbse_consistency(q_pert, C, p_hat)["residual_l1"]   # REUSED clean p_hat
    print(f"[anti-causal] consistency residual_l1: clean={res_clean:.4f} "
          f"vs perturbed p(x|y)={res_pert:.4f} (materially larger, REUSED clean p_hat)")
    assert res_pert > res_clean                          # DETECTS only via the reused reference

    # HONESTY MEASUREMENT (not a new guarantee): the DEPLOYMENT-realistic single-target
    # read is BLIND. Solve p_hat from the perturbed target's own q_T (as the field must)
    # and from a pure covariate-shift target (identical x-marginal): both read ≈ clean.
    _, p_hat_pert = bbse_weights(C, q_pert, gen.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
    res_pert_own = bbse_consistency(q_pert, C, p_hat_pert)["residual_l1"]
    genc = _gen_k2(p_t=CFG.k2.p_s, tilt="x1", theta=CFG.k2.theta_ent)   # pure covariate shift
    _, _, yhc, _, _ = genc.sample(CFG.mc.consistency_n, r, "T")
    q_cov = pred_hist(yhc, gen.K)
    _, p_hat_cov = bbse_weights(C, q_cov, gen.p_s, CFG.b.w_lab_min, CFG.b.w_lab_max)
    res_cov_own = bbse_consistency(q_cov, C, p_hat_cov)["residual_l1"]
    print(f"[anti-causal/BLIND] deployment single-target reads: x1-tilt(own p_hat)="
          f"{res_pert_own:.4f}, pure-cov-shift(own p_hat)={res_cov_own:.4f} "
          f"~ clean={res_clean:.4f} (the diagnostic is BLIND to these in single-target usage)")
    # DOCUMENT the blindness as a measured fact: the single-target read is within a
    # small band of the clean residual for a violation the reused-reference path flags.
    assert abs(res_pert_own - res_clean) <= 0.05         # BLIND in deployment usage (measured)
    assert res_pert_own < 0.1 * res_pert                 # << the reused-reference read


def test_diagnostic_n_eff_falls_with_shift():
    """DIAGNOSTIC (method_note §1.7, §7.5): ``n_eff`` (Kish) beside the Hájek risk.

    Self-test that ``n_eff`` FALLS as the prevalence shift is made more extreme (a
    heavier ``w_lab`` tail). REPORTED, never gated."""
    neffs = []
    for p_t in CFG.mc.neff_p_t:
        gen = _gen_k2(p_t=p_t)
        r = np.random.default_rng(CFG.seed + 10000)
        X, y, _, _, _ = gen.sample(CFG.mc.neff_n, r, "S")
        w = combine_weights(gen.oracle_w_lab()[y], gen.oracle_w_cov(X), gen.oracle_Z(X))
        neffs.append(kish_n_eff(w) / len(w))             # as a fraction of n
    print("[n_eff] n_eff/n vs |prevalence shift|: "
          + ", ".join(f"p_t={pt}:{ne:.3f}" for pt, ne in zip(CFG.mc.neff_p_t, neffs)))
    assert neffs[0] >= neffs[1] >= neffs[2] - 1e-2       # falls as the shift grows


def test_diagnostic_interval_coverage_report():
    """DIAGNOSTIC (prereg §6.8): synthetic interval-coverage of the ``Ẑ``-combined interval.

    On data with KNOWN ground-truth weighted risk and KNOWN weights (ORACLE combined
    weight ``w_lab*(y)`` under pure label shift, so the ground-truth weighted risk is
    exactly ``R_T``), report realized coverage of the WSR UCB vs nominal ``1-δ`` across
    two prevalence-shift regimes. The operative §6.8 action is the RELABEL, emitted
    where it under-covers; never a gate (only plumbing is asserted)."""
    for p_t, tag in ((CFG.k2.p_t, "moderate"), ([0.95, 0.05], "strong")):
        gen = _gen_k2(p_t=p_t)
        tau0 = CFG.k2.tau0
        rr = np.random.default_rng(CFG.seed + 11000)
        R_T = gen.accepted_risk(tau0, "T", rr, n=CFG.mc.oracle_n)
        covered, neffs = 0, []
        for t in range(CFG.mc.cov_T):
            r = np.random.default_rng(CFG.seed + 11000 + t)
            X, y, _, lc, uc = gen.sample(CFG.mc.neff_n, r, "S")
            acc = uc <= tau0
            L = lc[acc]
            w = gen.oracle_w_lab()[y[acc]]               # KNOWN weights (pure label shift)
            wm = float(w.max())                          # no clip => ground truth is exactly R_T
            ucb = wsr_ucb_weighted(L, w, CFG.b.delta, wm)
            covered += int(R_T <= ucb)
            neffs.append(kish_n_eff(w))
        realized = covered / CFG.mc.cov_T
        label = ("nominal interval, coverage validated empirically"
                 if realized < 1 - CFG.b.delta else "nominal interval")
        print(f"[interval-cov/{tag}] realized={realized:.4f} vs nominal={1 - CFG.b.delta} "
              f"(R_T={R_T:.4f}, n_eff~{np.mean(neffs):.0f}) -> RELABEL: '{label}'")
        assert 0.0 <= realized <= 1.0                    # plumbing only, never gated


def test_diagnostic_mlls_without_bcts_degrades():
    """DIAGNOSTIC (prereg §4A item 4; method_note §1.5): MLLS-without-BCTS degrades.

    When the frozen classifier is miscalibrated, running MLLS on the RAW softmax
    (no BCTS) yields a worse prevalence estimate than MLLS on the BCTS-recalibrated
    softmax -- BCTS is load-bearing. Gap reported; REPORTED, never gated."""
    gen = _gen_k2()
    b_mis = np.asarray(CFG.bcts.b_mis, dtype=float)
    err_without, err_with = [], []
    for t in range(CFG.mc.mapls_T):
        r = np.random.default_rng(CFG.seed + 12000 + t)
        Xs, ys, _, _, _ = gen.sample(CFG.mc.mlls_n, r, "S")
        Xt, _, _, _, _ = gen.sample(CFG.mc.mlls_n, r, "T")
        ls = gen.logits(Xs) / CFG.bcts.T_mis + b_mis
        lt = gen.logits(Xt) / CFG.bcts.T_mis + b_mis
        p_without = mlls_em(softmax(lt), gen.p_s)                       # raw (miscalibrated)
        T, b = fit_bcts(ls, ys)
        p_with = mlls_em(recalibrate_softmax(lt, T, b), gen.p_s)        # BCTS-recalibrated
        err_without.append(float(np.max(np.abs(p_without - gen.p_t))))
        err_with.append(float(np.max(np.abs(p_with - gen.p_t))))
    ew, wi = float(np.mean(err_without)), float(np.mean(err_with))
    print(f"[without-BCTS] MLLS prevalence err: without-BCTS={ew:.4f} > with-BCTS={wi:.4f} "
          f"(BCTS is load-bearing; {ew / max(wi, 1e-9):.1f}x)")
    assert ew > wi                                       # BCTS materially helps (reported)


def test_diagnostic_mapls_small_m():
    """DIAGNOSTIC (prereg §4A item 1): MAPLS Dirichlet-prior ablation in the small-``m`` regime.

    Reports the MAPLS-vs-MLLS+BCTS prevalence error across small target sizes ``m``.
    build_gates.md §6 permits either outcome -- "reduces label-weight error ... OR
    reports 'no measured small-m advantage'." Here the true prevalence is far from
    ``p_S``, so a prior toward ``p_S`` gives NO advantage; that honest result is
    reported. REPORTED, never gated."""
    gen = _gen_k2()
    alpha = 1.0 + CFG.mc.mapls_alpha * np.asarray(gen.p_s)
    rows = []
    for m in CFG.mc.mapls_grid:
        e_mlls, e_mapls = [], []
        for t in range(CFG.mc.mapls_T):
            r = np.random.default_rng(CFG.seed + 13000 + 41 * m + t)
            Xs, ys, _, _, _ = gen.sample(4000, r, "S")
            Xt, _, _, _, _ = gen.sample(m, r, "T")
            T, b = fit_bcts(gen.logits(Xs), ys)
            sm_t = recalibrate_softmax(gen.logits(Xt), T, b)
            e_mlls.append(float(np.max(np.abs(mlls_em(sm_t, gen.p_s) - gen.p_t))))
            e_mapls.append(float(np.max(np.abs(mapls_em(sm_t, gen.p_s, alpha) - gen.p_t))))
        rows.append((m, float(np.mean(e_mlls)), float(np.mean(e_mapls))))
    verdict = ("MAPLS reduces small-m error"
               if any(ma < ml for _, ml, ma in rows) else "no measured small-m advantage")
    print("[mapls] m -> (MLLS, MAPLS) prevalence err: "
          + ", ".join(f"{m}:({ml:.4f},{ma:.4f})" for m, ml, ma in rows)
          + f" -> {verdict}")
    assert all(np.isfinite(ml) and np.isfinite(ma) for _, ml, ma in rows)   # plumbing only


def test_diagnostic_zcombine_5arm_decomposition():
    """DIAGNOSTIC (prereg §4A item 3): the ``Ẑ``-combine 5-arm decomposition.

    Reports the Hájek accepted risk of five reweightings at a MATCHED answer-rate on
    the pure-label K=2 regime (oracle inputs): (a) no-reweight, (b) covariate-only,
    (c) label-only, (d) naïve product, (e) ``Ẑ``-divide. The only HARD sub-claim (that
    (e) matches the oracle and beats (d)) is gated separately in
    :func:`test_zdivide_recovers_oracle_product_doublecounts`; here all five are just
    reported vs the known ``R_S`` / ``R_T``. REPORTED, never gated."""
    gen = _gen_k2()
    tau0 = CFG.k2.tau0
    rr = np.random.default_rng(CFG.seed + 14000)
    R_S = gen.accepted_risk(tau0, "S", rr, n=CFG.mc.oracle_n)
    R_T = gen.accepted_risk(tau0, "T", rr, n=CFG.mc.oracle_n)
    r = np.random.default_rng(CFG.seed + 14001)
    X, y, _, lc, uc = gen.sample(CFG.mc.n_big, r, "S")
    acc = uc <= tau0
    L = lc[acc]
    wl_y = gen.oracle_w_lab()[y[acc]]
    wc = gen.oracle_w_cov(X)[acc]
    Z = gen.oracle_Z(X)[acc]
    arms = {
        "a_none": hajek_risk(L, np.ones_like(L)),
        "b_cov": hajek_risk(L, wc),
        "c_lab": hajek_risk(L, wl_y),
        "d_product": hajek_risk(L, wl_y * wc),
        "e_zdivide": hajek_risk(L, combine_weights(wl_y, wc, Z)),
    }
    print(f"[5-arm] R_S={R_S:.4f} R_T={R_T:.4f} | "
          + ", ".join(f"{k}={v:.4f}" for k, v in arms.items()))
    assert all(np.isfinite(v) for v in arms.values())    # plumbing only
    # sanity: the Ẑ-divide arm tracks R_T (the load-bearing exactness is gated in HG3)
    assert abs(arms["e_zdivide"] - R_T) <= CFG.mc.track_tol


def test_diagnostic_factorization_entanglement():
    """DIAGNOSTIC (method_note §3.4 step 3, §7.4; prereg §5.4): factorization-entanglement.

    The direct joint weight ``ŵ_joint(x,y)`` (here the ORACLE ``w_joint*``) vs the
    factorized ``ŵ_lab·ŵ_cov/Ẑ``: report the mean-squared log-ratio divergence in a
    FACTORIZABLE regime (covariate tilt ⟂ the label signal, on x2) vs an ENTANGLED
    regime (tilt along the label signal, on x1, so ``p(x|y)`` moves with the shift).
    Self-test that the divergence is ≈0 when factorizable and materially larger when
    entangled -- so the diagnostic DETECTS when the factorizable-shift premise fails
    (on a real pair, large divergence => downgrade the combined claim, prereg §3.1).
    The divergence is a REPORTED diagnostic in deployment (never gates the rung);
    THIS TEST is a HARD plumbing self-test asserting the divergence separates the
    KNOWN factorizable vs entangled synthetics -- consistent with the other
    ``test_diagnostic_*`` self-tests. (Scope note: 'factorizable' here is the
    invariant-density-ratio condition, strictly stronger than the FJS sense
    w(x,y)=U(x)V(y); see REVIEW_FINDINGS F12.)"""
    def _divergence(gen):
        r = np.random.default_rng(CFG.seed + 15000)
        X, y, _, _, _ = gen.sample(CFG.mc.entangle_n, r, "S")
        w_joint = gen.oracle_w_joint(X, y)
        fac = combine_weights(gen.oracle_w_lab()[y], gen.oracle_w_cov(X), gen.oracle_Z(X))
        return float(np.mean((np.log(fac) - np.log(w_joint)) ** 2))

    genf = _gen_k2(tilt="x2", theta=CFG.k2.theta_cov)    # factorizable
    gene = _gen_k2(tilt="x1", theta=CFG.k2.theta_ent)    # entangled
    d_fac = _divergence(genf)
    d_ent = _divergence(gene)
    print(f"[entanglement] factorized-vs-joint mean-sq log-ratio: "
          f"factorizable={d_fac:.2e} (~0) vs entangled={d_ent:.4f} (materially larger)")
    assert d_fac < 1e-10                                  # exact factorization => ~machine zero
    assert d_ent > 100 * max(d_fac, 1e-12)               # entanglement is DETECTED


def test_diagnostic_z_plugin_bias_sensitivity():
    """DIAGNOSTIC (build_gates.md §6 honesty rail 7; method_note §3.5, §7.4): the
    ``Ẑ(x)`` softmax plug-in bias is NOT removed by recalibration.

    ``Ẑ = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}`` uses a recalibrated softmax as a plug-in for
    ``p_S(y|x)``; BCTS REDUCES but does not ELIMINATE the plug-in bias, and the
    residual is amplified on rare / high-``w_lab`` classes. Self-test the sensitivity
    sweep: distort the frozen classifier with a PER-CLASS temperature (a
    mis-specification BCTS's single-temperature family cannot fully invert), then
    report ``mean|Ẑ_hat - Z*|`` for the RAW-softmax plug-in vs the BCTS-recalibrated
    plug-in across an increasingly extreme prevalence grid. BCTS shrinks the residual
    sharply, but it stays > 0 and GROWS with the high-``w_lab`` tail -- the
    "non-vanishing, amplified on rare classes" behaviour the rail names. REPORTED,
    never gated (asserts only direction / finiteness / sign, never a threshold)."""
    T_cls = np.array([2.0, 4.0])                          # per-class temp: mis-specified for BCTS
    b_mis = np.asarray(CFG.bcts.b_mis, dtype=float)
    rows = []
    for p_t in CFG.mc.neff_p_t:                           # increasingly extreme prevalence
        gen = _gen_k2(p_t=p_t)
        wlab = gen.oracle_w_lab()
        r = np.random.default_rng(CFG.seed + 16000)
        Xs, ys, _, _, _ = gen.sample(CFG.mc.mlls_n, r, "S")
        Xt, _, _, _, _ = gen.sample(CFG.mc.mlls_n, r, "T")
        Zstar = gen.oracle_Z(Xt)
        ls = gen.logits(Xs) / T_cls + b_mis
        lt = gen.logits(Xt) / T_cls + b_mis
        err_raw = float(np.mean(np.abs(z_corrector(wlab, softmax(lt)) - Zstar)))
        T, b = fit_bcts(ls, ys)
        err_bcts = float(np.mean(np.abs(z_corrector(wlab, recalibrate_softmax(lt, T, b)) - Zstar)))
        rows.append((p_t, wlab.max(), err_raw, err_bcts))
    print("[z-plugin-bias] p_t (w_lab_max) -> (raw Zhat err, BCTS Zhat err) vs oracle Z*: "
          + ", ".join(f"{pt}({wm:.1f}):({er:.4f},{eb:.4f})" for pt, wm, er, eb in rows))
    # BCTS reduces the plug-in bias (direction only, never gated) ...
    assert all(eb <= er + 1e-9 for _, _, er, eb in rows)
    # ... but does NOT eliminate it -- residual stays > 0 (non-vanishing) ...
    resid = [eb for _, _, _, eb in rows]
    assert all(rb > 1e-4 for rb in resid)
    # ... and is amplified as the high-w_lab tail grows (rarer minority class).
    assert resid[-1] >= resid[0] - 1e-4
