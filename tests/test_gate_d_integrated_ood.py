"""Gate-D -- integrated OOD budget & routing (fourth rung, the ladder's synthetic terminus).

Realizes the Definition of Done of ``build_gates.md §7``. G-D stacks the OOD screen and
the decoupled decision rule on top of the Gate-A/B/C weighted selective pipeline:

  * the feature-space OOD score ``o(x)`` -- **Mahalanobis++** L2-normalized
    tied-covariance primary (Müller et al. 2025), with plain Lee et al. (2018) / kNN /
    energy detector-agnostic ablations (``ood/detector.py``);
  * the leakage-budget screen ``t_ood`` set on the exposure set ``O`` to spend a target
    far-OOD leakage ``alpha_ood`` (``conformal/budget.py::set_t_ood``);
  * the integrated answered event
    ``A(x) = (u <= tau) & (o <= t_ood) & (w_cov <= w_max)`` and its three-way
    coverage decomposition (route-on-OOD / abstain-on-weight / defer-on-uncertainty),

and verifies -- on synthetic features with KNOWN in-scope vs far-OOD ground truth
(``tests/_synth.py::GaussianOODFeatures``) -- that each cited method reproduces its
stated property against ground truth, and that the two scope guards compose correctly.

Honesty rails enforced by construction (``build_gates.md §7``):
  * SYNTHETIC wiring checks on known ground truth. OOD detection is NOT
    distribution-free learnable (Fang et al. 2022); NO distribution-free / finite-sample
    OOD guarantee -- ``t_ood`` is a MEASURED screen with a reported leakage rate on a
    stated, swappable ``O``.
  * NO certified additive ``alpha = alpha_acc + alpha_ood`` union split; the two budgets
    are tracked and reported SEPARATELY.
  * ``t_ood`` is the SOLE far-OOD guard; ``w_max`` is a variance/boundedness control
    whose routed tail costs only coverage -- the two are DECOUPLED.
  * ``kappa(Sigma_hat)``, effective rank, ``n_eff`` and the interval-coverage report are
    REPORTED DIAGNOSTICS, never pass/fail gates.

Each test maps to one Definition-of-Done bullet; HARD GATES assert, DIAGNOSTICS
measure/print and only sanity-check plumbing. Numeric budgets come from
``configs/gate_d.toml`` (the committed "proposed" run config). All print output is
ASCII-only (cp1252-safe stdout; the Unicode lives only in docstrings).
"""

import pathlib
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from conformal.rcps import (
    wsr_ucb,
    hajek_risk,
    kish_n_eff,
    wsr_ucb_weighted,
    wsr_lcb_weighted,
)
from conformal.weights import cov_weight_from_d
from conformal.label_shift import bbse_weights, confusion_matrix, pred_hist
from conformal.budget import (
    leakage_rate,
    set_t_ood,
    accept_mask,
    route_decomposition,
    report_budgets,
    OperatingBudgets,
)
from conformal.folds import assert_group_disjoint
from ood.detector import (
    l2_normalize,
    TiedMahalanobis,
    knn_score,
    energy_score,
    kappa_sigma,
    ood_auroc,
    fpr_at_tpr,
)
from _synth import GaussianOODFeatures


# --------------------------------------------------------------------------- #
# Run config (committed proposed numbers -- build_gates.md §7 open question).   #
# --------------------------------------------------------------------------- #
_CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gate_d.toml"


def _load_cfg():
    with open(_CFG_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    return SimpleNamespace(
        b=SimpleNamespace(**raw["budgets"]),
        f=SimpleNamespace(**raw["features"]),
        mc=SimpleNamespace(**raw["monte_carlo"]),
        seed=raw["seed"]["base"],
    )


CFG = _load_cfg()


def _sd_vector(collapse=0):
    """Tied-covariance std per dim: class + shift dims std 1, noise dims ``sd_noise``.

    ``collapse > 0`` squashes the first ``collapse`` noise dims to near-zero variance
    (a low-rank / feature-collapsed embedding) so ``Sigma_hat`` becomes ill-conditioned
    -- the kappa/effective-rank feature-collapse diagnostic.
    """
    D, K = CFG.f.D, CFG.f.K
    sd = np.ones(D)
    sd[K:D - 1] = CFG.f.sd_noise
    if collapse > 0:
        sd[K:K + int(collapse)] = 0.03
    return sd


def _gen(collapse=0, **over):
    """Gate-D OOD-feature generator from configs/gate_d.toml [features], with overrides."""
    kw = dict(
        D=CFG.f.D, K=CFG.f.K, class_sep=CFG.f.class_sep, shift_delta=CFG.f.shift_delta,
        far_radius=CFG.f.far_radius, near_radius=CFG.f.near_radius, id_std=CFG.f.id_std,
        sd=_sd_vector(collapse),
    )
    kw.update(over)
    return GaussianOODFeatures(**kw)


def _fit_detector(gen, rng, normalize=True, n=None):
    """Fit the tied-covariance detector on a fresh in-scope (source) draw."""
    n = CFG.mc.detector_fit_n if n is None else n
    phi, y, _ = gen.sample_id(n, rng, "S")
    return TiedMahalanobis(normalize=normalize).fit(phi, y)


def _inscope_screen(det, gen, rng, n=8000, q=0.999):
    """A clean OOD screen set just above the in-scope score BULK (``q``-quantile of ID).

    For the routing-LOGIC gates (HG5/HG6) the leakage BUDGET is not the object under
    test -- the decoupling / conjunction logic is. A screen at a high in-scope
    percentile routes ~all far-OOD out (far-OOD ``o`` sits well above the in-scope
    bulk) while keeping the in-scope points, and -- unlike the raw max -- is not driven
    by a single ID tail outlier. The budget-spending ``t_ood`` is exercised in HG3/HG4.
    """
    o_id = det.score(gen.sample_id(n, rng, "S")[0])
    return float(np.quantile(o_id, q))


# =========================================================================== #
# HARD GATES                                                                    #
# =========================================================================== #
def _ref_maha(phi_train, y_train, phi_test, normalize):
    """Brute-force numpy reference for the tied-covariance Mahalanobis min-score.

    Independent of ``ood.detector`` internals: (optionally) L2-normalize, per-class
    means, tied covariance ``Sigma = (1/N) sum_c sum_{i in c}(x-mu_c)(x-mu_c)^T`` (the
    GDA MLE), then ``min_c (x-mu_c)^T Sigma^-1 (x-mu_c)``.
    """
    if normalize:
        phi_train = phi_train / np.linalg.norm(phi_train, axis=1, keepdims=True)
        phi_test = phi_test / np.linalg.norm(phi_test, axis=1, keepdims=True)
    classes = np.unique(y_train)
    D = phi_train.shape[1]
    N = phi_train.shape[0]
    means = np.array([phi_train[y_train == c].mean(axis=0) for c in classes])
    S = np.zeros((D, D))
    for c in classes:
        d = phi_train[y_train == c] - phi_train[y_train == c].mean(axis=0)
        S += d.T @ d
    prec = np.linalg.inv(S / N)
    diff = phi_test[:, None, :] - means[None, :, :]
    return np.min(np.einsum("nkd,de,nke->nk", diff, prec, diff), axis=1)


def test_maha_pp_crosscheck_reference():
    """HARD (gate 1): Maha++ (and the plain-Lee ablation) cross-check a brute-force reference.

    build_gates.md §7 gate 1: on >=5 seeded draws the PRIMARY o(x) from ood/detector.py
    equals the L2-normalized tied-covariance Mahalanobis min-score from an independent
    numpy reference to atol=1e-8 (normalization ON for the primary), and the plain
    Lee et al. ablation (normalization OFF) matches the unnormalized reference. Confirms
    the normalization path is genuinely ON for the primary (it changes the score).
    """
    for normalize, tag in ((True, "Maha++"), (False, "plain-Lee")):
        max_err = 0.0
        for s in range(CFG.mc.xcheck_seeds):
            r = np.random.default_rng(CFG.seed + 100 + s)
            gen = _gen()
            phi_tr, y_tr, _ = gen.sample_id(CFG.mc.xcheck_n, r, "S")
            phi_id, _, _ = gen.sample_id(300, r, "S")
            phi_od, _ = gen.sample_far_ood(300, r)
            phi_te = np.vstack([phi_id, phi_od])           # mix of ID and far-OOD test pts
            det = TiedMahalanobis(normalize=normalize).fit(phi_tr, y_tr)
            o = det.score(phi_te)
            o_ref = _ref_maha(phi_tr, y_tr, phi_te, normalize)
            max_err = max(max_err, float(np.max(np.abs(o - o_ref))))
        print(f"[xcheck/{tag}] max|o - o_ref| over {CFG.mc.xcheck_seeds} seeds = {max_err:.2e} "
              f"(atol={CFG.mc.xcheck_atol})")
        assert max_err <= CFG.mc.xcheck_atol

    # the normalization path is genuinely ON for the primary: it changes the score.
    r = np.random.default_rng(CFG.seed + 199)
    gen = _gen()
    phi_tr, y_tr, _ = gen.sample_id(CFG.mc.xcheck_n, r, "S")
    phi_te, _ = gen.sample_far_ood(200, r)
    o_pp = TiedMahalanobis(normalize=True).fit(phi_tr, y_tr).score(phi_te)
    o_plain = TiedMahalanobis(normalize=False).fit(phi_tr, y_tr).score(phi_te)
    assert np.max(np.abs(o_pp - o_plain)) > 1.0           # primary != plain (normalization is live)


def test_ood_score_separates_far():
    """HARD (gate 2): the primary o(x) separates in-scope from the KNOWN far-OOD O.

    build_gates.md §7 gate 2: far-tier AUROC >= auroc_min averaged over >=200 resamples;
    FPR95 also reported. (AUROC pass-line proposed; the docs mandate reporting
    AUROC/FPR95 but state no numeric pass-line -- prereg §5.2; method_note §4.3.)
    """
    gen = _gen()
    det = _fit_detector(gen, np.random.default_rng(CFG.seed + 200))
    aurocs, fprs = [], []
    for t in range(CFG.mc.auroc_T):
        r = np.random.default_rng(CFG.seed + 2000 + t)
        phi_id, _, _ = gen.sample_id(CFG.mc.auroc_n_id, r, "S")
        phi_od, _ = gen.sample_far_ood(CFG.mc.auroc_n_ood, r)
        s_id, s_od = det.score(phi_id), det.score(phi_od)
        lab = np.r_[np.zeros(len(s_id)), np.ones(len(s_od))]
        aurocs.append(ood_auroc(np.r_[s_id, s_od], lab))
        fprs.append(fpr_at_tpr(s_od, s_id, 0.95))
    print(f"[auroc] mean far-tier AUROC={np.mean(aurocs):.4f} (min={np.min(aurocs):.4f}, "
          f"pass>={CFG.mc.auroc_min}); mean FPR95={np.mean(fprs):.4f}")
    assert np.mean(aurocs) >= CFG.mc.auroc_min


def test_t_ood_spends_alpha_ood():
    """HARD (gate 3): t_ood is the cutoff on O spending leakage budget alpha_ood.

    build_gates.md §7 gate 3: for alpha_ood in {0.01, 0.05, 0.10}, the in-sample
    leakage on the O used to set t_ood is <= alpha_ood and within one scan step
    (1/leak_fit_n) of it, and the OUT-OF-SAMPLE leakage on a fresh large O (the analytic
    leakage-vs-cutoff curve) is within leak_tol of alpha_ood.
    """
    gen = _gen()
    det = _fit_detector(gen, np.random.default_rng(CFG.seed + 300))
    for alpha in CFG.b.alpha_ood:
        ins, outs = [], []
        for s in range(CFG.mc.leak_seeds):
            r = np.random.default_rng(CFG.seed + 30000 + int(1000 * alpha) + s)
            o_fit = det.score(gen.sample_far_ood(CFG.mc.leak_fit_n, r)[0])
            t = set_t_ood(o_fit, alpha)
            ins.append(leakage_rate(o_fit, t))
            r2 = np.random.default_rng(CFG.seed + 35000 + int(1000 * alpha) + s)
            o_ref = det.score(gen.sample_far_ood(CFG.mc.leak_ref_n, r2)[0])
            outs.append(leakage_rate(o_ref, t))          # analytic leakage at the chosen t_ood
        m_in, m_out = float(np.mean(ins)), float(np.mean(outs))
        step = 1.0 / CFG.mc.leak_fit_n
        print(f"[t_ood/alpha={alpha}] in-sample leak={m_in:.4f} (<= alpha, within {step:.2e}); "
              f"out-of-sample leak={m_out:.4f} (|.-alpha| <= {CFG.mc.leak_tol})")
        assert m_in <= alpha + 1e-12                     # spends but never exceeds the budget
        assert alpha - m_in <= step + 1e-12              # within one scan step of alpha
        assert abs(m_out - alpha) <= CFG.mc.leak_tol     # matches the analytic curve at t_ood


def test_screen_reduces_far_ood_leakage():
    """HARD (gate 4): the screen measurably reduces far-OOD leakage vs no screen.

    build_gates.md §7 gate 4 (the core Milestone-D claim): far-OOD is CONFIDENT
    (passes the u gate) with w_cov < w_max (passes the weight gate), so with the OOD
    screen OFF it is answered (leaks). Turning the screen ON (o <= t_ood at alpha_ood)
    strictly lowers the leakage into A(x), with NON-OVERLAPPING betting intervals over
    >=200 resamples. Leakage ON is a genuine out-of-sample measured number (~alpha_ood,
    not identically 0): some far-OOD leaks, so the reduction is a non-trivial delta.
    """
    gen = _gen()
    det = _fit_detector(gen, np.random.default_rng(CFG.seed + 400))
    tau, wmax, alpha = CFG.f.tau0, CFG.b.w_max, CFG.mc.red_alpha
    off_ind, on_ind = [], []
    for t in range(CFG.mc.red_T):
        r = np.random.default_rng(CFG.seed + 40000 + t)
        t_ood = set_t_ood(det.score(gen.sample_far_ood(CFG.mc.leak_fit_n, r)[0]), alpha)
        phi_od, _ = gen.sample_far_ood(CFG.mc.red_n_ood, r)      # fresh far-OOD test
        u, o, w = gen.uncertainty(phi_od), det.score(phi_od), gen.oracle_w_cov(phi_od)
        ans_off = (u <= tau) & (w <= wmax)                       # screen OFF (no OOD gate)
        ans_on = ans_off & (o <= t_ood)                          # screen ON
        off_ind.append(ans_off)
        on_ind.append(ans_on)
    off = np.concatenate(off_ind).astype(float)
    on = np.concatenate(on_ind).astype(float)
    one = np.ones_like(off)
    off_lcb = wsr_lcb_weighted(off, one, CFG.b.delta, 1.0)
    on_ucb = wsr_ucb_weighted(on, np.ones_like(on), CFG.b.delta, 1.0)
    print(f"[screen] leak OFF={off.mean():.4f} [LCB {off_lcb:.4f}] vs ON={on.mean():.4f} "
          f"[UCB {on_ucb:.4f}] over {CFG.mc.red_T} resamples (non-overlap => screen reduces leakage)")
    assert on.mean() < off.mean()                                # strictly lower
    assert on_ucb < off_lcb                                      # non-overlapping intervals


def test_t_ood_is_sole_far_ood_guard():
    """HARD (gate 5): t_ood is the SOLE far-OOD guard (w_max does not flag far-OOD-d0).

    build_gates.md §7 gate 5: on the engineered far-OOD points with d(x) ~ 0
    (w_cov ~ 0), the weight gate routes ~0 of them (w_cov ~ 0 <= w_max) while the OOD
    gate routes ~1.0 (o > t_ood). method_note §4.2, §1.5.1.
    """
    gen = _gen()
    r = np.random.default_rng(CFG.seed + 500)
    det = _fit_detector(gen, r)
    t_ood = _inscope_screen(det, gen, r)                         # screen above the in-scope bulk
    phi_d0, _ = gen.far_ood_d0(CFG.mc.probe_n, r)
    o, w = det.score(phi_d0), gen.oracle_w_cov(phi_d0)
    routed_by_weight = float(np.mean(w > CFG.b.w_max))
    routed_by_ood = float(np.mean(o > t_ood))
    print(f"[sole-guard] far-OOD-d0: max w_cov={w.max():.2e} (<= w_max={CFG.b.w_max}); "
          f"routed by w_max={routed_by_weight:.3f} (~0), by t_ood={routed_by_ood:.3f} (~1)")
    assert routed_by_weight == 0.0                               # w_max flags none of them
    assert routed_by_ood >= 0.99                                 # t_ood flags ~all of them


def test_routing_mechanisms_decoupled():
    """HARD (gate 6): the two routing mechanisms are DECOUPLED; the decomposition sums to 1-coverage.

    build_gates.md §7 gate 6: a labeled pool with four disjoint-by-construction groups
    (answered / defer-only / abstain-weight-only / route-OOD-only). The near-OOD
    abstain-weight group has w_cov > w_max but o <= t_ood, so it is routed by the WEIGHT
    gate and NOT the OOD gate. Each mechanism's count matches the constructed membership
    EXACTLY and the three not-answered fractions sum to 1 - coverage. o for the in-scope
    groups is the REAL detector score on real ID features; the far-OOD group's o is the
    real score on far-OOD features; the u / w memberships are constructed
    (on real data w_cov is the Gate-B discriminator raw weight).
    """
    gen = _gen()
    r = np.random.default_rng(CFG.seed + 600)
    det = _fit_detector(gen, r)
    tau, wmax = CFG.f.tau0, CFG.b.w_max
    t_ood = _inscope_screen(det, gen, r)                          # screen in the ID/far-OOD gap
    g = CFG.mc.group_n

    # filter to UNAMBIGUOUS membership: in-scope points strictly below the screen ...
    phi_id, _, _ = gen.sample_id(6 * g, r, "S")
    o_in = det.score(phi_id)
    o_in = o_in[o_in < t_ood]
    assert o_in.size >= 3 * g                                     # enough clean in-scope points
    o_in = o_in[:3 * g]
    # ... and far-OOD points strictly above it (the route-OOD group)
    phi_far, _ = gen.sample_far_ood(2 * g, r)
    o_far = det.score(phi_far)
    o_far = o_far[o_far > t_ood]
    assert o_far.size >= g                                        # enough clean far-OOD points
    o_far = o_far[:g]

    # four groups, each failing (at most) one gate, in order [answered, defer, weight, ood]
    u = np.concatenate([np.zeros(g), np.full(g, tau + 0.5), np.zeros(g), np.zeros(g)])
    o = np.concatenate([o_in[:g], o_in[g:2 * g], o_in[2 * g:3 * g], o_far])
    w = np.concatenate([np.ones(g), np.ones(g), np.full(g, wmax + 5.0), np.ones(g)])
    dec = route_decomposition(u, o, w, tau, t_ood, wmax)

    print(f"[decouple] counts answered={dec.mask_answered.sum()} defer={dec.mask_defer.sum()} "
          f"abstain_weight={dec.mask_abstain_weight.sum()} route_ood={dec.mask_route_ood.sum()} "
          f"(each should be {g})")
    assert dec.mask_answered.sum() == g
    assert dec.mask_defer.sum() == g
    assert dec.mask_abstain_weight.sum() == g
    assert dec.mask_route_ood.sum() == g
    # DECOUPLING: the near-OOD w>w_max group is booked to abstain_weight, NOT route_ood
    assert dec.mask_abstain_weight[2 * g:3 * g].all()
    assert not dec.mask_route_ood[2 * g:3 * g].any()
    # the three not-answered fractions sum to exactly 1 - coverage
    not_ans = dec.defer_uncertainty + dec.route_ood + dec.abstain_weight
    assert abs(not_ans - (1.0 - dec.answered)) <= 1e-12


def test_accept_rule_exact_conjunction():
    """HARD (gate 7): A(x) is the exact conjunction of the three gates.

    build_gates.md §7 gate 7: for >=1000 randomized (u, o, w_cov, tau, t_ood, w_max)
    inputs, A(x) == (u <= tau) & (o <= t_ood) & (w_cov <= w_max) elementwise (zero
    mismatches), and gate evaluation ORDER does not change the answered set.
    """
    r = np.random.default_rng(CFG.seed + 700)
    n = CFG.mc.conj_n
    u = r.uniform(0, 1, n)
    o = r.uniform(0, 300, n)
    w = r.uniform(0, 2 * CFG.b.w_max, n)
    # scalar thresholds
    tau, t, wm = 0.5, 130.0, CFG.b.w_max
    A = accept_mask(u, o, w, tau, t, wm)
    ref = (u <= tau) & (o <= t) & (w <= wm)
    assert int(np.sum(A != ref)) == 0
    assert np.array_equal(A, (w <= wm) & (o <= t) & (u <= tau))   # order-independent
    # per-point thresholds too
    tau_v, t_v, wm_v = r.uniform(0.2, 0.8, n), r.uniform(50, 200, n), r.uniform(4, 12, n)
    A2 = accept_mask(u, o, w, tau_v, t_v, wm_v)
    ref2 = (u <= tau_v) & (o <= t_v) & (w <= wm_v)
    n_mismatch = int(np.sum(A2 != ref2))
    print(f"[conjunction] {n} randomized inputs: scalar-threshold mismatches=0, "
          f"per-point mismatches={n_mismatch}")
    assert n_mismatch == 0


def test_budgets_tracked_separately_no_union_split():
    """HARD (gate 8): alpha_acc and alpha_ood are two SEPARATELY measured budgets.

    build_gates.md §7 gate 8: budget.py exposes them as distinct reported quantities;
    any alpha sum is a reporting-convenience headline only. Negative check: the code
    makes NO claim that realized accepted risk + realized leakage <= alpha_acc+alpha_ood
    with confidence 1-delta (certified_additive_split is False). method_note §5.3;
    prereg §6.2, §8.
    """
    rep = report_budgets(CFG.b.alpha_acc, 0.05, realized_accept_risk=0.082, realized_leakage=0.049)
    print(f"[budgets] alpha_acc={rep['alpha_acc']} (realized risk {rep['realized_accept_risk']}); "
          f"alpha_ood={rep['alpha_ood']} (realized leak {rep['realized_leakage']}); "
          f"alpha_sum={rep['alpha_sum']} REPORTING-ONLY; certified_additive_split="
          f"{rep['certified_additive_split']}")
    # the two budgets are DISTINCT reported quantities
    assert rep["alpha_acc"] == CFG.b.alpha_acc
    assert rep["alpha_ood"] == 0.05
    assert rep["realized_accept_risk"] == 0.082
    assert rep["realized_leakage"] == 0.049
    # NO certified additive union split
    assert rep["certified_additive_split"] is False
    assert rep["alpha_sum"] == CFG.b.alpha_acc + 0.05             # reporting convenience only
    # OperatingBudgets keeps them as two separate fields (never a summed certificate)
    ob = OperatingBudgets(CFG.b.alpha_acc, 0.05)
    assert ob.alpha_acc == CFG.b.alpha_acc and ob.alpha_ood == 0.05
    assert "certified" not in rep or rep.get("certified_additive_split") is False


def test_fold_disjointness_O_vs_inscope():
    """HARD (gate 9): the exposure set O is disjoint from the in-scope cal/test pools.

    build_gates.md §7 gate 9 / method_note §1.7 (set-intersection = 0 artifact) over
    O, D_disc, D_cal, held-out test.
    """
    n = 2000
    gids = SimpleNamespace(
        O=np.arange(0, n),
        D_disc=np.arange(n, 2 * n),
        D_cal=np.arange(2 * n, 3 * n),
        test=np.arange(3 * n, 4 * n),
    )
    assert assert_group_disjoint(O=gids.O, D_disc=gids.D_disc, D_cal=gids.D_cal,
                                 test=gids.test, verbose=True)
    with pytest.raises(AssertionError):
        assert_group_disjoint(O=[1, 2, 3], D_cal=[3, 4, 5])      # negative control


def test_no_regression_ladder_surface_intact():
    """HARD (gate 10): the A/B/C ladder surface is intact (smoke; full check = the combined run).

    build_gates.md §7 gate 10: tests/test_gate_a/b/c plus test_gate_d all pass in one
    pytest invocation ("A-D still reproduce"). This smoke test exercises a public
    call from each earlier rung so an import/signature regression trips here; the
    load-bearing regression check is the combined A+B+C+D pytest run.
    """
    # Gate A: WSR UCB on a known-mean sample
    ub = wsr_ucb(np.zeros(200), CFG.b.delta)
    assert 0.0 <= ub <= 1.0
    # Gate A/B: Hajek weighted risk + Kish n_eff
    assert abs(hajek_risk(np.array([0.0, 1.0]), np.array([1.0, 1.0])) - 0.5) < 1e-12
    assert kish_n_eff(np.ones(10)) == pytest.approx(10.0)
    # Gate B: clipped covariate weight
    assert cov_weight_from_d(np.array([0.5]), 1.0, 8.0)[0] == pytest.approx(1.0)
    # Gate C: BBSE recovers a trivial (no-shift) ratio ~ 1
    C = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1], 2)
    q = pred_hist([0, 1], 2)
    w_lab, _ = bbse_weights(C, q, np.array([0.5, 0.5]), 1e-3, 50.0)
    print(f"[regression] Gate-A/B/C surfaces intact; trivial w_lab={w_lab}")
    assert np.allclose(w_lab, 1.0, atol=1e-6)


# =========================================================================== #
# REPORTED DIAGNOSTICS (measured/printed, never a pass/fail gate)              #
# =========================================================================== #
def test_diagnostic_exposure_set_sensitivity():
    """DIAGNOSTIC (method_note §4.3; prereg §5.2): exposure-set sensitivity of t_ood.

    Re-run t_ood selection on >=2 distinct synthetic O and report how t_ood and its
    realized (out-of-sample) leakage move. REPORTED, never gated (plumbing only)."""
    gen = _gen()
    det = _fit_detector(gen, np.random.default_rng(CFG.seed + 800))
    alpha = CFG.mc.red_alpha
    rows = []
    for s in range(CFG.mc.exposure_seeds):
        r = np.random.default_rng(CFG.seed + 8100 + s)
        t = set_t_ood(det.score(gen.sample_far_ood(CFG.mc.leak_fit_n, r)[0]), alpha)
        r2 = np.random.default_rng(CFG.seed + 8600 + s)
        leak = leakage_rate(det.score(gen.sample_far_ood(CFG.mc.leak_ref_n, r2)[0]), t)
        rows.append((t, leak))
    print("[exposure] distinct O -> (t_ood, out-of-sample leak): "
          + ", ".join(f"({t:.2f},{lk:.4f})" for t, lk in rows))
    assert all(lk <= alpha + CFG.mc.leak_tol for _, lk in rows)   # plumbing only


def test_diagnostic_kappa_sigma_feature_collapse():
    """DIAGNOSTIC (method_note §4.1; prereg §5.4): kappa(Sigma_hat) + effective rank.

    On a deliberately feature-collapsed (low-rank) embedding the effective rank of the
    tied covariance DROPS and kappa RISES (the feature-collapse signature). REPORTED,
    never gates the rung."""
    r = np.random.default_rng(CFG.seed + 900)
    g0 = _gen()
    det0 = TiedMahalanobis(normalize=False).fit(*g0.sample_id(CFG.mc.detector_fit_n, r, "S")[:2])
    k0 = kappa_sigma(det0.cov_)
    r2 = np.random.default_rng(CFG.seed + 901)
    gc = _gen(collapse=CFG.mc.collapse_dims)
    detc = TiedMahalanobis(normalize=False).fit(*gc.sample_id(CFG.mc.detector_fit_n, r2, "S")[:2])
    kc = kappa_sigma(detc.cov_)
    print(f"[kappa] well-conditioned: kappa={k0['kappa']:.2f} eff_rank={k0['eff_rank']:.2f} | "
          f"collapsed({CFG.mc.collapse_dims} dims): kappa={kc['kappa']:.2f} eff_rank={kc['eff_rank']:.2f}")
    assert kc["kappa"] > k0["kappa"]                             # collapse inflates the condition number
    assert kc["eff_rank"] < k0["eff_rank"]                       # and drops the effective rank


def test_diagnostic_detector_agnostic_routing():
    """DIAGNOSTIC (method_note §4.1; prereg §5.2; flagship §6.1): routing rule is detector-agnostic.

    Swap o(x) among {Maha++, plain-Lee, energy, kNN} holding the routing logic fixed;
    each detector yields a valid AUROC/FPR95/leakage triple at its OWN leakage-matched
    t_ood, and the routing RULE (accept_mask) is identical. The answered SET is NOT
    expected to be identical across detectors -- only the rule/logic is. REPORTED, never
    gated."""
    gen = _gen()
    r = np.random.default_rng(CFG.seed + 1000)
    phi_fit, y_fit, _ = gen.sample_id(CFG.mc.detector_fit_n, r, "S")
    det = TiedMahalanobis(normalize=True).fit(phi_fit, y_fit)
    detp = TiedMahalanobis(normalize=False).fit(phi_fit, y_fit)
    phi_id, _, _ = gen.sample_id(2000, r, "S")
    phi_od, _ = gen.sample_far_ood(2000, r)
    lab = np.r_[np.zeros(len(phi_id)), np.ones(len(phi_od))]
    alpha, tau, wmax = CFG.mc.red_alpha, CFG.f.tau0, CFG.b.w_max
    scorers = {
        "maha++": lambda P: det.score(P),
        "plain-lee": lambda P: detp.score(P),
        "energy": lambda P: energy_score(det.logits(P)),
        "knn": lambda P: knn_score(P, phi_fit, CFG.f.knn_k),
    }
    for name, f in scorers.items():
        s_id, s_od = f(phi_id), f(phi_od)
        auroc = ood_auroc(np.r_[s_id, s_od], lab)
        fpr = fpr_at_tpr(s_od, s_id, 0.95)
        t = set_t_ood(s_od, alpha)
        leak = leakage_rate(s_od, t)
        # the routing RULE is the SAME accept_mask regardless of which detector produced o
        answered_far = accept_mask(gen.uncertainty(phi_od), s_od,
                                   gen.oracle_w_cov(phi_od), tau, t, wmax).mean()
        print(f"[detector-agnostic/{name}] AUROC={auroc:.3f} FPR95={fpr:.3f} "
              f"leak@{alpha}={leak:.3f} answered_far={answered_far:.3f}")
        assert auroc >= 0.90                                     # each detector separates the far tier
        assert leak <= alpha + CFG.mc.leak_tol                  # each yields a valid leakage-matched t_ood


def test_diagnostic_near_ood_tier_worse():
    """DIAGNOSTIC (prereg §5.2; method_note §4.4): near-OOD tier leakage, reported separately.

    The near-OOD tier (a closer shell with partial ID overlap) leaks MORE than far-OOD
    through the far-tuned t_ood, and is expected worse -- asserted NOT presented as a
    bound (near-OOD overlapping ID support is exactly the regime the impossibility
    covers, Fang et al. 2022). REPORTED, never gated."""
    gen = _gen()
    r = np.random.default_rng(CFG.seed + 1100)
    det = _fit_detector(gen, r)
    t = set_t_ood(det.score(gen.sample_far_ood(CFG.mc.leak_fit_n, r)[0]), CFG.mc.red_alpha)
    far_leak = leakage_rate(det.score(gen.sample_far_ood(CFG.mc.leak_ref_n, r)[0]), t)
    near_leak = leakage_rate(det.score(gen.sample_near_ood(CFG.mc.leak_ref_n, r)[0]), t)
    print(f"[near-tier] far-OOD leak={far_leak:.4f} vs near-OOD leak={near_leak:.4f} "
          f"(near-OOD worse; reported, NOT a bound)")
    assert near_leak > far_leak                                 # near-OOD leaks more (reported)


def test_diagnostic_n_eff_and_clip_abstention():
    """DIAGNOSTIC (method_note §1.7, §3.2, §4.2; prereg §5.4): n_eff + clip-abstention rate.

    On the answered-and-in-scope cohort report the Kish n_eff of the covariate weights
    and the clip-induced abstention rate (mass with raw w_cov > w_max). REPORTED, never
    gated (plumbing only)."""
    gen = _gen()
    r = np.random.default_rng(CFG.seed + 1200)
    det = _fit_detector(gen, r)
    tau, wmax = CFG.f.tau0, CFG.b.w_max
    t = set_t_ood(det.score(gen.sample_far_ood(CFG.mc.leak_fit_n, r)[0]), CFG.mc.red_alpha)
    phi, y, loss = gen.sample_id(CFG.mc.cov_n, r, "S")
    u, o, w = gen.uncertainty(phi), det.score(phi), gen.oracle_w_cov(phi)
    A = accept_mask(u, o, w, tau, t, wmax)
    w_acc = np.clip(w[A], 0.0, wmax)
    neff = kish_n_eff(w_acc)
    clip_rate = float(np.mean(w > wmax))
    print(f"[n_eff] answered-in-scope n={int(A.sum())}; n_eff={neff:.1f} "
          f"(n_eff/n={neff / max(int(A.sum()), 1):.3f}); clip-abstention rate={clip_rate:.4f}")
    assert 0.0 < neff <= len(w_acc) + 1e-9                      # plumbing only


def test_diagnostic_metadata_prescreen_not_in_synthetic():
    """DIAGNOSTIC (method_note §4.2): the DICOM/header metadata pre-screen is REAL-DATA only.

    On real deployment a metadata pre-screen (wrong modality / body-part) precedes the
    learned o(x) as a confident-flag (not a hard reject; missing/ambiguous tags fall
    through to o(x)), with the image-vs-metadata disagreement rate logged as a Stage-R
    number. It is NOT exercised in the synthetic gate (no DICOM headers) -- named here so
    its omission from the synthetic ladder is explicit, not an oversight. REPORTED."""
    print("[metadata-prescreen] NOT exercised in the synthetic gate (no DICOM headers); "
          "real-data-only confident-flag, disagreement rate logged at Stage R (method_note 4.2)")
    assert True                                                 # documentation diagnostic


def test_diagnostic_interval_coverage_report():
    """DIAGNOSTIC (prereg §6.8): synthetic interval-coverage on the answered-in-scope cohort.

    With KNOWN weights (oracle w_cov) and a KNOWN ground-truth weighted risk (the target
    accepted risk R_T), report realized coverage of the WSR UCB vs nominal 1-delta on the
    answered-and-in-scope cohort. The operative §6.8 action is the RELABEL, emitted where
    it under-covers; never a gate (only plumbing is asserted)."""
    gen = _gen()
    r0 = np.random.default_rng(CFG.seed + 1300)
    det = _fit_detector(gen, r0)
    tau, wmax = CFG.f.tau0, CFG.b.w_max
    rr = np.random.default_rng(CFG.seed + 1301)
    R_T = gen.accepted_risk(tau, "T", rr, n=CFG.mc.oracle_n)
    t = set_t_ood(det.score(gen.sample_far_ood(CFG.mc.leak_fit_n, r0)[0]), CFG.mc.red_alpha)
    covered, neffs = 0, []
    for tr in range(CFG.mc.cov_T):
        r = np.random.default_rng(CFG.seed + 13000 + tr)
        phi, y, loss = gen.sample_id(CFG.mc.cov_n, r, "S")
        u, o, w = gen.uncertainty(phi), det.score(phi), gen.oracle_w_cov(phi)
        A = accept_mask(u, o, w, tau, t, wmax)
        L = loss[A]
        W = np.clip(w[A], 0.0, wmax)
        ucb = wsr_ucb_weighted(L, W, CFG.b.delta, float(max(W.max(), 1e-9)))
        covered += int(R_T <= ucb)
        neffs.append(kish_n_eff(W))
    realized = covered / CFG.mc.cov_T
    label = ("nominal interval, coverage validated empirically"
             if realized < 1 - CFG.b.delta else "nominal interval")
    print(f"[interval-cov] realized={realized:.4f} vs nominal={1 - CFG.b.delta} "
          f"(R_T={R_T:.4f}, n_eff~{np.mean(neffs):.0f}) -> RELABEL: '{label}'")
    assert 0.0 <= realized <= 1.0                              # plumbing only, never gated
