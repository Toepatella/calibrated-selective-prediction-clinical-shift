"""Label-/prevalence-shift correction: BBSE, MLLS+BCTS, and the ``Ẑ``-divide combine (Gate C).

Realizes ``method_note §1.5`` / §3.3 / §3.4 step 2-3. Gate C adds the label ratio
``w_lab(y) = p_T(y)/p_S(y)`` and combines it with the Gate-B covariate weight
``ŵ_cov(x)`` through the per-``x`` normalizer ``Z(x)`` -- **not** by multiplication:

    w_lab = Ĉ_S⁻¹ q̂_T            (BBSE; Lipton, Wang & Smola, ICML 2018)
    p̂_T   via MLLS + BCTS         (Alexandari, Kundaje & Shrikumar, ICML 2020)
    Ẑ(x)  = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}     (σ̃ = BCTS-recalibrated softmax)
    ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x) / Ẑ(x)          (NOT ŵ_cov·ŵ_lab)

The combine is **not a new result**: ``Ẑ(x)`` is the ``Z(x)`` posterior-normalizer of
the prior-shift correction of Saerens, Latinne & Decaestecker (Neural Computation
2002), and ``w(x,y) = w_lab(y)·w_cov(x)/Z(x)`` is exactly Tasche (2022,
arXiv:2207.14514), Theorem 4 / Corollary 4 at ``ρ = 1`` -- the case of Factorizable
Joint Shift under the *invariant-density-ratio* assumption (Tasche 2014; JMLR 2017,
arXiv:1701.05512). We restate it here for the Hájek-weighted selective-risk
estimator; the module contributes the normalizer-estimation (BCTS ``σ̃``) and the
reported diagnostics, not the identity.

``Z(x)`` is exactly the double-count corrector the naïve product omits: even under
*pure* label shift the prevalence change moves ``p(x)`` by ``Z(x)``, so the product
``ŵ_lab·ŵ_cov`` inflates by ``Z(x)`` (``method_note §3.3`` worked 2-class example:
``p_S=(.5,.5)→p_T=(.9,.1)`` gives truth ``1.8``, product ``1.8·1.8=3.24``, and the
``Ẑ``-divide recovers ``1.8``). It collapses correctly in the two pure regimes
(``w_cov ≡ Z`` under pure label shift; ``Z ≡ 1`` **iff** ``w_lab ≡ 1`` -- i.e. under
pure covariate shift *with the prevalence held fixed*, NOT under covariate shift in
general: whenever the prevalence also moves, ``Z(x) = E_{p_S(·|x)}[w_lab]`` is
non-constant even in the covariate-tilt regime (on the committed K=2 config
``max_x|Z(x)-1| ≈ 0.80``; it vanishes to machine zero only once ``w_lab ≡ 1``).

Honesty rails (``build_gates.md §6``). BBSE/MLLS carry no computable ``(α,δ)`` /
distribution-free **certificate** for ``w_lab`` or the combined weight under label
shift given an invertible ``Ĉ_S``. (BBSE does have a finite-sample high-probability
error bound -- Lipton, Wang & Smola 2018, Thm. 3 -- but with unknown constants and
under an untestable premise, so it yields no computable operating guarantee here;
the honesty rail denies the *certificate*, not consistency or finite-sample theory.)
The two estimators are NOT interchangeable under this module's *factorizable*
premise: BBSE needs the anti-causal invariance ``P_T(ŷ|y)=P_S(ŷ|y)``, which the
factorizable / invariant-density-ratio condition does NOT imply (a class-free tilt
the classifier sees is factorizable yet biases BBSE by an ``O(1)`` amount
irreducible in ``n``); MLLS+BCTS is therefore the RECOMMENDED prevalence estimator
here and BBSE is KEPT as a baseline / anti-causal consistency diagnostic (see
:func:`bbse_weights`). Under *combined* covariate+label
shift the combined weight is **not identifiable from unlabeled target alone**; the
factorizable-shift premise is a modeling choice whose residual is only *measured* on
``D_tar^lab`` (never certified). The ``Ẑ(x)`` softmax plug-in bias is **not** assumed
removed by recalibration -- BCTS reduces but does not eliminate it, and the residual
is non-vanishing and amplified on rare / high-``w_lab`` classes (``method_note §3.5``,
§7.4), so it is *reported* via a sensitivity sweep rather than certified away.
``κ(Ĉ_S)``, the ``q̂_T`` vs ``Ĉ_S p̂_T`` check, and ``n_eff`` are REPORTED
diagnostics, never gates.

Implementation notes
--------------------
* Pure NumPy/SciPy, no scikit-learn -- consistent with the Gate-A/B convention
  (``requirements.txt``: "no confseq / sklearn dependency"). BCTS's small convex
  temperature+bias fit uses ``scipy.optimize.minimize`` (L-BFGS-B).
* BBSE uses the **joint** confusion ``C_S[i,k]=P_S(ŷ=i,y=k)`` so ``C_S⁻¹ q̂_T``
  returns ``w_lab=p_T/p_S`` directly (``(C_S w_lab)[i]=Σ_k P_S(ŷ=i|y=k)p_T(k)=q_T[i]``
  under the label-shift invariance ``P_T(ŷ|y)=P_S(ŷ|y)``).
* The estimate is stabilized exactly as ``method_note §3.4 step 2`` prescribes:
  project the implied ``p̂_T`` onto the probability simplex, then FLOOR/CEIL
  ``ŵ_lab ∈ [w_lab,min, w_lab,max]`` with ``w_lab,min > 0`` so an ill-conditioned
  ``Ĉ_S`` can never drive ``Z`` to zero or negative.

Terminology -- "factorizable"
----------------------------
Throughout this module and the Gate-C tests, **factorizable** means the
*invariant-density-ratio* condition: the class-conditional density ratio
``p_T(x|y)/p_S(x|y)`` is **class-free on overlapping support** (equivalently
``p_T(x|y)/p_S(x|y) = a(x)`` for a single ``a`` shared by all ``y``). This is exactly
what makes the ``Ẑ``-combine identity ``w = w_lab·w_cov/Z`` exact (Tasche 2022,
Thm 4 / Cor 4 at ``ρ=1``; Tasche 2014).

It is **NOT** the Factorizable-Joint-Shift (FJS) sense of He et al. (2022,
arXiv:2203.02902) / Tasche, where factorizable means only ``w(x,y) = U(x)·V(y)``.
The invariant-density-ratio condition is **strictly stronger**: a shift can satisfy
``w(x,y)=U(x)·V(y)`` exactly yet violate the combine. The repo's own *entangled*
``x1``-tilt regime (``tests/_synth.py::LabelShiftMixture(tilt='x1')``) is precisely
such a case -- it FJS-factorizes to machine precision yet breaks the ``Ẑ``-combine
(divergence ≈ 1.07); this is the distinction hard-gated in
``test_gate_c_label_shift.py::test_factorizable_means_invariant_density_ratio_not_fjs``.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "confusion_matrix",
    "pred_hist",
    "project_to_simplex",
    "bbse_weights",
    "softmax",
    "fit_bcts",
    "recalibrate_softmax",
    "mlls_em",
    "mapls_em",
    "mlls_weights",
    "z_corrector",
    "combine_weights",
    "kappa_cs",
    "bbse_consistency",
    "measure_slice_joint_weight",
    "residual_on_labeled_target",
]


# --------------------------------------------------------------------------- #
# Confusion matrix / predicted-label histogram (BBSE inputs).                  #
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred, K):
    """Joint source confusion ``C_S[i,k] = P_S(ŷ=i, y=k)`` (rows=pred, cols=true).

    Normalized to sum to 1 so ``C_S⁻¹ q̂_T`` returns ``w_lab = p_T/p_S`` directly
    (``method_note §1.5``). ``y_true`` / ``y_pred`` are integer class labels in
    ``[0, K)``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if (y_true.min(initial=0) < 0 or y_true.max(initial=0) >= K
            or y_pred.min(initial=0) < 0 or y_pred.max(initial=0) >= K):
        raise ValueError(                                 # [0, K) contract; -1 must NOT wrap silently
            f"confusion_matrix: y_true/y_pred must be integer class labels in [0, {K}); "
            "got an out-of-range label (e.g. a sentinel -1 would silently wrap into "
            "class K-1 via np.add.at's negative indexing)."
        )
    C = np.zeros((K, K), dtype=float)
    np.add.at(C, (y_pred, y_true), 1.0)
    return C / len(y_true)


def pred_hist(y_pred, K):
    """Target predicted-label distribution ``q̂_T[i] = P_T(ŷ=i)`` (sums to 1)."""
    y_pred = np.asarray(y_pred, dtype=int)
    q = np.bincount(y_pred, minlength=K).astype(float)
    return q / q.sum()


# --------------------------------------------------------------------------- #
# Simplex projection (Wang & Carreira-Perpiñán 2013).                          #
# --------------------------------------------------------------------------- #
def project_to_simplex(v):
    """Euclidean projection of ``v`` onto the probability simplex ``{p≥0, Σp=1}``.

    The ``method_note §3.4 step 2`` engineering fix: an ill-conditioned ``Ĉ_S``
    can push the raw ``p̂_T = ŵ_lab·p_S`` off the simplex (negative or unnormalized);
    projecting restores a valid prevalence vector. Exact ``O(K log K)`` algorithm of
    Wang & Carreira-Perpiñán (2013).
    """
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - css / ind > 0
    rho = np.nonzero(cond)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# --------------------------------------------------------------------------- #
# BBSE label-ratio estimator (Lipton, Wang & Smola 2018).                      #
# --------------------------------------------------------------------------- #
def bbse_weights(C_S, q_T, p_S, w_min=1e-3, w_max=1e3):
    """BBSE label ratio ``ŵ_lab = Ĉ_S⁻¹ q̂_T`` with simplex projection + floor/ceiling.

    Parameters
    ----------
    C_S : (K, K) joint source confusion ``P_S(ŷ=i, y=k)`` (from :func:`confusion_matrix`).
    q_T : (K,) target predicted-label distribution ``P_T(ŷ=i)`` (from :func:`pred_hist`).
    p_S : (K,) source prevalence ``p_S(y)``.
    w_min, w_max : floor/ceiling on ``ŵ_lab`` (``w_min > 0``; ``method_note §3.4 step 2``).

    Returns
    -------
    (w_lab, p_T_hat) : the clipped label ratio and the simplex-projected implied
        target prevalence ``p̂_T`` (a valid probability vector, ``method_note §1.7``).

    BBSE is a *consistent* estimator of ``w_lab`` under label shift given an
    invertible ``Ĉ_S`` -- consistency, NOT a finite-sample certificate
    (``build_gates.md §6`` honesty rails).

    SCOPE (do not treat as interchangeable with MLLS). BBSE's consistency needs
    the anti-causal invariance ``P_T(ŷ=i|y=k) = P_S(ŷ=i|y=k)`` (invariant
    ``C_S``). The *factorizable* / invariant-density-ratio premise this module is
    built for (``p_T(x|y)/p_S(x|y)`` class-free on overlapping support; Tasche
    2022, arXiv:2207.14514, Thm 4 / Cor 4 at ρ=1) does **NOT** imply that
    invariance: a class-free covariate tilt the frozen classifier *sees* (i.e.
    that moves the ``ŷ|y`` histogram) is factorizable yet breaks BBSE. On such a
    classifier-visible factorizable shift BBSE carries an ``O(1)`` bias that does
    NOT shrink in ``n`` (e.g. a class-free ``exp(β·tanh(x₁))`` tilt gives
    population BBSE ``≈(1.06, 0.94)`` for a truth of ``(1.8, 0.2)``). Under the
    factorizable premise the RECOMMENDED prevalence estimator is MLLS+BCTS
    (:func:`mlls_weights`), which is consistent given only calibration and does
    NOT require ``P(ŷ|y)`` invariance (Garg et al. 2020, arXiv:2003.07554). BBSE
    is KEPT as a baseline and as the anti-causal consistency diagnostic of §3
    (:func:`bbse_consistency`), not as an equal-standing estimator here.

    Clip-cost honesty. The ``[w_min, w_max]`` floor/ceiling is a stabilizer
    (``w_min > 0`` keeps ``Ẑ`` strictly positive), and its binding BIASES the combine
    -- a cost we report rather than hide. Two contributions are distinguished: the
    FLOOR (``w_min``) contributes negligibly (~0.02% of the ill-conditioning error at
    the committed shifts -- it binds only for provably-absent classes); the CEILING
    (``w_max``) is the material one, costing ~+1.2% relative on the combined weight at
    a ~5000x prevalence shift, where the true minority-class ratio exceeds the cap and
    is truncated. The binding is directly observable (compare ``ŵ_lab`` against the
    ``[w_min, w_max]`` boundary), so its effect is measured, not assumed harmless.
    """
    C_S = np.asarray(C_S, dtype=float)
    q_T = np.asarray(q_T, dtype=float)
    p_S = np.asarray(p_S, dtype=float)
    try:
        w_raw = np.linalg.solve(C_S, q_T)             # Ĉ_S⁻¹ q̂_T  (= p_T/p_S at population)
    except np.linalg.LinAlgError as exc:              # non-invertible Ĉ_S: report, don't crash bare
        info = kappa_cs(C_S)
        raise ValueError(
            "bbse_weights: Ĉ_S is singular (non-invertible); BBSE requires an "
            "invertible source confusion (method_note §1.5). Diagnostic "
            f"σ_min(Ĉ_S)={info['sigma_min']:.3e}, κ(Ĉ_S)={info['kappa']:.3e}. "
            "Typical cause: a class never predicted (a zero row) or never a true "
            "label (a zero column) in the source fold. Grow the source sample, "
            "merge/drop the degenerate class, or use mlls_weights (which does not "
            "invert Ĉ_S)."
        ) from exc
    p_T_hat = project_to_simplex(w_raw * p_S)         # implied prevalence, back onto the simplex
    w_lab = np.clip(p_T_hat / p_S, w_min, w_max)      # floor/ceiling: w_min>0 keeps Z>0
    return w_lab, p_T_hat


# --------------------------------------------------------------------------- #
# BCTS recalibration + MLLS/MAPLS EM (Alexandari et al. 2020).                 #
# --------------------------------------------------------------------------- #
def softmax(logits):
    """Row-wise softmax of a ``(n, K)`` logit matrix (numerically stable)."""
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_bcts(logits, y, max_iter=500):
    """Bias-Corrected Temperature Scaling: fit ``(T, b)`` minimizing source NLL.

    Recalibrates the frozen classifier's logits as ``σ̃ = softmax(logits/T + b)``
    with a single temperature ``T > 0`` and a per-class bias ``b`` (Alexandari,
    Kundaje & Shrikumar, ICML 2020). MLLS is consistent only when ``f`` is
    calibrated; BCTS supplies that calibration and so REDUCES the ``Ẑ`` softmax
    plug-in bias in the denominator ``σ̃ ≈ p_S(y|x)``. Whether the residual then
    vanishes depends on WELL-SPECIFICATION of the miscalibration:

    * IN-FAMILY (the true distortion is a single global temperature + per-class
      bias, the exact ``(T, b)`` family below): BCTS is CONSISTENT and the residual
      **vanishes with n**. In Gate C's own gate-2 regime (``T_mis=3``, ``b=[0,1]``)
      the fit recovers ``T→1/3``, ``b₁→−3`` and the ``Ẑ`` residual falls
      ``~0.015→0.007→0.001`` at ``n=5e3→5e4→5e5`` -- it is NOT non-vanishing there.
    * OUT-OF-FAMILY (e.g. a per-class temperature the single-``T`` family cannot
      invert): a bias FLOOR remains that does NOT shrink with ``n`` -- ``~0.008``
      (per-class ``T`` on the k2 fixture) up to ``~0.04`` at the extreme tail --
      amplified on rare / high-``w_lab`` classes (``method_note §3.5``, §7.4).

    So the plug-in bias is *reported* via the per-class-temperature sensitivity
    sweep (:func:`test_diagnostic_z_plugin_bias_sensitivity`), which deliberately
    lives in the OUT-OF-FAMILY regime so the residual is a genuine floor, not
    finite-sample noise; it is never assumed away. ``b`` is identified up to an
    additive constant (softmax is shift-invariant), so ``b[0]`` is pinned to 0.
    Non-convergence of the inner L-BFGS-B solve is surfaced as a warning (its
    ``success`` flag is otherwise discarded by the ``(T, b)`` return).

    Returns ``(T, b)`` for use with :func:`recalibrate_softmax`.
    """
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    y = np.asarray(y, dtype=int)
    n, K = logits.shape

    def unpack(params):
        T = np.exp(params[0])                         # T = exp(t) > 0
        b = np.concatenate([[0.0], params[1:]])       # b[0] pinned to 0
        return T, b

    def nll(params):
        T, b = unpack(params)
        z = logits / T + b[None, :]
        z = z - z.max(axis=1, keepdims=True)
        logZ = np.log(np.exp(z).sum(axis=1))
        ll = z[np.arange(n), y] - logZ
        return -np.mean(ll)

    x0 = np.zeros(K)                                  # t=0 (T=1), b_free=0
    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    if not res.success:                               # do NOT silently discard convergence
        warnings.warn(f"fit_bcts: L-BFGS-B did not converge ({res.message!r}); "
                      f"(T, b) may be unreliable", RuntimeWarning, stacklevel=2)
    T, b = unpack(res.x)
    return float(T), b


def recalibrate_softmax(logits, T, b):
    """BCTS-recalibrated softmax ``σ̃ = softmax(logits/T + b)`` (``method_note §3.4 step 3``)."""
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    return softmax(logits / float(T) + np.asarray(b, dtype=float)[None, :])


def mlls_em(probs_T, p_S, max_iter=1000, tol=1e-8, alpha=None):
    """MLLS prevalence estimate ``p̂_T(y)`` by EM on calibrated target posteriors.

    ``probs_T`` : (m, K) CALIBRATED source-domain posteriors ``σ̃(f(x_j))`` on the
    unlabeled target sample. ``p_S`` : (K,) source prevalence. The
    Saerens/MLLS fixed point (Alexandari et al. 2020): reweight each posterior by the
    current prior ratio and renormalize, then average --

        γ_j(y) ∝ σ̃_j(y)·(p_T^(t)(y)/p_S(y)),     p_T^(t+1)(y) = mean_j γ_j(y).

    With ``alpha`` a Dirichlet concentration vector (or scalar), the M-step becomes
    the MAP update ``p_T ∝ Σ_j γ_j + (α − 1)`` -- this is the **MAPLS** small-``m``
    regularizer (``prereg §4A.1``); ``alpha=None`` is plain MLLS. Consistent under
    label shift *when the posteriors are calibrated* (hence BCTS first).
    """
    probs_T = np.atleast_2d(np.asarray(probs_T, dtype=float))
    p_S = np.asarray(p_S, dtype=float)
    K = probs_T.shape[1]
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim == 0:
            alpha = alpha * np.ones(K)
    p_T = p_S.copy()
    for _ in range(int(max_iter)):
        ratio = p_T / p_S
        num = probs_T * ratio[None, :]
        gamma = num / num.sum(axis=1, keepdims=True)
        counts = gamma.sum(axis=0)
        if alpha is not None:
            counts = np.clip(counts + (alpha - 1.0), 0.0, None)   # Dirichlet MAP
        p_new = counts / counts.sum()
        if np.max(np.abs(p_new - p_T)) < tol:
            p_T = p_new
            break
        p_T = p_new
    return p_T


def mapls_em(probs_T, p_S, alpha, max_iter=1000, tol=1e-8):
    """MAPLS = MLLS with a Dirichlet(``alpha``) prior (``prereg §4A.1``).

    Thin wrapper over :func:`mlls_em` with ``alpha`` set; the prior shrinks the
    prevalence estimate toward uniform in the small-``m`` regime where the plain
    MLLS EM is high-variance. Reported ablation, never a gate.
    """
    return mlls_em(probs_T, p_S, max_iter=max_iter, tol=tol, alpha=alpha)


def mlls_weights(probs_T, p_S, w_min=1e-3, w_max=1e3, alpha=None,
                 max_iter=1000, tol=1e-8):
    """MLLS(+MAPLS) label ratio ``ŵ_lab`` with simplex projection + floor/ceiling.

    Runs :func:`mlls_em` for ``p̂_T``, projects it onto the simplex, and floors/ceils
    ``ŵ_lab = p̂_T/p_S ∈ [w_min, w_max]`` -- the same stabilization as
    :func:`bbse_weights`, so the two estimators are directly comparable. Returns
    ``(w_lab, p_T_hat)``.
    """
    p_T_hat = mlls_em(probs_T, p_S, max_iter=max_iter, tol=tol, alpha=alpha)
    p_T_hat = project_to_simplex(p_T_hat)
    p_S = np.asarray(p_S, dtype=float)
    w_lab = np.clip(p_T_hat / p_S, w_min, w_max)
    return w_lab, p_T_hat


# --------------------------------------------------------------------------- #
# The Ẑ-divide combine (do NOT multiply).                                      #
# --------------------------------------------------------------------------- #
def z_corrector(w_lab, sigma_tilde):
    """Per-``x`` double-count corrector ``Ẑ(x) = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}``.

    ``w_lab`` : (K,) label ratio; ``sigma_tilde`` : (n, K) BCTS-recalibrated softmax.
    Returns ``Ẑ`` shape (n,) -- the ``E_{p_S(·|x)}[w_lab]`` normalizer of
    ``method_note §3.3``. With ``w_lab ≥ w_min > 0`` the output is strictly positive.

    This is the ``Z(x)`` normalizer of Saerens, Latinne & Decaestecker's prior-shift
    posterior correction (Neural Computation 2002) -- the same normalizer that appears
    in :func:`mlls_em`'s E-step; not a new construction.
    """
    sigma_tilde = np.atleast_2d(np.asarray(sigma_tilde, dtype=float))
    return sigma_tilde @ np.asarray(w_lab, dtype=float)


def combine_weights(w_lab_y, w_cov, Z):
    """Combined weight ``ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x) / Ẑ(x)`` -- **NOT** the product.

    All three inputs are per-point arrays aligned by sample: ``w_lab_y`` is the label
    ratio indexed at each point's class ``y`` (``ŵ_lab[y_i]``), ``w_cov`` is the
    covariate weight ``ŵ_cov(x_i)``, and ``Z`` is ``Ẑ(x_i)``. Dividing by ``Ẑ``
    removes the double-count the naïve product ``ŵ_lab·ŵ_cov`` introduces
    (``method_note §3.3``). ``Ẑ`` is guarded away from 0.

    Not a new identity: this is Tasche (2022, arXiv:2207.14514) Theorem 4 / Corollary 4
    at ``ρ = 1`` (Saerens et al. 2002 for the ``Z`` normalizer). It is **exact iff** the
    class-conditional density ratio ``p_T(x|y)/p_S(x|y)`` is class-free on overlapping
    support (the *invariant-density-ratio* condition, Tasche 2014) -- STRICTLY STRONGER
    than the He et al. (2022, arXiv:2203.02902) FJS sense ``w(x,y)=U(x)·V(y)`` (see the
    module ``Terminology`` note; a shift can be FJS-factorizable yet break this combine).

    Range convention (NO cap is applied here -- capping would bias the estimand).
    The combined weight is deliberately left UNCAPPED: ``w_max`` is a ``ŵ_cov``
    routing/clip cap, and it does NOT bound the combined weight. Because
    ``Ẑ(x) = Σ_{y'} ŵ_lab(y')·σ̃_{y'} ∈ [w_lab_min, w_lab_max]``, the a-priori
    algebraic sup is ``w_lab_max·w_cov_max / w_lab_min`` (with the shipped
    ``configs/gate_c.toml`` floor/ceiling ``[1e-3, 50]`` this is ``5e4·w_cov_max`` --
    a ~5.7e-25-per-point corner event, never a realized draw: genuine factorizable
    draws max at ~1.1-1.2x ``w_cov_max``, e.g. 42.9 vs ``w_cov_max``=36.2 at the
    committed K=2 combined regime). The WSR bound's range parameter is therefore set
    from the REALIZED sample max ``w.max()`` (as the gates pass it), not this sup; the
    self-normalized (Hájek) risk is invariant to the overall weight scale, so the sup
    matters only through the betting bound's range normalizer, not the point estimate.
    """
    w_lab_y = np.asarray(w_lab_y, dtype=float)
    w_cov = np.asarray(w_cov, dtype=float)
    Z = np.clip(np.asarray(Z, dtype=float), 1e-12, None)
    return w_lab_y * w_cov / Z


# --------------------------------------------------------------------------- #
# Reported diagnostics (never gate the rung).                                  #
# --------------------------------------------------------------------------- #
def kappa_cs(C_S):
    """Conditioning of ``Ĉ_S`` -- ``{kappa, sigma_min, eff_rank}`` (REPORTED only).

    BBSE/MLLS error grows as ``Ĉ_S`` becomes ill-conditioned, so ``κ(Ĉ_S)=σ_max/σ_min``
    (and ``σ_min``, and the participation-ratio effective rank ``(Σσ)²/Σσ²``) are
    reported as a reliability signal. NEVER a pass/fail gate (``method_note §7.5``;
    ``prereg §5.4``).
    """
    s = np.linalg.svd(np.asarray(C_S, dtype=float), compute_uv=False)
    sigma_min = float(s[-1])
    kappa = float(s[0] / s[-1]) if sigma_min > 0 else float("inf")
    eff_rank = float((s.sum() ** 2) / np.sum(s ** 2)) if np.any(s > 0) else 0.0
    return {"kappa": kappa, "sigma_min": sigma_min, "eff_rank": eff_rank}


def bbse_consistency(q_T, C_S, p_T_hat):
    """Anti-causal consistency residual ``q̂_T − Ĉ_S(·|·) p̂_T`` (REPORTED only).

    Reconstructs the predicted-label distribution the estimated prevalence implies,
    ``q̃[i] = Σ_k P_S(ŷ=i|y=k)·p̂_T(k)`` (the conditional confusion, column-normalized
    from the joint ``C_S``), and returns ``{residual_vec, residual_l1}`` vs the
    observed ``q̂_T``. This flags GROSS violations only, and its power is limited by
    a structural blindness the caller must understand:

    * In SINGLE-TARGET deployment usage -- the only usage available in the field --
      ``p̂_T`` is solved from *this same* target's ``q̂_T`` (``p̂_T = project(Ĉ_S⁻¹ q̂_T·p_S)``).
      Then ``q̃ = Ĉ_S(·|·) p̂_T`` is an in-simplex re-projection of ``q̂_T`` through
      ``Ĉ_S``, so the residual is ≈ 0 (near-tautological) WHENEVER the projection does
      not bind -- even under a genuine anti-causal ``p(x|y)`` violation. The repo's own
      entangled ``x1``-tilt reads ``residual_l1 ≈ 0.002`` (INDISTINGUISHABLE from clean).
    * The residual is materially larger ONLY when ``p̂_T`` is held FIXED from a
      *different (clean) paired run* and then confronted with a perturbed ``q̂_T`` --
      which is what :func:`test_diagnostic_anticausal_consistency` does (it reuses the
      clean-run ``p̂_T``), and which is not available at deployment.
    * Any violation that preserves the target ``x``-marginal (hence ``q̂_T``) is
      INVISIBLE by construction (e.g. a pure covariate-shift target reads the clean
      residual). This is a limitation of an unlabeled-target check, not a bug.

    So this diagnostic DETECTS only violations that move ``q̂_T`` *away from the
    column space* of ``Ĉ_S`` off a reference ``p̂_T`` -- a coarse cross-run tripwire,
    not a per-pair anti-causal test. The falsifiable per-pair premise check is the
    labeled-slice factorization/joint-weight divergence on ``D_tar^lab`` (``§3.4 step
    3``, §7.4; :func:`residual_on_labeled_target`), not this residual. A continuous
    diagnostic, never a gate (``method_note §3.4 step 2``, §3.5, §7.5).
    """
    q_T = np.asarray(q_T, dtype=float)
    C_S = np.asarray(C_S, dtype=float)
    col = C_S.sum(axis=0, keepdims=True)
    cond = np.divide(C_S, col, out=np.zeros_like(C_S), where=col > 0)  # P(ŷ=i|y=k)
    q_recon = cond @ np.asarray(p_T_hat, dtype=float)
    resid = q_T - q_recon
    return {"residual_vec": resid, "residual_l1": float(np.abs(resid).sum())}


# --------------------------------------------------------------------------- #
# The D_tar^lab residual: MEASURE the factorizable premise from a labeled slice #
# (method_note §1.7, §3.4 step 3, §7; build_gates.md §6; prereg §7).             #
# --------------------------------------------------------------------------- #
def measure_slice_joint_weight(X_src, y_src, X_tar, y_tar, p_S, K, var_floor=1e-3):
    """Joint importance weight ``ᵡ_joint(x,y)=p_T(x,y)/p_S(x,y)`` MEASURED from labeled samples.

    Estimates the joint density ratio DIRECTLY from a small LABELED target slice
    ``(X_tar, y_tar) ∼ P_T`` against a labeled source cohort ``(X_src, y_src) ∼ P_S`` --
    **never** from an oracle formula. The estimate factors as the per-class covariate
    ratio ``p_T(x|y)/p_S(x|y)`` times the empirical label ratio
    ``ŵ_lab(y)=ĵ_T(y)/p_S(y)`` (counts on the slice):

        ᵡ_joint(x_i, y_i) = ŵ_lab(y_i) · [ N(x_i; μ̂_T^{y_i}, σ̂_T^{y_i}) / N(x_i; μ̂_S^{y_i}, σ̂_S^{y_i}) ]

    with per-class diagonal-Gaussian moments (mean/variance) fit on the labeled
    source and target subsets of each class. This is a deliberately simple,
    dependency-free (no sklearn) slice-level estimator -- high-variance at
    ``m_lab ≈ 50–200``, a *measurement* object only (never deployed, never a
    certificate). Because it reads the TRUE class label on the target slice it can
    see a class-conditional covariate move ``p_T(x|y)≠p_S(x|y)`` that the factorized
    weight (built from ``ŵ_lab`` × a class-free ``ŵ_cov`` / Ẑ) cannot represent --
    which is exactly the entangled-shift violation the divergence in
    :func:`residual_on_labeled_target` is meant to expose.

    Parameters
    ----------
    X_src, y_src : labeled SOURCE features (n_s, d) and integer classes in ``[0, K)``.
    X_tar, y_tar : labeled TARGET-slice features (m, d) and integer classes ``D_tar^lab``.
    p_S : (K,) source prevalence ``p_S(y)``.
    K : number of classes.
    var_floor : floor on each per-class/per-dim variance (guards tiny slice subsets).

    Returns
    -------
    (w_joint, w_lab_slice) : the per-target-point joint weight (shape (m,)) and the
        empirical slice label ratio ``ŵ_lab`` (shape (K,)). Classes absent from the
        target slice get weight 0 there (no mass to transport); a source class with
        < 2 points falls back to the label ratio alone (covariate ratio unidentified).
    """
    X_src = np.atleast_2d(np.asarray(X_src, dtype=float))
    X_tar = np.atleast_2d(np.asarray(X_tar, dtype=float))
    y_src = np.asarray(y_src, dtype=int)
    y_tar = np.asarray(y_tar, dtype=int)
    p_S = np.asarray(p_S, dtype=float)
    counts = np.bincount(y_tar, minlength=K).astype(float)
    p_T_slice = counts / counts.sum() if counts.sum() > 0 else counts
    w_lab_slice = np.divide(p_T_slice, p_S, out=np.zeros_like(p_S), where=p_S > 0)
    w_joint = np.zeros(X_tar.shape[0], dtype=float)
    for k in range(K):
        src_k = X_src[y_src == k]
        tar_mask = y_tar == k
        if not np.any(tar_mask):
            continue                                        # no target mass in class k
        if src_k.shape[0] < 2:
            w_joint[tar_mask] = w_lab_slice[k]              # covariate ratio unidentified
            continue
        tar_k = X_tar[tar_mask]
        mu_s = src_k.mean(axis=0)
        var_s = np.maximum(src_k.var(axis=0), var_floor)
        mu_t = tar_k.mean(axis=0)
        var_t = np.maximum(tar_k.var(axis=0), var_floor)
        # log N(x; mu_t, var_t) / N(x; mu_s, var_s) for diagonal Gaussians
        log_ratio = (0.5 * np.sum(np.log(var_s / var_t))
                     - 0.5 * np.sum((tar_k - mu_t) ** 2 / var_t, axis=1)
                     + 0.5 * np.sum((tar_k - mu_s) ** 2 / var_s, axis=1))
        w_joint[tar_mask] = w_lab_slice[k] * np.exp(log_ratio)
    return w_joint, w_lab_slice


def residual_on_labeled_target(w_fac_src, loss_src, accept_src,
                               X_src, y_src, X_tar, y_tar, w_fac_tar,
                               p_S, K, var_floor=1e-3):
    """MEASURE the factorizable-premise residual on a labeled target slice ``D_tar^lab``.

    The single honest artifact behind the module's ``D_tar^lab`` honesty rail
    (``method_note §1.7``, §3.4 step 3, §7; ``build_gates.md §6``; ``prereg §7``). It
    consumes a small LABELED target slice ``(X_tar, y_tar) ∼ P_T`` (with the
    per-point factorized combined weight ``w_fac_tar`` on the same slice) and a
    labeled source accepted cohort, and REPORTS two measurements -- it certifies and
    identifies **nothing**:

    (a) **residual risk degradation.** ``risk_plugin`` is the target accepted-region
        risk the factorized weight PREDICTS (the Hájek self-normalized risk of the
        source accepted cohort reweighted by ``w_fac_src``). The caller compares it to
        the EMPIRICAL accepted-region risk measured directly on the labeled target
        slice; a large gap means the factorized combined weight fails to transport
        the risk on this pair.

    (b) **factorization divergence.** ``divergence`` is the mean-squared log-ratio
        between the factorized weight ``w_fac_tar`` and a joint weight measured
        DIRECTLY on the slice by :func:`measure_slice_joint_weight` (NOT the oracle).
        Near zero when the shift factorizes on this pair; materially larger when the
        class-conditional covariate move entangles the two mechanisms.

    Both numbers are *measurements with their own slice-size uncertainty*, making the
    factorizable premise FALSIFIABLE PER PAIR from real labeled data -- but they are
    REPORTED only, never a ``(α,δ)`` / distribution-free certificate and never an
    identification of the combined-shift nuisance (``build_gates.md §6`` honesty
    rails; the combined weight stays non-identifiable from unlabeled target alone).

    Parameters
    ----------
    w_fac_src : (n_s,) factorized combined weight ``ŵ_lab(y)·ŵ_cov(x)/Ẑ(x)`` on the
        SOURCE accepted cohort (used to PREDICT the target accepted risk).
    loss_src : (n_s,) source 0-1 losses aligned with ``w_fac_src``.
    accept_src : (n_s,) boolean accept mask (``u(x) ≤ τ0``) on the source cohort.
    X_src, y_src : labeled source features/classes (for the slice-measured joint weight).
    X_tar, y_tar : the LABELED target slice ``D_tar^lab`` (features + true classes).
    w_fac_tar : (m,) factorized combined weight on the target slice.
    p_S : (K,) source prevalence; K : number of classes.
    var_floor : per-class variance floor forwarded to :func:`measure_slice_joint_weight`.

    Returns
    -------
    dict with ``risk_plugin`` (the factorized-weight predicted target accepted risk),
        ``divergence`` (factorized vs slice-measured joint), ``w_lab_slice`` (empirical
        slice label ratio), and ``m_lab`` (slice size). The caller supplies the
        empirical slice risk and forms ``risk_gap = |risk_plugin − risk_empirical|``.
    """
    loss_src = np.asarray(loss_src, dtype=float)
    accept_src = np.asarray(accept_src, dtype=bool)
    w_acc = np.asarray(w_fac_src, dtype=float)[accept_src]
    l_acc = loss_src[accept_src]
    if w_acc.sum() > 0:
        risk_plugin = float(np.sum(w_acc * l_acc) / np.sum(w_acc))   # Hájek self-normalized
    else:
        risk_plugin = float("nan")
    w_joint, w_lab_slice = measure_slice_joint_weight(
        X_src, y_src, X_tar, y_tar, p_S, K, var_floor=var_floor)
    w_fac_tar = np.asarray(w_fac_tar, dtype=float)
    ok = np.isfinite(w_joint) & (w_joint > 0) & (w_fac_tar > 0)
    if np.any(ok):
        divergence = float(np.mean(
            (np.log(w_fac_tar[ok]) - np.log(w_joint[ok])) ** 2))
    else:
        divergence = float("nan")
    return {
        "risk_plugin": risk_plugin,
        "divergence": divergence,
        "w_lab_slice": w_lab_slice,
        "m_lab": int(np.asarray(y_tar).size),
    }
