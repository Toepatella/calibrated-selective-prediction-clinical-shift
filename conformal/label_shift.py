"""Label-/prevalence-shift correction: BBSE, MLLS+BCTS, and the ``Ẑ``-divide combine (Gate C).

Realizes ``method_note §1.5`` / §3.3 / §3.4 step 2-3. Gate C adds the label ratio
``w_lab(y) = p_T(y)/p_S(y)`` and combines it with the Gate-B covariate weight
``ŵ_cov(x)`` through the per-``x`` normalizer ``Z(x)`` -- **not** by multiplication:

    w_lab = Ĉ_S⁻¹ q̂_T            (BBSE; Lipton, Wang & Smola, ICML 2018)
    p̂_T   via MLLS + BCTS         (Alexandari, Kundaje & Shrikumar, ICML 2020)
    Ẑ(x)  = Σ_{y'} ŵ_lab(y')·σ̃(f(x))_{y'}     (σ̃ = BCTS-recalibrated softmax)
    ŵ(x,y) = ŵ_lab(y)·ŵ_cov(x) / Ẑ(x)          (NOT ŵ_cov·ŵ_lab)

``Z(x)`` is exactly the double-count corrector the naïve product omits: even under
*pure* label shift the prevalence change moves ``p(x)`` by ``Z(x)``, so the product
``ŵ_lab·ŵ_cov`` inflates by ``Z(x)`` (``method_note §3.3`` worked 2-class example:
``p_S=(.5,.5)→p_T=(.9,.1)`` gives truth ``1.8``, product ``1.8·1.8=3.24``, and the
``Ẑ``-divide recovers ``1.8``). It collapses correctly in the two pure regimes
(``w_cov ≡ Z`` under pure label shift; ``Z ≡ 1`` under pure covariate shift).

Honesty rails (``build_gates.md §6``). BBSE/MLLS carry only **consistency** under
label shift given an invertible ``Ĉ_S`` -- no ``(α,δ)`` / distribution-free
certificate for ``w_lab`` or the combined weight. Under *combined* covariate+label
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
"""

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
    """
    C_S = np.asarray(C_S, dtype=float)
    q_T = np.asarray(q_T, dtype=float)
    p_S = np.asarray(p_S, dtype=float)
    w_raw = np.linalg.solve(C_S, q_T)                 # Ĉ_S⁻¹ q̂_T  (= p_T/p_S at population)
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
    calibrated; BCTS supplies that calibration and so REDUCES -- but does not
    eliminate -- the ``Ẑ`` softmax plug-in bias in the denominator ``σ̃ ≈ p_S(y|x)``.
    The residual is non-vanishing and amplified on rare / high-``w_lab`` classes
    (``method_note §3.5``, §7.4); it is *reported* via a sensitivity sweep, never
    assumed away. ``b`` is identified up to an additive constant (softmax is
    shift-invariant), so ``b[0]`` is pinned to 0.

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
    observed ``q̂_T``. Near zero when label shift holds; materially larger when
    ``p(x|y)`` is perturbed (appearance / anti-causal violation). A continuous
    diagnostic, never a gate (``method_note §3.4 step 2``, §3.5, §7.5).
    """
    q_T = np.asarray(q_T, dtype=float)
    C_S = np.asarray(C_S, dtype=float)
    col = C_S.sum(axis=0, keepdims=True)
    cond = np.divide(C_S, col, out=np.zeros_like(C_S), where=col > 0)  # P(ŷ=i|y=k)
    q_recon = cond @ np.asarray(p_T_hat, dtype=float)
    resid = q_T - q_recon
    return {"residual_vec": resid, "residual_l1": float(np.abs(resid).sum())}
