"""Domain-discriminator covariate-shift weights ``ŵ_cov(x)`` (Gate B).

Realizes ``method_note §1.5`` / §3.2 step 1: the covariate density ratio
``w_cov(x) = p_T(x)/p_S(x)`` is estimated from a **domain discriminator**
``d(x) = P(domain = T | x)`` via the conditional odds ratio, with the Remark-3
base-rate correction and a routing/clip cap:

    ŵ_cov(x) = clip( (d(x)/(1 − d(x)))·ĉ , 0 , w_max ),     ĉ = n_S/n_T.

This is the discriminator density-ratio estimator of Tibshirani, Foygel Barber,
Candès & Ramdas (NeurIPS 2019, their eq. 12) with the Remark-3 base-rate
constant. The base-rate constant ``ĉ`` **cancels in the self-normalized (Hájek)
weighted risk** and therefore matters only through the ``w_max`` clip boundary
(``method_note §1.5``, §7.3) -- a property Gate B checks directly.

Honesty rails (``build_gates.md §5``): the weight is ESTIMATED, so no
finite-sample / distribution-free coverage certificate carries over (Tibshirani
et al.'s Cor. 1 holds only under pure covariate shift with the *oracle*
likelihood ratio). Gate B **measures** the restoration; it certifies nothing.

Implementation notes
--------------------
* The discriminator is a plain L2-regularized logistic regression fit by Newton
  / IRLS in NumPy -- the project deliberately avoids a scikit-learn dependency
  (``requirements.txt``: "no confseq / sklearn dependency"). For the Gate-B
  synthetic construction the domain log-density-ratio is linear in the feature,
  so a logistic model is well-specified and recovers the oracle ratio at large
  sample.
* ``d`` is fit by **K-fold cross-fitting at the group-id level**: every point is
  scored by a model that did NOT see its fold, so weights are not optimistically
  fit on the points they reweight (``method_note §1.5``, §3.4 step 1). External
  points (a disjoint ``D_cal`` / test fold) are scored by the **bagged** average
  of the K fold-models -- a fixed function of ``x`` independent of the reweighted
  losses, which keeps the weighted betting bound's martingale property clean.
"""

import math
from collections import namedtuple

import numpy as np

__all__ = [
    "auroc",
    "cov_weight_from_d",
    "weight_summary",
    "CrossfitDiscriminator",
    "fit_discriminator",
    "aipw_risk",
]


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def auroc(scores, labels):
    """Area under the ROC curve via the Mann-Whitney U statistic (no sklearn).

    ``scores`` higher for the positive class; ``labels`` in {0, 1}. Ties get
    average ranks (the standard AUROC tie handling). Returns 0.5 when either
    class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    i = 0
    n = len(scores)
    while i < n:                                    # average ranks within tie groups
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # 1-based average rank
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    u_pos = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Logistic discriminator (L2, Newton/IRLS) -- hand-rolled, no sklearn.        #
# --------------------------------------------------------------------------- #
def _as_2d(X):
    """Coerce features to shape ``(n, p)``: a scalar or 1-D array becomes ``(n, 1)``.

    ``np.atleast_1d`` first, so a bare scalar (0-d, ``ndim == 0``) is promoted to a
    single ``(1, 1)`` point rather than falling through unshaped -- the external
    scoring API (``score`` / ``weights``) accepts a single point (`p == 1` case).
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))     # numerically stable both tails


class _Logistic:
    """L2-regularized logistic regression fit by Newton's method on standardized X.

    Features are standardized (mean/std from the training fold) for conditioning;
    an intercept is appended and is NOT penalized. Pure NumPy so the synthetic
    gates carry no scikit-learn dependency.
    """

    def __init__(self, l2=1.0, max_iter=100, tol=1e-9):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        Xs = (X - self.mean_) / self.std_
        Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])   # intercept last
        p = Xa.shape[1]
        beta = np.zeros(p)
        # penalty on all coefs except the intercept (last column)
        pen = self.l2 * np.ones(p)
        pen[-1] = 0.0
        for _ in range(self.max_iter):
            eta = Xa @ beta
            mu = _sigmoid(eta)
            w = np.clip(mu * (1.0 - mu), 1e-9, None)
            grad = Xa.T @ (mu - y) + pen * beta
            H = (Xa * w[:, None]).T @ Xa + np.diag(pen)
            step = np.linalg.solve(H, grad)
            beta = beta - step
            if np.max(np.abs(step)) < self.tol:
                break
        self.beta_ = beta
        return self

    def proba(self, X):
        X = _as_2d(X)
        Xs = (X - self.mean_) / self.std_
        Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        return _sigmoid(Xa @ self.beta_)


# --------------------------------------------------------------------------- #
# Covariate weight from the discriminator posterior (Remark-3 form).          #
# --------------------------------------------------------------------------- #
def cov_weight_from_d(d, c, w_max):
    """``ŵ_cov = clip((d/(1−d))·c, 0, w_max)`` (``method_note §1.5``, eq. 12 + Remark 3).

    ``d`` = discriminator posterior ``P(domain = T | x)``; ``c = ĉ = n_S/n_T`` the
    base-rate correction; ``w_max`` the clip / routing cap. ``d`` is clipped away
    from 1 so the odds ratio stays finite.
    """
    d = np.clip(np.asarray(d, dtype=float), 1e-12, 1.0 - 1e-12)
    odds = d / (1.0 - d)
    return np.clip(odds * float(c), 0.0, float(w_max))


def weight_summary(w):
    """Weight-distribution summary (median / 95 / 99 / max) -- ``method_note §3.4``.

    ``n_eff`` alone hides tails (``prereg §5.4``), so the tail percentiles are
    reported beside it. REPORTED diagnostic, never gated.
    """
    w = np.asarray(w, dtype=float)
    return {
        "median": float(np.median(w)),
        "p95": float(np.percentile(w, 95)),
        "p99": float(np.percentile(w, 99)),
        "max": float(w.max()),
        "mean": float(w.mean()),
    }


# --------------------------------------------------------------------------- #
# Cross-fitted discriminator + covariate weights.                             #
# --------------------------------------------------------------------------- #
# fields:
#   d_oof     -- out-of-fold P(T|x) for every pooled (source+target) point
#   fold_id   -- fold index each point was ASSIGNED to (= the fold it was scored
#                out of; the model for that fold was trained WITHOUT this point)
#   domain    -- 0 = source, 1 = target (the pooled domain labels)
#   c         -- base-rate constant ĉ = n_S/n_T
#   models    -- the K fold-models (each trained on the other K-1 folds)
#   w_max     -- clip / routing cap used for weights
CrossfitDiscriminator = namedtuple(
    "CrossfitDiscriminator",
    ["d_oof", "fold_id", "domain", "c", "models", "w_max"],
)


def _group_folds(groups, n_splits, rng):
    """Assign each point to one of ``n_splits`` folds at the GROUP-ID level.

    Points sharing a group id always land in the same fold, so a group is never
    split across train/score -- the group-level cross-fitting discipline of
    ``method_note §1.7`` / §1.5.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    fold_of_group = {g: (perm[i] % n_splits) for i, g in enumerate(uniq)}
    return np.array([fold_of_group[g] for g in groups], dtype=int)


def fit_discriminator(X_src, X_tar, w_max, c=None, n_splits=5, l2=1.0, seed=0,
                      groups_src=None, groups_tar=None):
    """Cross-fit a source-vs-target discriminator and return its OOF machinery.

    Pools ``X_src`` (domain 0) and ``X_tar`` (domain 1), fits an L2 logistic
    discriminator by **K-fold cross-fitting at the group level**, and returns a
    :class:`CrossfitDiscriminator`. ``ĉ`` defaults to ``n_S/n_T``. Use
    :meth:`score` / :meth:`weights` on the returned object to weight a DISJOINT
    fold (``D_cal`` / test) via the bagged fold-models.

    ``build_gates.md §5`` gate: every pooled point's ``d_oof`` comes from a model
    that did NOT see its fold (assert ``fold_id`` is the scored-out fold), so
    weights are not optimistically fit on the points they reweight.
    """
    X_src = _as_2d(X_src)
    X_tar = _as_2d(X_tar)
    n_s, n_t = X_src.shape[0], X_tar.shape[0]
    if c is None:
        c = n_s / n_t

    X = np.vstack([X_src, X_tar])
    domain = np.concatenate([np.zeros(n_s, int), np.ones(n_t, int)])
    if groups_src is None:
        groups_src = np.arange(n_s)
    if groups_tar is None:
        groups_tar = np.arange(n_s, n_s + n_t)
    groups = np.concatenate([np.asarray(groups_src), np.asarray(groups_tar)])

    rng = np.random.default_rng(seed)
    fold_id = _group_folds(groups, n_splits, rng)

    # Degeneracy guard (build_gates.md honesty rail): group-level cross-fitting needs
    # every fold's TRAINING split (fold_id != k) to contain BOTH domains. If a fold
    # trains on a single domain -- the one-group-per-domain case, or too few group ids
    # per domain to spread across folds -- the logistic discriminator sees one class,
    # converges to a constant posterior, and the odds ratio saturates so w_cov collapses
    # to all-equal (== w_max): Kish n_eff reads n (maximally healthy) while the covariate
    # correction is SILENTLY nulled to an unweighted mean. That is unrecoverable here, so
    # fail LOUD rather than return degenerate weights (method_note §1.5, §1.7).
    for k in range(n_splits):
        train = fold_id != k
        if train.any():
            dom_train = np.unique(domain[train])
            assert dom_train.size >= 2, (
                f"fit_discriminator: fold {k}'s training split is single-domain "
                f"(domains present={dom_train.tolist()}); group-level cross-fitting cannot "
                f"learn a source-vs-target boundary and would silently null w_cov to an "
                f"unweighted mean (all weights == w_max). Provide >=2 group ids per domain "
                f"that spread across folds (n_splits={n_splits}); one group per domain is "
                f"impossible in principle."
            )

    d_oof = np.full(len(X), np.nan)
    models = []
    for k in range(n_splits):
        train = fold_id != k
        held = fold_id == k
        model = _Logistic(l2=l2).fit(X[train], domain[train])
        models.append(model)
        if held.any():
            d_oof[held] = model.proba(X[held])

    return CrossfitDiscriminator(
        d_oof=d_oof, fold_id=fold_id, domain=domain, c=float(c),
        models=models, w_max=float(w_max),
    )


def _bagged_proba(disc, X):
    """Bagged discriminator posterior: mean ``P(T|x)`` over the K fold-models."""
    X = _as_2d(X)
    preds = np.array([m.proba(X) for m in disc.models])
    return preds.mean(axis=0)


def _disc_score(disc, X):
    return _bagged_proba(disc, X)


def _disc_weights(disc, X):
    """``ŵ_cov`` for external points ``X`` via the bagged discriminator.

    Use this for a DISJOINT fold (``D_cal`` / test): ``X`` was not in the
    discriminator's fitting pool, so the weight is a fixed function of ``x`` and is
    not optimistically fit on the points it reweights (``method_note §1.5``).
    """
    return cov_weight_from_d(_disc_score(disc, X), disc.c, disc.w_max)


def _disc_oof_weights(disc):
    """Cross-fitted ``ŵ_cov`` for the pooled fitting points, from the OOF posteriors.

    Each pooled point's weight comes from the fold-model that did NOT see it (the
    honest in-pool weight). Slice by ``disc.domain == 0`` for the source points.
    """
    return cov_weight_from_d(disc.d_oof, disc.c, disc.w_max)


# expose as methods on the namedtuple for ergonomic call sites
CrossfitDiscriminator.score = _disc_score
CrossfitDiscriminator.weights = _disc_weights
CrossfitDiscriminator.oof_weights = _disc_oof_weights


# --------------------------------------------------------------------------- #
# Doubly-robust / AIPW covariate correction (key secondary analysis).         #
# --------------------------------------------------------------------------- #
def aipw_risk(losses_src, weights_src, mhat_src, mhat_tar):
    """Self-normalized AIPW (doubly-robust) estimate of the target accepted risk.

    Augments the estimated covariate weight with a source outcome model
    ``m̂(x) ≈ E_S[ℓ | φ(x)]`` (``prereg §4A`` item 2; ``method_note §7.1``):

        R̂_DR = mean_{x∈target}[ m̂(x) ]  +  Σ_S ŵ·(ℓ − m̂) / Σ_S ŵ

    (over the accepted, in-scope source set). This is **doubly robust**: it recovers
    the target accepted risk when EITHER the weight OR ``m̂`` is correct --

      * ``m̂`` correct: ``mean_tar(m̂) = E_T[m̂] = E_T[ℓ]`` and, since ``ℓ − m̂`` is
        conditionally mean-zero under the (covariate-shift-invariant) outcome
        model, the source correction term vanishes;
      * ``ŵ`` correct: the source correction term equals ``E_T[ℓ − m̂]`` and adds
        back to ``mean_tar(m̂) = E_T[m̂]`` to give ``E_T[ℓ]``.

    SCOPE (both branches). This double-robustness holds under PURE COVARIATE shift
    -- where ``ŵ`` is the sole importance weight and ``E_S[ℓ | φ(x)] = E_T[ℓ | φ(x)]``
    (outcome-model invariance). It does NOT hold when a residual LABEL-marginal
    change survives conditioning on ``φ(x)``: under label shift the invariance premise
    fails in BOTH branches at once -- ``mean_tar(m̂)`` no longer equals ``E_T[ℓ]`` for a
    source-fit ``m̂`` (the ``m̂``-correct leg breaks), and a covariate-only ``ŵ`` is not
    the correct joint weight (the ``ŵ``-correct leg breaks) -- so AIPW is no longer
    unbiased. Under COMBINED shift the doubly-robust property is recovered only when
    ``ŵ`` is the library's ``Ẑ``-divide JOINT weight
    (:func:`conformal.label_shift.combine_weights`), not the bare covariate weight.
    This is a scoping note on the invariance premise, not a new guarantee.

    The Hájek single-model estimate stays the auditable PRIMARY; this is the
    residual-shrinkage DIAGNOSTIC (asymptotic, not a finite-sample certificate),
    never gated (``build_gates.md §5``).
    """
    l = np.asarray(losses_src, dtype=float)
    w = np.asarray(weights_src, dtype=float)
    ms = np.asarray(mhat_src, dtype=float)
    mt = np.asarray(mhat_tar, dtype=float)
    sw = float(w.sum())
    correction = float(np.sum(w * (l - ms)) / sw) if sw > 0 else 0.0
    return float(mt.mean() + correction)


def crossfit_outcome_model(X, losses, n_splits=5, l2=1.0, seed=0, groups=None):
    """Cross-fitted source outcome model ``m̂(x) ≈ E_S[ℓ | x]`` for the AIPW diagnostic.

    K-fold group-level cross-fitting (mirrors :func:`fit_discriminator`): returns
    ``(mhat_oof, models)`` where ``mhat_oof[i]`` is point ``i``'s loss-probability
    from a fold-model that did NOT see it, and ``models`` are the K fold-models for
    scoring external (target) points by :func:`bagged_outcome`. This is the
    DML/AIPW cross-fitting that makes the source correction term ``Σ ŵ·(ℓ − m̂)``
    free of own-observation optimism (``prereg §4A`` item 2; ``method_note §7.1``).
    """
    X = _as_2d(X)
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if groups is None:
        groups = np.arange(n)
    rng = np.random.default_rng(seed)
    fold = _group_folds(groups, n_splits, rng)
    mhat = np.full(n, np.nan)
    models = []
    for k in range(n_splits):
        train = fold != k
        held = fold == k
        model = _Logistic(l2=l2).fit(X[train], losses[train])
        models.append(model)
        if held.any():
            mhat[held] = model.proba(X[held])
    return mhat, models


def bagged_outcome(models, X):
    """Bagged source outcome model ``m̂(x)``: mean over the K cross-fit fold-models.

    Used to score EXTERNAL (target) points for the AIPW target-mean term -- those
    points are not in the source fitting pool, so a bagged average is the honest
    out-of-sample prediction.
    """
    X = _as_2d(X)
    return np.mean([m.proba(X) for m in models], axis=0)


__all__ += ["crossfit_outcome_model", "bagged_outcome"]
