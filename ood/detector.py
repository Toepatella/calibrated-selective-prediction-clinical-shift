"""Feature-space OOD scoring: Mahalanobis++ primary + plain-Lee / kNN / energy ablations (Gate D).

Realizes ``method_note §4.1``. The deployed / primary OOD score ``o(x)`` is the
**Mahalanobis++** L2-normalized tied-covariance Gaussian-discriminant score of
Müller et al. (2025, ``mueller-mp/maha-norm``):

    φ̃ = φ / ‖φ‖₂                                    (in-distribution-only L2-normalize)
    o(x) = min_c (φ̃ − μ̂_c)ᵀ Σ̂⁻¹ (φ̃ − μ̂_c)          (μ̂_c, Σ̂ fit on NORMALIZED source)

``o`` is a DISTANCE -- higher means more OOD -- and a point passes the screen (is
answered / in-scope) iff ``o ≤ t_ood`` (``conformal/budget.py``). Maha++ **keeps**
the *in-distribution-only, parameter-free* L2-normalization (each ``φ`` is scaled by
its own norm; ``μ̂_c``/``Σ̂`` are fit on normalized *source* features, so the
detector sees no OOD and the exposure set ``O`` stays disjoint / swappable) and
**drops** only the OOD-coupled add-ons of the original Maha++ recipe (FGSM input
perturbation, the per-layer logistic-regression ensemble) -- those alone would
couple the detector to ``O`` (``build_gates.md §7`` honesty rails; ``method_note
§4.1``).

Ablations, holding the routing logic fixed (``method_note §4.1``; ``prereg §5.2``):
  * **plain Lee et al. (2018)** tied-covariance Mahalanobis (normalization OFF) --
    the canonical-baseline ablation (``TiedMahalanobis(normalize=False)``);
  * **kNN** distance to the k-th nearest in-distribution neighbour (Sun et al. 2022);
  * **energy** ``E(x) = −T·logsumexp(f(x)/T)`` (Liu et al. 2020), read off the GDA
    log-likelihood logits.
The routing RULE is detector-agnostic; each detector yields its own valid
AUROC / FPR95 / leakage triple at its own leakage-matched ``t_ood`` (the answered
SET is not expected to be identical across detectors -- only the rule is).

Honesty rails (``build_gates.md §7``). OOD detection is NOT distribution-free
learnable in the unrestricted setting (Fang et al. 2022); nothing here is a
distribution-free / finite-sample OOD guarantee. ``o(x)`` is a MEASURED screen whose
leakage on a stated, swappable ``O`` is reported against ``α_ood`` (``method_note
§4.4``, §7.1). ``κ(Σ̂)`` and effective rank are REPORTED feature-collapse diagnostics,
never pass/fail gates (``method_note §4.1``; ``prereg §5.4``).

Implementation notes
--------------------
* Pure NumPy/SciPy, no scikit-learn -- consistent with the Gate-A/B/C convention.
* The tied covariance is the GDA maximum-likelihood estimate
  ``Σ̂ = (1/N) Σ_c Σ_{i: y_i=c} (φ̃_i − μ̂_c)(φ̃_i − μ̂_c)ᵀ`` (Lee et al. 2018 eq. 2);
  the Gate-D cross-check gate (``build_gates.md §7`` hard gate 1) re-derives exactly
  this from a brute-force numpy reference to ``atol = 1e-8`` for both the normalized
  (Maha++) and unnormalized (plain-Lee) paths.
"""

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "l2_normalize",
    "TiedMahalanobis",
    "knn_score",
    "energy_score",
    "kappa_sigma",
    "ood_auroc",
    "fpr_at_tpr",
]


def l2_normalize(phi, eps=1e-12):
    """Row-wise L2 normalization ``φ̃ = φ/‖φ‖₂`` (the Maha++ in-distribution-only step).

    Each row is scaled by its OWN norm, so the transform sees no OOD and does not
    couple the detector to the exposure set ``O`` (``method_note §4.1``). A (near-)zero
    row is left unscaled (guarded by ``eps``).
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return phi / norms


class TiedMahalanobis:
    """Tied-covariance Gaussian-discriminant Mahalanobis OOD score.

    ``normalize=True`` (default) is **Mahalanobis++** (Müller et al. 2025): the
    features are L2-normalized before the per-class means ``μ̂_c`` and the shared tied
    covariance ``Σ̂`` are estimated, and before scoring. ``normalize=False`` is the
    **plain Lee et al. (2018)** tied-covariance score -- the canonical-baseline
    ablation. Either way the score is the min-over-classes Mahalanobis DISTANCE
    ``o(x) = min_c (φ̃ − μ̂_c)ᵀ Σ̂⁻¹ (φ̃ − μ̂_c)`` (higher = more OOD).

    Parameters
    ----------
    normalize : bool
        L2-normalize features before fit/score (Maha++ on / plain-Lee off).
    shrinkage : float
        Optional ridge ``Σ̂ += shrinkage·I`` for conditioning. Default 0.0 so the
        score matches the brute-force reference exactly (``build_gates.md §7`` hard
        gate 1); set > 0 only on a genuinely ill-conditioned embedding.
    """

    def __init__(self, normalize=True, shrinkage=0.0):
        self.normalize = bool(normalize)
        self.shrinkage = float(shrinkage)

    def _prep(self, phi):
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        return l2_normalize(phi) if self.normalize else phi

    def fit(self, phi, y):
        """Fit ``μ̂_c`` and the tied covariance ``Σ̂`` on (optionally normalized) source.

        ``Σ̂`` is the GDA MLE: the within-class scatter summed over classes and divided
        by the TOTAL count ``N`` (Lee et al. 2018 eq. 2). ``phi`` are in-distribution
        (source) features only -- the detector never sees ``O``.
        """
        phi = self._prep(phi)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        D = phi.shape[1]
        N = phi.shape[0]
        means = np.empty((self.classes_.size, D))
        scatter = np.zeros((D, D))
        for j, c in enumerate(self.classes_):
            Xc = phi[y == c]
            mu = Xc.mean(axis=0)
            means[j] = mu
            diff = Xc - mu
            scatter += diff.T @ diff
        cov = scatter / N
        if self.shrinkage > 0.0:
            cov = cov + self.shrinkage * np.eye(D)
        self.means_ = means
        self.cov_ = cov
        self.prec_ = np.linalg.inv(cov)
        return self

    def maha_per_class(self, phi):
        """Per-class squared Mahalanobis distances ``(n, K)`` under ``Σ̂⁻¹``."""
        phi = self._prep(phi)
        diff = phi[:, None, :] - self.means_[None, :, :]          # (n, K, D)
        return np.einsum("nkd,de,nke->nk", diff, self.prec_, diff)

    def logits(self, phi):
        """GDA log-likelihood logits ``f_c(x) = −½·maha_c(x)`` (for the energy ablation)."""
        return -0.5 * self.maha_per_class(phi)

    def score(self, phi):
        """OOD score ``o(x) = min_c maha_c(x)`` (higher = more OOD)."""
        return self.maha_per_class(phi).min(axis=1)


def knn_score(phi_query, phi_bank, k=10, normalize=True):
    """kNN OOD score: Euclidean distance to the ``k``-th nearest ID neighbour (Sun et al. 2022).

    Higher = more OOD. ``phi_bank`` is an in-distribution feature bank; both query and
    bank are L2-normalized when ``normalize=True`` (the standard kNN-OOD setup). A
    detector-agnostic ablation over the SAME routing rule (``method_note §4.1``).
    """
    q = np.atleast_2d(np.asarray(phi_query, dtype=float))
    b = np.atleast_2d(np.asarray(phi_bank, dtype=float))
    if normalize:
        q = l2_normalize(q)
        b = l2_normalize(b)
    # squared Euclidean distances (n_q, n_b) via the |a-b|² expansion (no big temporaries per row)
    q2 = np.sum(q * q, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(q2 + b2 - 2.0 * (q @ b.T), 0.0)
    kk = min(int(k), b.shape[0])
    kth = np.partition(d2, kk - 1, axis=1)[:, kk - 1]
    return np.sqrt(kth)


def energy_score(logits, T=1.0):
    """Energy OOD score ``E(x) = −T·logsumexp(logits/T)`` (Liu et al. 2020).

    Higher energy = more OOD (in-distribution points have a high max logit → high
    logsumexp → low, very-negative energy). A detector-agnostic ablation
    (``method_note §4.1``); here ``logits`` are the GDA log-likelihoods
    ``TiedMahalanobis.logits``.
    """
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    return -float(T) * logsumexp(logits / float(T), axis=1)


def kappa_sigma(cov):
    """Conditioning of ``Σ̂`` -- ``{kappa, sigma_min, eff_rank}`` (REPORTED only).

    ``κ(Σ̂) = σ_max/σ_min``, ``σ_min``, and the participation-ratio effective rank
    ``(Σσ)²/Σσ²`` of the tied covariance's singular values: a FEATURE-COLLAPSE signal
    (a low-rank / collapsed embedding drops the effective rank and inflates ``κ``).
    NEVER a pass/fail gate (``method_note §4.1``; ``prereg §5.4``). Mirrors
    ``conformal.label_shift.kappa_cs`` for ``Ĉ_S``.
    """
    s = np.linalg.svd(np.asarray(cov, dtype=float), compute_uv=False)
    sigma_min = float(s[-1])
    kappa = float(s[0] / s[-1]) if sigma_min > 0 else float("inf")
    eff_rank = float((s.sum() ** 2) / np.sum(s ** 2)) if np.any(s > 0) else 0.0
    return {"kappa": kappa, "sigma_min": sigma_min, "eff_rank": eff_rank}


def ood_auroc(scores, is_ood):
    """AUROC of an OOD score (higher score = OOD) via the Mann-Whitney U statistic.

    ``is_ood`` in {0, 1} with 1 = OOD (the positive class). Ties get average ranks.
    Returns 0.5 when either class is empty. No sklearn (``method_note §4.3``;
    ``prereg §5.2``).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(is_ood).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    n = len(scores)
    while i < n:                                    # average ranks within tie groups
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # 1-based average rank
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    u_pos = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def fpr_at_tpr(scores_ood, scores_id, tpr=0.95):
    """FPR at a target TPR (the standard OOD "FPR95"): ID flagged-OOD rate at TPR=``tpr``.

    Sets the detection threshold where a ``tpr`` fraction of OOD is caught
    (``scores_ood >= thr``), then reports the fraction of ID that exceeds it (the
    false-positive rate). Higher OOD score = more OOD. REPORTED beside AUROC
    (``method_note §4.3``; ``prereg §5.2``).
    """
    so = np.asarray(scores_ood, dtype=float)
    si = np.asarray(scores_id, dtype=float)
    if so.size == 0 or si.size == 0:
        return float("nan")
    thr = float(np.quantile(so, 1.0 - float(tpr)))   # (1-tpr) quantile of OOD scores
    return float(np.mean(si >= thr))
