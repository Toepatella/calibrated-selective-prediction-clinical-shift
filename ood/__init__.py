"""Feature-space OOD screen for the trustworthy selective-prediction pipeline.

Stage D (integrated OOD budget & routing) scores frozen features with a
tied-covariance Gaussian-discriminant Mahalanobis detector and routes far-OOD out of
the answered path. The primary ``o(x)`` is **Mahalanobis++** (L2-normalized;
Müller et al. 2025); the plain Lee et al. (2018) tied-covariance score, kNN, and
energy are detector-agnostic ablations over the SAME routing rule (``method_note
§4.1``). ``o`` is a MEASURED screen with a reported leakage rate on a stated,
swappable exposure set ``O`` -- no distribution-free OOD guarantee (Fang et al. 2022;
``method_note §4.4``, §7.1). The ``t_ood`` budget selection and the integrated
decision rule ``A(x)`` live in ``conformal/budget.py``.
"""

from .detector import (
    l2_normalize,
    TiedMahalanobis,
    knn_score,
    energy_score,
    kappa_sigma,
    ood_auroc,
    fpr_at_tpr,
)

__all__ = [
    "l2_normalize",
    "TiedMahalanobis",
    "knn_score",
    "energy_score",
    "kappa_sigma",
    "ood_auroc",
    "fpr_at_tpr",
]
