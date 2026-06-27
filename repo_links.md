# Repos referenced in this project

| Repo | Used for | Where referenced |
|---|---|---|
| [aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk) | CRC (Conformal Risk Control, ICLR 2024) — Step 0.1.2 | phase0_learning.ipynb |
| [ryantibs/conformal](https://github.com/ryantibs/conformal) | Weighted conformal `weighted.quantile` (R/common.R, R/split.R) — Step 0.1.5 | phase0_learning.ipynb, gap3-project-playbook.md §0.1.5 note |
| [herbps10/AdaptiveConformal](https://github.com/herbps10/AdaptiveConformal) | ACI (R/aci_algorithm.R) — Step 0.1.5, and the real Phase 2 ACI implementation | phase0_learning.ipynb, gap3-project-playbook.md §0.1.5 note |
| [aangelopoulos/rcps](https://github.com/aangelopoulos/rcps) | RCPS — PAC (α,δ) risk-control backbone (Bates et al., JACM 2021, [arXiv:2101.02703](https://arxiv.org/abs/2101.02703)); `core/` has concentration bounds + λ̂. **v1 spine (Stage A)** | flagship-playbook.md §3.5, Stage A |
| [aangelopoulos/ltt](https://github.com/aangelopoulos/ltt) | LTT — multiple-testing wrapper for jointly tuning (τ,λ,t_ood) (Angelopoulos et al., AOAS 2025, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)). **v1** when joint-tuning thresholds | flagship-playbook.md §3.5 |
| [gostevehoward/confseq](https://github.com/gostevehoward/confseq) | WSR betting bound — tighter UCB for weighted RCPS (Waudby-Smith & Ramdas, JRSS-B 2024, [arXiv:2010.09686](https://arxiv.org/abs/2010.09686)); betting code in `src/confseq/betting.py`. **v1** (replaces Hoeffding–Bentkus) | flagship-playbook.md §3.5, Stage A |
| [iamownt/TRUECAM](https://github.com/iamownt/TRUECAM) | Closest competitor method (CRC + SNGP + OOD elimination) — reference only, not implemented | Gap3_audit_calibrated_selective_prediction_under_shift.md, row 1 |
| [p-lambda/wilds](https://github.com/p-lambda/wilds) (`wilds` pip package) | CAMELYON17-WILDS data loaders, Phase 1 | gap3-project-playbook.md §1.4.1, Appendix G |
| [mueller-mp/maha-norm](https://github.com/mueller-mp/maha-norm) | Mahalanobis++ (L2-normalized Mahalanobis OOD score, Müller et al. 2025) — Step 0.1.6 | phase0_learning.ipynb |
| [deeplearning-wisc/knn-ood](https://github.com/deeplearning-wisc/knn-ood) | kNN OOD detector (Sun et al. 2022, `run_cifar.py`) — Step 0.1.6 | phase0_learning.ipynb |
| [Toepatella/calibrated-selective-prediction-clinical-shift](https://github.com/Toepatella/calibrated-selective-prediction-clinical-shift) | Your own project repo (hosts phase0_learning.ipynb via Colab badge) | phase0_learning.ipynb header |
| [rohanhore/RLCP](https://github.com/rohanhore/RLCP) | Randomly localized conformal prediction (Hore & Barber, arXiv:2310.07850) — local (not just marginal) coverage guarantees, model-agnostic. **v1 Stage S (optional)** — strengthens the subgroup-validity audit toward approx. conditional coverage | flagship-playbook.md §5 Stage S |

Not GitHub repos (credentialed data-access portals):
- CheXpert — Stanford AIMI registration
- MIMIC-CXR — PhysioNet + CITI training
