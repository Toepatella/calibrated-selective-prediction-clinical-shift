# calibrated-selective-prediction-clinical-shift

Calibrated selective prediction under clinical distribution shift — flagship journal build.

See [flagship-playbook.md](flagship-playbook.md) for the full method, build sequence, and gate ladder (G-A -> G-R). See [repo_links.md](repo_links.md) for every external repo this project draws on and why.

## Data access
- **CAMELYON17-WILDS**: auto-downloads via the `wilds` pip package (see Appendix G in the playbook).
- **CheXpert**: requires Stanford AIMI registration.
- **MIMIC-CXR**: requires PhysioNet credentialing + CITI training.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Layout
See **Appendix E** in `flagship-playbook.md` for the full annotated tree.
