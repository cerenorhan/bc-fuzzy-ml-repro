# Breast cancer fuzzy-logic vs machine-learning reproducibility repo

This repository reproduces and benchmarks a breast-cancer endpoint prediction pipeline using:
- **Fuzzy logic**: Mamdani inference with data-driven **Wang–Mendel rule extraction**.
- **Machine learning baselines**: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting (scikit-learn).

## Data availability
The dataset is **not** included due to data-sharing restrictions.
It can be requested from the **corresponding author** of the associated manuscript.

After obtaining access, place the Excel file at:
`data/AI_B_CANCER_STUDY.xlsx`

The pipeline expects the sheet name: `ModelReady`.
Expected feature columns are defined in `src/config.py`.

## Quickstart (WSL2 / Ubuntu)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
