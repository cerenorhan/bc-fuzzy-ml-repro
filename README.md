# Breast cancer fuzzy + ML reproducibility repo

This repository reproduces (as closely as possible given available materials) the manuscript pipeline using:
- **Fuzzy Logic**: Mamdani inference with triangular/trapezoidal membership functions, centroid defuzzification, and **Wang–Mendel-style rule extraction** with **top-250 rules**.
- **Machine learning baselines**: Logistic Regression, SVM (RBF), Random Forest, and Gradient Boosting (sklearn).

## Data availability
The dataset is **not** included in this repository due to data-sharing restrictions.
It can be requested from the **corresponding author** of the associated manuscript.

After obtaining access, place the Excel file at:
`data/AI_B_CANCER_STUDY_TNZ_Niyazi_CLEANED_STAGE2.xlsx`

The pipeline expects the sheet name: `ModelReady`.
Expected column names are defined in `src/config.py`.

## Quickstart (WSL2 / Ubuntu)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    python -m src.run_all

## Outputs
Results are written to `outputs/`:
- `metrics_cv_fuzzy.csv`, `metrics_test_fuzzy.csv`
- `metrics_cv_ml.csv`, `metrics_test_ml.csv`
- Confusion matrices per target
