# Benchmarking expert-rule fuzzy, data-driven fuzzy, and machine-learning models for breast cancer classification in Tanzanian women

This repository reproduces and benchmarks a breast-cancer endpoint prediction pipeline using:

- **Expert-rule fuzzy logic**: a multi-output Mamdani inference system based on predefined interpretable clinical rules.
- **Data-driven fuzzy logic**: Mamdani inference with data-driven **Wang–Mendel rule extraction**.
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


## Figure organization for manuscript and supplementary materials

To preserve computational reproducibility, the original benchmark workflow and native output filenames were kept intact wherever possible.  
The manuscript, however, follows a curated figure order built on top of those reproducible outputs.

This study compares three model families:

1. **Expert-rule fuzzy**
2. **Data-driven fuzzy**
3. **Machine-learning baselines**

### Main manuscript figures

The current main-text figure order is:

1. `main/Fig01_workflow_schematic`  
   Workflow overview of preprocessing, repeated splits, expert-rule fuzzy, data-driven fuzzy, machine-learning models, evaluation, and unknown-stage secondary analyses.

2. `main/Fig02_class_distribution`  
   Class distributions of the four modeled endpoints: Stage, Diagnosis, Laterality, and IHC.

3. `main/Fig03_performance_heatmap`  
   Overall model performance heatmap based on mean weighted F1-score across repeated splits.

4. `main/Fig04_expert_data_ml_comparison`  
   Per-endpoint comparison of expert-rule fuzzy, data-driven fuzzy, and the best-performing machine-learning model.

5. `main/Fig05_metric_distributions`  
   Fold/seed-wise weighted F1-score distributions across repeated train/test splits.

6. `main/Fig06_confusion_matrices`  
   Confusion-matrix comparison of data-driven fuzzy, expert-rule fuzzy, and the best-performing machine-learning model for each endpoint.

7. `main/Fig07_unknown_stage_methods`  
   Secondary analyses for Unknown-Stage samples, including initial prediction, self-training, and label propagation perspectives.

8. `main/Fig08_feature_importance`  
   Feature-importance summary for Stage classification models.

### Supplementary figures

The current supplementary figure set is:

- `supplementary/FigS01_split_overview`
- `supplementary/FigS02_scatter_comparison`
- `supplementary/FigS03_stage_roc`
- `supplementary/FigS04_stage_pr_curve`
- `supplementary/FigS05_correlation_heatmap`
- `supplementary/FigS06_method_agreement_heatmap`
- `supplementary/FigS07_agreement_lollipop`

### Reproducibility note

The manuscript-oriented figure numbering above is a presentation layer and does not redefine the internal benchmark workflow.  
Re-running the benchmark regenerates the analysis outputs, while the manuscript uses the curated figure order listed here.

