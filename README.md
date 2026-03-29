# Benchmarking interpretable fuzzy logic and machine-learning approaches for breast cancer classification in Tanzanian women

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


## Figure organization for manuscript and supplementary materials

To preserve computational reproducibility, the original figure-generation code and its native output filenames were kept unchanged.  
The benchmark pipeline therefore continues to generate the original figure files under `outputs/figures/`.

For manuscript preparation, additional organized copies of the final figures were created under:

- `outputs/figures/main/`
- `outputs/figures/supplementary/`
- `outputs/figures/panel_sources/`

This structure was introduced without changing the underlying benchmark logic, analysis workflow, or figure-generation code. The goal is to keep the original pipeline reproducible while also providing a clean manuscript-oriented presentation layer.

### Main manuscript figures

The current manuscript figure order is:

1. `Fig01_workflow_schematic`  
   Workflow schematic used in the Materials and Methods section.

2. `Fig02_class_distribution`  
   Class distribution overview for Stage, Diagnosis, Laterality, and IHC.

3. `Fig03_metric_distributions`  
   Fold/seed-wise performance distributions across repeated train/test splits.

4. `Fig04_confusion_matrices_combined`  
   Combined confusion-matrix panel comparing fuzzy logic and the best-performing machine-learning model across targets.

5. `Fig05_stage_feature_importance`  
   Feature-importance summary for the Stage classification task.

### Supplementary figures

The current supplementary figure set is:

- `FigS01_unknown_stage_methods`
- `FigS02_method_agreement_heatmap`
- `FigS03_split_overview`
- `FigS04_stage_roc`
- `FigS05_stage_precision_recall`
- `FigS06_correlation_heatmap`
- `FigS07_agreement_lollipop`
- `FigS08_model_scatter`

### Panel source figures

To retain transparency for multi-panel figure assembly, source panels for the combined confusion-matrix figure are also stored separately:

- `Fig04_panel_source_diagnosis`
- `Fig04_panel_source_ihc`
- `Fig04_panel_source_laterality`
- `Fig04_panel_source_stage`

These files are retained as source components for figure assembly and review, while the combined panel remains the main manuscript version.

### Reproducibility note

The directories `main/`, `supplementary/`, and `panel_sources/` reflect manuscript-oriented curation only.  
They do not replace the original benchmark outputs and do not alter the reproducible workflow. Re-running the pipeline still regenerates the original output files with their native names.

