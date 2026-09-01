# Interpretable Clinical AI Benchmarking and Multilayer Genomic Contextualization in Tanzanian Breast Cancer

This repository contains reproducibility materials for the clinical AI benchmarking and independent molecular contextualization components of the study.

- **Expert-rule fuzzy logic**: a multi-output Mamdani inference system based on predefined interpretable clinical rules.
- **Data-driven fuzzy logic**: Mamdani inference with data-driven **Wang–Mendel rule extraction**.
- **Machine learning baselines**: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting (scikit-learn).

The molecular analyses are independent contextual layers and were not used as input features for the clinical fuzzy-logic or machine-learning models.


## Data availability

The local clinical dataset and local patient-level sequencing data are not distributed in this public repository because of data-sharing restrictions.

The clinical dataset may be requested from the corresponding author of the associated manuscript, subject to applicable ethical and institutional requirements. After obtaining access, place the Excel file at `data/AI_B_CANCER_STUDY.xlsx`. The pipeline expects the sheet name `ModelReady`, with feature definitions provided in `src/config.py`.

Public molecular datasets used in the study should be retrieved from their original repositories using the accession information documented under `molecular/`.

Selected aggregate and public-data-derived manuscript outputs are mirrored in `molecular/manuscript_outputs/`. Local sample-linked supplementary tables are intentionally not mirrored publicly.

## Quickstart (WSL2 / Ubuntu)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

## Molecular analyses

The molecular component includes local non-cancer germline WES, local breast-cancer tumour-only WES, public paired tumour-normal WES from PRJNA913947, public RNA-seq from GSE142258, and cross-layer genomic prioritization.

Because matched normal samples were not available for the local tumour WES cohort, variants from that layer are treated as tumour-associated candidate variants rather than definitively somatic or pathogenic variants.

Detailed molecular workflow documentation, configuration, software versions, and manuscript-linked outputs are provided in `molecular/README.md`.

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

