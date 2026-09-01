# Molecular analyses

This directory contains the reproducibility materials for the molecular component of:

**Interpretable Clinical AI Benchmarking and Multilayer Genomic Contextualization in Tanzanian Breast Cancer**

The molecular analyses were performed as independent contextual layers and were not used as input features for the clinical fuzzy-logic or machine-learning models.

## Analysis layers

The molecular component consists of:

1. Local non-cancer germline WES
2. Local breast-cancer tumour-only WES
3. Public paired tumour-normal WES from PRJNA913947
4. Public RNA-seq from GSE142258
5. Cross-layer genomic prioritization

## Directory structure

```text
molecular/
├── config/
├── environments/
├── wes/
│   ├── local_germline/
│   ├── local_tumor/
│   └── PRJNA913947/
├── rnaseq/
│   └── GSE142258/
├── integration/
└── manuscript_outputs/
```

## 1. Local non-cancer germline WES

Location: `wes/local_germline/`

This analysis provides germline and population-genomic context for the local cohort and is not intended as a breast-cancer association analysis.

The workflow includes FASTQ manifest generation, institutional metadata normalization, lane-level manifest construction, fastp preprocessing, BWA-MEM2 alignment, germline variant calling, joint genotyping, autosomal variant preparation, PLINK-based PCA, variant-filtering summaries, rare functional variant contextualization, and manuscript figure generation.

Restricted institutional metadata are not included in the repository.

## 2. Local breast-cancer tumour WES

Location: `wes/local_tumor/`

Matched normal samples were not available for this local breast-cancer WES layer. Accordingly, variants identified through this workflow are treated as tumour-associated candidate variants rather than definitively somatic variants.

The workflow includes input audit, tumour-only Mutect2 calling, contamination-related filtering, population/background filtering, functional annotation, focused candidate selection, and manuscript-oriented visualization.

Principal scripts:

```text
79A_audit_local_tumor_WES_inputs.sh
79B_local_tumor_WES_tumor_only_Mutect2_from_existing_BAMs.sh
79C_local_tumor_WES_candidate_filtering_and_figures.sh
79D_local_tumor_WES_strict_interpretable_candidates.sh
79E_local_tumor_WES_FINAL_focus_candidate_figures.sh
```

## 3. PRJNA913947 paired tumour-normal WES

Location: `wes/PRJNA913947/`

This public dataset provides the paired tumour-normal WES layer.

The workflow includes ENA/NCBI metadata retrieval, run-level metadata reconstruction, FASTQ download-list generation, FASTQ integrity checking, tumour-normal pair resolution, paired Mutect2 calling, functional annotation, recurrent functional-gene summaries, pathway-level contextualization, and manuscript figure generation.

Public accession metadata retained in the repository:

```text
metadata/PRJNA913947_run_accessions.txt
metadata/PRJNA913947_fastq_inventory.tsv
```

Large FASTQ, BAM, and intermediate variant files are not included.

## 4. GSE142258 RNA-seq

Location: `rnaseq/GSE142258/`

The canonical RNA-seq workflow used for the manuscript was:

```text
FASTQ retrieval
    ↓
Trimmomatic
    ↓
TopHat2
    ↓
featureCounts
    ↓
DESeq2
    ↓
gene-symbol annotation
    ↓
ranked pathway analysis
    ↓
publication figures
```

Primary analysis script: `84_GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2.sh`

Resume helper: `helpers/85_GSE142258_resume_from_trimmed_TopHat_featureCounts_DESeq2.sh`

Downstream manuscript-analysis scripts:

```text
74C_paper_results_04A_rnaseq_GSE142258_DESeq2_ranked_pathway_noinstall_fullrank.sh
75B_paper_results_04B_RNA_ranked_pathway_interpretation_flags_FIXED.sh
76_paper_results_04C_RNA_DESeq2_publication_volcano.sh
77A2_publication_polish_GSE142258_RNA_figures.sh
77A3_publication_final_RNA_selected_pathway_dotplot.sh
```

The RNA-seq layer provides independent transcriptomic context and is not included in the final genomic-only support score.

## 5. Integrated genomic prioritization

Location: `integration/scripts/`

Principal scripts:

```text
80A_integrated_multilayer_candidate_prioritization.sh
80B_integrated_FINAL_clean_focus_figures.sh
80C_integrated_FINAL_genomic_only_figures.sh
```

The final manuscript support matrix is genomic-only. RNA-seq results are retained as biological context rather than being counted in the final genomic support score.

## Configuration

All molecular shell workflows use `config/load_config.sh`.

A portable configuration template is provided at `config/config.example.sh`.

Machine-specific `config.local.sh` is excluded from version control. Optional execution notifications are disabled by default.

## Software environments

Software and version information is documented in `environments/README.md`.

Micromamba environment names retained in the scripts are:

- `hadza-wes`
- `cancer_anno`
- `rnaseq`
- `rnaseq_tophat`

## Manuscript-linked outputs

Main molecular figure source panels are located in `manuscript_outputs/figures/main/` and correspond to molecular source panels used in manuscript Figures 5–8.

Supplementary molecular figures corresponding to Supplementary Figures S8–S17 are located in `manuscript_outputs/figures/supplementary/`.

The following manuscript-linked molecular supplementary tables are mirrored in `manuscript_outputs/supplementary_data/`:

- Supplementary Data 12 — local WES VCF filtering cascade
- Supplementary Data 14 — PRJNA913947 unbiased functional gene summary
- Supplementary Data 15 — GSE142258 ranked pathway summary

Supplementary Data 13, 16, 17, and 18 contain local sample-level identifiers or local sample-linked molecular information and are therefore not mirrored in the public repository.

These restricted supplementary tables remain part of the manuscript submission materials and may be made available subject to applicable ethical and institutional data-sharing requirements.

## Data-sharing boundaries

The repository intentionally excludes raw local FASTQ files, BAM/CRAM files, restricted institutional metadata, raw or intermediate local VCF/gVCF files, large temporary analysis files, and workstation-specific configuration files.

Public sequencing data should be retrieved from their original repositories.

Restricted local data may be requested from the corresponding author, subject to applicable ethical and institutional data-sharing requirements.
