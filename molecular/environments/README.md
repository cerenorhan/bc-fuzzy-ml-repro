# Molecular analysis software environments

The molecular analyses were executed with Micromamba-managed environments.
The environment names used in the original workflow are retained in the scripts
for provenance and reproducibility.

## hadza-wes

Primary WES preprocessing, variant calling, QC, and population-genomic analyses.

Key software versions:

- GATK 4.6.2.0
- bcftools 1.24
- samtools 1.23.1
- BWA-MEM2 2.2.1
- fastp 1.3.6
- PLINK 2.0 a.6.9LM
- Python 3.11.15

## cancer_anno

Variant annotation environment.

Key software versions:

- SnpEff 5.4c
- Python 3.14.6

## rnaseq

RNA-seq downstream analysis, statistical analysis, plotting, pathway analysis,
and general R/Python utilities.

Key software versions:

- R 4.5.3
- Python 3.11.15
- MultiQC 1.35
- DESeq2 1.50.2
- ggplot2 4.0.3
- dplyr 1.2.1
- readr 2.2.0
- tidyr 1.3.2
- stringr 1.6.0
- forcats 1.0.1
- purrr 1.2.2
- msigdbr 26.1.0

## rnaseq_tophat

Primary GSE142258 read preprocessing and alignment environment.

Key software versions:

- Trimmomatic 0.40
- TopHat2 2.1.1
- featureCounts 2.1.1

## Notes

The repository documents the software versions used for the reported analyses.
Exact dependency resolution may vary across operating systems and package
channels. Users reproducing the workflows should prioritize the major tool
versions listed above and the analysis parameters encoded in the scripts.
