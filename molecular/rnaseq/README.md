# RNA-seq analyses

This directory contains reproducibility materials for the transcriptomic component of the study.

The final manuscript RNA-seq analysis uses:

- GSE142258

The canonical workflow is:

Trimmomatic → TopHat2 → featureCounts → DESeq2 → ranked pathway analysis → manuscript-oriented figures.

Raw sequencing data and large intermediate files are not included in the repository.

Public sequencing data should be retrieved from the original repository using the accession information provided in the workflow.

Detailed molecular workflow documentation is provided in:

`../README.md`
