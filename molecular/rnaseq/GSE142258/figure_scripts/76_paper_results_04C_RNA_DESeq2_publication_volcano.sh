#!/usr/bin/env bash

# Load portable molecular-analysis paths.
_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${_REPO_ROOT}" ]]; then
    echo "ERROR: Run this script from within the bc-fuzzy-ml-repro repository." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${_REPO_ROOT}/molecular/config/load_config.sh"

set -euo pipefail

cd $PROJECT_ROOT

PROJECT="$PROJECT_ROOT"
PAPER="$PAPER_RESULTS_ROOT"

DESEQ_FILE="$PROJECT/results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2/DESeq2/annotated/GSE142258_DESeq2_late_vs_early_results.annotated.tsv"

RNA_OUT="$PAPER/04_rnaseq_GSE142258"
OUTDIR="$RNA_OUT/05_publication_volcano"
FIGS="$RNA_OUT/03_figures"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUTDIR" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/76_paper_results_04C_RNA_DESeq2_publication_volcano_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 04C: RNA DESeq2 PUBLICATION VOLCANO ==="
date

if [[ ! -s "$DESEQ_FILE" ]]; then
  echo "ERROR: DESeq2 full annotated file not found:"
  echo "$DESEQ_FILE"
  exit 1
fi

echo
echo "Input:"
ls -lh "$DESEQ_FILE"
wc -l "$DESEQ_FILE"

export DESEQ_FILE OUTDIR FIGS STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(stringr)
})

deseq_file <- Sys.getenv("DESEQ_FILE")
outdir <- Sys.getenv("OUTDIR")
figs <- Sys.getenv("FIGS")
status <- Sys.getenv("STATUS")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(figs, recursive = TRUE, showWarnings = FALSE)

padj_thr <- 0.05
lfc_thr <- 1

df <- read_tsv(deseq_file, show_col_types = FALSE)

required <- c("Geneid", "gene_name", "baseMean", "log2FoldChange", "pvalue", "padj")
missing <- setdiff(required, names(df))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

vol <- df %>%
  mutate(
    gene_symbol = as.character(gene_name),
    gene_symbol = ifelse(is.na(gene_symbol) | gene_symbol == "" | gene_symbol == ".", as.character(Geneid), gene_symbol),
    baseMean = as.numeric(baseMean),
    log2FoldChange = as.numeric(log2FoldChange),
    pvalue = as.numeric(pvalue),
    padj = as.numeric(padj),

    # padj=NA satırlarını grafikten düşürmüyoruz; independent filtering / low-information gibi kabul edip y=0'a koyuyoruz.
    padj_for_plot = ifelse(is.na(padj) | padj <= 0, 1, padj),
    neglog10_padj = -log10(padj_for_plot),

    expression_class = case_when(
      !is.na(padj) & padj < padj_thr & log2FoldChange >= lfc_thr ~ "Upregulated",
      !is.na(padj) & padj < padj_thr & log2FoldChange <= -lfc_thr ~ "Downregulated",
      TRUE ~ "Not significant / unchanged"
    )
  ) %>%
  filter(!is.na(log2FoldChange), !is.na(neglog10_padj))

vol$expression_class <- factor(
  vol$expression_class,
  levels = c("Downregulated", "Not significant / unchanged", "Upregulated")
)

class_counts <- vol %>%
  count(expression_class, name = "n_genes") %>%
  mutate(
    padj_threshold = padj_thr,
    abs_log2FC_threshold = lfc_thr
  )

write_tsv(class_counts, file.path(outdir, "GSE142258_DESeq2_volcano_gene_class_counts.tsv"))
write_tsv(vol, file.path(outdir, "GSE142258_DESeq2_volcano_classified_genes.tsv"))

label_df <- vol %>%
  filter(expression_class != "Not significant / unchanged") %>%
  arrange(padj, desc(abs(log2FoldChange))) %>%
  slice_head(n = 20)

# Renkler sabit: yayın figürü için sınıflar net ayrılıyor.
vol_colors <- c(
  "Downregulated" = "#2B6CB0",
  "Not significant / unchanged" = "grey75",
  "Upregulated" = "#C53030"
)

p <- ggplot(vol, aes(x = log2FoldChange, y = neglog10_padj)) +
  geom_point(aes(color = expression_class), alpha = 0.55, size = 1.1) +
  geom_vline(xintercept = c(-lfc_thr, lfc_thr), linetype = "dashed", linewidth = 0.35) +
  geom_hline(yintercept = -log10(padj_thr), linetype = "dashed", linewidth = 0.35) +
  scale_color_manual(values = vol_colors, drop = FALSE) +
  labs(
    title = "GSE142258 RNA-seq DESeq2: late vs early",
    subtitle = paste0("Thresholds: adjusted p < ", padj_thr, ", |log2FC| ≥ ", lfc_thr),
    x = "log2 fold change",
    y = "-log10 adjusted p-value",
    color = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top",
    legend.justification = "left"
  )

if (nrow(label_df) > 0) {
  p <- p +
    geom_text(
      data = label_df,
      aes(label = gene_symbol),
      size = 2.8,
      vjust = -0.6,
      check_overlap = TRUE,
      show.legend = FALSE
    )
}

png_out <- file.path(figs, "GSE142258_DESeq2_publication_volcano_up_down_unchanged.png")
pdf_out <- file.path(figs, "GSE142258_DESeq2_publication_volcano_up_down_unchanged.pdf")

ggsave(png_out, p, width = 7.2, height = 6.0, dpi = 600)
ggsave(pdf_out, p, width = 7.2, height = 6.0)

# Daha temiz labelsiz versiyon
p_nolabel <- p + labs(subtitle = paste0("Upregulated, downregulated, and unchanged genes; adjusted p < ", padj_thr, ", |log2FC| ≥ ", lfc_thr))

png_nolabel_out <- file.path(figs, "GSE142258_DESeq2_publication_volcano_up_down_unchanged_nolabel.png")
pdf_nolabel_out <- file.path(figs, "GSE142258_DESeq2_publication_volcano_up_down_unchanged_nolabel.pdf")

ggsave(png_nolabel_out, p_nolabel, width = 7.2, height = 6.0, dpi = 600)
ggsave(pdf_nolabel_out, p_nolabel, width = 7.2, height = 6.0)

report <- file.path(status, "GSE142258_stage_04C_DESeq2_publication_volcano_status.txt")

sink(report)
cat("GSE142258 Stage 04C DESeq2 publication volcano\n")
cat("Generated:", as.character(Sys.time()), "\n")
cat("Input:", deseq_file, "\n")
cat("padj threshold:", padj_thr, "\n")
cat("abs log2FC threshold:", lfc_thr, "\n\n")
cat("Gene class counts\n")
print(class_counts)
cat("\nOutput figures\n")
cat(png_out, "\n")
cat(pdf_out, "\n")
cat(png_nolabel_out, "\n")
cat(pdf_nolabel_out, "\n")
sink()

cat("classified_genes", file.path(outdir, "GSE142258_DESeq2_volcano_classified_genes.tsv"), "\n")
cat("class_counts", file.path(outdir, "GSE142258_DESeq2_volcano_gene_class_counts.tsv"), "\n")
cat("volcano_png", png_out, "\n")
cat("volcano_pdf", pdf_out, "\n")
cat("volcano_nolabel_png", png_nolabel_out, "\n")
cat("volcano_nolabel_pdf", pdf_nolabel_out, "\n")
cat("report", report, "\n")
RS

echo
echo "=== VOLCANO GENE CLASS COUNTS ==="
column -t -s $'\t' "$OUTDIR/GSE142258_DESeq2_volcano_gene_class_counts.tsv"

echo
echo "=== VOLCANO FIGURES ==="
ls -lh "$FIGS"/GSE142258_DESeq2_publication_volcano_up_down_unchanged*

echo
echo "=== DONE: STAGE 04C ==="
date
