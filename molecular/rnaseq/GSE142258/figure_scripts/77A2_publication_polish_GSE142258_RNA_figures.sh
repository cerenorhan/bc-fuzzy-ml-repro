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

DESEQ="$PROJECT/results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2/DESeq2/annotated/GSE142258_DESeq2_late_vs_early_results.annotated.tsv"
RNA_SUM="$PAPER/04_rnaseq_GSE142258/04_interpretable_ranked_pathway_summary/GSE142258_RNA_ranked_pathway_biology_axis_summary.tsv"
RNA_TERMS="$PAPER/04_rnaseq_GSE142258/04_interpretable_ranked_pathway_summary/GSE142258_RNA_ranked_pathway_primary_interpretable_FDR025.tsv"

OUT="$PAPER/publication_figures/GSE142258_RNA_polished"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/77A2_publication_polish_GSE142258_RNA_figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 77A2: POLISHED PUBLICATION RNA FIGURES ==="
date

for f in "$DESEQ" "$RNA_SUM" "$RNA_TERMS"; do
  [[ -s "$f" ]] || { echo "ERROR missing: $f"; exit 1; }
done

export DESEQ RNA_SUM RNA_TERMS OUT STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
  library(tidyr)
})

deseq_file <- Sys.getenv("DESEQ")
axis_file  <- Sys.getenv("RNA_SUM")
terms_file <- Sys.getenv("RNA_TERMS")
outdir     <- Sys.getenv("OUT")
statusdir  <- Sys.getenv("STATUS")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 3),
      plot.subtitle = element_text(size = base_size),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.35, color = "grey20"),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(face = "bold")
    )
}

nice_axis <- function(x) {
  recode(
    x,
    "translation_ribosome_RNA" = "Translation / ribosome / rRNA",
    "immune_antigen_T_NK" = "Immune antigen / T-cell / NK",
    "cell_cycle_DNA_repair" = "Cell cycle / DNA repair",
    "development_Notch_TGF_WNT" = "Notch / TGF-beta / WNT",
    "hormone_luminal" = "Hormone / luminal signaling",
    "metabolism_hypoxia_lipid" = "Metabolism / hypoxia / lipid",
    "EMT_invasion_angiogenesis" = "EMT / invasion / angiogenesis",
    "EMT_invasion_angiogenesis;immune_antigen_T_NK" = "EMT + immune migration",
    "immune_antigen_T_NK;metabolism_hypoxia_lipid" = "Immune + metabolic",
    "cell_cycle_DNA_repair;immune_antigen_T_NK" = "Cell cycle + immune",
    .default = x
  )
}

dir_cols <- c(
  "Down in late" = "#D62728",
  "Up in late" = "#1F77B4"
)

axis_cols <- c(
  "Translation / ribosome / rRNA" = "#1F77B4",
  "Immune antigen / T-cell / NK" = "#009E73",
  "Cell cycle / DNA repair" = "#D55E00",
  "Notch / TGF-beta / WNT" = "#CC79A7",
  "Hormone / luminal signaling" = "#7E57C2",
  "Metabolism / hypoxia / lipid" = "#E69F00",
  "EMT / invasion / angiogenesis" = "#00A6D6",
  "EMT + immune migration" = "#00BFC4",
  "Immune + metabolic" = "#66A61E",
  "Cell cycle + immune" = "#A6761D"
)

# ---------------------------
# 1. Volcano, explicit zero downregulated
# ---------------------------
de <- read_tsv(deseq_file, show_col_types = FALSE)

vol <- de %>%
  mutate(
    gene_symbol = as.character(gene_name),
    gene_symbol = ifelse(is.na(gene_symbol) | gene_symbol == "" | gene_symbol == ".", as.character(Geneid), gene_symbol),
    log2FoldChange = suppressWarnings(as.numeric(log2FoldChange)),
    padj = suppressWarnings(as.numeric(padj)),
    padj_for_plot = ifelse(is.na(padj) | padj <= 0, 1, padj),
    neglog10_padj = -log10(padj_for_plot),
    regulation = case_when(
      !is.na(padj) & padj < 0.05 & log2FoldChange > 0 ~ "Upregulated",
      !is.na(padj) & padj < 0.05 & log2FoldChange < 0 ~ "Downregulated",
      TRUE ~ "Unchanged"
    ),
    regulation = factor(regulation, levels = c("Upregulated", "Downregulated", "Unchanged"))
  ) %>%
  filter(!is.na(log2FoldChange), is.finite(log2FoldChange), is.finite(neglog10_padj))

vol_counts <- vol %>%
  count(regulation, .drop = FALSE, name = "n_genes")

write_tsv(vol_counts, file.path(outdir, "RNA_volcano_counts_explicit_zero_down.tsv"))

label_df <- vol %>%
  filter(regulation != "Unchanged") %>%
  arrange(padj, desc(abs(log2FoldChange))) %>%
  slice_head(n = 10)

subtitle_txt <- paste0(
  "Adjusted p < 0.05; upregulated = ",
  vol_counts$n_genes[vol_counts$regulation == "Upregulated"],
  ", downregulated = ",
  vol_counts$n_genes[vol_counts$regulation == "Downregulated"],
  ", unchanged = ",
  vol_counts$n_genes[vol_counts$regulation == "Unchanged"]
)

p_vol <- ggplot(vol, aes(x = log2FoldChange, y = neglog10_padj)) +
  geom_point(
    data = vol %>% filter(regulation == "Unchanged"),
    aes(color = regulation),
    size = 0.9,
    alpha = 0.35
  ) +
  geom_point(
    data = vol %>% filter(regulation != "Unchanged"),
    aes(color = regulation),
    size = 2.4,
    alpha = 0.95
  ) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.35, color = "grey35") +
  geom_text(
    data = label_df,
    aes(label = gene_symbol),
    size = 3.2,
    vjust = -0.6,
    check_overlap = TRUE,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = c(
      "Upregulated" = "#1F77B4",
      "Downregulated" = "#D62728",
      "Unchanged" = "grey78"
    ),
    drop = FALSE
  ) +
  labs(
    title = "RNA-seq differential expression",
    subtitle = subtitle_txt,
    x = "log2 fold change",
    y = expression(-log[10](adjusted~p)),
    color = NULL
  ) +
  theme_pub(11)

ggsave(file.path(outdir, "Fig_RNA_01_volcano_polished.pdf"), p_vol, width = 7.6, height = 5.6)
ggsave(file.path(outdir, "Fig_RNA_01_volcano_polished.png"), p_vol, width = 7.6, height = 5.6, dpi = 600)

# ---------------------------
# 2. Bubble matrix instead of purple heatmap
# ---------------------------
axis_df <- read_tsv(axis_file, show_col_types = FALSE) %>%
  filter(reporting_tier == "interpretable_FDR025", biology_axis != "other") %>%
  mutate(
    n_terms = as.numeric(n_terms),
    best_padj = as.numeric(best_padj),
    direction_label = ifelse(direction == "up_in_late", "Up in late", "Down in late"),
    axis_label = nice_axis(biology_axis),
    support_score = -log10(best_padj + 1e-300)
  )

axis_main <- axis_df %>%
  group_by(axis_label, direction_label) %>%
  summarise(
    n_terms = sum(n_terms, na.rm = TRUE),
    best_padj = min(best_padj, na.rm = TRUE),
    support_score = -log10(best_padj + 1e-300),
    .groups = "drop"
  ) %>%
  mutate(
    axis_label = fct_reorder(axis_label, n_terms, .fun = max)
  )

write_tsv(axis_main, file.path(outdir, "RNA_axis_bubble_matrix_data.tsv"))

p_bubble <- ggplot(axis_main, aes(x = direction_label, y = axis_label)) +
  geom_point(aes(size = n_terms, fill = direction_label), shape = 21, color = "grey15", stroke = 0.35, alpha = 0.92) +
  geom_text(aes(label = n_terms), size = 3.3, fontface = "bold", color = "white") +
  scale_fill_manual(values = dir_cols) +
  scale_size_continuous(range = c(5, 22)) +
  labs(
    title = "RNA pathway axis support",
    subtitle = "Bubble size and label indicate the number of interpretable ranked pathway terms",
    x = NULL,
    y = NULL,
    fill = "Direction",
    size = "Terms"
  ) +
  theme_pub(11) +
  theme(
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.25),
    axis.text.y = element_text(size = 10.5),
    legend.position = "right"
  )

ggsave(file.path(outdir, "Fig_RNA_02_axis_bubble_matrix_polished.pdf"), p_bubble, width = 8.2, height = 5.8)
ggsave(file.path(outdir, "Fig_RNA_02_axis_bubble_matrix_polished.png"), p_bubble, width = 8.2, height = 5.8, dpi = 600)

# ---------------------------
# 3. Clean top pathway dotplot
# ---------------------------
terms <- read_tsv(terms_file, show_col_types = FALSE) %>%
  mutate(
    padj = as.numeric(padj),
    pvalue = as.numeric(pvalue),
    size = as.numeric(size),
    neglog10_fdr = -log10(padj + 1e-300),
    direction_label = ifelse(direction == "up_in_late", "Up in late", "Down in late"),
    axis_label = nice_axis(biology_axis),
    term_short = gs_name %>%
      str_replace("^HALLMARK_", "") %>%
      str_replace("^REACTOME_", "") %>%
      str_replace("^KEGG_", "") %>%
      str_replace("^GOBP_", "") %>%
      str_replace_all("_", " ") %>%
      str_to_title() %>%
      str_wrap(width = 42)
  ) %>%
  filter(reporting_tier == "interpretable_FDR025") %>%
  filter(biology_axis != "other") %>%
  filter(!str_detect(gs_name, "MEDICUS"))

top_terms <- terms %>%
  group_by(axis_label, direction_label) %>%
  arrange(padj, pvalue, .by_group = TRUE) %>%
  slice_head(n = 3) %>%
  ungroup() %>%
  arrange(direction_label, axis_label, padj) %>%
  group_by(direction_label) %>%
  slice_head(n = 24) %>%
  ungroup() %>%
  mutate(term_short = fct_reorder(term_short, neglog10_fdr))

write_tsv(top_terms, file.path(outdir, "RNA_top_pathway_dotplot_clean_data.tsv"))

p_dot <- ggplot(top_terms, aes(x = neglog10_fdr, y = term_short)) +
  geom_point(aes(size = size, color = axis_label), alpha = 0.92) +
  facet_grid(direction_label ~ ., scales = "free_y", space = "free_y") +
  scale_color_manual(values = axis_cols) +
  scale_size_continuous(range = c(2.8, 7.8)) +
  labs(
    title = "Top ranked RNA pathway terms",
    subtitle = "Top terms per biological axis and direction; other and MEDICUS terms excluded",
    x = expression(-log[10](FDR)),
    y = NULL,
    color = "Biological axis",
    size = "Set size"
  ) +
  theme_pub(10) +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 8.8),
    strip.text = element_text(size = 11, face = "bold")
  )

ggsave(file.path(outdir, "Fig_RNA_03_top_pathway_dotplot_polished.pdf"), p_dot, width = 9.2, height = 8.2)
ggsave(file.path(outdir, "Fig_RNA_03_top_pathway_dotplot_polished.png"), p_dot, width = 9.2, height = 8.2, dpi = 600)

# ---------------------------
# 4. Directional barplot with labels
# ---------------------------
bar_df <- axis_main %>%
  mutate(
    signed_terms = ifelse(direction_label == "Up in late", n_terms, -n_terms),
    label_x = ifelse(direction_label == "Up in late", signed_terms + 5, signed_terms - 5)
  )

p_bar <- ggplot(bar_df, aes(x = signed_terms, y = axis_label, fill = direction_label)) +
  geom_col(width = 0.68) +
  geom_vline(xintercept = 0, color = "grey25", linewidth = 0.35) +
  geom_text(aes(x = label_x, label = n_terms), size = 3.1, fontface = "bold") +
  scale_fill_manual(values = dir_cols) +
  scale_x_continuous(labels = abs) +
  labs(
    title = "Directional RNA pathway shifts",
    subtitle = "Interpretable ranked pathway terms at FDR <= 0.25",
    x = "Number of enriched pathway terms",
    y = NULL,
    fill = "Direction"
  ) +
  theme_pub(11) +
  theme(
    legend.position = "top",
    axis.text.y = element_text(size = 10.5)
  )

ggsave(file.path(outdir, "Fig_RNA_04_directional_axis_barplot_polished.pdf"), p_bar, width = 8.4, height = 5.8)
ggsave(file.path(outdir, "Fig_RNA_04_directional_axis_barplot_polished.png"), p_bar, width = 8.4, height = 5.8, dpi = 600)

report <- file.path(statusdir, "77A2_publication_polish_GSE142258_RNA_figures_status.txt")
sink(report)
cat("77A2 polished publication RNA figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Volcano counts:\n")
print(vol_counts)
cat("\nAxis bubble matrix data:\n")
print(axis_main)
cat("\nTop pathway dotplot terms:\n")
print(top_terms %>% select(database, gs_name, axis_label, direction_label, size, padj) %>% head(80))
sink()

cat("output_dir", outdir, "\n")
cat("report", report, "\n")
RS

echo
echo "=== POLISHED RNA FIGURES ==="
find "$OUT" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/77A2_publication_polish_GSE142258_RNA_figures_status.txt" | head -80

echo
echo "=== DONE 77A2 ==="
date
