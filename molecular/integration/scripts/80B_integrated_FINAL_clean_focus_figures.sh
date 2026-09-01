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

IN="$PAPER/06_integrated_prioritization/integrated_multilayer_gene_priority.tsv"
LOCAL="$PAPER/06_integrated_prioritization/integrated_local_focus_gene_support.tsv"

OUT="$PAPER/06_integrated_prioritization/01_final_clean_focus"
FIGS="$PAPER/publication_figures/integrated_prioritization_final"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/80B_integrated_FINAL_clean_focus_figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 80B: INTEGRATED FINAL CLEAN FOCUS FIGURES ==="
date

for f in "$IN" "$LOCAL"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing input:"
    echo "$f"
    exit 1
  fi
done

export IN LOCAL OUT FIGS STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(stringr)
})

infile <- Sys.getenv("IN")
localfile <- Sys.getenv("LOCAL")
outdir <- Sys.getenv("OUT")
figs <- Sys.getenv("FIGS")
status <- Sys.getenv("STATUS")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(figs, recursive = TRUE, showWarnings = FALSE)

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 3),
      plot.subtitle = element_text(size = base_size),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.35, color = "grey20"),
      legend.title = element_text(face = "bold"),
      legend.position = "top",
      legend.justification = "left"
    )
}

priority <- read_tsv(infile, show_col_types = FALSE)
local <- read_tsv(localfile, show_col_types = FALSE)

# Final interpretable set:
# 1) all local tumor focus genes
# 2) key PRJNA recurrent breast-cancer-relevant somatic genes seen in the integrated table
# Bu discovery listesi değil; final interpretive display setidir.
display_genes <- unique(c(
  local$gene,
  "TP53", "PIK3CA"
))

final <- priority %>%
  filter(gene %in% display_genes) %>%
  mutate(
    gene = factor(gene, levels = rev(c("KMT2C", "NF1", "EP300", "MYC", "TP53", "PIK3CA"))),
    local_tumor_support = as.logical(local_tumor_support),
    prjna_somatic_support = as.logical(prjna_somatic_support),
    rna_gene_level_support = !is.na(padj) & padj < 0.25,
    RNA_note = case_when(
      rna_gene_level_support ~ rna_DE_support,
      !is.na(log2FoldChange) ~ "gene-level not significant",
      TRUE ~ "not detected in RNA table"
    )
  ) %>%
  arrange(desc(integrated_score), gene)

write_tsv(final, file.path(outdir, "integrated_FINAL_clean_focus_gene_table.tsv"))

# -------------------------
# Fig 1: clean support matrix
# -------------------------
support <- final %>%
  transmute(
    gene,
    `Local tumor WES` = ifelse(local_tumor_support, local_focus_tumors, 0),
    `PRJNA somatic WES` = ifelse(prjna_somatic_support, prjna_functional_pairs, 0),
    `RNA-seq gene-level DE` = ifelse(rna_gene_level_support, pmin(-log10(padj), 10), 0)
  ) %>%
  pivot_longer(-gene, names_to = "layer", values_to = "support_value") %>%
  mutate(
    present = support_value > 0,
    layer = factor(layer, levels = c("Local tumor WES", "PRJNA somatic WES", "RNA-seq gene-level DE"))
  )

layer_cols <- c(
  "Local tumor WES" = "#E69F00",
  "PRJNA somatic WES" = "#1F77B4",
  "RNA-seq gene-level DE" = "#009E73"
)

p_matrix <- ggplot(support, aes(x = layer, y = gene)) +
  geom_point(data = support %>% filter(!present), size = 5.5, color = "grey88") +
  geom_point(
    data = support %>% filter(present),
    aes(size = support_value, fill = layer),
    shape = 21,
    color = "grey15",
    stroke = 0.35,
    alpha = 0.95
  ) +
  geom_text(
    data = support %>% filter(present),
    aes(label = round(support_value, 1)),
    size = 3.4,
    fontface = "bold",
    color = "white"
  ) +
  scale_fill_manual(values = layer_cols) +
  scale_size_continuous(range = c(6, 13)) +
  labs(
    title = "Final integrated candidate-gene support matrix",
    subtitle = "Local tumor WES focus genes with PRJNA somatic and RNA gene-level context",
    x = NULL,
    y = NULL,
    fill = "Layer",
    size = "Support value"
  ) +
  theme_pub(11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

ggsave(file.path(figs, "Fig_Integrated_FINAL_01_clean_support_matrix.pdf"), p_matrix, width = 8.2, height = 5.2)
ggsave(file.path(figs, "Fig_Integrated_FINAL_01_clean_support_matrix.png"), p_matrix, width = 8.2, height = 5.2, dpi = 600)

# -------------------------
# Fig 2: integrated score lollipop
# -------------------------
class_cols <- c(
  "Local + PRJNA" = "#7E57C2",
  "Local only" = "#E69F00",
  "PRJNA only" = "#1F77B4"
)

score_df <- final %>%
  mutate(
    support_class = case_when(
      local_tumor_support & prjna_somatic_support ~ "Local + PRJNA",
      local_tumor_support & !prjna_somatic_support ~ "Local only",
      !local_tumor_support & prjna_somatic_support ~ "PRJNA only",
      TRUE ~ "Other"
    ),
    label = case_when(
      local_tumor_support & prjna_somatic_support ~ paste0(local_focus_tumors, " local tumors + ", prjna_functional_pairs, " PRJNA pairs"),
      local_tumor_support ~ paste0(local_focus_tumors, " local tumors"),
      prjna_somatic_support ~ paste0(prjna_functional_pairs, " PRJNA pairs"),
      TRUE ~ ""
    )
  )

p_score <- ggplot(score_df, aes(x = integrated_score, y = gene)) +
  geom_segment(aes(x = 0, xend = integrated_score, yend = gene, color = support_class), linewidth = 1.15, alpha = 0.55) +
  geom_point(aes(size = multilayer_support_n, fill = support_class), shape = 21, color = "grey15", stroke = 0.35) +
  geom_text(aes(label = label), nudge_x = 4, hjust = 0, fontface = "bold", size = 3.4) +
  scale_fill_manual(values = class_cols) +
  scale_color_manual(values = class_cols) +
  scale_size_continuous(range = c(4.2, 8.8), breaks = c(1, 2)) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.38))) +
  labs(
    title = "Final integrated prioritization of interpretable genes",
    subtitle = "KMT2C shows the strongest cross-cohort genomic support",
    x = "Integrated prioritization score",
    y = NULL,
    fill = "Support class",
    color = "Support class",
    size = "Supported layers"
  ) +
  theme_pub(11) +
  guides(color = "none")

ggsave(file.path(figs, "Fig_Integrated_FINAL_02_clean_priority_lollipop.pdf"), p_score, width = 8.8, height = 5.4)
ggsave(file.path(figs, "Fig_Integrated_FINAL_02_clean_priority_lollipop.png"), p_score, width = 8.8, height = 5.4, dpi = 600)

# -------------------------
# Fig 3: RNA gene-level context for final genes
# -------------------------
rna_df <- final %>%
  mutate(
    gene = factor(gene, levels = rev(levels(gene))),
    direction = case_when(
      is.na(log2FoldChange) ~ "No RNA value",
      log2FoldChange > 0 ~ "Up in late",
      log2FoldChange < 0 ~ "Down in late",
      TRUE ~ "No change"
    )
  )

p_rna <- ggplot(rna_df, aes(x = log2FoldChange, y = gene, fill = direction)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey55") +
  geom_point(shape = 21, size = 5.2, color = "grey15", stroke = 0.35, alpha = 0.95) +
  geom_text(aes(label = RNA_note), nudge_x = 0.18, hjust = 0, size = 3.2) +
  scale_fill_manual(values = c(
    "Up in late" = "#D55E00",
    "Down in late" = "#0072B2",
    "No RNA value" = "grey70",
    "No change" = "grey70"
  )) +
  scale_x_continuous(expand = expansion(mult = c(0.10, 0.65))) +
  labs(
    title = "RNA-seq gene-level context for final integrated genes",
    subtitle = "No final focus gene reaches RNA gene-level FDR support; RNA pathway-level interpretation remains separate",
    x = "RNA-seq log2 fold-change, late vs early",
    y = NULL,
    fill = "Direction"
  ) +
  theme_pub(10)

ggsave(file.path(figs, "Fig_Integrated_FINAL_03_RNA_gene_level_context.pdf"), p_rna, width = 8.4, height = 5.2)
ggsave(file.path(figs, "Fig_Integrated_FINAL_03_RNA_gene_level_context.png"), p_rna, width = 8.4, height = 5.2, dpi = 600)

report <- file.path(status, "80B_integrated_FINAL_clean_focus_figures_status.txt")
sink(report)
cat("80B integrated final clean focus figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Final clean focus table:\n")
print(final)
sink()
RS

echo
echo "=== 80B FINAL CLEAN FOCUS TABLE ==="
column -t -s $'\t' "$OUT/integrated_FINAL_clean_focus_gene_table.tsv"

echo
echo "=== 80B FINAL FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/80B_integrated_FINAL_clean_focus_figures_status.txt" | head -120

echo
echo "=== DONE 80B ==="
date
