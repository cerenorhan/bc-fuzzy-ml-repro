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

IN="$PAPER/06_integrated_prioritization/01_final_clean_focus/integrated_FINAL_clean_focus_gene_table.tsv"

OUT="$PAPER/06_integrated_prioritization/02_final_genomic_only"
FIGS="$PAPER/publication_figures/integrated_prioritization_final_genomic_only"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/80C_integrated_FINAL_genomic_only_figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 80C: INTEGRATED FINAL GENOMIC-ONLY FIGURES ==="
date

if [[ ! -s "$IN" ]]; then
  echo "ERROR: missing input:"
  echo "$IN"
  exit 1
fi

export IN OUT FIGS STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
})

infile <- Sys.getenv("IN")
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

df <- read_tsv(infile, show_col_types = FALSE) %>%
  mutate(
    gene = as.character(gene),
    local_tumor_support = as.logical(local_tumor_support),
    prjna_somatic_support = as.logical(prjna_somatic_support),
    genomic_support_n = as.integer(local_tumor_support) + as.integer(prjna_somatic_support),
    genomic_class = case_when(
      local_tumor_support & prjna_somatic_support ~ "Local + PRJNA",
      local_tumor_support & !prjna_somatic_support ~ "Local only",
      !local_tumor_support & prjna_somatic_support ~ "PRJNA only",
      TRUE ~ "No genomic support"
    )
  ) %>%
  arrange(desc(genomic_support_n), desc(integrated_score), gene)

write_tsv(df, file.path(outdir, "integrated_FINAL_genomic_only_focus_gene_table.tsv"))

gene_order <- df %>%
  arrange(desc(genomic_support_n), desc(integrated_score), gene) %>%
  pull(gene)

support <- df %>%
  transmute(
    gene,
    `Local tumor WES` = ifelse(local_tumor_support, local_focus_tumors, 0),
    `PRJNA somatic WES` = ifelse(prjna_somatic_support, prjna_functional_pairs, 0)
  ) %>%
  pivot_longer(-gene, names_to = "layer", values_to = "support_value") %>%
  mutate(
    gene = factor(gene, levels = rev(gene_order)),
    layer = factor(layer, levels = c("Local tumor WES", "PRJNA somatic WES")),
    present = support_value > 0
  )

layer_cols <- c(
  "Local tumor WES" = "#E69F00",
  "PRJNA somatic WES" = "#1F77B4"
)

p_matrix <- ggplot(support, aes(x = layer, y = gene)) +
  geom_tile(fill = "grey94", color = "white", linewidth = 0.45) +
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
    aes(label = round(support_value, 0)),
    color = "white",
    fontface = "bold",
    size = 4
  ) +
  scale_fill_manual(values = layer_cols) +
  scale_size_continuous(range = c(8, 15)) +
  labs(
    title = "Final integrated genomic support matrix",
    subtitle = "Local tumor-only WES focus genes and PRJNA somatic WES support",
    x = NULL,
    y = NULL,
    fill = "Genomic layer",
    size = "Support count"
  ) +
  theme_pub(11)

ggsave(file.path(figs, "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA.pdf"), p_matrix, width = 6.8, height = 5.0)
ggsave(file.path(figs, "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA.png"), p_matrix, width = 6.8, height = 5.0, dpi = 600)

class_cols <- c(
  "Local + PRJNA" = "#7E57C2",
  "Local only" = "#E69F00",
  "PRJNA only" = "#1F77B4"
)

score_df <- df %>%
  mutate(
    gene = factor(gene, levels = rev(gene_order)),
    label = case_when(
      local_tumor_support & prjna_somatic_support ~ paste0(local_focus_tumors, " local tumors + ", prjna_functional_pairs, " PRJNA pairs"),
      local_tumor_support ~ paste0(local_focus_tumors, " local tumor"),
      prjna_somatic_support ~ paste0(prjna_functional_pairs, " PRJNA pairs"),
      TRUE ~ ""
    )
  )

p_score <- ggplot(score_df, aes(x = integrated_score, y = gene)) +
  geom_segment(aes(x = 0, xend = integrated_score, yend = gene, color = genomic_class), linewidth = 1.15, alpha = 0.55) +
  geom_point(aes(size = genomic_support_n, fill = genomic_class), shape = 21, color = "grey15", stroke = 0.35) +
  geom_text(aes(label = label), nudge_x = 4, hjust = 0, fontface = "bold", size = 3.5) +
  scale_fill_manual(values = class_cols) +
  scale_color_manual(values = class_cols) +
  scale_size_continuous(range = c(4.5, 9.5), breaks = c(1, 2)) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.38))) +
  labs(
    title = "Final integrated genomic prioritization",
    subtitle = "KMT2C shows the strongest cross-cohort genomic support",
    x = "Integrated prioritization score",
    y = NULL,
    fill = "Genomic support class",
    color = "Genomic support class",
    size = "Genomic layers"
  ) +
  theme_pub(11) +
  guides(color = "none")

ggsave(file.path(figs, "Fig_Integrated_FINAL_02_genomic_priority_lollipop_NO_RNA.pdf"), p_score, width = 8.8, height = 5.4)
ggsave(file.path(figs, "Fig_Integrated_FINAL_02_genomic_priority_lollipop_NO_RNA.png"), p_score, width = 8.8, height = 5.4, dpi = 600)

report <- file.path(status, "80C_integrated_FINAL_genomic_only_figures_status.txt")
sink(report)
cat("80C integrated final genomic-only figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Note:\n")
cat("RNA gene-level DE was removed from the main integrated matrix because no final focus gene reached RNA gene-level FDR support. RNA-seq will be represented separately as pathway-level context.\n\n")
cat("Final genomic-only table:\n")
print(df)
sink()
RS

echo
echo "=== 80C GENOMIC-ONLY TABLE ==="
column -t -s $'\t' "$OUT/integrated_FINAL_genomic_only_focus_gene_table.tsv"

echo
echo "=== 80C FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/80C_integrated_FINAL_genomic_only_figures_status.txt" | head -120

echo
echo "=== DONE 80C ==="
date
