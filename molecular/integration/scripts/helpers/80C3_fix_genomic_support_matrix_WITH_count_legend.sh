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

IN="$PAPER/06_integrated_prioritization/02_final_genomic_only/integrated_FINAL_genomic_only_focus_gene_table.tsv"
FIGS="$PAPER/publication_figures/integrated_prioritization_final_genomic_only"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/80C3_fix_genomic_support_matrix_WITH_count_legend_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 80C3: FIX GENOMIC SUPPORT MATRIX WITH COUNT LEGEND ==="
date

if [[ ! -s "$IN" ]]; then
  echo "ERROR: missing input:"
  echo "$IN"
  exit 1
fi

export IN FIGS STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
})

infile <- Sys.getenv("IN")
figs <- Sys.getenv("FIGS")
status <- Sys.getenv("STATUS")

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 4, hjust = 0),
      plot.subtitle = element_text(size = base_size + 1, hjust = 0, margin = margin(b = 8)),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.45, color = "grey20"),

      legend.position = "bottom",
      legend.box = "vertical",
      legend.justification = "left",
      legend.title = element_text(face = "bold", size = base_size + 1),
      legend.text = element_text(size = base_size),
      legend.margin = margin(t = 4, r = 4, b = 4, l = 0),
      legend.box.margin = margin(t = 8, r = 0, b = 0, l = 0),

      plot.margin = margin(t = 18, r = 28, b = 18, l = 18),
      panel.grid = element_blank()
    )
}

df <- read_tsv(infile, show_col_types = FALSE) %>%
  mutate(
    gene = as.character(gene),
    local_tumor_support = as.logical(local_tumor_support),
    prjna_somatic_support = as.logical(prjna_somatic_support),
    genomic_support_n = as.integer(local_tumor_support) + as.integer(prjna_somatic_support)
  ) %>%
  arrange(desc(genomic_support_n), desc(integrated_score), gene)

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
  geom_tile(
    fill = "grey94",
    color = "white",
    linewidth = 0.55,
    width = 0.96,
    height = 0.96
  ) +
  geom_point(
    data = support %>% filter(present),
    aes(size = support_value, fill = layer),
    shape = 21,
    color = "grey15",
    stroke = 0.45,
    alpha = 0.96
  ) +
  geom_text(
    data = support %>% filter(present),
    aes(label = round(support_value, 0)),
    color = "white",
    fontface = "bold",
    size = 4.8,
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = layer_cols,
    name = "Genomic layer"
  ) +
  scale_size_area(
    name = "Support count",
    max_size = 22,
    breaks = c(1, 2, 4, 5, 8),
    limits = c(0, 8)
  ) +
  guides(
    fill = guide_legend(
      order = 1,
      title.position = "left",
      title.hjust = 0,
      nrow = 1,
      override.aes = list(size = 5)
    ),
    size = guide_legend(
      order = 2,
      title.position = "left",
      title.hjust = 0,
      nrow = 1,
      override.aes = list(
        fill = "white",
        color = "grey15",
        alpha = 1,
        stroke = 0.45
      )
    )
  ) +
  labs(
    title = "Final integrated genomic support matrix",
    subtitle = "Local tumor-only WES focus genes and PRJNA somatic WES support",
    x = NULL,
    y = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_pub(12) +
  theme(
    axis.text.x = element_text(size = 12, face = "bold", margin = margin(t = 8)),
    axis.text.y = element_text(size = 12)
  )

ggsave(
  file.path(figs, "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA_FIXED_WITH_COUNT_LEGEND.pdf"),
  p_matrix,
  width = 9.2,
  height = 6.4,
  device = cairo_pdf
)

ggsave(
  file.path(figs, "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA_FIXED_WITH_COUNT_LEGEND.png"),
  p_matrix,
  width = 9.2,
  height = 6.4,
  dpi = 600
)

report <- file.path(status, "80C3_fix_genomic_support_matrix_WITH_count_legend_status.txt")
sink(report)
cat("80C3 fixed genomic support matrix with count legend\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Fixes applied:\n")
cat("- Restored Support count size legend.\n")
cat("- Moved legends to bottom to prevent clipping.\n")
cat("- Kept in-bubble exact support-count labels.\n")
cat("- Increased canvas width and height.\n")
sink()
RS

echo
echo "=== 80C3 FIXED FIGURES ==="
find "$FIGS" -maxdepth 1 -type f -name "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA_FIXED_WITH_COUNT_LEGEND.*" -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/80C3_fix_genomic_support_matrix_WITH_count_legend_status.txt"

echo
echo "=== DONE 80C3 ==="
date
