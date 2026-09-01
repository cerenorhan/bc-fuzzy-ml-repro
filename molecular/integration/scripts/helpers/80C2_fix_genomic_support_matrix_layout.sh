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

LOG="$LOGDIR/80C2_fix_genomic_support_matrix_layout_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 80C2: FIX GENOMIC SUPPORT MATRIX LAYOUT ==="
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

dir.create(figs, recursive = TRUE, showWarnings = FALSE)

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 4, hjust = 0),
      plot.subtitle = element_text(size = base_size + 1, hjust = 0, margin = margin(b = 10)),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.45, color = "grey20"),
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = base_size),
      legend.position = "top",
      legend.justification = "left",
      legend.box = "horizontal",
      legend.margin = margin(t = 2, r = 2, b = 8, l = 0),
      plot.margin = margin(t = 18, r = 28, b = 18, l = 18)
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
  geom_tile(fill = "grey94", color = "white", linewidth = 0.55, width = 0.96, height = 0.96) +
  geom_point(
    data = support %>% filter(present),
    aes(size = support_value, fill = layer),
    shape = 21,
    color = "grey15",
    stroke = 0.45,
    alpha = 0.96,
    show.legend = c(size = FALSE, fill = TRUE)
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
    name = "Genomic layer",
    guide = guide_legend(
      title.position = "left",
      title.hjust = 0,
      nrow = 1,
      override.aes = list(size = 5)
    )
  ) +
  scale_size_area(
    max_size = 22,
    limits = c(0, max(support$support_value, na.rm = TRUE)),
    guide = "none"
  ) +
  labs(
    title = "Final integrated genomic support matrix",
    subtitle = "Local tumor-only WES focus genes and PRJNA somatic WES support",
    x = NULL,
    y = NULL,
    caption = "Bubble size and in-bubble labels indicate support count."
  ) +
  coord_cartesian(clip = "off") +
  theme_pub(12) +
  theme(
    axis.text.x = element_text(size = 12, face = "bold", margin = margin(t = 8)),
    axis.text.y = element_text(size = 12),
    plot.caption = element_text(size = 9.5, color = "grey35", hjust = 0, margin = margin(t = 8)),
    panel.grid = element_blank()
  )

ggsave(
  file.path(figs, "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA_FIXED.pdf"),
  p_matrix,
  width = 8.2,
  height = 5.8,
  device = cairo_pdf
)

ggsave(
  file.path(figs, "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA_FIXED.png"),
  p_matrix,
  width = 8.2,
  height = 5.8,
  dpi = 600
)

report <- file.path(status, "80C2_fix_genomic_support_matrix_layout_status.txt")
sink(report)
cat("80C2 fixed genomic support matrix layout\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Fixes applied:\n")
cat("- Removed size legend to prevent clipping.\n")
cat("- Kept in-bubble support-count labels.\n")
cat("- Increased figure width and margins.\n")
cat("- Added caption explaining support-count encoding.\n")
sink()
RS

echo
echo "=== 80C2 FIXED FIGURES ==="
find "$FIGS" -maxdepth 1 -type f -name "Fig_Integrated_FINAL_01_genomic_support_matrix_NO_RNA_FIXED.*" -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/80C2_fix_genomic_support_matrix_layout_status.txt"

echo
echo "=== DONE 80C2 ==="
date
