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

FOCUS="$PAPER/05_local_breast_cancer_tumor_WES/03_strict_interpretable_candidates/local_tumor_WES_STRICT_breast_cancer_focus_candidates.tsv"
GENE_SUM="$PAPER/05_local_breast_cancer_tumor_WES/03_strict_interpretable_candidates/local_tumor_WES_STRICT_gene_summary.tsv"
SAMPLE_SHEET="$PAPER/05_local_breast_cancer_tumor_WES/local_tumor_wes_sample_sheet.tsv"

OUT="$PAPER/05_local_breast_cancer_tumor_WES/04_final_focus_summary"
FIGS="$PAPER/publication_figures/local_tumor_WES_final_focus"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/79E_local_tumor_WES_FINAL_focus_candidate_figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 79E: LOCAL TUMOR WES FINAL FOCUS CANDIDATE FIGURES ==="
date

for f in "$FOCUS" "$GENE_SUM" "$SAMPLE_SHEET"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing input:"
    echo "$f"
    exit 1
  fi
done

export FOCUS GENE_SUM SAMPLE_SHEET OUT FIGS STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(stringr)
})

focus_file <- Sys.getenv("FOCUS")
gene_file <- Sys.getenv("GENE_SUM")
sample_file <- Sys.getenv("SAMPLE_SHEET")
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
      legend.justification = "left",
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(face = "bold")
    )
}

to_bool <- function(x) {
  tolower(as.character(x)) %in% c("true", "1", "yes")
}

impact_cols <- c(
  "HIGH" = "#D62728",
  "MODERATE" = "#E69F00",
  "HIGH+MODERATE" = "#7E57C2"
)

gene_class_cols <- c(
  "Breast/cancer-focus" = "#D62728",
  "Cancer-seed only" = "#7E57C2"
)

focus <- read_tsv(focus_file, show_col_types = FALSE) %>%
  mutate(
    is_HIGH = to_bool(is_HIGH),
    is_MODERATE = to_bool(is_MODERATE),
    is_breast_focus_gene = to_bool(is_breast_focus_gene),
    is_cancer_seed_gene = to_bool(is_cancer_seed_gene),
    tumor_AF = as.numeric(tumor_AF),
    DP = as.numeric(DP),
    TLOD = as.numeric(TLOD),
    gene_class = case_when(
      is_breast_focus_gene & is_cancer_seed_gene ~ "Breast/cancer-focus",
      is_cancer_seed_gene ~ "Cancer-seed only",
      TRUE ~ "Other"
    )
  )

sample_sheet <- read_tsv(sample_file, show_col_types = FALSE) %>%
  mutate(
    sample_id = as.character(sample_id),
    sample_label = paste0(sample_id, "\n", molecular_subtype),
    stage_order = factor(stage, levels = c("II stage", "III stage", "IV stage"))
  ) %>%
  arrange(stage_order, sample_id)

sample_order <- sample_sheet$sample_id
sample_labels <- setNames(sample_sheet$sample_label, sample_sheet$sample_id)

gene_sum <- read_tsv(gene_file, show_col_types = FALSE) %>%
  mutate(
    is_breast_focus_gene = to_bool(is_breast_focus_gene),
    is_cancer_seed_gene = to_bool(is_cancer_seed_gene)
  ) %>%
  filter(is_breast_focus_gene | is_cancer_seed_gene) %>%
  arrange(desc(strict_priority_score), desc(n_samples), desc(n_HIGH), gene)

gene_order <- gene_sum$gene

# -------------------------
# manuscript tables
# -------------------------
focus_for_paper <- focus %>%
  transmute(
    sample_id,
    stage,
    molecular_subtype,
    er_status,
    pr_status,
    her_2_status,
    gene,
    CHROM,
    POS,
    REF,
    ALT,
    effect,
    impact,
    hgvsc,
    hgvsp,
    DP,
    alt_depth,
    tumor_AF,
    TLOD,
    local_tumor_recurrence_n,
    local_tumor_recurrence_samples,
    strict_candidate_tier
  ) %>%
  arrange(factor(gene, levels = gene_order), sample_id, desc(impact), POS)

gene_for_paper <- gene_sum %>%
  transmute(
    gene,
    n_variants,
    n_unique_loci,
    n_samples,
    samples,
    stages,
    subtypes,
    n_HIGH,
    n_MODERATE,
    median_tumor_AF,
    median_DP,
    max_TLOD,
    is_breast_focus_gene,
    is_cancer_seed_gene,
    strict_priority_score
  )

write_tsv(focus_for_paper, file.path(outdir, "local_tumor_WES_FINAL_focus_candidate_table_for_manuscript.tsv"))
write_tsv(gene_for_paper, file.path(outdir, "local_tumor_WES_FINAL_focus_gene_summary_for_manuscript.tsv"))

# -------------------------
# Fig 1: focus candidate matrix
# -------------------------
matrix <- focus %>%
  group_by(gene, sample_id) %>%
  summarise(
    n_variants = n(),
    best_impact = case_when(
      any(impact == "HIGH") & any(impact == "MODERATE") ~ "HIGH+MODERATE",
      any(impact == "HIGH") ~ "HIGH",
      any(impact == "MODERATE") ~ "MODERATE",
      TRUE ~ NA_character_
    ),
    label = paste0(n_variants),
    .groups = "drop"
  )

grid <- expand_grid(
  gene = gene_order,
  sample_id = sample_order
) %>%
  left_join(matrix, by = c("gene", "sample_id")) %>%
  mutate(
    gene = factor(gene, levels = rev(gene_order)),
    sample_id = factor(sample_id, levels = sample_order),
    sample_label = sample_labels[as.character(sample_id)]
  )

p_matrix <- ggplot(grid, aes(x = sample_id, y = gene)) +
  geom_tile(fill = "grey94", color = "white", linewidth = 0.35) +
  geom_tile(
    data = grid %>% filter(!is.na(best_impact)),
    aes(fill = best_impact),
    color = "white",
    linewidth = 0.35
  ) +
  geom_text(
    data = grid %>% filter(!is.na(best_impact)),
    aes(label = label),
    color = "white",
    fontface = "bold",
    size = 4
  ) +
  scale_fill_manual(values = impact_cols) +
  scale_x_discrete(labels = sample_labels) +
  labs(
    title = "Final local tumor WES focus-candidate matrix",
    subtitle = "Strict tumor-only breast/cancer-focus candidates; tile label indicates candidate count per gene-sample pair",
    x = "Local tumor WES sample",
    y = NULL,
    fill = "Best impact"
  ) +
  theme_pub(11) +
  theme(axis.text.x = element_text(size = 8.5))

ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_01_focus_candidate_matrix.pdf"), p_matrix, width = 8.2, height = 4.8)
ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_01_focus_candidate_matrix.png"), p_matrix, width = 8.2, height = 4.8, dpi = 600)

# -------------------------
# Fig 2: focus gene lollipop
# -------------------------
gene_plot <- gene_sum %>%
  mutate(
    gene = factor(gene, levels = rev(gene_order)),
    gene_class = ifelse(is_breast_focus_gene, "Breast/cancer-focus", "Cancer-seed only"),
    label = paste0(n_samples, " tumors / ", n_variants, " calls")
  )

p_gene <- ggplot(gene_plot, aes(x = strict_priority_score, y = gene)) +
  geom_segment(aes(x = 0, xend = strict_priority_score, yend = gene, color = gene_class), linewidth = 1.1, alpha = 0.55) +
  geom_point(aes(size = n_variants, fill = gene_class), shape = 21, color = "grey15", stroke = 0.35) +
  geom_text(aes(label = label), nudge_x = 3.2, size = 3.3, fontface = "bold", hjust = 0) +
  scale_fill_manual(values = gene_class_cols) +
  scale_color_manual(values = gene_class_cols) +
  scale_size_continuous(range = c(4.2, 9.5)) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.32))) +
  labs(
    title = "Final prioritized local tumor WES focus genes",
    subtitle = "Strict tumor-only candidates after local non-cancer background flagging",
    x = "Strict prioritization score",
    y = NULL,
    fill = "Gene class",
    color = "Gene class",
    size = "Candidate calls"
  ) +
  theme_pub(11) +
  guides(color = "none")

ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_02_focus_gene_lollipop.pdf"), p_gene, width = 8.2, height = 4.6)
ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_02_focus_gene_lollipop.png"), p_gene, width = 8.2, height = 4.6, dpi = 600)

# -------------------------
# Fig 3: sample focus burden
# -------------------------
sample_burden <- focus %>%
  mutate(
    sample_id = factor(sample_id, levels = sample_order),
    impact = factor(impact, levels = c("HIGH", "MODERATE"))
  ) %>%
  count(sample_id, impact, name = "n") %>%
  complete(sample_id = factor(sample_order, levels = sample_order), impact, fill = list(n = 0)) %>%
  mutate(sample_label = sample_labels[as.character(sample_id)])

p_sample <- ggplot(sample_burden, aes(x = sample_id, y = n, fill = impact)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.2) +
  geom_text(
    data = sample_burden %>% group_by(sample_id) %>% summarise(total = sum(n), .groups = "drop") %>% filter(total > 0),
    aes(x = sample_id, y = total, label = total),
    inherit.aes = FALSE,
    vjust = -0.35,
    fontface = "bold",
    size = 3.8
  ) +
  scale_fill_manual(values = impact_cols[c("HIGH", "MODERATE")]) +
  scale_x_discrete(labels = sample_labels) +
  labs(
    title = "Per-sample focus-candidate burden",
    subtitle = "Strict breast/cancer-focus tumor-only candidates",
    x = "Local tumor WES sample",
    y = "Number of focus candidates",
    fill = "Impact"
  ) +
  theme_pub(11) +
  theme(axis.text.x = element_text(size = 8.5))

ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_03_sample_focus_candidate_burden.pdf"), p_sample, width = 8.2, height = 4.8)
ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_03_sample_focus_candidate_burden.png"), p_sample, width = 8.2, height = 4.8, dpi = 600)

# -------------------------
# Fig 4: candidate allele fraction
# -------------------------
af_plot <- focus %>%
  mutate(
    gene = factor(gene, levels = rev(gene_order)),
    sample_id = factor(sample_id, levels = sample_order),
    impact = factor(impact, levels = c("HIGH", "MODERATE"))
  )

p_af <- ggplot(af_plot, aes(x = tumor_AF, y = gene, fill = impact)) +
  geom_point(shape = 21, size = 4.3, color = "grey15", stroke = 0.35, alpha = 0.92) +
  facet_wrap(~ sample_id, nrow = 1) +
  scale_fill_manual(values = impact_cols[c("HIGH", "MODERATE")]) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 0.70)) +
  labs(
    title = "Tumor allele fractions of final focus candidates",
    subtitle = "Tumor-only calls; matched-normal confirmation is unavailable",
    x = "Tumor allele fraction",
    y = NULL,
    fill = "Impact"
  ) +
  theme_pub(9) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_04_focus_candidate_AF.pdf"), p_af, width = 10.8, height = 4.8)
ggsave(file.path(figs, "Fig_LocalTumorWES_FINAL_04_focus_candidate_AF.png"), p_af, width = 10.8, height = 4.8, dpi = 600)

report <- file.path(status, "79E_local_tumor_WES_FINAL_focus_candidate_figures_status.txt")
sink(report)
cat("79E local tumor WES final focus candidate figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Tumor-only caveat:\n")
cat("Matched normals were unavailable; these are final focus tumor-only candidate variants, not definitive somatic calls.\n\n")
cat("Gene summary:\n")
print(gene_for_paper)
cat("\nFocus candidate table:\n")
print(focus_for_paper)
sink()
RS

echo
echo "=== 79E FINAL FOCUS GENE SUMMARY ==="
column -t -s $'\t' "$OUT/local_tumor_WES_FINAL_focus_gene_summary_for_manuscript.tsv"

echo
echo "=== 79E FINAL FOCUS CANDIDATE TABLE ==="
column -t -s $'\t' "$OUT/local_tumor_WES_FINAL_focus_candidate_table_for_manuscript.tsv"

echo
echo "=== 79E FINAL FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/79E_local_tumor_WES_FINAL_focus_candidate_figures_status.txt" | head -160

echo
echo "=== DONE 79E ==="
date
