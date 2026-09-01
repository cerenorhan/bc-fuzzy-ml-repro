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
SOM="$PAPER/03_somatic_wes_PRJNA913947"

OUT="$PAPER/publication_figures/PRJNA913947_somatic"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/77B2_publication_upgrade_PRJNA913947_somatic_figures_FIXED_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 77B2: PUBLICATION UPGRADE - PRJNA913947 SOMATIC FIGURES FIXED ==="
date

find_first() {
  local pattern="$1"
  find "$SOM" -type f -name "$pattern" 2>/dev/null | sort | head -1
}

VAR_LONG="$(find_first 'PRJNA913947_21pairs_somatic_PASS_variants.long.tsv.gz')"
[[ -z "$VAR_LONG" ]] && VAR_LONG="$(find_first 'PRJNA913947_21pairs_somatic_PASS_variants.long.tsv')"

FUNC_SUM="$(find_first 'PRJNA913947_unbiased_functional_gene_summary_no_caution_subset.tsv')"
[[ -z "$FUNC_SUM" ]] && FUNC_SUM="$(find_first 'PRJNA913947_unbiased_functional_gene_summary.tsv')"

GENE_PAIR="$(find_first 'PRJNA913947_unbiased_functional_gene_by_pair_long.tsv')"
ORA_AXIS="$(find_first 'PRJNA913947_somatic_ORA_biology_axis_summary.tsv')"

echo
echo "=== INPUTS FOUND ==="
printf "VAR_LONG\t%s\n" "${VAR_LONG:-MISSING}"
printf "FUNC_SUM\t%s\n" "${FUNC_SUM:-MISSING}"
printf "GENE_PAIR\t%s\n" "${GENE_PAIR:-MISSING}"
printf "ORA_AXIS\t%s\n" "${ORA_AXIS:-MISSING}"

[[ -s "$VAR_LONG" ]] || { echo "ERROR: VAR_LONG missing"; exit 1; }
[[ -s "$FUNC_SUM" ]] || { echo "ERROR: FUNC_SUM missing"; exit 1; }

export VAR_LONG FUNC_SUM GENE_PAIR ORA_AXIS OUT STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
})

var_long_file <- Sys.getenv("VAR_LONG")
func_sum_file <- Sys.getenv("FUNC_SUM")
gene_pair_file <- Sys.getenv("GENE_PAIR")
ora_axis_file <- Sys.getenv("ORA_AXIS")
outdir <- Sys.getenv("OUT")
statusdir <- Sys.getenv("STATUS")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 3),
      plot.subtitle = element_text(size = base_size),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.35, color = "grey20"),
      legend.title = element_text(face = "bold"),
      legend.position = "right",
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(face = "bold")
    )
}

pick_col <- function(df, patterns, required = TRUE) {
  nms <- names(df)
  low <- tolower(nms)

  for (p in patterns) {
    idx <- which(low == tolower(p))
    if (length(idx) > 0) return(nms[idx[1]])
  }

  for (p in patterns) {
    idx <- which(str_detect(low, regex(p, ignore_case = TRUE)))
    if (length(idx) > 0) return(nms[idx[1]])
  }

  if (required) {
    stop("Column not found: ", paste(patterns, collapse = ", "),
         "\nAvailable columns: ", paste(nms, collapse = ", "))
  }
  NA_character_
}

nice_pair <- function(x) {
  x <- as.character(x)
  num <- str_extract(x, "\\d+$")
  ifelse(!is.na(num), paste0("P", sprintf("%02d", as.integer(num))), x)
}

nice_axis <- function(x) {
  recode(
    x,
    "chromatin_epigenetic" = "Chromatin / epigenetic",
    "cell_cycle_mitotic" = "Cell cycle / mitotic",
    "DNA_repair_damage" = "DNA repair / damage response",
    "cytoskeleton_transport" = "Cytoskeleton / transport",
    "adhesion_ECM" = "Adhesion / ECM",
    "immune_inflammation" = "Immune / inflammation",
    "hormone_luminal" = "Hormone / luminal",
    "metabolism" = "Metabolism",
    "TGF_WNT_NOTCH" = "TGF / WNT / Notch",
    "other" = "Other",
    .default = str_to_title(str_replace_all(x, "_", " / "))
  )
}

variant_cols <- c(
  "SNV" = "#1F77B4",
  "INDEL" = "#E69F00",
  "MNP" = "#7E57C2",
  "OTHER" = "grey65"
)

impact_cols <- c(
  "HIGH" = "#D62728",
  "MODERATE" = "#E69F00",
  "LOW" = "#009E73",
  "FUNCTIONAL" = "#1F77B4"
)

driver_cols <- c(
  "Canonical breast/pan-cancer gene" = "#D62728",
  "Other recurrent functional gene" = "#1F77B4"
)

canonical_genes <- c(
  "TP53", "PIK3CA", "KMT2C", "KMT2D", "CDH1", "ESR1", "ERBB2", "GATA3",
  "FOXA1", "PTEN", "RB1", "NF1", "ATM", "BRCA1", "BRCA2", "PALB2",
  "MAP3K1", "ARID1A", "SMAD4", "SMARCA4", "EP300", "CREBBP", "MLH1",
  "MSH2", "SETD2", "TBX3"
)

# ---------------------------
# 1. Pair-level burden from long table
# ---------------------------
vl <- read_tsv(var_long_file, show_col_types = FALSE)

pair_col <- pick_col(vl, c("^pair$", "pair_id", "pair", "sample", "tumor_normal_pair"), required = TRUE)
class_col <- pick_col(vl, c("variant_type", "variant_class", "^type$", "class"), required = FALSE)
ref_col <- pick_col(vl, c("^ref$", "reference_allele", "REF"), required = FALSE)
alt_col <- pick_col(vl, c("^alt$", "alternate_allele", "ALT"), required = FALSE)

if (!is.na(class_col)) {
  pair_class <- vl %>%
    transmute(
      pair = as.character(.data[[pair_col]]),
      variant_class_raw = toupper(as.character(.data[[class_col]]))
    )
} else if (!is.na(ref_col) && !is.na(alt_col)) {
  pair_class <- vl %>%
    transmute(
      pair = as.character(.data[[pair_col]]),
      ref = as.character(.data[[ref_col]]),
      alt = as.character(.data[[alt_col]]),
      variant_class_raw = case_when(
        nchar(ref) == 1 & nchar(alt) == 1 ~ "SNV",
        nchar(ref) == nchar(alt) & nchar(ref) > 1 ~ "MNP",
        nchar(ref) != nchar(alt) ~ "INDEL",
        TRUE ~ "OTHER"
      )
    )
} else {
  stop("Could not infer variant class. Need variant_type/class or REF/ALT columns.")
}

pair_class <- pair_class %>%
  mutate(
    variant_class = case_when(
      str_detect(variant_class_raw, "SNV|SNP") ~ "SNV",
      str_detect(variant_class_raw, "MNP") ~ "MNP",
      str_detect(variant_class_raw, "INDEL|DEL|INS") ~ "INDEL",
      TRUE ~ "OTHER"
    )
  ) %>%
  count(pair, variant_class, name = "n")

pair_order <- pair_class %>%
  group_by(pair) %>%
  summarise(total = sum(n), .groups = "drop") %>%
  mutate(
    pair_label = nice_pair(pair),
    pair_num = suppressWarnings(as.integer(str_extract(pair, "\\d+$")))
  ) %>%
  arrange(pair_num, pair)

pair_class <- pair_class %>%
  left_join(pair_order, by = "pair") %>%
  mutate(pair_label = factor(pair_label, levels = pair_order$pair_label))

write_tsv(pair_class, file.path(outdir, "PRJNA913947_publication_pair_variant_class_data.tsv"))

p_burden <- ggplot(pair_class, aes(x = pair_label, y = n, fill = variant_class)) +
  geom_col(width = 0.78, color = "white", linewidth = 0.15) +
  scale_fill_manual(values = variant_cols, drop = FALSE) +
  labs(
    title = "Somatic mutation burden across tumor-normal pairs",
    subtitle = "PASS Mutect2 variants from 21 candidate Kenyan breast tumor-normal pairs",
    x = "Tumor-normal pair",
    y = "Number of PASS somatic variants",
    fill = "Variant class"
  ) +
  theme_pub(11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

ggsave(file.path(outdir, "Fig_PRJNA_01_pair_somatic_variant_burden_stacked.pdf"), p_burden, width = 9.2, height = 5.6)
ggsave(file.path(outdir, "Fig_PRJNA_01_pair_somatic_variant_burden_stacked.png"), p_burden, width = 9.2, height = 5.6, dpi = 600)

overall_class <- pair_class %>%
  group_by(variant_class) %>%
  summarise(n = sum(n), .groups = "drop") %>%
  mutate(
    pct = 100 * n / sum(n),
    variant_class = factor(variant_class, levels = c("SNV", "INDEL", "MNP", "OTHER"))
  ) %>%
  arrange(variant_class)

write_tsv(overall_class, file.path(outdir, "PRJNA913947_publication_overall_variant_class_summary.tsv"))

p_class <- ggplot(overall_class, aes(x = variant_class, y = n, fill = variant_class)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(aes(label = paste0(n, "\n", sprintf("%.1f", pct), "%")), vjust = -0.25, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = variant_cols, drop = FALSE) +
  labs(
    title = "Overall somatic variant class composition",
    subtitle = "Aggregate across 21 PASS somatic VCFs",
    x = NULL,
    y = "Number of variants"
  ) +
  theme_pub(11) +
  theme(legend.position = "none")

ggsave(file.path(outdir, "Fig_PRJNA_02_overall_variant_class_composition.pdf"), p_class, width = 6.4, height = 5.2)
ggsave(file.path(outdir, "Fig_PRJNA_02_overall_variant_class_composition.png"), p_class, width = 6.4, height = 5.2, dpi = 600)

# ---------------------------
# 2. Recurrent functional genes
# ---------------------------
fg <- read_tsv(func_sum_file, show_col_types = FALSE)

gene_col <- pick_col(fg, c("^gene_symbol$", "^gene$", "symbol"), required = TRUE)
pair_gene_col <- pick_col(fg, c("n_pairs_functional_extended", "n_pairs_functional", "n_pairs_any", "n_pairs"), required = FALSE)
var_gene_col <- pick_col(fg, c("n_functional_extended", "n_functional_variants", "n_variants_functional", "n_variants_any", "n_variants"), required = FALSE)
score_col <- pick_col(fg, c("priority_score", "score"), required = FALSE)

fg2 <- fg %>%
  transmute(
    gene = as.character(.data[[gene_col]]),
    n_pairs = if (!is.na(pair_gene_col)) as.numeric(.data[[pair_gene_col]]) else NA_real_,
    n_variants = if (!is.na(var_gene_col)) as.numeric(.data[[var_gene_col]]) else NA_real_,
    score = if (!is.na(score_col)) as.numeric(.data[[score_col]]) else NA_real_
  ) %>%
  mutate(
    n_pairs = ifelse(is.na(n_pairs), 0, n_pairs),
    n_variants = ifelse(is.na(n_variants), 0, n_variants),
    driver_status = ifelse(gene %in% canonical_genes, "Canonical breast/pan-cancer gene", "Other recurrent functional gene")
  ) %>%
  arrange(desc(n_pairs), desc(n_variants), desc(score), gene)

top_genes <- fg2 %>%
  slice_head(n = 25) %>%
  mutate(gene_label = fct_reorder(gene, n_pairs))

write_tsv(top_genes, file.path(outdir, "PRJNA913947_publication_top25_recurrent_functional_genes.tsv"))

p_genes <- ggplot(top_genes, aes(x = n_pairs, y = gene_label)) +
  geom_segment(aes(x = 0, xend = n_pairs, yend = gene_label, color = driver_status), linewidth = 1.0, alpha = 0.45) +
  geom_point(aes(size = n_variants, fill = driver_status), shape = 21, color = "grey15", stroke = 0.35, alpha = 0.95) +
  geom_text(aes(label = n_pairs), nudge_x = 0.35, size = 3.0, fontface = "bold") +
  scale_color_manual(values = driver_cols) +
  scale_fill_manual(values = driver_cols) +
  scale_size_continuous(range = c(3.5, 9.0)) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.12))) +
  labs(
    title = "Top recurrent functional somatic genes",
    subtitle = "Point size indicates functional variant count; label indicates affected pairs",
    x = "Number of tumor-normal pairs",
    y = NULL,
    fill = "Gene class",
    size = "Functional variants"
  ) +
  theme_pub(10) +
  guides(color = "none")

ggsave(file.path(outdir, "Fig_PRJNA_03_top_recurrent_functional_somatic_genes.pdf"), p_genes, width = 7.6, height = 7.2)
ggsave(file.path(outdir, "Fig_PRJNA_03_top_recurrent_functional_somatic_genes.png"), p_genes, width = 7.6, height = 7.2, dpi = 600)

# ---------------------------
# 3. Oncoprint-like matrix
# ---------------------------
if (nzchar(gene_pair_file) && file.exists(gene_pair_file)) {
  gp <- read_tsv(gene_pair_file, show_col_types = FALSE)

  gp_gene_col <- pick_col(gp, c("^gene_symbol$", "^gene$", "symbol"), required = TRUE)
  gp_pair_col <- pick_col(gp, c("^pair$", "pair_id", "pair", "sample", "tumor_normal_pair"), required = TRUE)
  gp_impact_col <- pick_col(gp, c("impact_class", "impact_group", "impact", "annotation_impact", "effect_impact"), required = FALSE)

  gp2 <- gp %>%
    transmute(
      gene = as.character(.data[[gp_gene_col]]),
      pair = as.character(.data[[gp_pair_col]]),
      impact_raw = if (!is.na(gp_impact_col)) as.character(.data[[gp_impact_col]]) else "FUNCTIONAL"
    ) %>%
    filter(gene %in% top_genes$gene) %>%
    mutate(
      impact_raw_upper = toupper(impact_raw),
      impact_class = case_when(
        str_detect(impact_raw_upper, "HIGH") ~ "HIGH",
        str_detect(impact_raw_upper, "MODERATE") ~ "MODERATE",
        str_detect(impact_raw_upper, "LOW") ~ "LOW",
        TRUE ~ "FUNCTIONAL"
      ),
      impact_score = case_when(
        impact_class == "HIGH" ~ 4,
        impact_class == "MODERATE" ~ 3,
        impact_class == "LOW" ~ 2,
        TRUE ~ 1
      )
    ) %>%
    group_by(gene, pair) %>%
    arrange(desc(impact_score), .by_group = TRUE) %>%
    slice_head(n = 1) %>%
    ungroup()

  pair_levels <- pair_order$pair
  gene_levels <- rev(top_genes$gene)

  gp_plot <- expand_grid(gene = top_genes$gene, pair = pair_levels) %>%
    left_join(gp2, by = c("gene", "pair")) %>%
    mutate(
      pair_label = factor(nice_pair(pair), levels = nice_pair(pair_levels)),
      gene = factor(gene, levels = gene_levels)
    )

  write_tsv(gp_plot, file.path(outdir, "PRJNA913947_publication_functional_oncoprint_matrix_data.tsv"))

  p_onco <- ggplot(gp_plot, aes(x = pair_label, y = gene)) +
    geom_tile(fill = "grey94", color = "white", linewidth = 0.25) +
    geom_tile(
      data = gp_plot %>% filter(!is.na(impact_class)),
      aes(fill = impact_class),
      color = "white",
      linewidth = 0.25
    ) +
    scale_fill_manual(values = impact_cols, na.value = "grey94") +
    labs(
      title = "Functional somatic alteration matrix",
      subtitle = "Top recurrent functional genes across 21 tumor-normal pairs",
      x = "Tumor-normal pair",
      y = NULL,
      fill = "Predicted impact"
    ) +
    theme_pub(9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "top"
    )

  ggsave(file.path(outdir, "Fig_PRJNA_04_functional_somatic_oncoprint_like_matrix.pdf"), p_onco, width = 9.8, height = 7.4)
  ggsave(file.path(outdir, "Fig_PRJNA_04_functional_somatic_oncoprint_like_matrix.png"), p_onco, width = 9.8, height = 7.4, dpi = 600)
}

# ---------------------------
# 4. Somatic pathway axis support
# ---------------------------
if (nzchar(ora_axis_file) && file.exists(ora_axis_file)) {
  oa <- read_tsv(ora_axis_file, show_col_types = FALSE)

  axis_col <- pick_col(oa, c("^biology_axis$", "axis"), required = TRUE)
  query_col <- pick_col(oa, c("^query_set$", "query", "input_set", "analysis_set", "gene_set"), required = FALSE)
  tier_col <- pick_col(oa, c("reporting_tier", "tier", "significance"), required = FALSE)
  n_col <- pick_col(oa, c("^n_terms$", "n_.*terms", "count"), required = TRUE)
  padj_col <- pick_col(oa, c("best_padj", "min_padj", "padj", "qvalue", "fdr"), required = FALSE)

  oa2 <- oa %>%
    transmute(
      query_set = if (!is.na(query_col)) as.character(.data[[query_col]]) else "functional_extended",
      biology_axis = as.character(.data[[axis_col]]),
      reporting_tier = if (!is.na(tier_col)) as.character(.data[[tier_col]]) else ".",
      n_terms = as.numeric(.data[[n_col]]),
      best_padj = if (!is.na(padj_col)) as.numeric(.data[[padj_col]]) else NA_real_
    ) %>%
    filter(!is.na(n_terms), n_terms > 0) %>%
    filter(biology_axis != "other") %>%
    filter((!is.na(best_padj) & best_padj <= 0.25) | str_detect(reporting_tier, regex("FDR025|interpretable", ignore_case = TRUE))) %>%
    mutate(
      axis_label = nice_axis(biology_axis),
      query_label = recode(
        query_set,
        "functional_extended" = "Functional",
        "strict_HIGH_MODERATE" = "Strict HIGH/MODERATE",
        "functional_extended_no_caution_sensitivity" = "No-caution sensitivity",
        .default = query_set
      ),
      support_score = -log10(best_padj + 1e-300)
    )

  if (nrow(oa2) > 0) {
    write_tsv(oa2, file.path(outdir, "PRJNA913947_publication_somatic_pathway_axis_data.tsv"))

    p_axis <- ggplot(oa2, aes(x = query_label, y = fct_reorder(axis_label, n_terms), size = n_terms, fill = support_score)) +
      geom_point(shape = 21, color = "grey15", stroke = 0.35, alpha = 0.92) +
      geom_text(aes(label = n_terms), size = 3.0, fontface = "bold", color = "white") +
      scale_size_continuous(range = c(5, 18)) +
      scale_fill_gradient(low = "#F2F2F2", high = "#4B0082") +
      labs(
        title = "Somatic mutated-gene pathway axis support",
        subtitle = "Bubble size and label indicate the number of enriched interpretable terms",
        x = NULL,
        y = NULL,
        size = "Terms",
        fill = expression(-log[10](FDR))
      ) +
      theme_pub(10) +
      theme(
        axis.text.x = element_text(angle = 20, hjust = 1),
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.25)
      )

    ggsave(file.path(outdir, "Fig_PRJNA_05_somatic_pathway_axis_support.pdf"), p_axis, width = 8.0, height = 5.8)
    ggsave(file.path(outdir, "Fig_PRJNA_05_somatic_pathway_axis_support.png"), p_axis, width = 8.0, height = 5.8, dpi = 600)
  }
}

report <- file.path(statusdir, "77B2_publication_upgrade_PRJNA913947_somatic_figures_FIXED_status.txt")
sink(report)
cat("77B2 publication upgrade - PRJNA913947 somatic figures FIXED\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

cat("Input files:\n")
cat("VAR_LONG:", var_long_file, "\n")
cat("FUNC_SUM:", func_sum_file, "\n")
cat("GENE_PAIR:", gene_pair_file, "\n")
cat("ORA_AXIS:", ora_axis_file, "\n\n")

cat("Pair burden summary:\n")
print(pair_order)

cat("\nOverall variant class summary:\n")
print(overall_class)

cat("\nTop recurrent functional genes:\n")
print(top_genes %>% select(gene, n_pairs, n_variants, driver_status) %>% head(25))

if (exists("oa2")) {
  cat("\nSomatic pathway axis data:\n")
  print(oa2)
}
sink()

cat("output_dir", outdir, "\n")
cat("report", report, "\n")
RS

echo
echo "=== PUBLICATION PRJNA FIGURES ==="
find "$OUT" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/77B2_publication_upgrade_PRJNA913947_somatic_figures_FIXED_status.txt" | head -160

echo
echo "=== DONE 77B2 ==="
date
