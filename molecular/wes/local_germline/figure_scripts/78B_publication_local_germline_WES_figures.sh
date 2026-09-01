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

META="metadata/local_wes.tsv"

RAW_VCF="results/germline/joint/local_wes_germline_only.raw.vcf.gz"
PASS_VCF="results/germline/filtered/local_wes_germline_only.pass.vcf.gz"
BIALLELIC="results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.vcf.gz"
AUTOSOMAL="results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.autosomal.vcf.gz"

PCA="results/germline/summary/wes_pca_smallN_readfreq_pruned.tsv"
PCA_EIG="results/germline/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.smallN_readfreq_pruned.pca.eigenval"

CALLABLE_SUM="results/germline/callable/local_wes.germline_only.empirical_callable.summary.tsv"

OUT="$PAPER/publication_figures/local_germline_WES"
TABLES="$PAPER/05_local_germline_WES/01_tables"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$TABLES" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/78B_publication_local_germline_WES_figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 78B: PUBLICATION LOCAL GERMLINE WES FIGURES ==="
date

for f in "$META" "$RAW_VCF" "$PASS_VCF" "$BIALLELIC" "$AUTOSOMAL" "$PCA" "$PCA_EIG"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: Missing required input:"
    echo "$f"
    exit 1
  fi
done

VCF_COUNTS="$TABLES/local_wes_vcf_filtering_cascade.tsv"
GT_COUNTS="$TABLES/local_wes_autosomal_biallelic_snp_sample_genotype_counts.tsv"

echo
echo "=== BUILD VCF FILTERING CASCADE TABLE ==="
{
  echo -e "step\tfile\tn_samples\tn_variants"
  for item in \
    "Raw joint VCF|$RAW_VCF" \
    "PASS variants|$PASS_VCF" \
    "PASS biallelic SNPs|$BIALLELIC" \
    "PASS autosomal biallelic SNPs|$AUTOSOMAL"
  do
    step="${item%%|*}"
    vcf="${item#*|}"
    ns=$(micromamba run -n hadza-wes bcftools query -l "$vcf" | wc -l)
    nv=$(micromamba run -n hadza-wes bcftools view -H "$vcf" | wc -l)
    echo -e "${step}\t${vcf}\t${ns}\t${nv}"
  done
} > "$VCF_COUNTS"

cat "$VCF_COUNTS"

echo
echo "=== BUILD SAMPLE GENOTYPE COUNT TABLE ==="
echo "This may take a few minutes for 441k variants x 31 samples."

micromamba run -n hadza-wes bcftools query -f '[%SAMPLE\t%GT\n]' "$AUTOSOMAL" \
  | awk '
BEGIN {
  OFS="\t"
}
{
  sample=$1
  gt=$2

  if (!(sample in seen)) {
    seen[sample]=1
    order[++n]=sample
  }

  total[sample]++

  if (gt == "./." || gt == ".|." || gt == "." || gt ~ /^\./ || gt ~ /\.$/) {
    missing[sample]++
  } else if (gt == "0/0" || gt == "0|0") {
    refhom[sample]++
  } else if (gt == "0/1" || gt == "1/0" || gt == "0|1" || gt == "1|0") {
    het[sample]++
  } else if (gt == "1/1" || gt == "1|1") {
    homalt[sample]++
  } else {
    other[sample]++
  }
}
END {
  print "sample_id","n_genotypes","n_ref_hom","n_het","n_hom_alt","n_missing","n_other","n_nonref"
  for (i=1; i<=n; i++) {
    s=order[i]
    print s,total[s]+0,refhom[s]+0,het[s]+0,homalt[s]+0,missing[s]+0,other[s]+0,(het[s]+0)+(homalt[s]+0)+(other[s]+0)
  }
}' > "$GT_COUNTS"

head "$GT_COUNTS"

export META VCF_COUNTS GT_COUNTS PCA PCA_EIG CALLABLE_SUM OUT TABLES STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
})

meta_file <- Sys.getenv("META")
vcf_counts_file <- Sys.getenv("VCF_COUNTS")
gt_counts_file <- Sys.getenv("GT_COUNTS")
pca_file <- Sys.getenv("PCA")
pca_eig_file <- Sys.getenv("PCA_EIG")
callable_file <- Sys.getenv("CALLABLE_SUM")
outdir <- Sys.getenv("OUT")
tables <- Sys.getenv("TABLES")
statusdir <- Sys.getenv("STATUS")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(tables, recursive = TRUE, showWarnings = FALSE)

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

normalize_group <- function(x) {
  y <- tolower(as.character(x))
  case_when(
    str_detect(y, "hadza|hadzabe") ~ "Hadza",
    str_detect(y, "tanzanian_control|control") ~ "Tanzanian control",
    str_detect(y, "breast") ~ "Breast cancer",
    TRUE ~ as.character(x)
  )
}

normalize_sex <- function(x) {
  y <- tolower(as.character(x))
  case_when(
    str_detect(y, "^f") ~ "Female",
    str_detect(y, "^m") ~ "Male",
    TRUE ~ as.character(x)
  )
}

group_cols <- c(
  "Hadza" = "#009E73",
  "Tanzanian control" = "#0072B2",
  "Breast cancer" = "#D55E00"
)

sex_shapes <- c(
  "Female" = 16,
  "Male" = 17
)

geno_cols <- c(
  "Heterozygous" = "#56B4E9",
  "Homozygous alternate" = "#E69F00",
  "Other non-reference" = "#7E57C2"
)

# ---------------------------
# metadata
# ---------------------------
meta <- read_tsv(meta_file, show_col_types = FALSE) %>%
  mutate(
    sample_id = as.character(sample_id),
    group_label = normalize_group(group),
    sex_label = normalize_sex(sex),
    origin_label = as.character(participants_origin)
  )

# ---------------------------
# PCA
# ---------------------------
pca <- read_tsv(pca_file, show_col_types = FALSE) %>%
  mutate(sample_id = as.character(sample_id)) %>%
  left_join(
    meta %>% select(sample_id, group_label, sex_label, origin_label),
    by = "sample_id"
  ) %>%
  mutate(
    group_label = ifelse(is.na(group_label), normalize_group(group), group_label),
    sex_label = ifelse(is.na(sex_label), normalize_sex(sex_from_id), sex_label),
    group_label = factor(group_label, levels = c("Hadza", "Tanzanian control", "Breast cancer")),
    sex_label = factor(sex_label, levels = c("Female", "Male"))
  )

eig <- scan(pca_eig_file, quiet = TRUE)
pc_pct <- 100 * eig / sum(eig)

write_tsv(pca, file.path(tables, "local_wes_pca_with_metadata.tsv"))

p_pca12 <- ggplot(pca, aes(x = PC1, y = PC2, color = group_label, shape = sex_label)) +
  stat_ellipse(aes(group = group_label, color = group_label), linewidth = 0.55, linetype = "solid", alpha = 0.75, show.legend = FALSE) +
  geom_point(size = 3.4, alpha = 0.92) +
  scale_color_manual(values = group_cols, drop = FALSE) +
  scale_shape_manual(values = sex_shapes, drop = FALSE) +
  labs(
    title = "Local germline WES population structure",
    subtitle = "LD-pruned autosomal biallelic SNP PCA; breast cancer samples excluded from germline-only VCF",
    x = paste0("PC1 (", sprintf("%.1f", pc_pct[1]), "%)"),
    y = paste0("PC2 (", sprintf("%.1f", pc_pct[2]), "%)"),
    color = "Group",
    shape = "Sex"
  ) +
  theme_pub(11)

ggsave(file.path(outdir, "Fig_LocalWES_01_PCA_PC1_PC2_population_structure.pdf"), p_pca12, width = 7.4, height = 5.8)
ggsave(file.path(outdir, "Fig_LocalWES_01_PCA_PC1_PC2_population_structure.png"), p_pca12, width = 7.4, height = 5.8, dpi = 600)

p_pca13 <- ggplot(pca, aes(x = PC1, y = PC3, color = group_label, shape = sex_label)) +
  stat_ellipse(aes(group = group_label, color = group_label), linewidth = 0.55, linetype = "solid", alpha = 0.75, show.legend = FALSE) +
  geom_point(size = 3.4, alpha = 0.92) +
  scale_color_manual(values = group_cols, drop = FALSE) +
  scale_shape_manual(values = sex_shapes, drop = FALSE) +
  labs(
    title = "Local germline WES PCA",
    subtitle = "PC1 versus PC3",
    x = paste0("PC1 (", sprintf("%.1f", pc_pct[1]), "%)"),
    y = paste0("PC3 (", sprintf("%.1f", pc_pct[3]), "%)"),
    color = "Group",
    shape = "Sex"
  ) +
  theme_pub(11)

ggsave(file.path(outdir, "Fig_LocalWES_01B_PCA_PC1_PC3_population_structure.pdf"), p_pca13, width = 7.4, height = 5.8)
ggsave(file.path(outdir, "Fig_LocalWES_01B_PCA_PC1_PC3_population_structure.png"), p_pca13, width = 7.4, height = 5.8, dpi = 600)

# ---------------------------
# filtering cascade
# ---------------------------
vcf_counts <- read_tsv(vcf_counts_file, show_col_types = FALSE) %>%
  mutate(
    step = factor(step, levels = c(
      "Raw joint VCF",
      "PASS variants",
      "PASS biallelic SNPs",
      "PASS autosomal biallelic SNPs"
    )),
    label = format(n_variants, big.mark = ",", scientific = FALSE)
  )

p_filter <- ggplot(vcf_counts, aes(x = step, y = n_variants, fill = step)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.25) +
  geom_text(aes(label = label), vjust = -0.35, size = 3.8, fontface = "bold") +
  scale_fill_manual(values = c(
    "Raw joint VCF" = "#4E79A7",
    "PASS variants" = "#59A14F",
    "PASS biallelic SNPs" = "#F28E2B",
    "PASS autosomal biallelic SNPs" = "#B07AA1"
  )) +
  labs(
    title = "Local germline WES variant filtering cascade",
    subtitle = "Joint germline callset filtering for PCA and background-context analyses",
    x = NULL,
    y = "Number of variants"
  ) +
  theme_pub(11) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave(file.path(outdir, "Fig_LocalWES_02_variant_filtering_cascade.pdf"), p_filter, width = 7.6, height = 5.4)
ggsave(file.path(outdir, "Fig_LocalWES_02_variant_filtering_cascade.png"), p_filter, width = 7.6, height = 5.4, dpi = 600)

# ---------------------------
# sample burden
# ---------------------------
gt <- read_tsv(gt_counts_file, show_col_types = FALSE) %>%
  mutate(sample_id = as.character(sample_id)) %>%
  left_join(
    meta %>% select(sample_id, group_label, sex_label, origin_label),
    by = "sample_id"
  ) %>%
  mutate(
    group_label = factor(group_label, levels = c("Hadza", "Tanzanian control", "Breast cancer")),
    sex_label = factor(sex_label, levels = c("Female", "Male")),
    sample_order = paste(group_label, sample_id)
  )

write_tsv(gt, file.path(tables, "local_wes_sample_autosomal_biallelic_snp_burden_with_metadata.tsv"))

gt_long <- gt %>%
  select(sample_id, group_label, sex_label, n_het, n_hom_alt, n_other) %>%
  pivot_longer(
    cols = c(n_het, n_hom_alt, n_other),
    names_to = "genotype_class",
    values_to = "n"
  ) %>%
  mutate(
    genotype_class = recode(
      genotype_class,
      "n_het" = "Heterozygous",
      "n_hom_alt" = "Homozygous alternate",
      "n_other" = "Other non-reference"
    ),
    sample_id = fct_reorder(sample_id, n, .fun = sum)
  )

p_sample_burden <- ggplot(gt_long, aes(x = sample_id, y = n, fill = genotype_class)) +
  geom_col(width = 0.78, color = "white", linewidth = 0.10) +
  facet_grid(. ~ group_label, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = geno_cols) +
  labs(
    title = "Sample-level germline SNP non-reference burden",
    subtitle = "Autosomal PASS biallelic SNP genotypes; shown for 31 germline-only samples",
    x = "Sample",
    y = "Number of non-reference genotypes",
    fill = "Genotype class"
  ) +
  theme_pub(10) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, size = 7.6),
    legend.position = "top"
  )

ggsave(file.path(outdir, "Fig_LocalWES_03_sample_level_nonreference_snp_burden.pdf"), p_sample_burden, width = 10.4, height = 5.7)
ggsave(file.path(outdir, "Fig_LocalWES_03_sample_level_nonreference_snp_burden.png"), p_sample_burden, width = 10.4, height = 5.7, dpi = 600)

p_group_burden <- ggplot(gt, aes(x = group_label, y = n_nonref, fill = group_label)) +
  geom_boxplot(width = 0.48, alpha = 0.65, outlier.shape = NA, color = "grey25") +
  geom_jitter(aes(shape = sex_label), width = 0.12, size = 2.8, alpha = 0.88, color = "grey15") +
  scale_fill_manual(values = group_cols, drop = FALSE) +
  scale_shape_manual(values = sex_shapes, drop = FALSE) +
  labs(
    title = "Group-level germline SNP burden",
    subtitle = "Non-reference autosomal biallelic SNP genotypes per sample",
    x = NULL,
    y = "Non-reference genotype count",
    fill = "Group",
    shape = "Sex"
  ) +
  theme_pub(11)

ggsave(file.path(outdir, "Fig_LocalWES_04_group_level_nonreference_snp_burden.pdf"), p_group_burden, width = 6.8, height = 5.4)
ggsave(file.path(outdir, "Fig_LocalWES_04_group_level_nonreference_snp_burden.png"), p_group_burden, width = 6.8, height = 5.4, dpi = 600)

# ---------------------------
# analyzed sample composition
# ---------------------------
comp <- pca %>%
  count(group_label, sex_label, name = "n") %>%
  filter(!is.na(group_label))

write_tsv(comp, file.path(tables, "local_wes_analyzed_sample_composition.tsv"))

p_comp <- ggplot(comp, aes(x = group_label, y = n, fill = sex_label)) +
  geom_col(width = 0.62, color = "white", linewidth = 0.25) +
  geom_text(aes(label = n), position = position_stack(vjust = 0.5), color = "white", fontface = "bold", size = 4) +
  scale_fill_manual(values = c("Female" = "#CC79A7", "Male" = "#0072B2")) +
  labs(
    title = "Analyzed local germline WES cohort",
    subtitle = "Samples included in the germline-only VCF and PCA",
    x = NULL,
    y = "Number of samples",
    fill = "Sex"
  ) +
  theme_pub(11)

ggsave(file.path(outdir, "Fig_LocalWES_05_analyzed_sample_composition.pdf"), p_comp, width = 6.4, height = 5.2)
ggsave(file.path(outdir, "Fig_LocalWES_05_analyzed_sample_composition.png"), p_comp, width = 6.4, height = 5.2, dpi = 600)

# ---------------------------
# callable summary table for reporting
# ---------------------------
if (file.exists(callable_file)) {
  callable <- read_tsv(callable_file, show_col_types = FALSE)
  write_tsv(callable, file.path(tables, "local_wes_callable_summary_for_publication.tsv"))
}

report <- file.path(statusdir, "78B_publication_local_germline_WES_figures_status.txt")

sink(report)
cat("78B publication local germline WES figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

cat("PCA file:", pca_file, "\n")
cat("PCA eigenval file:", pca_eig_file, "\n\n")

cat("Analyzed sample composition:\n")
print(comp)

cat("\nVCF filtering cascade:\n")
print(vcf_counts)

cat("\nSample burden summary by group:\n")
print(
  gt %>%
    group_by(group_label) %>%
    summarise(
      n_samples = n(),
      median_nonref = median(n_nonref, na.rm = TRUE),
      mean_nonref = mean(n_nonref, na.rm = TRUE),
      min_nonref = min(n_nonref, na.rm = TRUE),
      max_nonref = max(n_nonref, na.rm = TRUE),
      .groups = "drop"
    )
)

cat("\nCallable summary:\n")
if (exists("callable")) print(callable)
sink()

cat("output_dir", outdir, "\n")
cat("report", report, "\n")
RS

echo
echo "=== LOCAL WES PUBLICATION FIGURES ==="
find "$OUT" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/78B_publication_local_germline_WES_figures_status.txt" | head -140

echo
echo "=== DONE 78B ==="
date
