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

IN="$PAPER/05_local_breast_cancer_tumor_WES/02_candidate_filtering/local_tumor_WES_tumor_only_PASS_variants.annotated_with_flags.tsv.gz"
OUT="$PAPER/05_local_breast_cancer_tumor_WES/03_strict_interpretable_candidates"
FIGS="$PAPER/publication_figures/local_tumor_WES"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/79D_local_tumor_WES_strict_interpretable_candidates_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 79D: LOCAL TUMOR WES STRICT INTERPRETABLE CANDIDATES ==="
date

if [[ ! -s "$IN" ]]; then
  echo "ERROR: input not found:"
  echo "$IN"
  exit 1
fi

export IN OUT FIGS STATUS

micromamba run -n hadza-wes python - <<'PY'
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

infile = os.environ["IN"]
outdir = Path(os.environ["OUT"])
status = Path(os.environ["STATUS"])
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(infile, sep="\t", compression="gzip", low_memory=False)

# Boolean düzeltmeleri
for c in [
    "is_functional_HIGH_MODERATE",
    "is_local_background_overlap",
    "is_cancer_seed_gene",
    "is_breast_focus_gene",
    "is_HIGH",
    "is_MODERATE"
]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])

df["tumor_AF"] = pd.to_numeric(df["tumor_AF"], errors="coerce")
df["DP"] = pd.to_numeric(df["DP"], errors="coerce")
df["TLOD"] = pd.to_numeric(df["TLOD"], errors="coerce")
df["local_tumor_recurrence_n"] = pd.to_numeric(df["local_tumor_recurrence_n"], errors="coerce")

artifact_pattern = re.compile(
    r"^(MUC\d|MUC\d+[A-Z]?|PRAMEF|KRTAP|NBPF|FRG|SPANX|CGB|PSG|CT45|"
    r"OR\d|OR[0-9A-Z]+|HLA-|LINC|LOC|AC[0-9]|AL[0-9]|AP[0-9]|"
    r"RNA|MIR|RNU|RPL|RPS|KIR|ZNF[0-9]+)$",
    re.IGNORECASE
)

def artifact_gene(g):
    g = str(g)
    if g in [".", "", "nan"]:
        return True
    return bool(artifact_pattern.search(g))

df["artifact_prone_gene_pattern"] = df["gene"].map(artifact_gene)

# Strict primary candidate:
# Tumor-only olduğundan çok yüksek AF ve tüm örneklerde tekrarlayan non-driver çağrıları baskılıyoruz.
strict = df[
    (df["is_functional_HIGH_MODERATE"]) &
    (~df["is_local_background_overlap"]) &
    (df["tumor_AF"].between(0.05, 0.80, inclusive="both")) &
    (df["DP"].fillna(0) >= 20) &
    (
        (df["local_tumor_recurrence_n"].fillna(0) <= 3) |
        (df["is_breast_focus_gene"]) |
        (df["is_cancer_seed_gene"])
    ) &
    (
        (~df["artifact_prone_gene_pattern"]) |
        (df["is_breast_focus_gene"]) |
        (df["is_cancer_seed_gene"])
    )
].copy()

# Daha da yorumlanabilir: breast/cancer-focus öncelikli.
focus = strict[
    strict["is_breast_focus_gene"] | strict["is_cancer_seed_gene"]
].copy()

# Tumor-only artefact review table: neden bastırıldı?
suppressed = df[
    (df["is_functional_HIGH_MODERATE"]) &
    (~df["is_local_background_overlap"]) &
    (
        (df["tumor_AF"] > 0.80) |
        (df["local_tumor_recurrence_n"].fillna(0) >= 4) |
        (df["artifact_prone_gene_pattern"])
    )
].copy()

def tier(row):
    if row.get("is_breast_focus_gene", False) and row.get("impact") == "HIGH":
        return "Tier_A_HIGH_breast_focus"
    if row.get("is_breast_focus_gene", False):
        return "Tier_B_breast_focus"
    if row.get("is_cancer_seed_gene", False) and row.get("impact") == "HIGH":
        return "Tier_C_HIGH_cancer_seed"
    if row.get("is_cancer_seed_gene", False):
        return "Tier_D_cancer_seed"
    if row.get("impact") == "HIGH":
        return "Tier_E_HIGH_other"
    return "Tier_F_MODERATE_other"

strict["strict_candidate_tier"] = strict.apply(tier, axis=1)
focus["strict_candidate_tier"] = focus.apply(tier, axis=1)

sort_cols = [
    "is_breast_focus_gene",
    "is_cancer_seed_gene",
    "is_HIGH",
    "local_tumor_recurrence_n",
    "TLOD",
    "tumor_AF"
]
strict = strict.sort_values(sort_cols, ascending=[False, False, False, False, False, False])
focus = focus.sort_values(sort_cols, ascending=[False, False, False, False, False, False])

strict_path = outdir / "local_tumor_WES_STRICT_interpretable_candidates.tsv"
focus_path = outdir / "local_tumor_WES_STRICT_breast_cancer_focus_candidates.tsv"
suppressed_path = outdir / "local_tumor_WES_suppressed_tumor_only_likely_artifact_or_germline_review.tsv"

strict.to_csv(strict_path, sep="\t", index=False)
focus.to_csv(focus_path, sep="\t", index=False)
suppressed.to_csv(suppressed_path, sep="\t", index=False)

# Gene summary
if len(strict):
    gene = (
        strict.groupby("gene")
        .agg(
            n_variants=("variant_key", "count"),
            n_unique_loci=("variant_key", "nunique"),
            n_samples=("sample_id", "nunique"),
            samples=("sample_id", lambda x: ";".join(sorted(set(x)))),
            stages=("stage", lambda x: ";".join(sorted(set(map(str, x))))),
            subtypes=("molecular_subtype", lambda x: ";".join(sorted(set(map(str, x))))),
            n_HIGH=("is_HIGH", "sum"),
            n_MODERATE=("is_MODERATE", "sum"),
            median_tumor_AF=("tumor_AF", "median"),
            median_DP=("DP", "median"),
            max_TLOD=("TLOD", "max"),
            is_breast_focus_gene=("is_breast_focus_gene", "max"),
            is_cancer_seed_gene=("is_cancer_seed_gene", "max")
        )
        .reset_index()
    )
    gene["strict_priority_score"] = (
        gene["n_samples"] * 6 +
        gene["n_HIGH"] * 5 +
        gene["n_MODERATE"] * 2 +
        gene["is_breast_focus_gene"].astype(int) * 15 +
        gene["is_cancer_seed_gene"].astype(int) * 8
    )
    gene = gene.sort_values(
        ["strict_priority_score", "is_breast_focus_gene", "is_cancer_seed_gene", "n_samples", "n_HIGH", "gene"],
        ascending=[False, False, False, False, False, True]
    )
else:
    gene = pd.DataFrame()

gene_path = outdir / "local_tumor_WES_STRICT_gene_summary.tsv"
gene.to_csv(gene_path, sep="\t", index=False)

# Sample summary
sample = (
    strict.groupby(["sample_id", "stage", "molecular_subtype", "er_status", "pr_status", "her_2_status"])
    .agg(
        n_strict_candidates=("variant_key", "count"),
        n_HIGH=("is_HIGH", "sum"),
        n_MODERATE=("is_MODERATE", "sum"),
        n_breast_focus=("is_breast_focus_gene", "sum"),
        n_cancer_seed=("is_cancer_seed_gene", "sum")
    )
    .reset_index()
    if len(strict) else pd.DataFrame()
)

sample_path = outdir / "local_tumor_WES_STRICT_sample_summary.tsv"
sample.to_csv(sample_path, sep="\t", index=False)

summary = pd.DataFrame([
    {"metric": "input_PASS_variants", "value": len(df)},
    {"metric": "input_functional_nonbackground", "value": int(((df["is_functional_HIGH_MODERATE"]) & (~df["is_local_background_overlap"])).sum())},
    {"metric": "strict_interpretable_candidates", "value": len(strict)},
    {"metric": "strict_breast_or_cancer_focus_candidates", "value": len(focus)},
    {"metric": "strict_genes", "value": gene["gene"].nunique() if len(gene) else 0},
    {"metric": "strict_breast_focus_genes", "value": int(gene["is_breast_focus_gene"].sum()) if len(gene) else 0},
    {"metric": "strict_cancer_seed_genes", "value": int(gene["is_cancer_seed_gene"].sum()) if len(gene) else 0},
    {"metric": "suppressed_likely_artifact_or_germline_review", "value": len(suppressed)}
])

summary_path = outdir / "local_tumor_WES_STRICT_summary_counts.tsv"
summary.to_csv(summary_path, sep="\t", index=False)

report = status / "79D_local_tumor_WES_strict_interpretable_candidates_status.txt"
with open(report, "w") as f:
    f.write("79D local tumor WES strict interpretable candidates\n")
    f.write("Tumor-only caveat: no matched normal; these are candidate tumor variants, not definitive somatic calls.\n\n")
    f.write("Strict filter: HIGH/MODERATE, no exact local non-cancer background overlap, AF 0.05-0.80, DP>=20, recurrence<=3 unless breast/cancer-focus gene, artifact-prone gene pattern suppressed unless focus gene.\n\n")
    f.write("Summary counts:\n")
    f.write(summary.to_csv(sep="\t", index=False))
    f.write("\nStrict gene summary:\n")
    f.write(gene.head(80).to_csv(sep="\t", index=False) if len(gene) else "none\n")
    f.write("\nBreast/cancer-focus candidates:\n")
    f.write(focus.head(120).to_csv(sep="\t", index=False) if len(focus) else "none\n")

print("strict_candidates", strict_path)
print("focus_candidates", focus_path)
print("suppressed_review", suppressed_path)
print("gene_summary", gene_path)
print("sample_summary", sample_path)
print("summary_counts", summary_path)
print("report", report)
PY

echo
echo "=== BUILD 79D FIGURES ==="

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(stringr)
})

outdir <- Sys.getenv("OUT")
figs <- Sys.getenv("FIGS")
status <- Sys.getenv("STATUS")

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

summary <- read_tsv(file.path(outdir, "local_tumor_WES_STRICT_summary_counts.tsv"), show_col_types = FALSE)
strict <- read_tsv(file.path(outdir, "local_tumor_WES_STRICT_interpretable_candidates.tsv"), show_col_types = FALSE)
gene <- read_tsv(file.path(outdir, "local_tumor_WES_STRICT_gene_summary.tsv"), show_col_types = FALSE)
sample <- read_tsv(file.path(outdir, "local_tumor_WES_STRICT_sample_summary.tsv"), show_col_types = FALSE)

if (nrow(sample) > 0) {
  sample_long <- sample %>%
    select(sample_id, stage, molecular_subtype, n_HIGH, n_MODERATE) %>%
    pivot_longer(c(n_HIGH, n_MODERATE), names_to = "impact", values_to = "n") %>%
    mutate(
      impact = recode(impact, "n_HIGH" = "HIGH", "n_MODERATE" = "MODERATE"),
      sample_id = factor(sample_id, levels = sample$sample_id)
    )

  p_sample <- ggplot(sample_long, aes(x = sample_id, y = n, fill = impact)) +
    geom_col(width = 0.72, color = "white", linewidth = 0.2) +
    scale_fill_manual(values = c("HIGH" = "#D62728", "MODERATE" = "#E69F00")) +
    labs(
      title = "Strict local tumor WES interpretable candidates",
      subtitle = "Tumor-only HIGH/MODERATE candidates after local-background and artifact-prone filtering",
      x = "Local tumor WES sample",
      y = "Number of strict candidates",
      fill = "Impact"
    ) +
    theme_pub(11)

  ggsave(file.path(figs, "Fig_LocalTumorWES_05_STRICT_candidate_burden.pdf"), p_sample, width = 7.4, height = 5.2)
  ggsave(file.path(figs, "Fig_LocalTumorWES_05_STRICT_candidate_burden.png"), p_sample, width = 7.4, height = 5.2, dpi = 600)
}

if (nrow(gene) > 0) {
  top_gene <- gene %>%
    arrange(desc(strict_priority_score), desc(is_breast_focus_gene), desc(is_cancer_seed_gene), desc(n_samples), desc(n_HIGH), gene) %>%
    slice_head(n = 25) %>%
    mutate(
      gene_label = fct_reorder(gene, strict_priority_score),
      class = case_when(
        is_breast_focus_gene ~ "Breast-focus gene",
        is_cancer_seed_gene ~ "Cancer-seed gene",
        TRUE ~ "Other strict gene"
      )
    )

  p_gene <- ggplot(top_gene, aes(x = strict_priority_score, y = gene_label)) +
    geom_segment(aes(x = 0, xend = strict_priority_score, yend = gene_label, color = class), linewidth = 1, alpha = 0.5) +
    geom_point(aes(size = n_samples, fill = class), shape = 21, color = "grey15", stroke = 0.35) +
    geom_text(aes(label = n_samples), nudge_x = 1.0, size = 3.0, fontface = "bold") +
    scale_fill_manual(values = c(
      "Breast-focus gene" = "#D62728",
      "Cancer-seed gene" = "#E69F00",
      "Other strict gene" = "#1F77B4"
    )) +
    scale_color_manual(values = c(
      "Breast-focus gene" = "#D62728",
      "Cancer-seed gene" = "#E69F00",
      "Other strict gene" = "#1F77B4"
    )) +
    scale_x_continuous(expand = expansion(mult = c(0.01, 0.15))) +
    scale_size_continuous(range = c(3.2, 8.4)) +
    labs(
      title = "Strict prioritized local tumor WES genes",
      subtitle = "Label indicates number of affected local tumors",
      x = "Strict prioritization score",
      y = NULL,
      fill = "Gene class",
      color = "Gene class",
      size = "Tumors"
    ) +
    theme_pub(10) +
    guides(color = "none")

  ggsave(file.path(figs, "Fig_LocalTumorWES_06_STRICT_prioritized_genes.pdf"), p_gene, width = 7.6, height = 7.0)
  ggsave(file.path(figs, "Fig_LocalTumorWES_06_STRICT_prioritized_genes.png"), p_gene, width = 7.6, height = 7.0, dpi = 600)
}

report <- file.path(status, "79D_local_tumor_WES_strict_interpretable_candidates_figures_status.txt")
sink(report)
cat("79D figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Summary:\n")
print(summary)
cat("\nSample summary:\n")
print(sample)
cat("\nTop strict genes:\n")
print(head(gene, 50))
sink()
RS

echo
echo "=== 79D STRICT SUMMARY COUNTS ==="
column -t -s $'\t' "$OUT/local_tumor_WES_STRICT_summary_counts.tsv"

echo
echo "=== 79D STRICT SAMPLE SUMMARY ==="
column -t -s $'\t' "$OUT/local_tumor_WES_STRICT_sample_summary.tsv"

echo
echo "=== 79D STRICT TOP GENES ==="
column -t -s $'\t' "$OUT/local_tumor_WES_STRICT_gene_summary.tsv" | head -80

echo
echo "=== 79D STRICT BREAST/CANCER FOCUS CANDIDATES ==="
column -t -s $'\t' "$OUT/local_tumor_WES_STRICT_breast_cancer_focus_candidates.tsv" | head -80

echo
echo "=== 79D FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "Fig_LocalTumorWES_05*" -o -name "Fig_LocalTumorWES_06*" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/79D_local_tumor_WES_strict_interpretable_candidates_status.txt" | head -160

echo
echo "=== DONE 79D ==="
date
