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

LOCAL_GENE="$PAPER/05_local_breast_cancer_tumor_WES/04_final_focus_summary/local_tumor_WES_FINAL_focus_gene_summary_for_manuscript.tsv"
LOCAL_CAND="$PAPER/05_local_breast_cancer_tumor_WES/04_final_focus_summary/local_tumor_WES_FINAL_focus_candidate_table_for_manuscript.tsv"

PRJNA_GENE=$(find "$PAPER/03_somatic_wes_PRJNA913947" -type f \( \
  -name "PRJNA913947_unbiased_functional_gene_summary_no_caution_subset.tsv" -o \
  -name "PRJNA913947_unbiased_functional_gene_summary.tsv" -o \
  -name "*functional_gene_summary*.tsv" \
  \) | sort | head -1)

RNA_DE=$(find results paper_results -type f -name "GSE142258_DESeq2_late_vs_early_results.annotated.tsv" | sort | head -1)

OUT="$PAPER/06_integrated_prioritization"
FIGS="$PAPER/publication_figures/integrated_prioritization"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/80A_integrated_multilayer_candidate_prioritization_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 80A: INTEGRATED MULTILAYER CANDIDATE PRIORITIZATION ==="
date

echo
echo "Inputs:"
echo "LOCAL_GENE=$LOCAL_GENE"
echo "LOCAL_CAND=$LOCAL_CAND"
echo "PRJNA_GENE=$PRJNA_GENE"
echo "RNA_DE=$RNA_DE"

for f in "$LOCAL_GENE" "$LOCAL_CAND"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing required local tumor WES input:"
    echo "$f"
    exit 1
  fi
done

if [[ -z "${PRJNA_GENE:-}" || ! -s "$PRJNA_GENE" ]]; then
  echo "ERROR: PRJNA functional gene summary not found."
  exit 1
fi

if [[ -z "${RNA_DE:-}" || ! -s "$RNA_DE" ]]; then
  echo "ERROR: RNA DESeq2 annotated result not found."
  exit 1
fi

export LOCAL_GENE LOCAL_CAND PRJNA_GENE RNA_DE OUT FIGS STATUS

micromamba run -n rnaseq python - <<'PY'
import os
import math
import pandas as pd
import numpy as np
from pathlib import Path

local_gene_file = os.environ["LOCAL_GENE"]
local_cand_file = os.environ["LOCAL_CAND"]
prjna_gene_file = os.environ["PRJNA_GENE"]
rna_file = os.environ["RNA_DE"]
outdir = Path(os.environ["OUT"])
status = Path(os.environ["STATUS"])
outdir.mkdir(parents=True, exist_ok=True)

def clean_gene(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def to_bool(x):
    return str(x).lower() in ["true", "1", "yes"]

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def as_num(s):
    return pd.to_numeric(s, errors="coerce")

# -------------------------
# Local tumor WES focus genes
# -------------------------
local = pd.read_csv(local_gene_file, sep="\t", dtype=str)
local["gene_clean"] = local["gene"].map(clean_gene)

for c in ["n_variants", "n_unique_loci", "n_samples", "n_HIGH", "n_MODERATE", "median_tumor_AF", "median_DP", "max_TLOD", "strict_priority_score"]:
    if c in local.columns:
        local[c] = as_num(local[c])

for c in ["is_breast_focus_gene", "is_cancer_seed_gene"]:
    if c in local.columns:
        local[c] = local[c].map(to_bool)

local_keep = local[[
    "gene_clean", "gene", "n_variants", "n_unique_loci", "n_samples", "samples",
    "stages", "subtypes", "n_HIGH", "n_MODERATE", "median_tumor_AF",
    "median_DP", "max_TLOD", "is_breast_focus_gene", "is_cancer_seed_gene",
    "strict_priority_score"
]].copy()

local_keep = local_keep.rename(columns={
    "gene": "gene_symbol",
    "n_variants": "local_focus_calls",
    "n_unique_loci": "local_focus_unique_loci",
    "n_samples": "local_focus_tumors",
    "samples": "local_focus_samples",
    "stages": "local_focus_stages",
    "subtypes": "local_focus_subtypes",
    "n_HIGH": "local_focus_HIGH",
    "n_MODERATE": "local_focus_MODERATE",
    "median_tumor_AF": "local_focus_median_tumor_AF",
    "median_DP": "local_focus_median_DP",
    "max_TLOD": "local_focus_max_TLOD",
    "strict_priority_score": "local_focus_score"
})

# -------------------------
# PRJNA somatic WES gene summary
# -------------------------
prjna = pd.read_csv(prjna_gene_file, sep="\t", dtype=str)
gene_col = first_existing(prjna, ["gene", "Gene", "gene_symbol", "symbol"])
if gene_col is None:
    raise SystemExit("No gene column found in PRJNA gene summary.")

prjna["gene_clean"] = prjna[gene_col].map(clean_gene)

pairs_col = first_existing(prjna, [
    "n_pairs_functional_extended", "n_pairs_functional", "n_pairs_any",
    "n_pairs", "n_samples", "n_pair"
])
var_col = first_existing(prjna, [
    "n_functional_extended", "n_functional_variants", "n_variants_functional",
    "n_variants", "variant_count"
])
high_col = first_existing(prjna, ["n_HIGH", "n_high", "HIGH", "n_high_impact"])
mod_col = first_existing(prjna, ["n_MODERATE", "n_moderate", "MODERATE", "n_moderate_impact"])
score_col = first_existing(prjna, ["priority_score", "score", "candidate_score"])
seed_col = first_existing(prjna, ["is_cancer_seed_gene", "cancer_seed", "is_driver_seed", "driver_seed"])

prjna_keep = pd.DataFrame({"gene_clean": prjna["gene_clean"]})
prjna_keep["prjna_gene_symbol"] = prjna[gene_col]

prjna_keep["prjna_functional_pairs"] = as_num(prjna[pairs_col]) if pairs_col else np.nan
prjna_keep["prjna_functional_variants"] = as_num(prjna[var_col]) if var_col else np.nan
prjna_keep["prjna_HIGH"] = as_num(prjna[high_col]) if high_col else 0
prjna_keep["prjna_MODERATE"] = as_num(prjna[mod_col]) if mod_col else 0
prjna_keep["prjna_score_raw"] = as_num(prjna[score_col]) if score_col else np.nan
prjna_keep["prjna_cancer_seed"] = prjna[seed_col].map(to_bool) if seed_col else False

prjna_keep = (
    prjna_keep
    .sort_values(["gene_clean", "prjna_functional_pairs", "prjna_functional_variants"], ascending=[True, False, False])
    .drop_duplicates("gene_clean")
)

# -------------------------
# RNA DESeq2 annotated
# -------------------------
rna = pd.read_csv(rna_file, sep="\t", dtype=str)
rna_gene_col = first_existing(rna, ["gene_name", "gene", "symbol", "external_gene_name", "Gene"])
if rna_gene_col is None:
    raise SystemExit("No gene symbol column found in RNA DESeq2 annotated table.")

rna["gene_clean"] = rna[rna_gene_col].map(clean_gene)

for c in ["log2FoldChange", "stat", "pvalue", "padj", "baseMean"]:
    if c in rna.columns:
        rna[c] = as_num(rna[c])

rna_keep = rna[["gene_clean", rna_gene_col, "log2FoldChange", "pvalue", "padj"]].copy()
rna_keep = rna_keep.rename(columns={rna_gene_col: "rna_gene_symbol"})

rna_keep["rna_direction_late_vs_early"] = np.where(
    rna_keep["log2FoldChange"] > 0, "up_in_late",
    np.where(rna_keep["log2FoldChange"] < 0, "down_in_late", "no_change")
)

rna_keep["rna_DE_support"] = "not_significant"
rna_keep.loc[rna_keep["padj"] < 0.25, "rna_DE_support"] = "FDR<0.25"
rna_keep.loc[rna_keep["padj"] < 0.10, "rna_DE_support"] = "FDR<0.10"
rna_keep.loc[rna_keep["padj"] < 0.05, "rna_DE_support"] = "FDR<0.05"

rna_keep["rna_abs_log2FC"] = rna_keep["log2FoldChange"].abs()

rna_keep = (
    rna_keep
    .sort_values(["gene_clean", "padj", "rna_abs_log2FC"], ascending=[True, True, False])
    .drop_duplicates("gene_clean")
)

# -------------------------
# Integrated table
# -------------------------
all_genes = set(local_keep["gene_clean"]) | set(prjna_keep["gene_clean"])

# Add strongly significant RNA genes so RNA layer is visible in the global table.
rna_sig = set(rna_keep.loc[(rna_keep["padj"] < 0.05) | (rna_keep["rna_abs_log2FC"] >= 2), "gene_clean"])
all_genes |= rna_sig

base = pd.DataFrame({"gene_clean": sorted(g for g in all_genes if g)})

merged = (
    base
    .merge(local_keep, on="gene_clean", how="left")
    .merge(prjna_keep, on="gene_clean", how="left")
    .merge(rna_keep, on="gene_clean", how="left")
)

merged["gene"] = merged["gene_symbol"].combine_first(merged["prjna_gene_symbol"]).combine_first(merged["rna_gene_symbol"]).combine_first(merged["gene_clean"])

num_defaults = {
    "local_focus_calls": 0,
    "local_focus_unique_loci": 0,
    "local_focus_tumors": 0,
    "local_focus_HIGH": 0,
    "local_focus_MODERATE": 0,
    "local_focus_score": 0,
    "prjna_functional_pairs": 0,
    "prjna_functional_variants": 0,
    "prjna_HIGH": 0,
    "prjna_MODERATE": 0,
    "prjna_score_raw": 0,
    "log2FoldChange": np.nan,
    "padj": np.nan,
    "pvalue": np.nan,
    "rna_abs_log2FC": 0
}

for c, default in num_defaults.items():
    if c not in merged.columns:
        merged[c] = default
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(default)

for c in ["is_breast_focus_gene", "is_cancer_seed_gene", "prjna_cancer_seed"]:
    if c not in merged.columns:
        merged[c] = False
    merged[c] = merged[c].fillna(False).astype(bool)

merged["local_tumor_support"] = merged["local_focus_tumors"] > 0
merged["prjna_somatic_support"] = merged["prjna_functional_pairs"] > 0
merged["rna_DE_nominal_support"] = merged["padj"] < 0.25

merged["multilayer_support_n"] = (
    merged["local_tumor_support"].astype(int) +
    merged["prjna_somatic_support"].astype(int) +
    merged["rna_DE_nominal_support"].astype(int)
)

merged["integrated_score"] = (
    merged["local_tumor_support"].astype(int) * 25 +
    merged["local_focus_tumors"] * 8 +
    merged["local_focus_HIGH"] * 5 +
    merged["local_focus_MODERATE"] * 2 +
    merged["is_breast_focus_gene"].astype(int) * 15 +
    merged["is_cancer_seed_gene"].astype(int) * 8 +
    merged["prjna_somatic_support"].astype(int) * 8 +
    merged["prjna_functional_pairs"] * 4 +
    merged["prjna_HIGH"] * 4 +
    merged["prjna_MODERATE"] * 2 +
    merged["prjna_cancer_seed"].astype(int) * 6 +
    np.where(merged["padj"] < 0.05, 10, np.where(merged["padj"] < 0.10, 6, np.where(merged["padj"] < 0.25, 3, 0))) +
    np.where(merged["rna_abs_log2FC"] >= 1, 2, 0)
)

merged["primary_interpretation_class"] = "supporting_or_background_candidate"
merged.loc[merged["rna_DE_nominal_support"] & (~merged["local_tumor_support"]) & (~merged["prjna_somatic_support"]), "primary_interpretation_class"] = "RNA_only_signal"
merged.loc[merged["prjna_somatic_support"] & (~merged["local_tumor_support"]), "primary_interpretation_class"] = "PRJNA_somatic_supported"
merged.loc[merged["local_tumor_support"] & (~merged["prjna_somatic_support"]), "primary_interpretation_class"] = "local_tumor_focus"
merged.loc[merged["local_tumor_support"] & merged["prjna_somatic_support"], "primary_interpretation_class"] = "local_and_PRJNA_somatic_supported"
merged.loc[merged["local_tumor_support"] & merged["prjna_somatic_support"] & merged["rna_DE_nominal_support"], "primary_interpretation_class"] = "three_layer_supported"

merged = merged.sort_values(
    ["integrated_score", "multilayer_support_n", "local_focus_tumors", "prjna_functional_pairs", "gene"],
    ascending=[False, False, False, False, True]
)

cols = [
    "gene",
    "integrated_score",
    "multilayer_support_n",
    "primary_interpretation_class",
    "local_tumor_support",
    "local_focus_tumors",
    "local_focus_calls",
    "local_focus_unique_loci",
    "local_focus_HIGH",
    "local_focus_MODERATE",
    "local_focus_samples",
    "local_focus_subtypes",
    "is_breast_focus_gene",
    "is_cancer_seed_gene",
    "prjna_somatic_support",
    "prjna_functional_pairs",
    "prjna_functional_variants",
    "prjna_HIGH",
    "prjna_MODERATE",
    "prjna_cancer_seed",
    "log2FoldChange",
    "padj",
    "rna_direction_late_vs_early",
    "rna_DE_support"
]

cols = [c for c in cols if c in merged.columns]

priority_path = outdir / "integrated_multilayer_gene_priority.tsv"
merged[cols].to_csv(priority_path, sep="\t", index=False)

local_focus_path = outdir / "integrated_local_focus_gene_support.tsv"
merged[merged["local_tumor_support"]][cols].to_csv(local_focus_path, sep="\t", index=False)

summary = pd.DataFrame([
    {"metric": "integrated_genes_total", "value": len(merged)},
    {"metric": "local_tumor_focus_genes", "value": int(merged["local_tumor_support"].sum())},
    {"metric": "prjna_somatic_supported_genes", "value": int(merged["prjna_somatic_support"].sum())},
    {"metric": "rna_FDR025_supported_genes", "value": int(merged["rna_DE_nominal_support"].sum())},
    {"metric": "local_and_PRJNA_supported_genes", "value": int((merged["local_tumor_support"] & merged["prjna_somatic_support"]).sum())},
    {"metric": "three_layer_supported_genes", "value": int((merged["local_tumor_support"] & merged["prjna_somatic_support"] & merged["rna_DE_nominal_support"]).sum())}
])

summary_path = outdir / "integrated_multilayer_summary_counts.tsv"
summary.to_csv(summary_path, sep="\t", index=False)

report_path = status / "80A_integrated_multilayer_candidate_prioritization_status.txt"
with open(report_path, "w") as f:
    f.write("80A integrated multilayer candidate prioritization\n")
    f.write("Note: local tumor WES calls are tumor-only candidates, not definitive somatic variants.\n\n")
    f.write("Inputs:\n")
    f.write(f"LOCAL_GENE={local_gene_file}\n")
    f.write(f"LOCAL_CAND={local_cand_file}\n")
    f.write(f"PRJNA_GENE={prjna_gene_file}\n")
    f.write(f"RNA_DE={rna_file}\n\n")
    f.write("Summary counts:\n")
    f.write(summary.to_csv(sep="\t", index=False))
    f.write("\nTop integrated genes:\n")
    f.write(merged[cols].head(60).to_csv(sep="\t", index=False))
    f.write("\nLocal focus gene integrated support:\n")
    f.write(merged[merged["local_tumor_support"]][cols].to_csv(sep="\t", index=False))

print("priority_table", priority_path)
print("local_focus_support", local_focus_path)
print("summary_counts", summary_path)
print("report", report_path)
PY

echo
echo "=== BUILD 80A FIGURES ==="

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

priority <- read_tsv(file.path(outdir, "integrated_multilayer_gene_priority.tsv"), show_col_types = FALSE)
summary <- read_tsv(file.path(outdir, "integrated_multilayer_summary_counts.tsv"), show_col_types = FALSE)

top <- priority %>%
  arrange(desc(integrated_score), desc(multilayer_support_n), gene) %>%
  slice_head(n = 25) %>%
  mutate(
    gene = factor(gene, levels = rev(gene)),
    class = case_when(
      primary_interpretation_class == "three_layer_supported" ~ "Three-layer",
      primary_interpretation_class == "local_and_PRJNA_somatic_supported" ~ "Local + PRJNA",
      primary_interpretation_class == "local_tumor_focus" ~ "Local tumor focus",
      primary_interpretation_class == "PRJNA_somatic_supported" ~ "PRJNA somatic",
      primary_interpretation_class == "RNA_only_signal" ~ "RNA only",
      TRUE ~ "Other"
    )
  )

class_cols <- c(
  "Three-layer" = "#D62728",
  "Local + PRJNA" = "#7E57C2",
  "Local tumor focus" = "#E69F00",
  "PRJNA somatic" = "#1F77B4",
  "RNA only" = "#009E73",
  "Other" = "grey65"
)

p_lollipop <- ggplot(top, aes(x = integrated_score, y = gene)) +
  geom_segment(aes(x = 0, xend = integrated_score, yend = gene, color = class), linewidth = 1.0, alpha = 0.55) +
  geom_point(aes(size = multilayer_support_n, fill = class), shape = 21, color = "grey15", stroke = 0.35) +
  scale_fill_manual(values = class_cols) +
  scale_color_manual(values = class_cols) +
  scale_size_continuous(range = c(3.5, 8.8), breaks = c(1, 2, 3)) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.12))) +
  labs(
    title = "Integrated multi-layer candidate prioritization",
    subtitle = "Local tumor WES, PRJNA somatic WES and RNA-seq support",
    x = "Integrated prioritization score",
    y = NULL,
    fill = "Support class",
    color = "Support class",
    size = "Supported layers"
  ) +
  theme_pub(10) +
  guides(color = "none")

ggsave(file.path(figs, "Fig_Integrated_01_multilayer_priority_lollipop.pdf"), p_lollipop, width = 8.0, height = 7.0)
ggsave(file.path(figs, "Fig_Integrated_01_multilayer_priority_lollipop.png"), p_lollipop, width = 8.0, height = 7.0, dpi = 600)

support_long <- top %>%
  transmute(
    gene,
    `Local tumor WES` = ifelse(local_tumor_support, local_focus_tumors, 0),
    `PRJNA somatic WES` = ifelse(prjna_somatic_support, prjna_functional_pairs, 0),
    `RNA-seq DE` = ifelse(!is.na(padj) & padj < 0.25, pmin(-log10(padj), 10), 0)
  ) %>%
  pivot_longer(-gene, names_to = "layer", values_to = "support_strength") %>%
  mutate(
    present = support_strength > 0,
    layer = factor(layer, levels = c("Local tumor WES", "PRJNA somatic WES", "RNA-seq DE"))
  )

p_dot <- ggplot(support_long, aes(x = layer, y = gene)) +
  geom_point(data = support_long %>% filter(!present), size = 2.2, color = "grey88") +
  geom_point(
    data = support_long %>% filter(present),
    aes(size = support_strength, fill = layer),
    shape = 21,
    color = "grey15",
    stroke = 0.3,
    alpha = 0.92
  ) +
  scale_fill_manual(values = c(
    "Local tumor WES" = "#E69F00",
    "PRJNA somatic WES" = "#1F77B4",
    "RNA-seq DE" = "#009E73"
  )) +
  scale_size_continuous(range = c(3.0, 9.0)) +
  labs(
    title = "Layer-specific support for prioritized genes",
    subtitle = "Dot size reflects tumors, PRJNA pairs or RNA −log10(FDR)",
    x = NULL,
    y = NULL,
    fill = "Layer",
    size = "Support"
  ) +
  theme_pub(10)

ggsave(file.path(figs, "Fig_Integrated_02_multilayer_support_dotmatrix.pdf"), p_dot, width = 7.4, height = 7.0)
ggsave(file.path(figs, "Fig_Integrated_02_multilayer_support_dotmatrix.png"), p_dot, width = 7.4, height = 7.0, dpi = 600)

local_focus <- priority %>%
  filter(local_tumor_support) %>%
  arrange(desc(integrated_score)) %>%
  mutate(gene = factor(gene, levels = rev(gene)))

p_focus <- ggplot(local_focus, aes(x = integrated_score, y = gene)) +
  geom_segment(aes(x = 0, xend = integrated_score, yend = gene), linewidth = 1.1, alpha = 0.55, color = "#D62728") +
  geom_point(aes(size = local_focus_tumors), shape = 21, fill = "#D62728", color = "grey15", stroke = 0.35) +
  geom_text(aes(label = paste0(local_focus_tumors, " local tumors")), nudge_x = 3, hjust = 0, fontface = "bold", size = 3.4) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.35))) +
  scale_size_continuous(range = c(4, 9)) +
  labs(
    title = "Integrated context for final local tumor WES focus genes",
    subtitle = "Tumor-only local candidates prioritized against PRJNA WES and RNA-seq layers",
    x = "Integrated prioritization score",
    y = NULL,
    size = "Local tumors"
  ) +
  theme_pub(11)

ggsave(file.path(figs, "Fig_Integrated_03_local_focus_gene_integrated_context.pdf"), p_focus, width = 8.0, height = 4.8)
ggsave(file.path(figs, "Fig_Integrated_03_local_focus_gene_integrated_context.png"), p_focus, width = 8.0, height = 4.8, dpi = 600)

report <- file.path(status, "80A_integrated_multilayer_candidate_prioritization_figures_status.txt")
sink(report)
cat("80A integrated multilayer candidate prioritization figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Summary:\n")
print(summary)
cat("\nTop integrated genes:\n")
print(head(priority, 40))
sink()
RS

echo
echo "=== 80A SUMMARY COUNTS ==="
column -t -s $'\t' "$OUT/integrated_multilayer_summary_counts.tsv"

echo
echo "=== 80A TOP INTEGRATED GENES ==="
column -t -s $'\t' "$OUT/integrated_multilayer_gene_priority.tsv" | head -60

echo
echo "=== 80A LOCAL FOCUS INTEGRATED SUPPORT ==="
column -t -s $'\t' "$OUT/integrated_local_focus_gene_support.tsv"

echo
echo "=== 80A FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/80A_integrated_multilayer_candidate_prioritization_status.txt" | head -160

echo
echo "=== DONE 80A ==="
date
