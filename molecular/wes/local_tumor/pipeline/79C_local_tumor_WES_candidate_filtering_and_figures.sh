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

MUTECT_DIR="$PAPER/05_local_breast_cancer_tumor_WES/01_tumor_only_mutect2"
MANIFEST="$MUTECT_DIR/local_tumor_WES_tumor_only_Mutect2_manifest.tsv"
SAMPLE_SHEET="$PAPER/05_local_breast_cancer_tumor_WES/local_tumor_wes_sample_sheet.tsv"

BG_VCF="results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.autosomal.vcf.gz"

OUT="$PAPER/05_local_breast_cancer_tumor_WES/02_candidate_filtering"
FIGS="$PAPER/publication_figures/local_tumor_WES"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/79C_local_tumor_WES_candidate_filtering_and_figures_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 79C: LOCAL TUMOR WES CANDIDATE FILTERING AND FIGURES ==="
date

for f in "$MANIFEST" "$SAMPLE_SHEET" "$BG_VCF"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing input:"
    echo "$f"
    exit 1
  fi
done

BG_KEYS="$OUT/local_non_cancer_germline_background_autosomal_snp_keys.tsv"

echo
echo "=== BUILD LOCAL NON-CANCER GERMLINE BACKGROUND KEY TABLE ==="
if [[ ! -s "$BG_KEYS" ]]; then
  micromamba run -n hadza-wes bcftools +fill-tags "$BG_VCF" -Ou -- -t AC,AN,AF \
    | micromamba run -n hadza-wes bcftools query \
      -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AC\t%INFO/AN\t%INFO/AF\n' \
      > "$BG_KEYS"
fi

echo "Background key table:"
ls -lh "$BG_KEYS"
head "$BG_KEYS"

export MANIFEST SAMPLE_SHEET BG_KEYS OUT STATUS

micromamba run -n hadza-wes python - <<'PY'
import os
import re
import gzip
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd

manifest_file = os.environ["MANIFEST"]
sample_sheet_file = os.environ["SAMPLE_SHEET"]
bg_keys_file = os.environ["BG_KEYS"]
outdir = Path(os.environ["OUT"])
statusdir = Path(os.environ["STATUS"])

outdir.mkdir(parents=True, exist_ok=True)

canonical_genes = set("""
TP53 PIK3CA KMT2C KMT2D CDH1 ESR1 ERBB2 GATA3 FOXA1 PTEN RB1 NF1 ATM BRCA1 BRCA2 PALB2
MAP3K1 ARID1A SMAD4 SMARCA4 EP300 CREBBP MLH1 MSH2 SETD2 TBX3 CHEK2 RAD51C RAD51D BARD1
BRIP1 ERBB3 FGFR1 FGFR2 FGFR3 AKT1 AKT2 AKT3 MTOR PTEN KRAS NRAS HRAS BRAF RAF1
CCND1 CDK4 CDK6 MYC MDM2 TERT ARID1B ARID2
""".split())

breast_focus_genes = set("""
TP53 PIK3CA CDH1 GATA3 FOXA1 ESR1 ERBB2 PTEN RB1 NF1 KMT2C KMT2D MAP3K1 AKT1
BRCA1 BRCA2 PALB2 ATM CHEK2 RAD51C RAD51D BARD1 CCND1 CDK4 CDK6 MYC FGFR1
""".split())

def norm_chrom(chrom):
    c = str(chrom)
    if c.startswith("chr"):
        c = c[3:]
    if c == "M":
        c = "MT"
    return c

def variant_key(chrom, pos, ref, alt):
    return f"{norm_chrom(chrom)}:{pos}:{ref}:{alt}"

def parse_float_first(x):
    if x is None or x in ["", "."]:
        return math.nan
    try:
        return float(str(x).split(",")[0])
    except Exception:
        return math.nan

def parse_info(info):
    d = {}
    for item in str(info).split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
        else:
            d[item] = True
    return d

def parse_format(fmt, sample_value):
    keys = str(fmt).split(":")
    vals = str(sample_value).split(":")
    d = {k: vals[i] if i < len(vals) else "." for i, k in enumerate(keys)}

    def to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    dp = to_int(d.get("DP", "."))
    af = parse_float_first(d.get("AF", "."))
    ad = d.get("AD", ".")
    ref_dp = None
    alt_dp = None

    if ad not in [None, ".", ""]:
        try:
            parts = [int(float(z)) for z in ad.split(",") if z not in ["", "."]]
            if len(parts) >= 2:
                ref_dp = parts[0]
                alt_dp = sum(parts[1:])
        except Exception:
            pass

    return {
        "GT": d.get("GT", "."),
        "DP": dp,
        "AF": af,
        "AD": ad,
        "ref_depth": ref_dp,
        "alt_depth": alt_dp
    }

def parse_ann(ann):
    if ann is None or ann in ["", "."]:
        return {
            "effect": ".",
            "impact": ".",
            "gene": ".",
            "gene_id": ".",
            "feature": ".",
            "biotype": ".",
            "hgvsc": ".",
            "hgvsp": "."
        }

    priority = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
    best = None
    best_score = -1

    for rec in str(ann).split(","):
        fields = rec.split("|")
        while len(fields) < 11:
            fields.append(".")
        impact = fields[2] if fields[2] else "."
        score = priority.get(impact, 0)
        if score > best_score:
            best_score = score
            best = fields

    return {
        "effect": best[1] if len(best) > 1 and best[1] else ".",
        "impact": best[2] if len(best) > 2 and best[2] else ".",
        "gene": best[3] if len(best) > 3 and best[3] else ".",
        "gene_id": best[4] if len(best) > 4 and best[4] else ".",
        "feature": best[6] if len(best) > 6 and best[6] else ".",
        "biotype": best[7] if len(best) > 7 and best[7] else ".",
        "hgvsc": best[9] if len(best) > 9 and best[9] else ".",
        "hgvsp": best[10] if len(best) > 10 and best[10] else "."
    }

def variant_type(ref, alt):
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(ref) == len(alt):
        return "MNP"
    return "INDEL"

# -------------------------
# Metadata
# -------------------------
sample_sheet = pd.read_csv(sample_sheet_file, sep="\t", dtype=str)
meta = sample_sheet.set_index("sample_id").to_dict("index")

# -------------------------
# Local non-cancer background keys
# -------------------------
bg = {}
with open(bg_keys_file) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        chrom, pos, ref, alt, ac, an, af = line.split("\t")[:7]
        key = variant_key(chrom, pos, ref, alt)
        bg[key] = {
            "local_bg_AC": ac,
            "local_bg_AN": an,
            "local_bg_AF": af
        }

# -------------------------
# Parse local tumor-only VCFs
# -------------------------
manifest = pd.read_csv(manifest_file, sep="\t", dtype=str)
manifest = manifest[manifest["status"].astype(str).eq("DONE")].copy()

records = []

for _, r in manifest.iterrows():
    sample = r["sample_id"]
    vcf = r["pass_snpeff_vcf"]

    if not Path(vcf).exists():
        continue

    opener = gzip.open if str(vcf).endswith(".gz") else open
    header = None
    sample_col_name = sample

    with opener(vcf, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                header = line.rstrip("\n").lstrip("#").split("\t")
                if len(header) > 9:
                    sample_col_name = header[9]
                continue

            if not line.strip():
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, vid, ref, alt, qual, flt, info = fields[:8]
            fmt = fields[8] if len(fields) > 8 else "."
            sample_value = fields[9] if len(fields) > 9 else "."

            info_d = parse_info(info)
            ann_d = parse_ann(info_d.get("ANN", "."))

            fmt_d = parse_format(fmt, sample_value)

            key = variant_key(chrom, pos, ref, alt)
            bg_hit = key in bg

            rec = {
                "sample_id": sample,
                "vcf_sample_name": sample_col_name,
                "stage": meta.get(sample, {}).get("stage", "."),
                "molecular_subtype": meta.get(sample, {}).get("molecular_subtype", "."),
                "er_status": meta.get(sample, {}).get("er_status", "."),
                "pr_status": meta.get(sample, {}).get("pr_status", "."),
                "her_2_status": meta.get(sample, {}).get("her_2_status", "."),
                "CHROM": chrom,
                "POS": int(pos),
                "ID": vid,
                "REF": ref,
                "ALT": alt,
                "variant_key": key,
                "variant_type": variant_type(ref, alt),
                "QUAL": qual,
                "FILTER": flt,
                "TLOD": parse_float_first(info_d.get("TLOD")),
                "NLOD": parse_float_first(info_d.get("NLOD")),
                "POPAF": parse_float_first(info_d.get("POPAF")),
                "GERMQ": parse_float_first(info_d.get("GERMQ")),
                "STR": info_d.get("STR", False) is True,
                "gene": ann_d["gene"],
                "effect": ann_d["effect"],
                "impact": ann_d["impact"],
                "hgvsc": ann_d["hgvsc"],
                "hgvsp": ann_d["hgvsp"],
                "GT": fmt_d["GT"],
                "DP": fmt_d["DP"],
                "AD": fmt_d["AD"],
                "ref_depth": fmt_d["ref_depth"],
                "alt_depth": fmt_d["alt_depth"],
                "tumor_AF": fmt_d["AF"],
                "is_functional_HIGH_MODERATE": ann_d["impact"] in {"HIGH", "MODERATE"},
                "is_HIGH": ann_d["impact"] == "HIGH",
                "is_MODERATE": ann_d["impact"] == "MODERATE",
                "is_local_background_overlap": bg_hit,
                "local_bg_AC": bg.get(key, {}).get("local_bg_AC", "."),
                "local_bg_AN": bg.get(key, {}).get("local_bg_AN", "."),
                "local_bg_AF": bg.get(key, {}).get("local_bg_AF", "."),
                "is_cancer_seed_gene": ann_d["gene"] in canonical_genes,
                "is_breast_focus_gene": ann_d["gene"] in breast_focus_genes
            }
            records.append(rec)

all_df = pd.DataFrame(records)

if all_df.empty:
    raise SystemExit("No variants parsed from PASS SnpEff VCFs.")

# Exact recurrence across local tumor-only samples
recurrence = (
    all_df.groupby("variant_key")
    .agg(
        local_tumor_recurrence_n=("sample_id", "nunique"),
        local_tumor_recurrence_samples=("sample_id", lambda x: ";".join(sorted(set(x))))
    )
    .reset_index()
)

all_df = all_df.merge(recurrence, on="variant_key", how="left")

all_df["tumor_only_caution_flag"] = (
    all_df["is_local_background_overlap"] |
    (all_df["local_tumor_recurrence_n"] >= 4)
)

all_df["candidate_tier"] = "nonfunctional_or_background"
all_df.loc[
    all_df["is_functional_HIGH_MODERATE"] & (~all_df["is_local_background_overlap"]),
    "candidate_tier"
] = "functional_nonbackground"
all_df.loc[
    all_df["is_functional_HIGH_MODERATE"] &
    (~all_df["is_local_background_overlap"]) &
    all_df["is_cancer_seed_gene"],
    "candidate_tier"
] = "functional_nonbackground_cancer_seed"
all_df.loc[
    all_df["is_functional_HIGH_MODERATE"] &
    (~all_df["is_local_background_overlap"]) &
    all_df["is_breast_focus_gene"],
    "candidate_tier"
] = "functional_nonbackground_breast_focus"

all_df.loc[
    all_df["is_functional_HIGH_MODERATE"] &
    (~all_df["is_local_background_overlap"]) &
    all_df["is_breast_focus_gene"] &
    (all_df["impact"] == "HIGH"),
    "candidate_tier"
] = "HIGH_functional_nonbackground_breast_focus"

# Conservative display candidates: functional, not local background, and not very broadly recurrent,
# unless the gene is breast/cancer seed. This is a display flag, not hard deletion from all table.
all_df["display_conservative_candidate"] = (
    all_df["is_functional_HIGH_MODERATE"] &
    (~all_df["is_local_background_overlap"]) &
    (
        (all_df["local_tumor_recurrence_n"] <= 2) |
        all_df["is_breast_focus_gene"] |
        all_df["is_cancer_seed_gene"]
    )
)

all_path = outdir / "local_tumor_WES_tumor_only_PASS_variants.annotated_with_flags.tsv.gz"
func_path = outdir / "local_tumor_WES_functional_nonbackground_candidates.tsv"
conservative_path = outdir / "local_tumor_WES_display_conservative_candidates.tsv"
cancer_seed_path = outdir / "local_tumor_WES_cancer_seed_candidates.tsv"

all_df.to_csv(all_path, sep="\t", index=False, compression="gzip")

func_df = all_df[
    all_df["is_functional_HIGH_MODERATE"] &
    (~all_df["is_local_background_overlap"])
].copy()
func_df.to_csv(func_path, sep="\t", index=False)

cons_df = all_df[all_df["display_conservative_candidate"]].copy()
cons_df.to_csv(conservative_path, sep="\t", index=False)

seed_df = all_df[
    all_df["is_functional_HIGH_MODERATE"] &
    (~all_df["is_local_background_overlap"]) &
    all_df["is_cancer_seed_gene"]
].copy()
seed_df.to_csv(cancer_seed_path, sep="\t", index=False)

# Sample summary
sample_rows = []
for sample, sub in all_df.groupby("sample_id"):
    fsub = sub[sub["is_functional_HIGH_MODERATE"]]
    fnbg = sub[sub["is_functional_HIGH_MODERATE"] & (~sub["is_local_background_overlap"])]
    sample_rows.append({
        "sample_id": sample,
        "stage": meta.get(sample, {}).get("stage", "."),
        "molecular_subtype": meta.get(sample, {}).get("molecular_subtype", "."),
        "er_status": meta.get(sample, {}).get("er_status", "."),
        "pr_status": meta.get(sample, {}).get("pr_status", "."),
        "her_2_status": meta.get(sample, {}).get("her_2_status", "."),
        "n_PASS": len(sub),
        "n_SNV": int((sub["variant_type"] == "SNV").sum()),
        "n_INDEL": int((sub["variant_type"] == "INDEL").sum()),
        "n_MNP": int((sub["variant_type"] == "MNP").sum()),
        "n_local_background_overlap": int(sub["is_local_background_overlap"].sum()),
        "n_functional_HIGH_MODERATE": len(fsub),
        "n_functional_nonbackground": len(fnbg),
        "n_HIGH_nonbackground": int((fnbg["impact"] == "HIGH").sum()),
        "n_MODERATE_nonbackground": int((fnbg["impact"] == "MODERATE").sum()),
        "n_cancer_seed_functional_nonbackground": int(fnbg["is_cancer_seed_gene"].sum()),
        "n_breast_focus_functional_nonbackground": int(fnbg["is_breast_focus_gene"].sum()),
        "n_display_conservative_candidates": int(sub["display_conservative_candidate"].sum())
    })

sample_summary = pd.DataFrame(sample_rows).sort_values("sample_id")
sample_summary_path = outdir / "local_tumor_WES_candidate_sample_summary.tsv"
sample_summary.to_csv(sample_summary_path, sep="\t", index=False)

# Gene summary
gene_source = func_df[func_df["gene"].notna() & (func_df["gene"] != ".")].copy()

if not gene_source.empty:
    gene_summary = (
        gene_source.groupby("gene")
        .agg(
            n_variants=("variant_key", "count"),
            n_unique_loci=("variant_key", "nunique"),
            n_samples=("sample_id", "nunique"),
            samples=("sample_id", lambda x: ";".join(sorted(set(x)))),
            stages=("stage", lambda x: ";".join(sorted(set(map(str, x))))),
            subtypes=("molecular_subtype", lambda x: ";".join(sorted(set(map(str, x))))),
            n_HIGH=("is_HIGH", "sum"),
            n_MODERATE=("is_MODERATE", "sum"),
            max_local_tumor_recurrence_n=("local_tumor_recurrence_n", "max"),
            median_tumor_AF=("tumor_AF", "median"),
            median_DP=("DP", "median"),
            is_cancer_seed_gene=("is_cancer_seed_gene", "max"),
            is_breast_focus_gene=("is_breast_focus_gene", "max")
        )
        .reset_index()
    )
    gene_summary["priority_score"] = (
        gene_summary["n_samples"] * 5 +
        gene_summary["n_HIGH"] * 4 +
        gene_summary["n_MODERATE"] * 2 +
        gene_summary["is_breast_focus_gene"].astype(int) * 8 +
        gene_summary["is_cancer_seed_gene"].astype(int) * 5
    )
    gene_summary = gene_summary.sort_values(
        ["priority_score", "n_samples", "n_HIGH", "n_variants", "gene"],
        ascending=[False, False, False, False, True]
    )
else:
    gene_summary = pd.DataFrame()

gene_summary_path = outdir / "local_tumor_WES_gene_summary_functional_nonbackground.tsv"
gene_summary.to_csv(gene_summary_path, sep="\t", index=False)

# Gene-sample matrix
matrix_source = func_df[
    func_df["gene"].notna() &
    (func_df["gene"] != ".")
].copy()

if not matrix_source.empty:
    def best_impact(vals):
        vals = set(vals)
        if "HIGH" in vals and "MODERATE" in vals:
            return "HIGH+MODERATE"
        if "HIGH" in vals:
            return "HIGH"
        if "MODERATE" in vals:
            return "MODERATE"
        return "."

    gene_sample = (
        matrix_source.groupby(["gene", "sample_id"])
        .agg(
            n_variants=("variant_key", "count"),
            best_impact=("impact", best_impact),
            max_tumor_AF=("tumor_AF", "max"),
            is_breast_focus_gene=("is_breast_focus_gene", "max"),
            is_cancer_seed_gene=("is_cancer_seed_gene", "max")
        )
        .reset_index()
    )
else:
    gene_sample = pd.DataFrame(columns=[
        "gene", "sample_id", "n_variants", "best_impact",
        "max_tumor_AF", "is_breast_focus_gene", "is_cancer_seed_gene"
    ])

gene_sample_path = outdir / "local_tumor_WES_functional_nonbackground_gene_sample_matrix.tsv"
gene_sample.to_csv(gene_sample_path, sep="\t", index=False)

# Exact recurrent loci table
recurrent_loci = (
    all_df[all_df["local_tumor_recurrence_n"] >= 2]
    .sort_values(["local_tumor_recurrence_n", "is_functional_HIGH_MODERATE", "is_cancer_seed_gene", "gene"],
                 ascending=[False, False, False, True])
)
recurrent_loci_path = outdir / "local_tumor_WES_exact_recurrent_loci.tsv"
recurrent_loci.to_csv(recurrent_loci_path, sep="\t", index=False)

summary_counts = pd.DataFrame([
    {"metric": "samples", "value": all_df["sample_id"].nunique()},
    {"metric": "PASS_tumor_only_candidate_variants", "value": len(all_df)},
    {"metric": "local_background_exact_overlaps", "value": int(all_df["is_local_background_overlap"].sum())},
    {"metric": "functional_HIGH_MODERATE_all", "value": int(all_df["is_functional_HIGH_MODERATE"].sum())},
    {"metric": "functional_HIGH_MODERATE_nonbackground", "value": len(func_df)},
    {"metric": "display_conservative_candidates", "value": len(cons_df)},
    {"metric": "functional_nonbackground_genes", "value": gene_summary["gene"].nunique() if not gene_summary.empty else 0},
    {"metric": "functional_nonbackground_breast_focus_variants", "value": int(func_df["is_breast_focus_gene"].sum()) if not func_df.empty else 0},
    {"metric": "functional_nonbackground_cancer_seed_variants", "value": int(func_df["is_cancer_seed_gene"].sum()) if not func_df.empty else 0}
])

summary_counts_path = outdir / "local_tumor_WES_candidate_filtering_summary_counts.tsv"
summary_counts.to_csv(summary_counts_path, sep="\t", index=False)

report_path = statusdir / "79C_local_tumor_WES_candidate_filtering_status.txt"
with open(report_path, "w") as f:
    f.write("79C local tumor WES candidate filtering\n")
    f.write("Analysis note: tumor-only candidate variant analysis. Matched normals are absent; variants are not definitive somatic calls.\n")
    f.write("Local non-cancer germline background exact-overlap flag: 31 Hadza/Tanzanian-control germline VCF.\n\n")
    f.write("Summary counts:\n")
    f.write(summary_counts.to_csv(sep="\t", index=False))
    f.write("\nSample summary:\n")
    f.write(sample_summary.to_csv(sep="\t", index=False))
    f.write("\nTop functional nonbackground genes:\n")
    f.write(gene_summary.head(60).to_csv(sep="\t", index=False) if not gene_summary.empty else "none\n")
    f.write("\nCancer-seed candidates:\n")
    f.write(seed_df.head(100).to_csv(sep="\t", index=False) if not seed_df.empty else "none\n")

print("all_variants", all_path)
print("functional_nonbackground", func_path)
print("display_conservative", conservative_path)
print("cancer_seed", cancer_seed_path)
print("sample_summary", sample_summary_path)
print("gene_summary", gene_summary_path)
print("gene_sample_matrix", gene_sample_path)
print("recurrent_loci", recurrent_loci_path)
print("summary_counts", summary_counts_path)
print("report", report_path)
PY

echo
echo "=== BUILD LOCAL TUMOR WES PUBLICATION FIGURES ==="

export OUT FIGS SAMPLE_SHEET STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
})

outdir <- Sys.getenv("OUT")
figs <- Sys.getenv("FIGS")
sample_sheet_file <- Sys.getenv("SAMPLE_SHEET")
statusdir <- Sys.getenv("STATUS")

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

variant_cols <- c(
  "SNV" = "#1F77B4",
  "INDEL" = "#E69F00",
  "MNP" = "#7E57C2",
  "OTHER" = "grey65"
)

impact_cols <- c(
  "HIGH" = "#D62728",
  "MODERATE" = "#E69F00",
  "HIGH+MODERATE" = "#7E57C2"
)

gene_class_cols <- c(
  "Breast/cancer-focus gene" = "#D62728",
  "Other functional gene" = "#1F77B4"
)

all <- read_tsv(
  file.path(outdir, "local_tumor_WES_tumor_only_PASS_variants.annotated_with_flags.tsv.gz"),
  show_col_types = FALSE
)

sample_sum <- read_tsv(
  file.path(outdir, "local_tumor_WES_candidate_sample_summary.tsv"),
  show_col_types = FALSE
)

gene <- read_tsv(
  file.path(outdir, "local_tumor_WES_gene_summary_functional_nonbackground.tsv"),
  show_col_types = FALSE
)

gene_sample <- read_tsv(
  file.path(outdir, "local_tumor_WES_functional_nonbackground_gene_sample_matrix.tsv"),
  show_col_types = FALSE
)

summary <- read_tsv(
  file.path(outdir, "local_tumor_WES_candidate_filtering_summary_counts.tsv"),
  show_col_types = FALSE
)

sample_order <- sample_sum %>%
  arrange(factor(stage, levels = c("II stage", "III stage", "IV stage")), sample_id) %>%
  pull(sample_id)

sample_labels <- sample_sum %>%
  mutate(label = paste0(sample_id, "\n", molecular_subtype)) %>%
  select(sample_id, label)

# Figure 1: PASS tumor-only candidate burden
burden <- all %>%
  mutate(sample_id = factor(sample_id, levels = sample_order)) %>%
  count(sample_id, variant_type, name = "n")

p_burden <- ggplot(burden, aes(x = sample_id, y = n, fill = variant_type)) +
  geom_col(width = 0.72, color = "white", linewidth = 0.15) +
  scale_fill_manual(values = variant_cols, drop = FALSE) +
  labs(
    title = "Local tumor WES tumor-only candidate burden",
    subtitle = "Matched normals unavailable; variants are tumor-only candidate calls",
    x = "Local tumor WES sample",
    y = "Number of PASS candidate variants",
    fill = "Variant class"
  ) +
  theme_pub(11)

ggsave(file.path(figs, "Fig_LocalTumorWES_01_tumor_only_candidate_burden.pdf"), p_burden, width = 7.4, height = 5.4)
ggsave(file.path(figs, "Fig_LocalTumorWES_01_tumor_only_candidate_burden.png"), p_burden, width = 7.4, height = 5.4, dpi = 600)

# Figure 2: functional nonbackground burden
fun <- all %>%
  filter(is_functional_HIGH_MODERATE == TRUE, is_local_background_overlap == FALSE) %>%
  mutate(
    sample_id = factor(sample_id, levels = sample_order),
    impact = factor(impact, levels = c("HIGH", "MODERATE"))
  ) %>%
  count(sample_id, impact, name = "n")

p_fun <- ggplot(fun, aes(x = sample_id, y = n, fill = impact)) +
  geom_col(width = 0.72, color = "white", linewidth = 0.15) +
  scale_fill_manual(values = impact_cols, drop = FALSE) +
  labs(
    title = "Functional tumor-only candidates after local-background flagging",
    subtitle = "SnpEff HIGH/MODERATE variants excluding exact overlap with local non-cancer germline SNP background",
    x = "Local tumor WES sample",
    y = "Functional non-background candidate variants",
    fill = "Predicted impact"
  ) +
  theme_pub(11)

ggsave(file.path(figs, "Fig_LocalTumorWES_02_functional_nonbackground_candidate_burden.pdf"), p_fun, width = 7.6, height = 5.4)
ggsave(file.path(figs, "Fig_LocalTumorWES_02_functional_nonbackground_candidate_burden.png"), p_fun, width = 7.6, height = 5.4, dpi = 600)

# Figure 3: top genes
top_gene <- gene %>%
  filter(gene != ".") %>%
  arrange(desc(priority_score), desc(n_samples), desc(n_HIGH), desc(n_variants), gene) %>%
  slice_head(n = 25) %>%
  mutate(
    gene_label = fct_reorder(gene, priority_score),
    gene_class = ifelse(is_breast_focus_gene | is_cancer_seed_gene, "Breast/cancer-focus gene", "Other functional gene")
  )

if (nrow(top_gene) > 0) {
  p_gene <- ggplot(top_gene, aes(x = priority_score, y = gene_label)) +
    geom_segment(aes(x = 0, xend = priority_score, yend = gene_label, color = gene_class), linewidth = 1.0, alpha = 0.48) +
    geom_point(aes(size = n_samples, fill = gene_class), shape = 21, color = "grey15", stroke = 0.35, alpha = 0.95) +
    geom_text(aes(label = n_samples), nudge_x = 1.2, size = 3.0, fontface = "bold") +
    scale_fill_manual(values = gene_class_cols) +
    scale_color_manual(values = gene_class_cols) +
    scale_size_continuous(range = c(3.5, 8.8)) +
    scale_x_continuous(expand = expansion(mult = c(0.01, 0.12))) +
    labs(
      title = "Prioritized local tumor WES candidate genes",
      subtitle = "Functional non-background tumor-only candidates; label indicates number of affected local tumors",
      x = "Candidate prioritization score",
      y = NULL,
      fill = "Gene class",
      color = "Gene class",
      size = "Tumors"
    ) +
    theme_pub(10) +
    guides(color = "none")

  ggsave(file.path(figs, "Fig_LocalTumorWES_03_prioritized_candidate_genes.pdf"), p_gene, width = 7.6, height = 7.2)
  ggsave(file.path(figs, "Fig_LocalTumorWES_03_prioritized_candidate_genes.png"), p_gene, width = 7.6, height = 7.2, dpi = 600)
}

# Figure 4: oncoprint-like matrix
top_matrix_genes <- top_gene %>%
  slice_head(n = 22) %>%
  pull(gene)

if (length(top_matrix_genes) > 0 && nrow(gene_sample) > 0) {
  grid <- expand_grid(
    gene = top_matrix_genes,
    sample_id = sample_order
  ) %>%
    left_join(gene_sample, by = c("gene", "sample_id")) %>%
    mutate(
      sample_id = factor(sample_id, levels = sample_order),
      gene = factor(gene, levels = rev(top_matrix_genes)),
      best_impact = ifelse(is.na(best_impact), NA_character_, best_impact)
    )

  p_matrix <- ggplot(grid, aes(x = sample_id, y = gene)) +
    geom_tile(fill = "grey94", color = "white", linewidth = 0.25) +
    geom_tile(
      data = grid %>% filter(!is.na(best_impact)),
      aes(fill = best_impact),
      color = "white",
      linewidth = 0.25
    ) +
    scale_fill_manual(values = impact_cols, na.value = "grey94") +
    labs(
      title = "Local tumor WES functional candidate matrix",
      subtitle = "Top prioritized functional non-background genes across six local tumor WES samples",
      x = "Local tumor WES sample",
      y = NULL,
      fill = "Best impact"
    ) +
    theme_pub(9) +
    theme(legend.position = "top")

  ggsave(file.path(figs, "Fig_LocalTumorWES_04_functional_candidate_matrix.pdf"), p_matrix, width = 8.4, height = 6.8)
  ggsave(file.path(figs, "Fig_LocalTumorWES_04_functional_candidate_matrix.png"), p_matrix, width = 8.4, height = 6.8, dpi = 600)
}

report <- file.path(statusdir, "79C_local_tumor_WES_candidate_filtering_figures_status.txt")

sink(report)
cat("79C local tumor WES candidate filtering and figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Interpretation note:\n")
cat("Matched normals were not available. These are tumor-only candidate variants, not definitive somatic variants.\n\n")
cat("Summary counts:\n")
print(summary)
cat("\nSample summary:\n")
print(sample_sum)
cat("\nTop genes:\n")
print(top_gene)
sink()

cat("output_figures", figs, "\n")
cat("report", report, "\n")
RS

echo
echo "=== 79C SUMMARY COUNTS ==="
column -t -s $'\t' "$OUT/local_tumor_WES_candidate_filtering_summary_counts.tsv"

echo
echo "=== 79C SAMPLE SUMMARY ==="
column -t -s $'\t' "$OUT/local_tumor_WES_candidate_sample_summary.tsv"

echo
echo "=== 79C TOP FUNCTIONAL NONBACKGROUND GENES ==="
column -t -s $'\t' "$OUT/local_tumor_WES_gene_summary_functional_nonbackground.tsv" | head -60

echo
echo "=== LOCAL TUMOR WES FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "*.png" -o -name "*.pdf" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/79C_local_tumor_WES_candidate_filtering_figures_status.txt" | head -160

echo
echo "=== DONE 79C ==="
date
