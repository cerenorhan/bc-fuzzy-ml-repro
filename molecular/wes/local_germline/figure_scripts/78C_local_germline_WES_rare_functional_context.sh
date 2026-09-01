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
AUTOSOMAL="results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.autosomal.vcf.gz"

OUT="$PAPER/05_local_germline_WES/02_rare_functional_context"
FIGS="$PAPER/publication_figures/local_germline_WES"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/78C_local_germline_WES_rare_functional_context_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 78C: LOCAL GERMLINE WES RARE FUNCTIONAL CONTEXT ==="
date

if [[ ! -s "$META" ]]; then
  echo "ERROR: missing metadata: $META"
  exit 1
fi

if [[ ! -s "$AUTOSOMAL" ]]; then
  echo "ERROR: missing autosomal VCF: $AUTOSOMAL"
  exit 1
fi

echo
echo "Input VCF:"
ls -lh "$AUTOSOMAL"

FILLED="$OUT/local_wes.autosomal_biallelic_snps.filltags.vcf.gz"
ANNO="$OUT/local_wes.autosomal_biallelic_snps.filltags.snpeff.vcf.gz"
QUERY="$OUT/local_wes.autosomal_biallelic_snps.snpeff.query.tsv"

echo
echo "=== FILL TAGS ==="
if [[ ! -s "$FILLED" ]]; then
  micromamba run -n hadza-wes bcftools +fill-tags "$AUTOSOMAL" -Oz -o "$FILLED" -- -t AC,AN,AF,NS
  micromamba run -n hadza-wes bcftools index -t -f "$FILLED"
fi

echo
echo "=== SNPEFF ANNOTATION ==="
if [[ ! -s "$ANNO" ]]; then
  micromamba run -n cancer_anno snpEff -Xmx24g -v GRCh38.99 "$FILLED" \
    | micromamba run -n hadza-wes bgzip -c > "$ANNO"
  micromamba run -n hadza-wes bcftools index -t -f "$ANNO"
fi

echo
echo "=== QUERY ANNOTATED VCF ==="
if [[ ! -s "$QUERY" ]]; then
  {
    printf "CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\tANN"
    micromamba run -n hadza-wes bcftools query -l "$ANNO" | awk '{printf "\tGT_"$1}'
    printf "\n"
    micromamba run -n hadza-wes bcftools query \
      -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%INFO/AC\t%INFO/AN\t%INFO/AF\t%INFO/ANN[\t%GT]\n' \
      "$ANNO"
  } > "$QUERY"
fi

echo
echo "Query table:"
ls -lh "$QUERY"
head -2 "$QUERY" | cut -c1-220

export META QUERY OUT FIGS STATUS

micromamba run -n hadza-wes python - <<'PY'
import os
import re
import math
import pandas as pd
from collections import defaultdict

meta_file = os.environ["META"]
query_file = os.environ["QUERY"]
outdir = os.environ["OUT"]
figs = os.environ["FIGS"]
status = os.environ["STATUS"]

os.makedirs(outdir, exist_ok=True)
os.makedirs(figs, exist_ok=True)

meta = pd.read_csv(meta_file, sep="\t", dtype=str)

def norm_group(x):
    x = str(x).lower()
    if "hadza" in x or "hadzabe" in x:
        return "Hadza"
    if "control" in x or "tanzanian_control" in x:
        return "Tanzanian control"
    if "breast" in x:
        return "Breast cancer"
    return x

def norm_sex(x):
    x = str(x).lower()
    if x.startswith("f"):
        return "Female"
    if x.startswith("m"):
        return "Male"
    return x

meta["group_label"] = meta["group"].map(norm_group)
meta["sex_label"] = meta["sex"].map(norm_sex)

meta_map = meta.set_index("sample_id")[["group_label", "sex_label", "participants_origin"]].to_dict("index")

with open(query_file) as f:
    header = f.readline().rstrip("\n").split("\t")

gt_cols = [c for c in header if c.startswith("GT_")]
samples = [c.replace("GT_", "", 1) for c in gt_cols]

sample_groups = {s: meta_map.get(s, {}).get("group_label", "Unknown") for s in samples}
sample_sexes = {s: meta_map.get(s, {}).get("sex_label", "Unknown") for s in samples}

groups_used = ["Hadza", "Tanzanian control"]

canonical_genes = set("""
TP53 PIK3CA KMT2C KMT2D CDH1 ESR1 ERBB2 GATA3 FOXA1 PTEN RB1 NF1 ATM BRCA1 BRCA2 PALB2
MAP3K1 ARID1A SMAD4 SMARCA4 EP300 CREBBP MLH1 MSH2 SETD2 TBX3 CHEK2 RAD51C RAD51D BARD1
""".split())

def parse_int_first(x):
    if x is None or x == "." or x == "":
        return None
    x = str(x).split(",")[0]
    try:
        return int(float(x))
    except Exception:
        return None

def parse_float_first(x):
    if x is None or x == "." or x == "":
        return None
    x = str(x).split(",")[0]
    try:
        return float(x)
    except Exception:
        return None

def gt_ac_an(gt):
    gt = str(gt)
    if gt in [".", "./.", ".|."] or gt.startswith(".") or gt.endswith("."):
        return 0, 0, False
    parts = re.split(r"[\/|]", gt)
    ac = 0
    an = 0
    for p in parts:
        if p == "." or p == "":
            continue
        an += 1
        if p != "0":
            ac += 1
    return ac, an, ac > 0

def parse_ann(ann):
    if ann is None or ann == "." or ann == "":
        return {
            "effect": ".",
            "impact": ".",
            "gene": ".",
            "gene_id": ".",
            "feature": ".",
            "transcript_biotype": ".",
            "hgvsc": ".",
            "hgvsp": "."
        }

    records = str(ann).split(",")
    priority = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
    best = None
    best_score = -1

    for rec in records:
        fields = rec.split("|")
        while len(fields) < 11:
            fields.append(".")
        impact = fields[2] if fields[2] else "."
        score = priority.get(impact, 0)
        if score > best_score:
            best_score = score
            best = fields

    return {
        "effect": best[1] if len(best) > 1 else ".",
        "impact": best[2] if len(best) > 2 else ".",
        "gene": best[3] if len(best) > 3 and best[3] else ".",
        "gene_id": best[4] if len(best) > 4 else ".",
        "feature": best[6] if len(best) > 6 else ".",
        "transcript_biotype": best[7] if len(best) > 7 else ".",
        "hgvsc": best[9] if len(best) > 9 else ".",
        "hgvsp": best[10] if len(best) > 10 else "."
    }

records = []
rare_records = []
sample_burden = {
    s: {
        "sample_id": s,
        "group_label": sample_groups.get(s, "Unknown"),
        "sex_label": sample_sexes.get(s, "Unknown"),
        "rare_functional_nonref": 0,
        "rare_high_nonref": 0,
        "rare_moderate_nonref": 0,
        "rare_cancer_seed_nonref": 0
    }
    for s in samples
}

n_total = 0
n_rare = 0
n_func = 0
n_rare_func = 0

with open(query_file) as f:
    header = f.readline().rstrip("\n").split("\t")

    for line in f:
        n_total += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue

        row = dict(zip(header[:9], parts[:9]))
        gts = parts[9:]

        ann = parse_ann(row.get("ANN", "."))
        impact = ann["impact"]
        gene = ann["gene"]

        total_ac = 0
        total_an = 0
        group_ac = {g: 0 for g in groups_used}
        group_an = {g: 0 for g in groups_used}
        group_carriers = {g: set() for g in groups_used}
        variant_carrier_samples = []

        for s, gt in zip(samples, gts):
            ac, an, carrier = gt_ac_an(gt)
            total_ac += ac
            total_an += an

            g = sample_groups.get(s, "Unknown")
            if g in groups_used:
                group_ac[g] += ac
                group_an[g] += an
                if carrier:
                    group_carriers[g].add(s)

            if carrier:
                variant_carrier_samples.append(s)

        overall_af = total_ac / total_an if total_an > 0 else math.nan
        info_af = parse_float_first(row.get("AF"))
        if math.isnan(overall_af) and info_af is not None:
            overall_af = info_af

        is_rare = (not math.isnan(overall_af)) and overall_af <= 0.05 and total_ac <= 3
        is_functional = impact in {"HIGH", "MODERATE"}
        is_cancer_seed = gene in canonical_genes

        if is_rare:
            n_rare += 1
        if is_functional:
            n_func += 1
        if is_rare and is_functional:
            n_rare_func += 1

            for s in variant_carrier_samples:
                if s in sample_burden:
                    sample_burden[s]["rare_functional_nonref"] += 1
                    if impact == "HIGH":
                        sample_burden[s]["rare_high_nonref"] += 1
                    if impact == "MODERATE":
                        sample_burden[s]["rare_moderate_nonref"] += 1
                    if is_cancer_seed:
                        sample_burden[s]["rare_cancer_seed_nonref"] += 1

        rec = {
            "variant_id": f"{row['CHROM']}:{row['POS']}:{row['REF']}:{row['ALT']}",
            "CHROM": row["CHROM"],
            "POS": row["POS"],
            "ID": row["ID"],
            "REF": row["REF"],
            "ALT": row["ALT"],
            "overall_AC": total_ac,
            "overall_AN": total_an,
            "overall_AF": overall_af,
            "gene": gene,
            "effect": ann["effect"],
            "impact": impact,
            "hgvsc": ann["hgvsc"],
            "hgvsp": ann["hgvsp"],
            "is_rare_internal_AF_le_0.05_AC_le_3": is_rare,
            "is_functional_HIGH_MODERATE": is_functional,
            "is_cancer_seed_gene": is_cancer_seed,
            "hadza_AC": group_ac["Hadza"],
            "hadza_AN": group_an["Hadza"],
            "hadza_AF": group_ac["Hadza"] / group_an["Hadza"] if group_an["Hadza"] > 0 else math.nan,
            "hadza_carriers_n": len(group_carriers["Hadza"]),
            "hadza_carriers": ";".join(sorted(group_carriers["Hadza"])) if group_carriers["Hadza"] else ".",
            "control_AC": group_ac["Tanzanian control"],
            "control_AN": group_an["Tanzanian control"],
            "control_AF": group_ac["Tanzanian control"] / group_an["Tanzanian control"] if group_an["Tanzanian control"] > 0 else math.nan,
            "control_carriers_n": len(group_carriers["Tanzanian control"]),
            "control_carriers": ";".join(sorted(group_carriers["Tanzanian control"])) if group_carriers["Tanzanian control"] else ".",
            "all_carriers": ";".join(sorted(variant_carrier_samples)) if variant_carrier_samples else "."
        }

        records.append(rec)

        if is_rare and is_functional:
            rare_records.append(rec)

all_df = pd.DataFrame(records)
rare_df = pd.DataFrame(rare_records)

all_path = os.path.join(outdir, "local_wes_autosomal_biallelic_snps_snpeff_group_counts_all.tsv.gz")
rare_path = os.path.join(outdir, "local_wes_rare_functional_snps_group_counts.tsv")

all_df.to_csv(all_path, sep="\t", index=False, compression="gzip")
rare_df.to_csv(rare_path, sep="\t", index=False)

sample_df = pd.DataFrame(sample_burden.values())
sample_df = sample_df[sample_df["group_label"].isin(groups_used)].copy()
sample_path = os.path.join(outdir, "local_wes_sample_rare_functional_snp_burden.tsv")
sample_df.to_csv(sample_path, sep="\t", index=False)

if len(rare_df) > 0:
    gene_rows = []
    for gene, sub in rare_df.groupby("gene"):
        hadza_carriers = set()
        control_carriers = set()
        all_carriers = set()

        for x in sub["hadza_carriers"].dropna():
            if x != ".":
                hadza_carriers.update(x.split(";"))
                all_carriers.update(x.split(";"))

        for x in sub["control_carriers"].dropna():
            if x != ".":
                control_carriers.update(x.split(";"))
                all_carriers.update(x.split(";"))

        gene_rows.append({
            "gene": gene,
            "n_rare_functional_variants": len(sub),
            "n_HIGH": int((sub["impact"] == "HIGH").sum()),
            "n_MODERATE": int((sub["impact"] == "MODERATE").sum()),
            "hadza_unique_carriers": len(hadza_carriers),
            "control_unique_carriers": len(control_carriers),
            "total_unique_carriers": len(all_carriers),
            "is_cancer_seed_gene": gene in canonical_genes
        })

    gene_df = pd.DataFrame(gene_rows).sort_values(
        ["total_unique_carriers", "n_rare_functional_variants", "is_cancer_seed_gene", "gene"],
        ascending=[False, False, False, True]
    )
else:
    gene_df = pd.DataFrame(columns=[
        "gene", "n_rare_functional_variants", "n_HIGH", "n_MODERATE",
        "hadza_unique_carriers", "control_unique_carriers",
        "total_unique_carriers", "is_cancer_seed_gene"
    ])

gene_path = os.path.join(outdir, "local_wes_rare_functional_snp_gene_summary.tsv")
gene_df.to_csv(gene_path, sep="\t", index=False)

impact_summary = (
    rare_df.groupby(["impact"], dropna=False)
    .agg(
        n_variants=("variant_id", "count"),
        hadza_carrier_events=("hadza_carriers_n", "sum"),
        control_carrier_events=("control_carriers_n", "sum")
    )
    .reset_index()
    if len(rare_df) > 0 else
    pd.DataFrame(columns=["impact", "n_variants", "hadza_carrier_events", "control_carrier_events"])
)
impact_path = os.path.join(outdir, "local_wes_rare_functional_snp_impact_summary.tsv")
impact_summary.to_csv(impact_path, sep="\t", index=False)

summary_path = os.path.join(outdir, "local_wes_rare_functional_context_summary_counts.tsv")
summary = pd.DataFrame([
    {"metric": "total_autosomal_biallelic_snps", "value": n_total},
    {"metric": "rare_internal_AF_le_0.05_AC_le_3_snps", "value": n_rare},
    {"metric": "functional_HIGH_MODERATE_snps", "value": n_func},
    {"metric": "rare_functional_HIGH_MODERATE_snps", "value": n_rare_func},
    {"metric": "rare_functional_genes", "value": gene_df["gene"].nunique() if len(gene_df) else 0},
    {"metric": "rare_functional_cancer_seed_genes", "value": int(gene_df["is_cancer_seed_gene"].sum()) if len(gene_df) else 0}
])
summary.to_csv(summary_path, sep="\t", index=False)

report_path = os.path.join(status, "78C_local_germline_WES_rare_functional_context_status.txt")
with open(report_path, "w") as f:
    f.write("78C local germline WES rare functional context\n")
    f.write("Internal rarity definition: cohort AF <= 0.05 and AC <= 3\n")
    f.write("Functional definition: SnpEff HIGH or MODERATE impact\n\n")
    f.write("Summary counts:\n")
    f.write(summary.to_csv(sep="\t", index=False))
    f.write("\nTop rare functional genes:\n")
    f.write(gene_df.head(50).to_csv(sep="\t", index=False))
    f.write("\nSample burden summary:\n")
    f.write(sample_df.to_csv(sep="\t", index=False))

print("all_variants", all_path)
print("rare_functional_variants", rare_path)
print("sample_burden", sample_path)
print("gene_summary", gene_path)
print("impact_summary", impact_path)
print("summary_counts", summary_path)
print("report", report_path)
print("n_total", n_total)
print("n_rare", n_rare)
print("n_functional", n_func)
print("n_rare_functional", n_rare_func)
PY

echo
echo "=== BUILD FIGURES ==="

export OUT FIGS STATUS

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

group_cols <- c(
  "Hadza" = "#009E73",
  "Tanzanian control" = "#0072B2"
)

impact_cols <- c(
  "HIGH" = "#D62728",
  "MODERATE" = "#E69F00"
)

sample <- read_tsv(file.path(outdir, "local_wes_sample_rare_functional_snp_burden.tsv"), show_col_types = FALSE)
gene <- read_tsv(file.path(outdir, "local_wes_rare_functional_snp_gene_summary.tsv"), show_col_types = FALSE)
impact <- read_tsv(file.path(outdir, "local_wes_rare_functional_snp_impact_summary.tsv"), show_col_types = FALSE)
summary <- read_tsv(file.path(outdir, "local_wes_rare_functional_context_summary_counts.tsv"), show_col_types = FALSE)

p_sample <- ggplot(sample, aes(x = group_label, y = rare_functional_nonref, fill = group_label)) +
  geom_boxplot(width = 0.48, alpha = 0.62, outlier.shape = NA, color = "grey25") +
  geom_jitter(aes(shape = sex_label), width = 0.12, size = 2.8, alpha = 0.88, color = "grey15") +
  scale_fill_manual(values = group_cols) +
  scale_shape_manual(values = c("Female" = 16, "Male" = 17)) +
  labs(
    title = "Local germline rare functional SNP burden",
    subtitle = "Internal AF <= 0.05 and AC <= 3; SnpEff HIGH/MODERATE impact",
    x = NULL,
    y = "Rare functional non-reference SNP count",
    fill = "Group",
    shape = "Sex"
  ) +
  theme_pub(11)

ggsave(file.path(figs, "Fig_LocalWES_06_rare_functional_snp_burden_by_group.pdf"), p_sample, width = 6.8, height = 5.4)
ggsave(file.path(figs, "Fig_LocalWES_06_rare_functional_snp_burden_by_group.png"), p_sample, width = 6.8, height = 5.4, dpi = 600)

if (nrow(impact) > 0) {
  impact_long <- impact %>%
    pivot_longer(
      cols = c(hadza_carrier_events, control_carrier_events),
      names_to = "group",
      values_to = "carrier_events"
    ) %>%
    mutate(
      group = recode(
        group,
        "hadza_carrier_events" = "Hadza",
        "control_carrier_events" = "Tanzanian control"
      ),
      impact = factor(impact, levels = c("HIGH", "MODERATE"))
    )

  p_impact <- ggplot(impact_long, aes(x = impact, y = carrier_events, fill = group)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64, color = "white", linewidth = 0.25) +
    geom_text(
      aes(label = carrier_events),
      position = position_dodge(width = 0.72),
      vjust = -0.35,
      size = 3.6,
      fontface = "bold"
    ) +
    scale_fill_manual(values = group_cols) +
    labs(
      title = "Carrier-event distribution of rare functional SNPs",
      subtitle = "Carrier events are descriptive and not interpreted as association tests",
      x = "SnpEff impact",
      y = "Carrier events",
      fill = "Group"
    ) +
    theme_pub(11)

  ggsave(file.path(figs, "Fig_LocalWES_07_rare_functional_snp_impact_carrier_events.pdf"), p_impact, width = 6.8, height = 5.2)
  ggsave(file.path(figs, "Fig_LocalWES_07_rare_functional_snp_impact_carrier_events.png"), p_impact, width = 6.8, height = 5.2, dpi = 600)
}

top_gene <- gene %>%
  filter(gene != ".", total_unique_carriers > 0) %>%
  arrange(desc(total_unique_carriers), desc(n_rare_functional_variants), desc(is_cancer_seed_gene), gene) %>%
  slice_head(n = 25) %>%
  mutate(
    gene_label = fct_reorder(gene, total_unique_carriers),
    gene_class = ifelse(is_cancer_seed_gene, "Cancer-seed gene", "Other gene")
  )

if (nrow(top_gene) > 0) {
  p_gene <- ggplot(top_gene, aes(x = total_unique_carriers, y = gene_label)) +
    geom_segment(aes(x = 0, xend = total_unique_carriers, yend = gene_label, color = gene_class), linewidth = 1.0, alpha = 0.50) +
    geom_point(aes(size = n_rare_functional_variants, fill = gene_class), shape = 21, color = "grey15", stroke = 0.35, alpha = 0.94) +
    geom_text(aes(label = total_unique_carriers), nudge_x = 0.25, size = 3.0, fontface = "bold") +
    scale_color_manual(values = c("Cancer-seed gene" = "#D62728", "Other gene" = "#1F77B4")) +
    scale_fill_manual(values = c("Cancer-seed gene" = "#D62728", "Other gene" = "#1F77B4")) +
    scale_size_continuous(range = c(3.2, 8.5)) +
    labs(
      title = "Genes carrying rare functional germline SNPs",
      subtitle = "Local germline background context; label indicates unique carrier count",
      x = "Unique carriers across 31 germline-only samples",
      y = NULL,
      fill = "Gene class",
      color = "Gene class",
      size = "Rare functional SNPs"
    ) +
    theme_pub(10) +
    guides(color = "none")

  ggsave(file.path(figs, "Fig_LocalWES_08_top_rare_functional_snp_genes.pdf"), p_gene, width = 7.4, height = 7.0)
  ggsave(file.path(figs, "Fig_LocalWES_08_top_rare_functional_snp_genes.png"), p_gene, width = 7.4, height = 7.0, dpi = 600)
}

report <- file.path(status, "78C_local_germline_WES_rare_functional_context_figures_status.txt")
sink(report)
cat("78C local germline WES rare functional context figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Summary counts:\n")
print(summary)
cat("\nSample burden by group:\n")
print(
  sample %>%
    group_by(group_label) %>%
    summarise(
      n_samples = n(),
      median_rare_functional = median(rare_functional_nonref, na.rm = TRUE),
      mean_rare_functional = mean(rare_functional_nonref, na.rm = TRUE),
      min_rare_functional = min(rare_functional_nonref, na.rm = TRUE),
      max_rare_functional = max(rare_functional_nonref, na.rm = TRUE),
      .groups = "drop"
    )
)
cat("\nTop genes:\n")
print(top_gene)
sink()
RS

echo
echo "=== 78C SUMMARY COUNTS ==="
column -t -s $'\t' "$OUT/local_wes_rare_functional_context_summary_counts.tsv"

echo
echo "=== 78C TOP RARE FUNCTIONAL GENES ==="
column -t -s $'\t' "$OUT/local_wes_rare_functional_snp_gene_summary.tsv" | head -40

echo
echo "=== LOCAL WES RARE FUNCTIONAL FIGURES ==="
find "$FIGS" -maxdepth 1 -type f \( -name "Fig_LocalWES_06*" -o -name "Fig_LocalWES_07*" -o -name "Fig_LocalWES_08*" \) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/78C_local_germline_WES_rare_functional_context_figures_status.txt" | head -120

echo
echo "=== DONE 78C ==="
date
