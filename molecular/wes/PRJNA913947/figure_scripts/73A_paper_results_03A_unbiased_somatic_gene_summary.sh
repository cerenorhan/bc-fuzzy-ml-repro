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


PROJECT="${PROJECT:-$PROJECT_ROOT}"
PAPER="$PAPER_RESULTS_ROOT"

SOMATIC="$PAPER/03_somatic_wes_PRJNA913947"
DRIVER="$SOMATIC/05_driver_tables"
OUTDIR="$SOMATIC/06_unbiased_gene_summary"
FIGS="$SOMATIC/02_figures"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUTDIR" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/73A_paper_results_03A_unbiased_somatic_gene_summary_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 03A: UNBIASED SOMATIC GENE-LEVEL SUMMARY ==="
date

ANNOT="$DRIVER/PRJNA913947_somatic_snpeff_annotated_all.tsv.gz"

if [[ ! -s "$ANNOT" ]]; then
  echo "ERROR: Missing SnpEff annotated table:"
  echo "$ANNOT"
  exit 1
fi

echo
echo "Input:"
ls -lh "$ANNOT"

export ANNOT OUTDIR FIGS STATUS

micromamba run -n cancer_anno python - <<'PY'
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

annot = os.environ["ANNOT"]
outdir = os.environ["OUTDIR"]
figs = os.environ["FIGS"]
status = os.environ["STATUS"]

os.makedirs(outdir, exist_ok=True)
os.makedirs(figs, exist_ok=True)

df = pd.read_csv(annot, sep="\t", dtype=str).fillna(".")

required = ["row_id", "pair", "gene", "Annotation", "Annotation_Impact", "impact_class", "variant_type"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

df["gene"] = df["gene"].astype(str).replace("", ".")
df["pair"] = df["pair"].astype(str)

for c in ["tumor_AF", "tumor_DP", "normal_AF", "normal_DP"]:
    if c in df.columns:
        df[c + "_num"] = pd.to_numeric(df[c], errors="coerce")
    else:
        df[c + "_num"] = pd.NA

df["is_high"] = df["Annotation_Impact"].eq("HIGH")
df["is_moderate"] = df["Annotation_Impact"].eq("MODERATE")
df["is_low"] = df["Annotation_Impact"].eq("LOW")
df["is_modifier"] = df["Annotation_Impact"].eq("MODIFIER")
df["is_low_splice_region"] = df["impact_class"].eq("LOW_SPLICE_REGION")

df["is_functional_strict"] = df["Annotation_Impact"].isin(["HIGH", "MODERATE"])
df["is_functional_extended"] = df["is_functional_strict"] | df["is_low_splice_region"]

def caution_reason_symbol(gene):
    gene = str(gene)
    reasons = []

    if gene in ["", ".", "nan", "None"]:
        reasons.append("missing_gene_symbol")

    if "-" in gene:
        reasons.append("multi_gene_or_interval_annotation")

    if re.match(r"^(AC|AL|AP|LINC|MIR|SNORD|SNORA|RN7|RNU|IGH|IGK|IGL)", gene):
        reasons.append("noncoding_or_immunoglobulin_like_prefix")

    if re.search(r"P\d*$", gene) and gene not in {"TP53", "PTEN"}:
        reasons.append("possible_pseudogene_symbol")

    return ";".join(reasons) if reasons else "."

df["caution_reason_symbol"] = df["gene"].apply(caution_reason_symbol)
df["has_symbol_caution"] = df["caution_reason_symbol"].ne(".")

work = df[df["gene"].ne(".")].copy()

base = (
    work.groupby("gene")
    .agg(
        n_pairs_any=("pair", lambda x: len(set(x))),
        n_variants_any=("row_id", "count"),
        median_tumor_AF=("tumor_AF_num", "median"),
        median_tumor_DP=("tumor_DP_num", "median"),
        median_normal_AF=("normal_AF_num", "median"),
        median_normal_DP=("normal_DP_num", "median"),
        pairs_any=("pair", lambda x: ",".join(sorted(set(x)))),
        caution_reason_symbol=("caution_reason_symbol", lambda x: ";".join(sorted(set([v for v in x if v != "."]))) if any(v != "." for v in x) else "."),
    )
    .reset_index()
)

def flag_stats(flag_col, prefix):
    sub = work[work[flag_col]].copy()
    if sub.empty:
        return pd.DataFrame({"gene": [], f"n_pairs_{prefix}": [], f"n_{prefix}": []})
    return (
        sub.groupby("gene")
        .agg(
            **{
                f"n_pairs_{prefix}": ("pair", lambda x: len(set(x))),
                f"n_{prefix}": ("row_id", "count"),
            }
        )
        .reset_index()
    )

for flag_col, prefix in [
    ("is_high", "high"),
    ("is_moderate", "moderate"),
    ("is_low", "low"),
    ("is_modifier", "modifier"),
    ("is_low_splice_region", "low_splice_region"),
    ("is_functional_strict", "functional_strict"),
    ("is_functional_extended", "functional_extended"),
]:
    base = base.merge(flag_stats(flag_col, prefix), on="gene", how="left")

count_cols = [c for c in base.columns if c.startswith("n_")]
for c in count_cols:
    base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0).astype(int)

gene_summary = base.copy()

gene_summary["has_symbol_caution"] = gene_summary["caution_reason_symbol"].ne(".")

gene_summary["caution_reason_data"] = "."
gene_summary.loc[
    (gene_summary["n_pairs_any"] >= 15) & (gene_summary["n_functional_extended"] == 0),
    "caution_reason_data"
] = "high_recurrence_without_functional_signal"

gene_summary.loc[
    (gene_summary["n_variants_any"] >= 75) & (gene_summary["n_functional_extended"] == 0),
    "caution_reason_data"
] = gene_summary["caution_reason_data"].apply(
    lambda x: "high_variant_count_without_functional_signal" if x == "." else x + ";high_variant_count_without_functional_signal"
)

def merge_cautions(row):
    vals = []
    for c in ["caution_reason_symbol", "caution_reason_data"]:
        v = str(row[c])
        if v != ".":
            vals.extend(v.split(";"))
    vals = sorted(set(vals))
    return ";".join(vals) if vals else "."

gene_summary["caution_reason"] = gene_summary.apply(merge_cautions, axis=1)
gene_summary["has_caution_flag"] = gene_summary["caution_reason"].ne(".")

gene_summary["unbiased_somatic_gene_score"] = (
    gene_summary["n_pairs_functional_extended"] * 3
    + gene_summary["n_pairs_high"] * 4
    + gene_summary["n_pairs_moderate"] * 2
    + gene_summary["n_high"] * 2
    + gene_summary["n_moderate"] * 1
)

gene_summary["unbiased_somatic_gene_score_caution_adjusted"] = gene_summary["unbiased_somatic_gene_score"]
gene_summary.loc[gene_summary["has_caution_flag"], "unbiased_somatic_gene_score_caution_adjusted"] -= 5

gene_summary = gene_summary.sort_values(
    [
        "unbiased_somatic_gene_score_caution_adjusted",
        "n_pairs_functional_extended",
        "n_pairs_high",
        "n_pairs_moderate",
        "n_functional_extended",
        "gene"
    ],
    ascending=[False, False, False, False, False, True]
)

all_out = os.path.join(outdir, "PRJNA913947_unbiased_somatic_gene_summary_all.tsv")
functional_out = os.path.join(outdir, "PRJNA913947_unbiased_functional_gene_summary.tsv")
strict_out = os.path.join(outdir, "PRJNA913947_unbiased_strict_HIGH_MODERATE_gene_summary.tsv")
caution_out = os.path.join(outdir, "PRJNA913947_unbiased_gene_caution_flags_no_removal.tsv")
top_no_caution_out = os.path.join(outdir, "PRJNA913947_unbiased_functional_gene_summary_no_caution_subset.tsv")

functional_genes = gene_summary[gene_summary["n_functional_extended"] > 0].copy()
strict_functional_genes = gene_summary[gene_summary["n_functional_strict"] > 0].copy()
top_no_caution = functional_genes[~functional_genes["has_caution_flag"]].copy()

gene_summary.to_csv(all_out, sep="\t", index=False)
functional_genes.to_csv(functional_out, sep="\t", index=False)
strict_functional_genes.to_csv(strict_out, sep="\t", index=False)
gene_summary[gene_summary["has_caution_flag"]].to_csv(caution_out, sep="\t", index=False)
top_no_caution.to_csv(top_no_caution_out, sep="\t", index=False)

def unique_gene_count_for_flag(sub_df, flag_col):
    return len(set(sub_df.loc[sub_df[flag_col], "gene"]) - {"."})

pair_rows = []
for pair, sub in df.groupby("pair"):
    pair_rows.append({
        "pair": pair,
        "n_variants_any": len(sub),
        "n_genes_any": len(set(sub["gene"]) - {"."}),
        "n_high": int(sub["is_high"].sum()),
        "n_moderate": int(sub["is_moderate"].sum()),
        "n_low": int(sub["is_low"].sum()),
        "n_modifier": int(sub["is_modifier"].sum()),
        "n_low_splice_region": int(sub["is_low_splice_region"].sum()),
        "n_functional_strict": int(sub["is_functional_strict"].sum()),
        "n_functional_extended": int(sub["is_functional_extended"].sum()),
        "n_functional_strict_genes": unique_gene_count_for_flag(sub, "is_functional_strict"),
        "n_functional_extended_genes": unique_gene_count_for_flag(sub, "is_functional_extended"),
        "median_tumor_AF": sub["tumor_AF_num"].median(),
        "median_tumor_DP": sub["tumor_DP_num"].median(),
    })

pair_burden = pd.DataFrame(pair_rows).sort_values("pair")
pair_out = os.path.join(outdir, "PRJNA913947_unbiased_pair_level_gene_burden.tsv")
pair_burden.to_csv(pair_out, sep="\t", index=False)

impact_pair = (
    df.pivot_table(index="pair", columns="Annotation_Impact", values="row_id", aggfunc="count", fill_value=0)
    .reset_index()
)
impact_pair_out = os.path.join(outdir, "PRJNA913947_unbiased_pair_by_impact_matrix.tsv")
impact_pair.to_csv(impact_pair_out, sep="\t", index=False)

func_gene_pair = (
    df[df["is_functional_extended"] & df["gene"].ne(".")]
    .groupby(["gene", "pair"])
    .agg(
        n_functional_variants=("row_id", "count"),
        n_high=("is_high", "sum"),
        n_moderate=("is_moderate", "sum"),
        n_low_splice_region=("is_low_splice_region", "sum"),
        median_tumor_AF=("tumor_AF_num", "median"),
        median_tumor_DP=("tumor_DP_num", "median"),
    )
    .reset_index()
)

func_gene_pair_out = os.path.join(outdir, "PRJNA913947_unbiased_functional_gene_by_pair_long.tsv")
func_gene_pair.to_csv(func_gene_pair_out, sep="\t", index=False)

universe_genes = sorted(set(gene_summary["gene"]) - {"."})
query_functional_extended = sorted(set(functional_genes["gene"]) - {"."})
query_strict = sorted(set(strict_functional_genes["gene"]) - {"."})
query_no_caution = sorted(set(top_no_caution["gene"]) - {"."})

def write_list(path, genes):
    with open(path, "w") as f:
        for g in genes:
            f.write(g + "\n")

universe_out = os.path.join(outdir, "PRJNA913947_somatic_background_all_annotated_genes.txt")
query_ext_out = os.path.join(outdir, "PRJNA913947_somatic_query_functional_extended_genes.txt")
query_strict_out = os.path.join(outdir, "PRJNA913947_somatic_query_strict_HIGH_MODERATE_genes.txt")
query_no_caution_out = os.path.join(outdir, "PRJNA913947_somatic_query_functional_extended_no_caution_genes.txt")

write_list(universe_out, universe_genes)
write_list(query_ext_out, query_functional_extended)
write_list(query_strict_out, query_strict)
write_list(query_no_caution_out, query_no_caution)

summary_counts = pd.DataFrame([
    ["annotated_variants", len(df)],
    ["unique_annotated_genes", len(universe_genes)],
    ["functional_extended_variants", int(df["is_functional_extended"].sum())],
    ["strict_HIGH_MODERATE_variants", int(df["is_functional_strict"].sum())],
    ["functional_extended_genes", len(query_functional_extended)],
    ["strict_HIGH_MODERATE_genes", len(query_strict)],
    ["functional_extended_no_caution_genes", len(query_no_caution)],
    ["caution_flagged_genes", int(gene_summary["has_caution_flag"].sum())],
], columns=["metric", "value"])

summary_counts_out = os.path.join(outdir, "PRJNA913947_unbiased_somatic_gene_summary_counts.tsv")
summary_counts.to_csv(summary_counts_out, sep="\t", index=False)

top = functional_genes.head(30).sort_values("unbiased_somatic_gene_score_caution_adjusted")
plt.figure(figsize=(9, max(5, len(top) * 0.28)))
plt.barh(top["gene"], top["unbiased_somatic_gene_score_caution_adjusted"])
plt.xlabel("Unbiased somatic gene score, caution-adjusted")
plt.ylabel("Gene")
plt.title("Top unbiased functional somatic genes")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_03A_top_unbiased_functional_somatic_genes.png"), dpi=300)
plt.close()

top_nc = top_no_caution.head(30).sort_values("unbiased_somatic_gene_score_caution_adjusted")
if len(top_nc) > 0:
    plt.figure(figsize=(9, max(5, len(top_nc) * 0.28)))
    plt.barh(top_nc["gene"], top_nc["unbiased_somatic_gene_score_caution_adjusted"])
    plt.xlabel("Unbiased somatic gene score, caution-adjusted")
    plt.ylabel("Gene")
    plt.title("Top unbiased functional somatic genes without caution flags")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_03A_top_unbiased_functional_somatic_genes_no_caution.png"), dpi=300)
    plt.close()

pb = pair_burden.copy()
pb["pair_short"] = pb["pair"].str.replace("candidate_kenya_pair_", "P", regex=False)

plt.figure(figsize=(12, 5))
plt.bar(pb["pair_short"], pb["n_functional_extended"])
plt.xlabel("Pair")
plt.ylabel("Functional extended variant count")
plt.title("Pair-level functional somatic burden")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_03A_pair_functional_variant_burden.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(pb["pair_short"], pb["n_functional_extended_genes"])
plt.xlabel("Pair")
plt.ylabel("Functional altered gene count")
plt.title("Pair-level functional altered gene burden")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_03A_pair_functional_gene_burden.png"), dpi=300)
plt.close()

impact_counts = df.groupby("Annotation_Impact").size().reset_index(name="n_variants").sort_values("n_variants", ascending=False)
plt.figure(figsize=(7, 5))
plt.bar(impact_counts["Annotation_Impact"], impact_counts["n_variants"])
plt.xlabel("SnpEff impact")
plt.ylabel("Variant count")
plt.title("Somatic variant impact distribution")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_03A_snpeff_impact_distribution.png"), dpi=300)
plt.close()

report = os.path.join(status, "PRJNA913947_stage_03A_unbiased_somatic_gene_summary_status.txt")
with open(report, "w") as f:
    f.write("PRJNA913947 Stage 03A unbiased somatic gene-level summary\n")
    f.write(f"Generated: {pd.Timestamp.now()}\n\n")
    f.write("Summary counts:\n")
    f.write(summary_counts.to_csv(sep="\t", index=False))
    f.write("\nTop unbiased functional genes:\n")
    f.write(functional_genes.head(50).to_csv(sep="\t", index=False))
    f.write("\nTop unbiased functional genes without caution flags:\n")
    f.write(top_no_caution.head(50).to_csv(sep="\t", index=False))
    f.write("\nPair-level burden:\n")
    f.write(pair_burden.to_csv(sep="\t", index=False))
    f.write("\nCaution flags are reporting flags only; no genes were removed from the full tables.\n")

print("summary_counts", summary_counts_out)
print("gene_summary_all", all_out)
print("functional_gene_summary", functional_out)
print("strict_HIGH_MODERATE_gene_summary", strict_out)
print("caution_flags_no_removal", caution_out)
print("functional_no_caution_subset", top_no_caution_out)
print("pair_level_gene_burden", pair_out)
print("pair_by_impact_matrix", impact_pair_out)
print("functional_gene_by_pair_long", func_gene_pair_out)
print("background_genes_for_enrichment", universe_out)
print("query_functional_extended_genes", query_ext_out)
print("query_strict_HIGH_MODERATE_genes", query_strict_out)
print("query_functional_extended_no_caution_genes", query_no_caution_out)
print("report", report)
PY

echo
echo "=== SUMMARY COUNTS ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_unbiased_somatic_gene_summary_counts.tsv"

echo
echo "=== TOP UNBIASED FUNCTIONAL GENES ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_unbiased_functional_gene_summary.tsv" | head -40

echo
echo "=== TOP UNBIASED FUNCTIONAL GENES WITHOUT CAUTION FLAGS ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_unbiased_functional_gene_summary_no_caution_subset.tsv" | head -40

echo
echo "=== PAIR-LEVEL GENE BURDEN ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_unbiased_pair_level_gene_burden.tsv"

echo
echo "=== OUTPUT FILES ==="
find "$OUTDIR" -maxdepth 1 -type f -printf "%f\t%k KB\n" | sort

echo
echo "=== FIGURES ==="
find "$FIGS" -maxdepth 1 -type f -name "PRJNA913947_03A_*png" -printf "%f\t%k KB\n" | sort

echo
echo "=== DONE: STAGE 03A ==="
date
