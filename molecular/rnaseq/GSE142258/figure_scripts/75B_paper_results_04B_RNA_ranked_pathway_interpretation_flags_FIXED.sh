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

RNA_OUT="$PAPER/04_rnaseq_GSE142258"
RNA_GSEA="$RNA_OUT/02_ranked_pathway_enrichment_noinstall"
OUTDIR="$RNA_OUT/04_interpretable_ranked_pathway_summary"
FIGS="$RNA_OUT/03_figures"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUTDIR" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/75B_paper_results_04B_RNA_ranked_pathway_interpretation_flags_FIXED_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 04B-FIXED: RNA RANKED PATHWAY INTERPRETATION FLAGS ==="
date

IN="$RNA_GSEA/GSE142258_RNA_ranked_pathway_enrichment_all_results.tsv"
LEAD="$RNA_GSEA/GSE142258_RNA_ranked_pathway_leading_edge_genes_long.tsv"

if [[ ! -s "$IN" ]]; then
  echo "ERROR: Missing RNA ranked pathway results:"
  echo "$IN"
  exit 1
fi

export IN LEAD OUTDIR FIGS STATUS

micromamba run -n rnaseq python - <<'PY'
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

inp = os.environ["IN"]
lead = os.environ["LEAD"]
outdir = os.environ["OUTDIR"]
figs = os.environ["FIGS"]
status = os.environ["STATUS"]

os.makedirs(outdir, exist_ok=True)
os.makedirs(figs, exist_ok=True)

df = pd.read_csv(inp, sep="\t")

for c in ["padj", "pvalue", "rank_shift_z"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "direction" not in df.columns:
    df["direction"] = df["rank_shift_z"].apply(lambda x: "up_in_late" if x >= 0 else "down_in_late")

df["term_upper"] = df["gs_name"].astype(str).str.upper()

def axis(term):
    t = str(term).upper()
    rules = [
        ("EMT_invasion_angiogenesis", ["EPITHELIAL_MESENCHYMAL", "ANGIOGENESIS", "EXTRACELLULAR", "COLLAGEN", "MIGRATION", "INVASION", "COAGULATION"]),
        ("immune_antigen_T_NK", ["ANTIGEN", "T_CELL", "B_CELL", "NK", "NATURAL_KILLER", "ALLOGRAFT", "INTERFERON", "INFLAMMATORY", "CYTOKINE", "LEUKOCYTE", "IMMUNE"]),
        ("translation_ribosome_RNA", ["TRANSLATION", "RIBOSOME", "RIBONUCLEOPROTEIN", "RRNA", "RNA_PROCESSING"]),
        ("cell_cycle_DNA_repair", ["CELL_CYCLE", "MITOTIC", "CHECKPOINT", "DNA_REPAIR", "DOUBLE_STRAND", "BRCA", "P53", "E2F", "G2M"]),
        ("metabolism_hypoxia_lipid", ["HYPOXIA", "CHOLESTEROL", "FATTY_ACID", "ADIPOGENESIS", "HEME", "GLYCOLYSIS", "OXIDATIVE", "MTORC1", "PI3K"]),
        ("hormone_luminal", ["ESTROGEN", "ANDROGEN", "APICAL"]),
        ("development_Notch_TGF_WNT", ["NOTCH", "TGF", "WNT", "HEDGEHOG"]),
    ]
    out = []
    for name, keys in rules:
        if any(k in t for k in keys):
            out.append(name)
    return ";".join(sorted(set(out))) if out else "other"

def caution(term):
    t = str(term).upper()
    caution_patterns = [
        "OLFACTORY", "SMELL", "SENSORY", "TASTE",
        "SPERM", "OOCYTE", "GAMETE",
        "SYNCYTIUM", "MYOBLAST", "MYOGENESIS",
        "VIRAL", "AUTOIMMUNE", "GRAFT_VERSUS_HOST",
        "MEDICUS_VARIANT", "MEDICUS_PATHOGEN"
    ]
    hits = [x for x in caution_patterns if x in t]
    return ";".join(hits) if hits else "."

df["biology_axis"] = df["gs_name"].apply(axis)
df["term_caution_reason"] = df["gs_name"].apply(caution)
df["has_term_caution"] = df["term_caution_reason"].ne(".")

df["reporting_tier"] = "exploratory_all"
df.loc[(df["padj"] <= 0.25) & (~df["has_term_caution"]), "reporting_tier"] = "interpretable_FDR025"
df.loc[(df["padj"] > 0.25) & (df["pvalue"] <= 0.05) & (~df["has_term_caution"]), "reporting_tier"] = "nominal_interpretable"
df.loc[df["has_term_caution"], "reporting_tier"] = "caution_term_retained_not_primary"

all_out = os.path.join(outdir, "GSE142258_RNA_ranked_pathway_all_results_with_interpretation_flags.tsv")
df.to_csv(all_out, sep="\t", index=False)

primary = df[df["reporting_tier"].eq("interpretable_FDR025")].copy()
primary = primary.sort_values(["direction", "padj", "pvalue"], ascending=[True, True, True])

primary_out = os.path.join(outdir, "GSE142258_RNA_ranked_pathway_primary_interpretable_FDR025.tsv")
primary.to_csv(primary_out, sep="\t", index=False)

paper = df[df["reporting_tier"].isin(["interpretable_FDR025", "nominal_interpretable"])].copy()
paper = paper.sort_values(["reporting_tier", "direction", "padj", "pvalue"], ascending=[True, True, True, True])

paper_out = os.path.join(outdir, "GSE142258_RNA_ranked_pathway_paper_friendly_top_terms.tsv")
paper.to_csv(paper_out, sep="\t", index=False)

caution_df = df[df["has_term_caution"]].copy()
caution_out = os.path.join(outdir, "GSE142258_RNA_ranked_pathway_caution_terms_retained.tsv")
caution_df.to_csv(caution_out, sep="\t", index=False)

axis_summary = (
    df[df["reporting_tier"].isin(["interpretable_FDR025", "nominal_interpretable"])]
    .groupby(["biology_axis", "direction", "reporting_tier"], dropna=False)
    .agg(
        n_terms=("gs_name", "count"),
        best_padj=("padj", "min"),
        best_pvalue=("pvalue", "min"),
        top_terms=("gs_name", lambda x: ",".join(list(x.head(12))))
    )
    .reset_index()
    .sort_values(["reporting_tier", "direction", "best_padj", "best_pvalue"])
)

axis_out = os.path.join(outdir, "GSE142258_RNA_ranked_pathway_biology_axis_summary.tsv")
axis_summary.to_csv(axis_out, sep="\t", index=False)

# Leading-edge gene summary, robust to direction_x/direction_y/padj.x/padj.y column suffixes
if os.path.exists(lead) and os.path.getsize(lead) > 0:
    le = pd.read_csv(lead, sep="\t")

    # normalize pathway direction
    if "pathway_direction" not in le.columns:
        if "direction" in le.columns:
            le["pathway_direction"] = le["direction"]
        elif "direction.x" in le.columns:
            le["pathway_direction"] = le["direction.x"]
        elif "direction_x" in le.columns:
            le["pathway_direction"] = le["direction_x"]
        elif "direction_DE" in le.columns:
            le["pathway_direction"] = le["direction_DE"]
        else:
            le["pathway_direction"] = "."

    # normalize pathway padj
    if "pathway_padj" not in le.columns:
        if "padj.x" in le.columns:
            le["pathway_padj"] = pd.to_numeric(le["padj.x"], errors="coerce")
        elif "padj_x" in le.columns:
            le["pathway_padj"] = pd.to_numeric(le["padj_x"], errors="coerce")
        elif "padj" in le.columns:
            le["pathway_padj"] = pd.to_numeric(le["padj"], errors="coerce")
        else:
            le["pathway_padj"] = pd.NA

    annot = df[["database", "gs_name", "biology_axis", "reporting_tier"]].drop_duplicates()
    le2 = le.merge(annot, on=["database", "gs_name"], how="left")

    le2 = le2[le2["reporting_tier"].isin(["interpretable_FDR025", "nominal_interpretable"])].copy()

    le_out = os.path.join(outdir, "GSE142258_RNA_ranked_pathway_interpretable_leading_edge_genes_long.tsv")
    le2.to_csv(le_out, sep="\t", index=False)

    if "gene_symbol" in le2.columns and len(le2) > 0:
        agg_dict = {
            "n_terms": ("gs_name", "nunique"),
            "best_pathway_padj": ("pathway_padj", "min")
        }

        if "log2FoldChange" in le2.columns:
            le2["log2FoldChange"] = pd.to_numeric(le2["log2FoldChange"], errors="coerce")
            agg_dict["median_log2FC"] = ("log2FoldChange", "median")

        gene_axis = (
            le2.groupby(["gene_symbol", "biology_axis", "pathway_direction"], dropna=False)
            .agg(**agg_dict)
            .reset_index()
            .sort_values(["n_terms", "best_pathway_padj"], ascending=[False, True])
        )

        gene_axis.to_csv(
            os.path.join(outdir, "GSE142258_RNA_interpretable_leading_edge_gene_axis_summary.tsv"),
            sep="\t",
            index=False
        )

# Figures
plot_df = primary.copy()
if len(plot_df) > 0:
    for direction in sorted(plot_df["direction"].dropna().unique()):
        dat = plot_df[plot_df["direction"] == direction].copy()
        dat = dat.sort_values(["padj", "pvalue"]).head(25).iloc[::-1]
        dat["term_short"] = (
            dat["gs_name"].astype(str)
            .str.replace("^HALLMARK_", "", regex=True)
            .str.replace("^REACTOME_", "", regex=True)
            .str.replace("^KEGG_", "", regex=True)
            .str.replace("^GOBP_", "", regex=True)
            .str.replace("_", " ", regex=False)
            .str.title()
        )
        plt.figure(figsize=(9, max(5, len(dat) * 0.32)))
        plt.barh(dat["term_short"], [-math.log10(x + 1e-300) for x in dat["padj"]])
        plt.xlabel("-log10 FDR")
        plt.ylabel("")
        plt.title(f"RNA ranked pathway interpretable terms: {direction}")
        plt.tight_layout()
        plt.savefig(os.path.join(figs, f"GSE142258_04B_RNA_interpretable_ranked_pathway_{direction}.png"), dpi=300)
        plt.close()

axis_plot = axis_summary[axis_summary["reporting_tier"].eq("interpretable_FDR025")].copy()
if len(axis_plot) > 0:
    collapsed = (
        axis_plot.groupby(["biology_axis", "direction"])
        .agg(n_terms=("n_terms", "sum"), best_padj=("best_padj", "min"))
        .reset_index()
        .sort_values("n_terms")
    )
    collapsed["label"] = collapsed["biology_axis"] + " / " + collapsed["direction"]
    plt.figure(figsize=(9, max(4, len(collapsed) * 0.32)))
    plt.barh(collapsed["label"], collapsed["n_terms"])
    plt.xlabel("Number of interpretable FDR≤0.25 RNA ranked terms")
    plt.ylabel("")
    plt.title("RNA pathway axes")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "GSE142258_04B_RNA_ranked_pathway_axis_summary.png"), dpi=300)
    plt.close()

report = os.path.join(status, "GSE142258_stage_04B_RNA_ranked_pathway_interpretation_flags_FIXED_status.txt")
with open(report, "w") as f:
    f.write("GSE142258 Stage 04B RNA ranked pathway interpretation flags FIXED\n\n")
    f.write(f"n_all_terms\t{len(df)}\n")
    f.write(f"n_interpretable_FDR025\t{len(primary)}\n")
    f.write(f"n_nominal_interpretable\t{len(df[df['reporting_tier'].eq('nominal_interpretable')])}\n")
    f.write(f"n_caution_terms_retained\t{len(caution_df)}\n\n")
    f.write("Biology axis summary:\n")
    f.write(axis_summary.to_csv(sep="\t", index=False))
    f.write("\nTop paper-friendly RNA terms:\n")
    f.write(paper.head(120).to_csv(sep="\t", index=False))

print("all_flagged", all_out)
print("primary_interpretable_FDR025", primary_out)
print("paper_friendly_top_terms", paper_out)
print("caution_terms_retained", caution_out)
print("biology_axis_summary", axis_out)
print("report", report)
print("n_all_terms", len(df))
print("n_interpretable_FDR025", len(primary))
print("n_nominal_interpretable", len(df[df["reporting_tier"].eq("nominal_interpretable")]))
print("n_caution_terms_retained", len(caution_df))
PY

echo
echo "=== RNA STAGE 04B FIXED SUMMARY ==="
grep -E "n_all_terms|n_interpretable_FDR025|n_nominal_interpretable|n_caution_terms_retained" "$STATUS/GSE142258_stage_04B_RNA_ranked_pathway_interpretation_flags_FIXED_status.txt"

echo
echo "=== RNA BIOLOGY AXIS SUMMARY ==="
column -t -s $'\t' "$OUTDIR/GSE142258_RNA_ranked_pathway_biology_axis_summary.tsv" | head -100

echo
echo "=== RNA PAPER-FRIENDLY TOP TERMS ==="
column -t -s $'\t' "$OUTDIR/GSE142258_RNA_ranked_pathway_paper_friendly_top_terms.tsv" | head -100

echo
echo "=== OUTPUT FILES ==="
find "$OUTDIR" -maxdepth 1 -type f -printf "%f\t%k KB\n" | sort

echo
echo "=== DONE: STAGE 04B-FIXED ==="
date
