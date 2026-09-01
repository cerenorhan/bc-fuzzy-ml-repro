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
ORA="$SOMATIC/07_database_pathway_enrichment"
OUTDIR="$SOMATIC/08_interpretable_somatic_pathway_summary"
FIGS="$SOMATIC/02_figures"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUTDIR" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/73C_paper_results_03C_somatic_ORA_interpretation_flags_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 03C: SOMATIC ORA INTERPRETATION FLAGS ==="
date

ALL_ORA="$ORA/PRJNA913947_somatic_database_ORA_all_results.tsv"
TOP_ORA="$ORA/PRJNA913947_somatic_database_ORA_top25_per_query_database.tsv"
FDR_ORA="$ORA/PRJNA913947_somatic_database_ORA_FDR025.tsv"

for f in "$ALL_ORA" "$TOP_ORA"
do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: Missing ORA file: $f"
    exit 1
  fi
done

export ALL_ORA TOP_ORA FDR_ORA OUTDIR FIGS STATUS

micromamba run -n rnaseq python - <<'PY'
import os
import re
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

all_ora = os.environ["ALL_ORA"]
top_ora = os.environ["TOP_ORA"]
fdr_ora = os.environ["FDR_ORA"]
outdir = os.environ["OUTDIR"]
figs = os.environ["FIGS"]
status = os.environ["STATUS"]

os.makedirs(outdir, exist_ok=True)
os.makedirs(figs, exist_ok=True)

df = pd.read_csv(all_ora, sep="\t")
df["term_upper"] = df["gs_name"].astype(str).str.upper()

def term_caution(term):
    t = str(term).upper()
    caution_patterns = [
        "OLFACTORY", "SMELL", "SENSORY", "TASTE",
        "OOCYTE", "SPERM", "SPERMATOGENESIS", "GAMETE",
        "REPRODUCTIVE", "REPRODUCTION",
        "NEURON", "SYNAPSE", "AXON", "DENDRITE",
        "VIRAL INFECTION", "VIRUS", "MEDICUS_VARIANT", "MEDICUS_PATHOGEN"
    ]
    hits = [p for p in caution_patterns if p in t]
    return ";".join(hits) if hits else "."

def biology_axis(term):
    t = str(term).upper()

    axes = []
    rules = [
        ("chromatin_epigenetic", ["CHROMATIN", "EPIGENETIC", "HISTONE", "NUCLEOSOME"]),
        ("DNA_repair_damage", ["DNA_REPAIR", "DNA DAMAGE", "REPLICATION MAINTENANCE OF FIDELITY", "DOUBLE STRAND", "MISMATCH", "RPA", "END RESECTION"]),
        ("cell_cycle_mitotic", ["CELL_CYCLE", "CELL CYCLE", "MITOTIC", "SPINDLE", "G2M", "E2F", "CHECKPOINT", "P53", "P300_P21"]),
        ("hormone_luminal", ["ESTROGEN", "PROGESTERONE", "OOCYTE"]),
        ("PI3K_AKT_mTOR", ["PI3K", "AKT", "MTOR", "MTORC1"]),
        ("adhesion_ECM", ["ADHESION", "JUNCTION", "CAM", "EXTRACELLULAR", "ECM"]),
        ("immune_inflammation", ["IMMUNE", "LEUKOCYTE", "INTERLEUKIN", "IL2", "STAT5", "COMPLEMENT", "CYTOTOXICITY", "TNF", "INTERFERON", "VIRAL"]),
        ("TGF_WNT_NOTCH", ["TGF", "WNT", "BETA_CATENIN", "NOTCH", "HEDGEHOG"]),
        ("metabolism_stress", ["METABOLISM", "HYPOXIA", "ROS", "REACTIVE_OXYGEN", "UNFOLDED", "FATTY_ACID", "HEME"]),
        ("cytoskeleton_transport", ["MICROTUBULE", "CYTOSKELETON", "DYNEIN", "TRANSPORT", "KINETOCHORE"]),
    ]

    for axis, keys in rules:
        if any(k in t for k in keys):
            axes.append(axis)

    return ";".join(sorted(set(axes))) if axes else "other"

df["term_caution_reason"] = df["gs_name"].apply(term_caution)
df["has_term_caution"] = df["term_caution_reason"].ne(".")
df["biology_axis"] = df["gs_name"].apply(biology_axis)

df["reporting_tier"] = "exploratory_all"
df.loc[
    (df["qvalue_BH_within_database"] <= 0.25) &
    (df["overlap_n"] >= 2) &
    (~df["has_term_caution"]),
    "reporting_tier"
] = "interpretable_FDR025"

df.loc[
    (df["pvalue"] <= 0.05) &
    (df["qvalue_BH_within_database"] > 0.25) &
    (~df["has_term_caution"]),
    "reporting_tier"
] = "nominal_interpretable"

df.loc[df["has_term_caution"], "reporting_tier"] = "caution_term_retained_not_primary"

all_flagged = os.path.join(outdir, "PRJNA913947_somatic_ORA_all_results_with_interpretation_flags.tsv")
df.to_csv(all_flagged, sep="\t", index=False)

interpretable = df[df["reporting_tier"].isin(["interpretable_FDR025", "nominal_interpretable"])].copy()
interpretable = interpretable.sort_values(
    ["query_label", "database", "reporting_tier", "qvalue_BH_within_database", "pvalue"],
    ascending=[True, True, True, True, True]
)

interpretable_out = os.path.join(outdir, "PRJNA913947_somatic_ORA_interpretable_terms.tsv")
interpretable.to_csv(interpretable_out, sep="\t", index=False)

primary = df[df["reporting_tier"].eq("interpretable_FDR025")].copy()
primary = primary.sort_values(
    ["query_label", "database", "qvalue_BH_within_database", "pvalue"],
    ascending=[True, True, True, True]
)

primary_out = os.path.join(outdir, "PRJNA913947_somatic_ORA_primary_interpretable_FDR025.tsv")
primary.to_csv(primary_out, sep="\t", index=False)

caution = df[df["has_term_caution"]].copy()
caution = caution.sort_values(["query_label", "database", "qvalue_BH_within_database", "pvalue"])

caution_out = os.path.join(outdir, "PRJNA913947_somatic_ORA_caution_terms_retained.tsv")
caution.to_csv(caution_out, sep="\t", index=False)

axis_summary = (
    df[df["reporting_tier"].isin(["interpretable_FDR025", "nominal_interpretable"])]
    .groupby(["query_label", "biology_axis", "reporting_tier"])
    .agg(
        n_terms=("gs_name", "count"),
        best_qvalue=("qvalue_BH_within_database", "min"),
        best_pvalue=("pvalue", "min"),
        top_terms=("gs_name", lambda x: ",".join(list(x.head(10))))
    )
    .reset_index()
    .sort_values(["query_label", "reporting_tier", "best_qvalue", "best_pvalue"])
)

axis_out = os.path.join(outdir, "PRJNA913947_somatic_ORA_biology_axis_summary.tsv")
axis_summary.to_csv(axis_out, sep="\t", index=False)

# Paper-friendly concise top table
paper_top = (
    df[
        (df["query_label"].isin(["functional_extended", "strict_HIGH_MODERATE", "functional_extended_no_caution_sensitivity"])) &
        (df["reporting_tier"].isin(["interpretable_FDR025", "nominal_interpretable"]))
    ]
    .sort_values(["reporting_tier", "qvalue_BH_within_database", "pvalue", "overlap_n"], ascending=[True, True, True, False])
    .loc[:, [
        "query_label", "database", "gs_name", "biology_axis", "reporting_tier",
        "overlap_n", "gene_set_size", "odds_ratio", "pvalue",
        "qvalue_BH_within_database", "overlap_genes"
    ]]
)

paper_top_out = os.path.join(outdir, "PRJNA913947_somatic_ORA_paper_friendly_top_terms.tsv")
paper_top.to_csv(paper_top_out, sep="\t", index=False)

# Figures
plot_df = primary.copy()
if len(plot_df) == 0:
    plot_df = interpretable[interpretable["qvalue_BH_within_database"] <= 0.25].copy()

if len(plot_df) > 0:
    for query in sorted(plot_df["query_label"].unique()):
        pdat = plot_df[plot_df["query_label"] == query].copy()
        pdat = pdat.sort_values(["qvalue_BH_within_database", "pvalue"]).head(25)
        pdat = pdat.iloc[::-1].copy()
        pdat["term_short"] = (
            pdat["gs_name"].astype(str)
            .str.replace("^HALLMARK_", "", regex=True)
            .str.replace("^REACTOME_", "", regex=True)
            .str.replace("^KEGG_", "", regex=True)
            .str.replace("^GOBP_", "", regex=True)
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        plt.figure(figsize=(9, max(5, len(pdat) * 0.32)))
        plt.barh(pdat["term_short"], -pdat["qvalue_BH_within_database"].apply(lambda x: math.log10(x) if x > 0 else -300))
        plt.xlabel("-log10 within-database FDR")
        plt.ylabel("")
        plt.title(f"Somatic ORA interpretable terms: {query}")
        plt.tight_layout()
        plt.savefig(os.path.join(figs, f"PRJNA913947_03C_somatic_ORA_interpretable_{query}.png"), dpi=300)
        plt.close()

axis_plot = axis_summary[axis_summary["reporting_tier"].eq("interpretable_FDR025")].copy()
if len(axis_plot) > 0:
    collapsed = (
        axis_plot.groupby("biology_axis")
        .agg(n_terms=("n_terms", "sum"), best_qvalue=("best_qvalue", "min"))
        .reset_index()
        .sort_values("n_terms", ascending=True)
    )
    plt.figure(figsize=(8, max(4, len(collapsed) * 0.35)))
    plt.barh(collapsed["biology_axis"], collapsed["n_terms"])
    plt.xlabel("Number of interpretable FDR≤0.25 terms")
    plt.ylabel("Biology axis")
    plt.title("Somatic pathway axes from database-driven ORA")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_03C_somatic_ORA_biology_axis_summary.png"), dpi=300)
    plt.close()

report = os.path.join(status, "PRJNA913947_stage_03C_somatic_ORA_interpretation_flags_status.txt")
with open(report, "w") as f:
    f.write("PRJNA913947 Stage 03C somatic ORA interpretation flags\n\n")
    f.write(f"n_all_terms\t{len(df)}\n")
    f.write(f"n_interpretable_FDR025\t{len(primary)}\n")
    f.write(f"n_nominal_interpretable\t{len(df[df['reporting_tier'].eq('nominal_interpretable')])}\n")
    f.write(f"n_caution_terms_retained\t{len(caution)}\n\n")
    f.write("Biology axis summary:\n")
    f.write(axis_summary.to_csv(sep="\t", index=False))
    f.write("\nTop paper-friendly terms:\n")
    f.write(paper_top.head(80).to_csv(sep="\t", index=False))
    f.write("\nCaution terms are retained in full output but should not be interpreted as primary cancer-biology findings without additional validation.\n")

print("all_flagged", all_flagged)
print("interpretable_terms", interpretable_out)
print("primary_interpretable_FDR025", primary_out)
print("caution_terms_retained", caution_out)
print("biology_axis_summary", axis_out)
print("paper_friendly_top_terms", paper_top_out)
print("report", report)
print("n_all_terms", len(df))
print("n_interpretable_FDR025", len(primary))
print("n_nominal_interpretable", len(df[df["reporting_tier"].eq("nominal_interpretable")]))
print("n_caution_terms_retained", len(caution))
PY

echo
echo "=== STAGE 03C SUMMARY ==="
grep -E "n_all_terms|n_interpretable_FDR025|n_nominal_interpretable|n_caution_terms_retained" "$STATUS/PRJNA913947_stage_03C_somatic_ORA_interpretation_flags_status.txt"

echo
echo "=== BIOLOGY AXIS SUMMARY ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_ORA_biology_axis_summary.tsv" | head -80

echo
echo "=== PRIMARY INTERPRETABLE FDR <= 0.25 TERMS ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_ORA_primary_interpretable_FDR025.tsv" | head -80

echo
echo "=== PAPER-FRIENDLY TOP TERMS ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_ORA_paper_friendly_top_terms.tsv" | head -80

echo
echo "=== CAUTION TERMS RETAINED, NOT PRIMARY ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_ORA_caution_terms_retained.tsv" | head -40

echo
echo "=== OUTPUT FILES ==="
find "$OUTDIR" -maxdepth 1 -type f -printf "%f\t%k KB\n" | sort

echo
echo "=== DONE: STAGE 03C ==="
date
