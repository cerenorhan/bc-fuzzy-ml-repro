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
TABLES="$SOMATIC/01_tables"
FIGS="$SOMATIC/02_figures"
ANNOT_IN="$SOMATIC/03_annotation_input"
ANNOT_OUT="$SOMATIC/04_annotation"
DRIVER_OUT="$SOMATIC/05_driver_tables"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$ANNOT_IN" "$ANNOT_OUT" "$DRIVER_OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/71_paper_results_02_somatic_annotation_driver_filtering_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 02: SOMATIC ANNOTATION + DRIVER FILTERING ==="
date

LONG_TSV="$TABLES/PRJNA913947_21pairs_somatic_PASS_variants.long.tsv.gz"

if [[ ! -s "$LONG_TSV" ]]; then
  echo "ERROR: Missing input long somatic table:"
  echo "$LONG_TSV"
  exit 1
fi

echo
echo "Input long table:"
ls -lh "$LONG_TSV"

echo
echo "=== 1. Locate ANNOVAR ==="

TABLE_ANNOVAR=""

for p in \
  "$PROJECT/annovar/table_annovar.pl" \
  "$PROJECT/tools/annovar/table_annovar.pl" \
  "$PROJECT/reference/annovar/table_annovar.pl" \
  "$HOME/annovar/table_annovar.pl" \
  "$HOME/tools/annovar/table_annovar.pl" \
  "$HOME/software/annovar/table_annovar.pl" \
  "/opt/annovar/table_annovar.pl"
do
  if [[ -f "$p" ]]; then
    TABLE_ANNOVAR="$p"
    break
  fi
done

if [[ -z "$TABLE_ANNOVAR" ]]; then
  if command -v table_annovar.pl >/dev/null 2>&1; then
    TABLE_ANNOVAR="$(command -v table_annovar.pl)"
  fi
fi

if [[ -z "$TABLE_ANNOVAR" ]]; then
  echo "ERROR: table_annovar.pl bulunamadı."
  echo "ANNOVAR yolu farklıysa script içinde TABLE_ANNOVAR değişkenini elle düzenle."
  exit 1
fi

ANNOVAR_DIR="$(dirname "$TABLE_ANNOVAR")"

HUMANDB=""

for d in \
  "$ANNOVAR_DIR/humandb" \
  "$PROJECT/annovar/humandb" \
  "$PROJECT/reference/annovar/humandb" \
  "$PROJECT/tools/annovar/humandb" \
  "$HOME/annovar/humandb" \
  "$HOME/tools/annovar/humandb" \
  "$HOME/software/annovar/humandb"
do
  if [[ -d "$d" ]]; then
    HUMANDB="$d"
    break
  fi
done

if [[ -z "$HUMANDB" ]]; then
  echo "ERROR: ANNOVAR humandb klasörü bulunamadı."
  echo "Beklenen örnek: $ANNOVAR_DIR/humandb veya $HOME/annovar/humandb"
  exit 1
fi

echo "TABLE_ANNOVAR=$TABLE_ANNOVAR"
echo "HUMANDB=$HUMANDB"

if ! compgen -G "$HUMANDB/hg38_refGene.txt*" >/dev/null; then
  echo "ERROR: hg38_refGene database bulunamadı:"
  echo "$HUMANDB/hg38_refGene.txt"
  echo
  echo "Bu aşama için en az refGene gerekli."
  exit 1
fi

echo
echo "=== 2. Build clean ANNOVAR input without header ==="

AVINPUT="$ANNOT_IN/PRJNA913947_21pairs_somatic_PASS_for_annovar.clean.avinput"
AVINPUT_MAP="$ANNOT_IN/PRJNA913947_21pairs_somatic_PASS_for_annovar.row_metadata.tsv"

export LONG_TSV AVINPUT AVINPUT_MAP

micromamba run -n rnaseq python - <<'PY'
import os
import gzip
import csv

long_tsv = os.environ["LONG_TSV"]
avinput = os.environ["AVINPUT"]
avmap = os.environ["AVINPUT_MAP"]

with gzip.open(long_tsv, "rt") as f, open(avinput, "w", newline="") as av, open(avmap, "w", newline="") as mp:
    reader = csv.DictReader(f, delimiter="\t")
    avw = csv.writer(av, delimiter="\t", lineterminator="\n")
    mpw = csv.writer(mp, delimiter="\t", lineterminator="\n")

    mpw.writerow([
        "row_id", "pair", "pair_no", "chrom", "start", "end", "ref", "alt",
        "variant_type", "tumor_AF", "tumor_DP", "tumor_AD_REF", "tumor_AD_ALT",
        "normal_AF", "normal_DP", "normal_AD_REF", "normal_AD_ALT",
        "INFO_TLOD", "INFO_NLOD", "INFO_POPAF", "vcf_path"
    ])

    n = 0
    for r in reader:
        n += 1
        row_id = f"somatic_{n:08d}"
        chrom = r["chrom"]
        start = int(r["pos"])
        end = start + len(r["ref"]) - 1
        ref = r["ref"]
        alt = r["alt"]

        # ANNOVAR avinput: Chr Start End Ref Alt + optional comments.
        avw.writerow([
            chrom, start, end, ref, alt,
            row_id,
            r.get("pair", "."),
            r.get("pair_no", "."),
            r.get("variant_type", "."),
            r.get("tumor_AF", "."),
            r.get("tumor_DP", "."),
            r.get("normal_AF", "."),
            r.get("normal_DP", ".")
        ])

        mpw.writerow([
            row_id,
            r.get("pair", "."),
            r.get("pair_no", "."),
            chrom,
            start,
            end,
            ref,
            alt,
            r.get("variant_type", "."),
            r.get("tumor_AF", "."),
            r.get("tumor_DP", "."),
            r.get("tumor_AD_REF", "."),
            r.get("tumor_AD_ALT", "."),
            r.get("normal_AF", "."),
            r.get("normal_DP", "."),
            r.get("normal_AD_REF", "."),
            r.get("normal_AD_ALT", "."),
            r.get("INFO_TLOD", "."),
            r.get("INFO_NLOD", "."),
            r.get("INFO_POPAF", "."),
            r.get("vcf_path", ".")
        ])

print(f"written_avinput_rows={n}")
print(avinput)
print(avmap)
PY

echo
echo "AVINPUT:"
ls -lh "$AVINPUT" "$AVINPUT_MAP"
echo "First 3 avinput rows:"
head -3 "$AVINPUT"

echo
echo "=== 3. Build available ANNOVAR protocol list ==="

PROTOCOLS=()
OPERATIONS=()

add_protocol_exact () {
  local proto="$1"
  local op="$2"
  if compgen -G "$HUMANDB/hg38_${proto}.txt*" >/dev/null; then
    PROTOCOLS+=("$proto")
    OPERATIONS+=("$op")
  fi
}

add_protocol_pattern_latest () {
  local pattern="$1"
  local op="$2"
  local match
  match=$(find "$HUMANDB" -maxdepth 1 -type f -name "$pattern" \
    | sed 's#.*/##' \
    | sed 's/^hg38_//' \
    | sed 's/\.txt.*$//' \
    | sort -V \
    | tail -1 || true)

  if [[ -n "$match" ]]; then
    PROTOCOLS+=("$match")
    OPERATIONS+=("$op")
  fi
}

add_protocol_exact "refGene" "g"
add_protocol_exact "cytoBand" "r"
add_protocol_pattern_latest "hg38_avsnp*.txt*" "f"
add_protocol_pattern_latest "hg38_clinvar_*.txt*" "f"
add_protocol_exact "exac03" "f"
add_protocol_pattern_latest "hg38_gnomad*_exome.txt*" "f"
add_protocol_pattern_latest "hg38_gnomad*_genome.txt*" "f"
add_protocol_pattern_latest "hg38_dbnsfp*.txt*" "f"
add_protocol_pattern_latest "hg38_cosmic*.txt*" "f"

if [[ "${#PROTOCOLS[@]}" -eq 0 ]]; then
  echo "ERROR: No ANNOVAR protocols detected."
  exit 1
fi

PROTOCOL_STR=$(IFS=, ; echo "${PROTOCOLS[*]}")
OPERATION_STR=$(IFS=, ; echo "${OPERATIONS[*]}")

echo "PROTOCOLS=$PROTOCOL_STR"
echo "OPERATIONS=$OPERATION_STR"

echo -e "protocol\toperation" > "$ANNOT_OUT/annovar_protocols_used.tsv"
for idx in "${!PROTOCOLS[@]}"
do
  echo -e "${PROTOCOLS[$idx]}\t${OPERATIONS[$idx]}" >> "$ANNOT_OUT/annovar_protocols_used.tsv"
done

echo
echo "=== 4. Run ANNOVAR table_annovar ==="

ANN_PREFIX="$ANNOT_OUT/PRJNA913947_21pairs_somatic_PASS"

perl "$TABLE_ANNOVAR" \
  "$AVINPUT" \
  "$HUMANDB" \
  -buildver hg38 \
  -out "$ANN_PREFIX" \
  -remove \
  -protocol "$PROTOCOL_STR" \
  -operation "$OPERATION_STR" \
  -nastring . \
  -polish \
  -otherinfo

MULTIANNO="${ANN_PREFIX}.hg38_multianno.txt"

if [[ ! -s "$MULTIANNO" ]]; then
  echo "ERROR: ANNOVAR multianno output not produced:"
  echo "$MULTIANNO"
  exit 1
fi

echo
echo "ANNOVAR output:"
ls -lh "$MULTIANNO"

echo
echo "=== 5. Write curated cancer / breast cancer seed gene list ==="

CANCER_SEED="$DRIVER_OUT/curated_breast_pan_cancer_seed_genes.tsv"

cat > "$CANCER_SEED" <<'TSV'
gene	category	notes
TP53	breast/pan-cancer	tumor_suppressor
PIK3CA	breast/pan-cancer	PI3K_pathway
GATA3	breast	 luminal_breast_driver
MAP3K1	breast	MAPK_pathway
CDH1	breast	adhesion_lobular_breast_cancer
PTEN	breast/pan-cancer	PI3K_pathway_tumor_suppressor
AKT1	breast/pan-cancer	PI3K_pathway
ERBB2	breast/pan-cancer	HER2_receptor
ESR1	breast	estrogen_receptor
BRCA1	breast/ovarian	DNA_repair
BRCA2	breast/ovarian	DNA_repair
PALB2	breast	DNA_repair
ATM	breast/pan-cancer	DNA_damage_response
CHEK2	breast	DNA_damage_response
RB1	breast/pan-cancer	cell_cycle
NF1	breast/pan-cancer	RAS_pathway
ARID1A	breast/pan-cancer	chromatin_remodeling
FBXW7	breast/pan-cancer	ubiquitin_ligase
KMT2C	breast/pan-cancer	chromatin_remodeling
KMT2D	breast/pan-cancer	chromatin_remodeling
CCND1	breast/pan-cancer	cell_cycle
MYC	breast/pan-cancer	transcription
FGFR1	breast/pan-cancer	RTK
FGFR2	breast/pan-cancer	RTK
MDM2	breast/pan-cancer	p53_pathway
CCNE1	breast/pan-cancer	cell_cycle
KRAS	pan-cancer	RAS_pathway
NRAS	pan-cancer	RAS_pathway
HRAS	pan-cancer	RAS_pathway
BRAF	pan-cancer	MAPK_pathway
EGFR	pan-cancer	RTK
ALK	pan-cancer	RTK_fusion
MET	pan-cancer	RTK
RET	pan-cancer	RTK
NTRK1	pan-cancer	RTK_fusion
NTRK2	pan-cancer	RTK_fusion
NTRK3	pan-cancer	RTK_fusion
APC	pan-cancer	WNT_pathway
CTNNB1	pan-cancer	WNT_pathway
SMAD4	pan-cancer	TGF_beta
TGFBR2	pan-cancer	TGF_beta
STK11	pan-cancer	AMPK_pathway
SMARCA4	pan-cancer	chromatin_remodeling
SMARCB1	pan-cancer	chromatin_remodeling
EP300	pan-cancer	chromatin_transcription
CREBBP	pan-cancer	chromatin_transcription
NOTCH1	pan-cancer	Notch_pathway
NOTCH2	pan-cancer	Notch_pathway
NOTCH3	pan-cancer	Notch_pathway
JAK1	pan-cancer	JAK_STAT
JAK2	pan-cancer	JAK_STAT
STAT3	pan-cancer	JAK_STAT
IDH1	pan-cancer	metabolism
IDH2	pan-cancer	metabolism
TERT	pan-cancer	telomerase
BAP1	pan-cancer	chromatin_DNA_repair
SETD2	pan-cancer	chromatin
VHL	pan-cancer	hypoxia
POLE	pan-cancer	DNA_replication_repair
POLD1	pan-cancer	DNA_replication_repair
MLH1	pan-cancer	mismatch_repair
MSH2	pan-cancer	mismatch_repair
MSH6	pan-cancer	mismatch_repair
PMS2	pan-cancer	mismatch_repair
RAD51C	breast/ovarian	DNA_repair
RAD51D	breast/ovarian	DNA_repair
BARD1	breast	DNA_repair
BRIP1	breast/ovarian	DNA_repair
CDK4	pan-cancer	cell_cycle
CDK6	breast/pan-cancer	cell_cycle
AURKA	breast/pan-cancer	cell_cycle
FOXA1	breast	luminal_breast
RUNX1	breast/pan-cancer	transcription
TBX3	breast	transcription
SF3B1	breast/pan-cancer	splicing
CBFB	breast	transcription
MUC16	pan-cancer	large_recurrently_mutated_gene
TSHZ3	breast	candidate_context
NCOR1	breast/pan-cancer	transcription_repressor
TSV

echo
echo "=== 6. Post-process ANNOVAR output into paper-ready tables ==="

export MULTIANNO
export AVINPUT_MAP
export CANCER_SEED
export DRIVER_OUT
export FIGS

micromamba run -n rnaseq python - <<'PY'
import os
import re
import gzip
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

multianno = os.environ["MULTIANNO"]
avmap = os.environ["AVINPUT_MAP"]
cancer_seed = os.environ["CANCER_SEED"]
driver_out = os.environ["DRIVER_OUT"]
figs = os.environ["FIGS"]

os.makedirs(driver_out, exist_ok=True)
os.makedirs(figs, exist_ok=True)

df = pd.read_csv(multianno, sep="\t", dtype=str).fillna(".")
meta = pd.read_csv(avmap, sep="\t", dtype=str).fillna(".")
seed = pd.read_csv(cancer_seed, sep="\t", dtype=str).fillna(".")

# Recover row_id and pair metadata from Otherinfo columns if available.
other_cols = [c for c in df.columns if c.startswith("Otherinfo")]

if "row_id" not in df.columns:
    if len(other_cols) >= 1:
        df["row_id"] = df[other_cols[0]]
    else:
        df["row_id"] = "."

df = df.merge(meta, on="row_id", how="left", suffixes=("", "_meta"))

# Robust column selection
gene_col = None
for c in ["Gene.refGene", "Gene.knownGene", "Gene.ensGene"]:
    if c in df.columns:
        gene_col = c
        break

func_col = None
for c in ["Func.refGene", "Func.knownGene", "Func.ensGene"]:
    if c in df.columns:
        func_col = c
        break

exonic_col = None
for c in ["ExonicFunc.refGene", "ExonicFunc.knownGene", "ExonicFunc.ensGene"]:
    if c in df.columns:
        exonic_col = c
        break

aachange_col = None
for c in ["AAChange.refGene", "AAChange.knownGene", "AAChange.ensGene"]:
    if c in df.columns:
        aachange_col = c
        break

if gene_col is None:
    df["Gene_symbol_raw"] = "."
else:
    df["Gene_symbol_raw"] = df[gene_col].astype(str)

if func_col is None:
    df["Func_annotation"] = "."
else:
    df["Func_annotation"] = df[func_col].astype(str)

if exonic_col is None:
    df["ExonicFunc_annotation"] = "."
else:
    df["ExonicFunc_annotation"] = df[exonic_col].astype(str)

if aachange_col is None:
    df["AAChange_annotation"] = "."
else:
    df["AAChange_annotation"] = df[aachange_col].astype(str)

def split_genes(x):
    x = str(x)
    if x in ["", ".", "nan", "None"]:
        return []
    # Remove transcript/detail parentheses lightly but keep gene tokens.
    toks = re.split(r"[;,]", x)
    clean = []
    for t in toks:
        t = t.strip()
        t = re.sub(r"\(.*?\)", "", t).strip()
        if t and t not in [".", "NONE", "UNKNOWN"]:
            clean.append(t)
    return sorted(set(clean))

df["gene_list"] = df["Gene_symbol_raw"].apply(split_genes)
df["gene_primary"] = df["gene_list"].apply(lambda x: x[0] if x else ".")

def impact_class(row):
    func = str(row.get("Func_annotation", ".")).lower()
    exo = str(row.get("ExonicFunc_annotation", ".")).lower()

    if "splicing" in func:
        return "HIGH_SPLICE"
    if "stopgain" in exo or "stoploss" in exo:
        return "HIGH_STOP"
    if "frameshift" in exo:
        return "HIGH_FRAMESHIFT"
    if "nonsynonymous" in exo:
        return "MODERATE_MISSENSE"
    if "nonframeshift" in exo:
        return "MODERATE_INFRAME_INDEL"
    if "synonymous" in exo:
        return "LOW_SYNONYMOUS"
    if "exonic" in func:
        return "EXONIC_OTHER"
    if any(k in func for k in ["utr", "intronic", "intergenic", "upstream", "downstream", "ncrna"]):
        return "NONCODING"
    return "OTHER_OR_UNKNOWN"

df["impact_class"] = df.apply(impact_class, axis=1)

functional_classes = {
    "HIGH_SPLICE",
    "HIGH_STOP",
    "HIGH_FRAMESHIFT",
    "MODERATE_MISSENSE",
    "MODERATE_INFRAME_INDEL",
    "EXONIC_OTHER",
}

df["is_functional_candidate"] = df["impact_class"].isin(functional_classes)

seed_map = seed.set_index("gene").to_dict(orient="index")
seed_genes = set(seed["gene"].dropna().astype(str))

def seed_hit(genes):
    return any(g in seed_genes for g in genes)

def seed_hit_genes(genes):
    return ",".join([g for g in genes if g in seed_genes]) if genes else "."

df["cancer_seed_hit"] = df["gene_list"].apply(seed_hit)
df["cancer_seed_hit_genes"] = df["gene_list"].apply(seed_hit_genes)

# Write annotated all table
all_out = os.path.join(driver_out, "PRJNA913947_somatic_annotated_all.tsv.gz")
df.to_csv(all_out, sep="\t", index=False, compression="gzip")

functional = df[df["is_functional_candidate"]].copy()
functional_out = os.path.join(driver_out, "PRJNA913947_somatic_functional_candidates.tsv")
functional.to_csv(functional_out, sep="\t", index=False)

seed_hits = df[df["cancer_seed_hit"]].copy()
seed_out = os.path.join(driver_out, "PRJNA913947_somatic_cancer_seed_gene_hits.tsv")
seed_hits.to_csv(seed_out, sep="\t", index=False)

functional_seed = df[df["cancer_seed_hit"] & df["is_functional_candidate"]].copy()
functional_seed_out = os.path.join(driver_out, "PRJNA913947_somatic_functional_cancer_seed_gene_hits.tsv")
functional_seed.to_csv(functional_seed_out, sep="\t", index=False)

# Explode gene table for recurrence
exp = df[[
    "row_id", "pair", "gene_list", "gene_primary",
    "variant_type", "tumor_AF", "tumor_DP", "normal_AF", "normal_DP",
    "Func_annotation", "ExonicFunc_annotation", "AAChange_annotation",
    "impact_class", "is_functional_candidate", "cancer_seed_hit", "cancer_seed_hit_genes"
]].copy()

exp = exp.explode("gene_list")
exp = exp.rename(columns={"gene_list": "gene"})
exp = exp[(exp["gene"].notna()) & (exp["gene"] != ".") & (exp["gene"] != "")].copy()

gene_all = (
    exp.groupby("gene")
    .agg(
        n_pairs=("pair", lambda x: len(set(x))),
        n_variants=("row_id", "count"),
        pairs=("pair", lambda x: ",".join(sorted(set(x)))),
        n_functional=("is_functional_candidate", lambda x: int(sum(x.astype(bool)))),
        n_cancer_seed_variants=("cancer_seed_hit", lambda x: int(sum(x.astype(bool)))),
    )
    .reset_index()
    .sort_values(["n_pairs", "n_variants", "gene"], ascending=[False, False, True])
)

gene_all["is_cancer_seed_gene"] = gene_all["gene"].isin(seed_genes)
gene_all["seed_category"] = gene_all["gene"].map(lambda g: seed_map.get(g, {}).get("category", "."))
gene_all["seed_notes"] = gene_all["gene"].map(lambda g: seed_map.get(g, {}).get("notes", "."))

gene_all_out = os.path.join(driver_out, "PRJNA913947_somatic_gene_recurrence_all.tsv")
gene_all.to_csv(gene_all_out, sep="\t", index=False)

exp_func = exp[exp["is_functional_candidate"]].copy()

gene_func = (
    exp_func.groupby("gene")
    .agg(
        n_pairs_functional=("pair", lambda x: len(set(x))),
        n_functional_variants=("row_id", "count"),
        pairs=("pair", lambda x: ",".join(sorted(set(x)))),
        high_impact_variants=("impact_class", lambda x: int(sum(str(v).startswith("HIGH") for v in x))),
        moderate_variants=("impact_class", lambda x: int(sum(str(v).startswith("MODERATE") for v in x))),
    )
    .reset_index()
    .sort_values(["n_pairs_functional", "n_functional_variants", "gene"], ascending=[False, False, True])
)

gene_func["is_cancer_seed_gene"] = gene_func["gene"].isin(seed_genes)
gene_func["seed_category"] = gene_func["gene"].map(lambda g: seed_map.get(g, {}).get("category", "."))
gene_func["seed_notes"] = gene_func["gene"].map(lambda g: seed_map.get(g, {}).get("notes", "."))

gene_func_out = os.path.join(driver_out, "PRJNA913947_somatic_gene_recurrence_functional.tsv")
gene_func.to_csv(gene_func_out, sep="\t", index=False)

driver_recurrence = gene_func[gene_func["is_cancer_seed_gene"]].copy()
driver_recurrence_out = os.path.join(driver_out, "PRJNA913947_somatic_driver_seed_gene_recurrence.tsv")
driver_recurrence.to_csv(driver_recurrence_out, sep="\t", index=False)

# Preliminary somatic gene priority score
priority = gene_all.merge(
    gene_func[["gene", "n_pairs_functional", "n_functional_variants", "high_impact_variants", "moderate_variants"]],
    on="gene",
    how="left"
).fillna({
    "n_pairs_functional": 0,
    "n_functional_variants": 0,
    "high_impact_variants": 0,
    "moderate_variants": 0
})

for c in ["n_pairs_functional", "n_functional_variants", "high_impact_variants", "moderate_variants"]:
    priority[c] = priority[c].astype(int)

priority["somatic_priority_score"] = (
    priority["n_pairs"].astype(int) * 1
    + priority["n_pairs_functional"].astype(int) * 2
    + priority["high_impact_variants"].astype(int) * 3
    + priority["moderate_variants"].astype(int) * 1
    + priority["is_cancer_seed_gene"].astype(bool).astype(int) * 5
)

priority = priority.sort_values(
    ["somatic_priority_score", "n_pairs_functional", "n_pairs", "n_functional_variants", "gene"],
    ascending=[False, False, False, False, True]
)

priority_out = os.path.join(driver_out, "PRJNA913947_somatic_gene_priority_preliminary.tsv")
priority.to_csv(priority_out, sep="\t", index=False)

# Impact summary
impact_summary = (
    df.groupby("impact_class")
    .agg(n_variants=("row_id", "count"), n_pairs=("pair", lambda x: len(set(x))))
    .reset_index()
    .sort_values("n_variants", ascending=False)
)

impact_out = os.path.join(driver_out, "PRJNA913947_somatic_impact_class_summary.tsv")
impact_summary.to_csv(impact_out, sep="\t", index=False)

# Pair x driver matrix
if len(driver_recurrence) > 0:
    driver_genes = list(driver_recurrence.sort_values(["n_pairs_functional", "n_functional_variants"], ascending=False)["gene"].head(30))
    mat_src = exp_func[exp_func["gene"].isin(driver_genes)].copy()
    mat_src["value"] = 1
    mat = mat_src.pivot_table(index="gene", columns="pair", values="value", aggfunc="max", fill_value=0)
    mat = mat.reindex(driver_genes)
else:
    mat = pd.DataFrame()

matrix_out = os.path.join(driver_out, "PRJNA913947_somatic_driver_seed_gene_by_pair_matrix.tsv")
mat.to_csv(matrix_out, sep="\t")

# Figures
top = gene_func.head(25).copy()
if len(top) > 0:
    plt.figure(figsize=(8, max(4, len(top) * 0.28)))
    plt.barh(top["gene"][::-1], top["n_pairs_functional"][::-1])
    plt.xlabel("Number of pairs with functional candidate variants")
    plt.ylabel("Gene")
    plt.title("Top recurrent functionally altered genes")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_top_functional_gene_recurrence.png"), dpi=300)
    plt.close()

top_priority = priority.head(25).copy()
if len(top_priority) > 0:
    plt.figure(figsize=(8, max(4, len(top_priority) * 0.28)))
    plt.barh(top_priority["gene"][::-1], top_priority["somatic_priority_score"][::-1])
    plt.xlabel("Preliminary somatic priority score")
    plt.ylabel("Gene")
    plt.title("Preliminary somatic gene prioritization")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_somatic_gene_priority_preliminary_top25.png"), dpi=300)
    plt.close()

if len(impact_summary) > 0:
    plt.figure(figsize=(9, 5))
    plt.bar(impact_summary["impact_class"], impact_summary["n_variants"])
    plt.xlabel("Impact class")
    plt.ylabel("Variant count")
    plt.title("Somatic variant impact class summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_somatic_impact_class_summary.png"), dpi=300)
    plt.close()

if mat.shape[0] > 0 and mat.shape[1] > 0:
    plt.figure(figsize=(12, max(4, mat.shape[0] * 0.3)))
    plt.imshow(mat.values, aspect="auto")
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.xticks(range(mat.shape[1]), [c.replace("candidate_kenya_pair_", "P") for c in mat.columns], rotation=45, ha="right")
    plt.xlabel("Pair")
    plt.ylabel("Cancer seed gene")
    plt.title("Functional cancer seed gene alteration matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_functional_cancer_seed_gene_matrix.png"), dpi=300)
    plt.close()

print("annotated_all", all_out)
print("functional_candidates", functional_out)
print("cancer_seed_hits", seed_out)
print("functional_cancer_seed_hits", functional_seed_out)
print("gene_recurrence_all", gene_all_out)
print("gene_recurrence_functional", gene_func_out)
print("driver_recurrence", driver_recurrence_out)
print("priority", priority_out)
print("impact_summary", impact_out)
print("driver_matrix", matrix_out)
print("n_annotated_variants", len(df))
print("n_functional_candidates", len(functional))
print("n_cancer_seed_hits", len(seed_hits))
print("n_functional_cancer_seed_hits", len(functional_seed))
print("n_recurrent_genes_all", len(gene_all))
print("n_recurrent_genes_functional", len(gene_func))
print("n_driver_seed_genes_functional", len(driver_recurrence))
PY

echo
echo "=== 7. Write stage status report ==="

REPORT="$STATUS/PRJNA913947_somatic_stage_02_annotation_driver_status.txt"

{
  echo "PRJNA913947 somatic WES stage 02 annotation + driver filtering"
  echo "Generated: $(date)"
  echo
  echo "ANNOVAR:"
  echo "TABLE_ANNOVAR=$TABLE_ANNOVAR"
  echo "HUMANDB=$HUMANDB"
  echo
  echo "Protocols used:"
  column -t -s $'\t' "$ANNOT_OUT/annovar_protocols_used.tsv"
  echo
  echo "Annotation output:"
  ls -lh "$ANNOT_OUT"
  echo
  echo "Driver output:"
  ls -lh "$DRIVER_OUT"
  echo
  echo "Impact class summary:"
  column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_impact_class_summary.tsv"
  echo
  echo "Top functional recurrent genes:"
  column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_gene_recurrence_functional.tsv" | head -40
  echo
  echo "Functional cancer seed gene recurrence:"
  column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_driver_seed_gene_recurrence.tsv" | head -80
  echo
  echo "Top preliminary somatic priority genes:"
  column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_gene_priority_preliminary.tsv" | head -50
} > "$REPORT"

echo
echo "=== OUTPUT TREE: STAGE 02 ==="
find "$SOMATIC" -maxdepth 3 -type f | sort

echo
echo "=== IMPACT CLASS SUMMARY ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_impact_class_summary.tsv"

echo
echo "=== TOP FUNCTIONAL RECURRENT GENES ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_gene_recurrence_functional.tsv" | head -30

echo
echo "=== FUNCTIONAL CANCER SEED GENE RECURRENCE ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_driver_seed_gene_recurrence.tsv" | head -80

echo
echo "=== TOP PRELIMINARY SOMATIC PRIORITY GENES ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_gene_priority_preliminary.tsv" | head -30

echo
echo "=== DONE: STAGE 02 ==="
date
