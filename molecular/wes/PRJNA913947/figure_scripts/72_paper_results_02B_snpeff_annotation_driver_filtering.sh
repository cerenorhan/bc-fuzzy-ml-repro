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
ANNOT_OUT="$SOMATIC/04_annotation_snpeff"
DRIVER_OUT="$SOMATIC/05_driver_tables"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$ANNOT_OUT" "$DRIVER_OUT" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/72_paper_results_02B_snpeff_annotation_driver_filtering_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 02B: SnpEff SOMATIC ANNOTATION + DRIVER FILTERING ==="
date

LONG_TSV="$TABLES/PRJNA913947_21pairs_somatic_PASS_variants.long.tsv.gz"

if [[ ! -s "$LONG_TSV" ]]; then
  echo "ERROR: Missing input long somatic table:"
  echo "$LONG_TSV"
  exit 1
fi

echo
echo "=== 1. Prepare annotation environment ==="

if ! micromamba env list | awk '{print $1}' | grep -qx "cancer_anno"; then
  echo "Creating micromamba env: cancer_anno"
  micromamba create -y -n cancer_anno -c conda-forge -c bioconda snpeff snpsift pandas matplotlib openjdk
else
  echo "Env exists: cancer_anno"
fi

echo
echo "SnpEff version:"
micromamba run -n cancer_anno snpEff -version || true

echo
echo "=== 2. Build SnpEff input VCF from long somatic table ==="

SNPEFF_INPUT="$ANNOT_OUT/PRJNA913947_21pairs_somatic_PASS_for_snpeff.nochr.vcf"
SNPEFF_MAP="$ANNOT_OUT/PRJNA913947_21pairs_somatic_PASS_snpeff_row_metadata.tsv"

export LONG_TSV SNPEFF_INPUT SNPEFF_MAP

micromamba run -n cancer_anno python - <<'PY'
import os
import gzip
import csv
import re

long_tsv = os.environ["LONG_TSV"]
vcf_out = os.environ["SNPEFF_INPUT"]
map_out = os.environ["SNPEFF_MAP"]

def clean_info_value(x):
    x = "." if x is None else str(x)
    x = x.replace(";", ",").replace("=", "-").replace("\t", "_").replace(" ", "_")
    return x

with gzip.open(long_tsv, "rt") as f, open(vcf_out, "w", newline="") as vcf, open(map_out, "w", newline="") as mp:
    reader = csv.DictReader(f, delimiter="\t")

    vcf.write("##fileformat=VCFv4.2\n")
    vcf.write("##source=PRJNA913947_candidate_kenya_somatic_PASS_long_table\n")
    vcf.write('##INFO=<ID=PAIR,Number=1,Type=String,Description="Candidate pair ID">\n')
    vcf.write('##INFO=<ID=ROWID,Number=1,Type=String,Description="Internal row ID for merging annotation back to table">\n')
    vcf.write('##INFO=<ID=ORIGCHROM,Number=1,Type=String,Description="Original chromosome name in source VCF">\n')
    vcf.write('##INFO=<ID=VT,Number=1,Type=String,Description="Variant type from parsed VCF">\n')
    vcf.write('##INFO=<ID=T_AF,Number=1,Type=String,Description="Tumor allele fraction">\n')
    vcf.write('##INFO=<ID=T_DP,Number=1,Type=String,Description="Tumor depth">\n')
    vcf.write('##INFO=<ID=N_AF,Number=1,Type=String,Description="Normal allele fraction">\n')
    vcf.write('##INFO=<ID=N_DP,Number=1,Type=String,Description="Normal depth">\n')
    vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

    mpw = csv.writer(mp, delimiter="\t", lineterminator="\n")
    mpw.writerow([
        "row_id", "pair", "pair_no", "chrom_original", "chrom_snpeff", "pos", "ref", "alt",
        "variant_type", "tumor_AF", "tumor_DP", "normal_AF", "normal_DP",
        "tumor_AD_REF", "tumor_AD_ALT", "normal_AD_REF", "normal_AD_ALT",
        "INFO_TLOD", "INFO_NLOD", "INFO_POPAF", "vcf_path"
    ])

    n = 0
    for r in reader:
        n += 1
        row_id = f"somatic_{n:08d}"

        chrom_orig = r["chrom"]
        chrom_snpeff = re.sub(r"^chr", "", chrom_orig)

        pos = r["pos"]
        ref = r["ref"]
        alt = r["alt"]
        pair = r["pair"]

        info = [
            f"ROWID={row_id}",
            f"PAIR={clean_info_value(pair)}",
            f"ORIGCHROM={clean_info_value(chrom_orig)}",
            f"VT={clean_info_value(r.get('variant_type', '.'))}",
            f"T_AF={clean_info_value(r.get('tumor_AF', '.'))}",
            f"T_DP={clean_info_value(r.get('tumor_DP', '.'))}",
            f"N_AF={clean_info_value(r.get('normal_AF', '.'))}",
            f"N_DP={clean_info_value(r.get('normal_DP', '.'))}",
        ]

        vcf.write(
            f"{chrom_snpeff}\t{pos}\t{row_id}\t{ref}\t{alt}\t.\tPASS\t" +
            ";".join(info) + "\n"
        )

        mpw.writerow([
            row_id,
            pair,
            r.get("pair_no", "."),
            chrom_orig,
            chrom_snpeff,
            pos,
            ref,
            alt,
            r.get("variant_type", "."),
            r.get("tumor_AF", "."),
            r.get("tumor_DP", "."),
            r.get("normal_AF", "."),
            r.get("normal_DP", "."),
            r.get("tumor_AD_REF", "."),
            r.get("tumor_AD_ALT", "."),
            r.get("normal_AD_REF", "."),
            r.get("normal_AD_ALT", "."),
            r.get("INFO_TLOD", "."),
            r.get("INFO_NLOD", "."),
            r.get("INFO_POPAF", "."),
            r.get("vcf_path", "."),
        ])

print(f"written_vcf_records={n}")
print(vcf_out)
print(map_out)
PY

echo
echo "SnpEff input:"
ls -lh "$SNPEFF_INPUT" "$SNPEFF_MAP"
head -20 "$SNPEFF_INPUT"

echo
echo "=== 3. Select/download SnpEff GRCh38 database ==="

SNPEFF_DB_FILE="$ANNOT_OUT/snpeff_database_used.txt"

DB_CANDIDATES=(
  "GRCh38.105"
  "GRCh38.104"
  "GRCh38.103"
  "GRCh38.99"
  "GRCh38.86"
  "GRCh38.p13"
  "GRCh38.p12"
  "GRCh38"
)

DB_OK=""

for db in "${DB_CANDIDATES[@]}"
do
  echo
  echo "Trying SnpEff database: $db"

  if micromamba run -n cancer_anno snpEff download -v "$db" >/dev/null 2>&1; then
    DB_OK="$db"
    echo "Selected DB: $DB_OK"
    break
  else
    echo "Could not download/use: $db"
  fi
done

if [[ -z "$DB_OK" ]]; then
  echo "ERROR: No usable SnpEff GRCh38 database could be downloaded."
  echo "Internet bağlantısı veya SnpEff database erişimi kontrol edilmeli."
  echo "Alternatif olarak mevcut lokal snpEff data klasörü varsa bildir."
  exit 1
fi

echo "$DB_OK" > "$SNPEFF_DB_FILE"

echo
echo "=== 4. Run SnpEff annotation ==="

SNPEFF_VCF="$ANNOT_OUT/PRJNA913947_21pairs_somatic_PASS.snpeff.${DB_OK}.vcf"
SNPEFF_HTML="$ANNOT_OUT/PRJNA913947_21pairs_somatic_PASS.snpeff.${DB_OK}.summary.html"
SNPEFF_GENES="$ANNOT_OUT/PRJNA913947_21pairs_somatic_PASS.snpeff.${DB_OK}.genes.txt"

micromamba run -n cancer_anno snpEff \
  -v \
  -canon \
  -stats "$SNPEFF_HTML" \
  "$DB_OK" \
  "$SNPEFF_INPUT" \
  > "$SNPEFF_VCF"

if [[ ! -s "$SNPEFF_VCF" ]]; then
  echo "ERROR: SnpEff output VCF not created:"
  echo "$SNPEFF_VCF"
  exit 1
fi

echo
echo "SnpEff output:"
ls -lh "$SNPEFF_VCF" "$SNPEFF_HTML" 2>/dev/null || true

echo
echo "=== 5. Write curated cancer / breast cancer seed gene list ==="

CANCER_SEED="$DRIVER_OUT/curated_breast_pan_cancer_seed_genes.tsv"

cat > "$CANCER_SEED" <<'TSV'
gene	category	notes
TP53	breast/pan-cancer	tumor_suppressor
PIK3CA	breast/pan-cancer	PI3K_pathway
GATA3	breast	luminal_breast_driver
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
NCOR1	breast/pan-cancer	transcription_repressor
TSHZ3	breast	candidate_context
MUC16	pan-cancer	large_recurrently_mutated_gene
TSV

echo
echo "=== 6. Parse SnpEff ANN and generate driver/prioritization tables ==="

export SNPEFF_VCF SNPEFF_MAP CANCER_SEED DRIVER_OUT FIGS ANNOT_OUT STATUS DB_OK

micromamba run -n cancer_anno python - <<'PY'
import os
import re
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

snpeff_vcf = os.environ["SNPEFF_VCF"]
meta_path = os.environ["SNPEFF_MAP"]
seed_path = os.environ["CANCER_SEED"]
driver_out = os.environ["DRIVER_OUT"]
figs = os.environ["FIGS"]
annot_out = os.environ["ANNOT_OUT"]
status = os.environ["STATUS"]
db_ok = os.environ["DB_OK"]

os.makedirs(driver_out, exist_ok=True)
os.makedirs(figs, exist_ok=True)

def parse_info(info):
    d = {}
    for item in info.split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
        else:
            d[item] = "True"
    return d

ann_fields = [
    "Allele", "Annotation", "Annotation_Impact", "Gene_Name", "Gene_ID",
    "Feature_Type", "Feature_ID", "Transcript_BioType", "Rank",
    "HGVS_c", "HGVS_p", "cDNA_pos_len", "CDS_pos_len", "AA_pos_len",
    "Distance", "ERRORS_WARNINGS_INFO"
]

impact_rank = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1, ".": 0, "": 0}

def parse_ann_one(ann):
    parts = ann.split("|")
    if len(parts) < len(ann_fields):
        parts += ["."] * (len(ann_fields) - len(parts))
    return dict(zip(ann_fields, parts[:len(ann_fields)]))

def choose_best_ann(ann_value):
    if not ann_value or ann_value == ".":
        return {k: "." for k in ann_fields}
    anns = [parse_ann_one(x) for x in ann_value.split(",")]
    anns = sorted(
        anns,
        key=lambda x: (
            impact_rank.get(x.get("Annotation_Impact", "."), 0),
            1 if x.get("Transcript_BioType", "") == "protein_coding" else 0,
            -len(x.get("Annotation", ""))
        ),
        reverse=True
    )
    return anns[0]

records = []

with open(snpeff_vcf) as f:
    for line in f:
        if line.startswith("#"):
            continue
        line = line.rstrip("\n")
        if not line:
            continue
        chrom, pos, vid, ref, alt, qual, filt, info = line.split("\t")[:8]
        inf = parse_info(info)
        rowid = inf.get("ROWID", vid)
        best = choose_best_ann(inf.get("ANN", "."))
        row = {
            "row_id": rowid,
            "snpeff_chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "snpeff_db": db_ok,
            "ANN_raw": inf.get("ANN", "."),
        }
        row.update(best)
        records.append(row)

ann = pd.DataFrame(records).fillna(".")
meta = pd.read_csv(meta_path, sep="\t", dtype=str).fillna(".")
seed = pd.read_csv(seed_path, sep="\t", dtype=str).fillna(".")

df = meta.merge(ann, on="row_id", how="left").fillna(".")

def impact_class(row):
    impact = str(row.get("Annotation_Impact", "."))
    annot = str(row.get("Annotation", ".")).lower()

    if impact == "HIGH":
        if "splice" in annot:
            return "HIGH_SPLICE"
        if "stop" in annot:
            return "HIGH_STOP"
        if "frameshift" in annot:
            return "HIGH_FRAMESHIFT"
        return "HIGH_OTHER"
    if impact == "MODERATE":
        if "missense" in annot:
            return "MODERATE_MISSENSE"
        if "inframe" in annot:
            return "MODERATE_INFRAME_INDEL"
        return "MODERATE_OTHER"
    if impact == "LOW":
        if "splice_region" in annot:
            return "LOW_SPLICE_REGION"
        if "synonymous" in annot:
            return "LOW_SYNONYMOUS"
        return "LOW_OTHER"
    if impact == "MODIFIER":
        return "NONCODING_MODIFIER"
    return "OTHER_OR_UNKNOWN"

df["impact_class"] = df.apply(impact_class, axis=1)

functional_classes = {
    "HIGH_SPLICE",
    "HIGH_STOP",
    "HIGH_FRAMESHIFT",
    "HIGH_OTHER",
    "MODERATE_MISSENSE",
    "MODERATE_INFRAME_INDEL",
    "MODERATE_OTHER",
    "LOW_SPLICE_REGION",
}

df["is_functional_candidate"] = df["impact_class"].isin(functional_classes)

seed_map = seed.set_index("gene").to_dict(orient="index")
seed_genes = set(seed["gene"].astype(str))

df["gene"] = df["Gene_Name"].replace("", ".")
df["cancer_seed_hit"] = df["gene"].isin(seed_genes)
df["seed_category"] = df["gene"].map(lambda g: seed_map.get(g, {}).get("category", "."))
df["seed_notes"] = df["gene"].map(lambda g: seed_map.get(g, {}).get("notes", "."))

annotated_all = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_annotated_all.tsv.gz")
functional_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_functional_candidates.tsv")
seed_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_cancer_seed_gene_hits.tsv")
functional_seed_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_functional_cancer_seed_gene_hits.tsv")

df.to_csv(annotated_all, sep="\t", index=False, compression="gzip")
df[df["is_functional_candidate"]].to_csv(functional_out, sep="\t", index=False)
df[df["cancer_seed_hit"]].to_csv(seed_out, sep="\t", index=False)
df[df["cancer_seed_hit"] & df["is_functional_candidate"]].to_csv(functional_seed_out, sep="\t", index=False)

gene_all = (
    df[df["gene"] != "."]
    .groupby("gene")
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

gene_all_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_gene_recurrence_all.tsv")
gene_all.to_csv(gene_all_out, sep="\t", index=False)

func = df[(df["gene"] != ".") & (df["is_functional_candidate"])].copy()

gene_func = (
    func.groupby("gene")
    .agg(
        n_pairs_functional=("pair", lambda x: len(set(x))),
        n_functional_variants=("row_id", "count"),
        pairs=("pair", lambda x: ",".join(sorted(set(x)))),
        high_impact_variants=("Annotation_Impact", lambda x: int(sum(v == "HIGH" for v in x))),
        moderate_impact_variants=("Annotation_Impact", lambda x: int(sum(v == "MODERATE" for v in x))),
        low_splice_region_variants=("impact_class", lambda x: int(sum(v == "LOW_SPLICE_REGION" for v in x))),
    )
    .reset_index()
    .sort_values(["n_pairs_functional", "n_functional_variants", "gene"], ascending=[False, False, True])
)

gene_func["is_cancer_seed_gene"] = gene_func["gene"].isin(seed_genes)
gene_func["seed_category"] = gene_func["gene"].map(lambda g: seed_map.get(g, {}).get("category", "."))
gene_func["seed_notes"] = gene_func["gene"].map(lambda g: seed_map.get(g, {}).get("notes", "."))

gene_func_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_gene_recurrence_functional.tsv")
gene_func.to_csv(gene_func_out, sep="\t", index=False)

driver_recurrence = gene_func[gene_func["is_cancer_seed_gene"]].copy()
driver_recurrence_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_driver_seed_gene_recurrence.tsv")
driver_recurrence.to_csv(driver_recurrence_out, sep="\t", index=False)

priority = gene_all.merge(
    gene_func[[
        "gene", "n_pairs_functional", "n_functional_variants",
        "high_impact_variants", "moderate_impact_variants", "low_splice_region_variants"
    ]],
    on="gene",
    how="left"
).fillna({
    "n_pairs_functional": 0,
    "n_functional_variants": 0,
    "high_impact_variants": 0,
    "moderate_impact_variants": 0,
    "low_splice_region_variants": 0
})

for c in ["n_pairs_functional", "n_functional_variants", "high_impact_variants", "moderate_impact_variants", "low_splice_region_variants"]:
    priority[c] = priority[c].astype(int)

priority["somatic_priority_score"] = (
    priority["n_pairs"].astype(int) * 1
    + priority["n_pairs_functional"].astype(int) * 2
    + priority["high_impact_variants"].astype(int) * 3
    + priority["moderate_impact_variants"].astype(int) * 1
    + priority["low_splice_region_variants"].astype(int) * 1
    + priority["is_cancer_seed_gene"].astype(bool).astype(int) * 5
)

priority = priority.sort_values(
    ["somatic_priority_score", "n_pairs_functional", "n_pairs", "n_functional_variants", "gene"],
    ascending=[False, False, False, False, True]
)

priority_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_gene_priority_preliminary.tsv")
priority.to_csv(priority_out, sep="\t", index=False)

impact_summary = (
    df.groupby(["Annotation_Impact", "impact_class"])
    .agg(n_variants=("row_id", "count"), n_pairs=("pair", lambda x: len(set(x))))
    .reset_index()
    .sort_values(["Annotation_Impact", "n_variants"], ascending=[True, False])
)

impact_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_impact_class_summary.tsv")
impact_summary.to_csv(impact_out, sep="\t", index=False)

# Pair x driver matrix
if len(driver_recurrence) > 0:
    driver_genes = list(driver_recurrence.sort_values(["n_pairs_functional", "n_functional_variants"], ascending=False)["gene"].head(40))
    mat_src = func[func["gene"].isin(driver_genes)].copy()
    mat_src["value"] = 1
    mat = mat_src.pivot_table(index="gene", columns="pair", values="value", aggfunc="max", fill_value=0)
    mat = mat.reindex(driver_genes)
else:
    mat = pd.DataFrame()

matrix_out = os.path.join(driver_out, "PRJNA913947_somatic_snpeff_driver_seed_gene_by_pair_matrix.tsv")
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
    plt.savefig(os.path.join(figs, "PRJNA913947_snpeff_top_functional_gene_recurrence.png"), dpi=300)
    plt.close()

top_priority = priority.head(25).copy()
if len(top_priority) > 0:
    plt.figure(figsize=(8, max(4, len(top_priority) * 0.28)))
    plt.barh(top_priority["gene"][::-1], top_priority["somatic_priority_score"][::-1])
    plt.xlabel("Preliminary somatic priority score")
    plt.ylabel("Gene")
    plt.title("Preliminary somatic gene prioritization")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_snpeff_somatic_gene_priority_preliminary_top25.png"), dpi=300)
    plt.close()

impact_plot = impact_summary.groupby("impact_class", as_index=False)["n_variants"].sum().sort_values("n_variants", ascending=False)
if len(impact_plot) > 0:
    plt.figure(figsize=(10, 5))
    plt.bar(impact_plot["impact_class"], impact_plot["n_variants"])
    plt.xlabel("Impact class")
    plt.ylabel("Variant count")
    plt.title("SnpEff somatic impact class summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "PRJNA913947_snpeff_impact_class_summary.png"), dpi=300)
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
    plt.savefig(os.path.join(figs, "PRJNA913947_snpeff_functional_cancer_seed_gene_matrix.png"), dpi=300)
    plt.close()

report = os.path.join(status, "PRJNA913947_somatic_stage_02B_snpeff_annotation_driver_status.txt")
with open(report, "w") as f:
    f.write("PRJNA913947 somatic WES stage 02B SnpEff annotation + driver filtering\n")
    f.write(f"SnpEff database: {db_ok}\n\n")
    f.write(f"n_annotated_variants\t{len(df)}\n")
    f.write(f"n_functional_candidates\t{int(df['is_functional_candidate'].sum())}\n")
    f.write(f"n_cancer_seed_hits\t{int(df['cancer_seed_hit'].sum())}\n")
    f.write(f"n_functional_cancer_seed_hits\t{len(df[df['cancer_seed_hit'] & df['is_functional_candidate']])}\n")
    f.write(f"n_gene_recurrence_all\t{len(gene_all)}\n")
    f.write(f"n_gene_recurrence_functional\t{len(gene_func)}\n")
    f.write(f"n_driver_seed_genes_functional\t{len(driver_recurrence)}\n\n")
    f.write("Impact class summary:\n")
    f.write(impact_summary.to_csv(sep="\t", index=False))
    f.write("\nTop functional recurrent genes:\n")
    f.write(gene_func.head(40).to_csv(sep="\t", index=False))
    f.write("\nFunctional cancer seed gene recurrence:\n")
    f.write(driver_recurrence.head(80).to_csv(sep="\t", index=False))
    f.write("\nTop preliminary somatic priority genes:\n")
    f.write(priority.head(50).to_csv(sep="\t", index=False))

print("annotated_all", annotated_all)
print("functional_candidates", functional_out)
print("cancer_seed_hits", seed_out)
print("functional_cancer_seed_hits", functional_seed_out)
print("gene_recurrence_all", gene_all_out)
print("gene_recurrence_functional", gene_func_out)
print("driver_recurrence", driver_recurrence_out)
print("priority", priority_out)
print("impact_summary", impact_out)
print("driver_matrix", matrix_out)
print("report", report)
print("n_annotated_variants", len(df))
print("n_functional_candidates", int(df["is_functional_candidate"].sum()))
print("n_cancer_seed_hits", int(df["cancer_seed_hit"].sum()))
print("n_functional_cancer_seed_hits", len(df[df["cancer_seed_hit"] & df["is_functional_candidate"]]))
print("n_gene_recurrence_all", len(gene_all))
print("n_gene_recurrence_functional", len(gene_func))
print("n_driver_seed_genes_functional", len(driver_recurrence))
PY

echo
echo "=== 7. Preview outputs ==="

echo
echo "=== SnpEff DB used ==="
cat "$SNPEFF_DB_FILE"

echo
echo "=== IMPACT CLASS SUMMARY ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_snpeff_impact_class_summary.tsv" | head -40

echo
echo "=== TOP FUNCTIONAL RECURRENT GENES ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_snpeff_gene_recurrence_functional.tsv" | head -30

echo
echo "=== FUNCTIONAL CANCER SEED GENE RECURRENCE ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_snpeff_driver_seed_gene_recurrence.tsv" | head -80

echo
echo "=== TOP PRELIMINARY SOMATIC PRIORITY GENES ==="
column -t -s $'\t' "$DRIVER_OUT/PRJNA913947_somatic_snpeff_gene_priority_preliminary.tsv" | head -30

echo
echo "=== OUTPUT TREE: STAGE 02B ==="
find "$SOMATIC" -maxdepth 3 -type f | sort

echo
echo "=== DONE: STAGE 02B ==="
date
