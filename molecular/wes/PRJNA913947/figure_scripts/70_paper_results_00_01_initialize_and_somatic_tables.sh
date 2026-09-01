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

PRJNA_BASE="$PROJECT/results/EA_BC_AI_MultiOmics/PRJNA913947_candidate_kenya_wes"
PAPER="$PAPER_RESULTS_ROOT"

OVERVIEW="$PAPER/00_overview"
STATUS="$PAPER/00_status"
SOMATIC="$PAPER/03_somatic_wes_PRJNA913947"
SOMATIC_QC="$SOMATIC/00_final_qc"
SOMATIC_TABLES="$SOMATIC/01_tables"
SOMATIC_FIGS="$SOMATIC/02_figures"
SOMATIC_ANNOT_INPUT="$SOMATIC/03_annotation_input"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OVERVIEW" "$STATUS" "$SOMATIC_QC" "$SOMATIC_TABLES" "$SOMATIC_FIGS" "$SOMATIC_ANNOT_INPUT" "$LOGDIR"

LOG="$LOGDIR/70_paper_results_00_01_initialize_and_somatic_tables_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== PAPER RESULTS INITIALIZATION + PRJNA SOMATIC TABLES ==="
date
echo "PROJECT=$PROJECT"
echo "PAPER=$PAPER"
echo "PRJNA_BASE=$PRJNA_BASE"

echo
echo "=== 0. Write final paper analysis plan ==="

cat > "$OVERVIEW/AI_Paper_Kurgu_final_analysis_plan.md" <<'MD'
# AI_Paper_Kurgu — Final Analysis Plan

## Main concept

Clinical AI-guided multi-layer molecular characterization of East African breast cancer.

## Main research question

Can a clinical AI framework developed in a Tanzanian breast cancer cohort be extended with East African germline background, Kenyan tumor-normal somatic WES, and transcriptomic pathway analysis to prioritize candidate molecular biomarkers and pathways relevant to breast cancer in East Africa?

## Layers

### 1. Clinical-AI layer
Tanzanian breast cancer cohort with clinical, pathological, and IHC variables.

Expected outputs:
- Model performance table
- Feature-importance / SHAP table
- Clinical phenotype summary
- AI-derived clinical risk/phenotype signal

### 2. Local germline genomics layer
Hadza + Tanzanian control WES as East African germline variation background.

Expected outputs:
- Germline variant background table
- Candidate cancer-gene rare variant burden
- Pathway-level rare variant burden
- East African genomic-context interpretation

This is not a GWAS layer. Formal GWAS claims will be avoided.

### 3. East African tumor genomics layer
PRJNA913947 Kenyan breast cancer tumor/adjacent-normal WES subset.

Expected outputs:
- Somatic PASS variant cohort table
- Pair-level mutation burden
- Variant-class spectrum
- Recurrent mutated genes/loci
- Cancer driver/pathway enrichment

### 4. Transcriptomic/pathway layer
Kenyan/African ancestry RNA-seq data.

Expected outputs:
- DESeq2 late-vs-early differential expression
- Ranked gene list
- GSEA/ORA pathway enrichment
- Leading-edge genes
- Pathway activation interpretation

### 5. Integrated biomarker prioritization
Somatic mutation, RNA expression/pathway signal, germline background, and clinical AI context will be combined.

Expected outputs:
- Integrated candidate gene table
- Integrated candidate pathway table
- Prioritization score
- Figure-ready result summaries
MD

echo
echo "=== 1. Collect final PRJNA PASS VCFs ==="

VCF_LIST="$SOMATIC_QC/PRJNA913947_21pairs_PASS_vcf_paths.txt"
find -L "$PRJNA_BASE/mutect2" -maxdepth 2 -name "candidate_kenya_pair_*.PASS.vcf.gz" \
  | sort -V > "$VCF_LIST"

N=$(wc -l < "$VCF_LIST")
echo "PASS VCF count: $N"

if [[ "$N" -ne 21 ]]; then
  echo "ERROR: Expected 21 PASS VCFs, found $N"
  cat "$VCF_LIST"
  exit 1
fi

echo
echo "=== 2. Copy existing final QC files when present ==="

if [[ -d "$PRJNA_BASE/final_qc" ]]; then
  cp -av "$PRJNA_BASE/final_qc/"*.tsv "$SOMATIC_QC/" 2>/dev/null || true
fi

echo
echo "=== 3. Build SHA256 manifest ==="

SHA_OUT="$SOMATIC_QC/PRJNA913947_21pairs_PASS_vcf_sha256.tsv"
echo -e "pair\tsha256\tvcf_path" > "$SHA_OUT"

while read -r vcf
do
  pair=$(basename "$(dirname "$vcf")")
  sha=$(sha256sum "$vcf" | awk '{print $1}')
  echo -e "${pair}\t${sha}\t${vcf}" >> "$SHA_OUT"
done < "$VCF_LIST"

echo
echo "=== 4. Parse 21 PASS VCFs into analysis-ready somatic tables ==="

export VCF_LIST
export SOMATIC_TABLES
export SOMATIC_ANNOT_INPUT

micromamba run -n hadza-wes python - <<'PY'
import os
import re
import gzip
import csv
from collections import defaultdict, Counter
from statistics import median

vcf_list = os.environ["VCF_LIST"]
tables = os.environ["SOMATIC_TABLES"]
annot_input = os.environ["SOMATIC_ANNOT_INPUT"]

os.makedirs(tables, exist_ok=True)
os.makedirs(annot_input, exist_ok=True)

long_tsv_gz = os.path.join(tables, "PRJNA913947_21pairs_somatic_PASS_variants.long.tsv.gz")
pair_metrics_tsv = os.path.join(tables, "PRJNA913947_21pairs_somatic_PASS_pair_metrics.tsv")
type_summary_tsv = os.path.join(tables, "PRJNA913947_21pairs_somatic_PASS_variant_type_summary.tsv")
recurrent_loci_tsv = os.path.join(tables, "PRJNA913947_21pairs_somatic_PASS_recurrent_loci.tsv")
annovar_input = os.path.join(annot_input, "PRJNA913947_21pairs_somatic_PASS_for_annovar.avinput")

def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def parse_info(info_str):
    d = {}
    if not info_str or info_str == ".":
        return d
    for item in info_str.split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
        else:
            d[item] = "True"
    return d

def parse_format(fmt, sample_str):
    keys = fmt.split(":")
    vals = sample_str.split(":")
    return dict(zip(keys, vals))

def pick_alt_value(value, alt_i):
    if value is None or value in ("", "."):
        return "."
    parts = value.split(",")
    if len(parts) == 1:
        return value
    return parts[alt_i] if alt_i < len(parts) else value

def pick_ad_ref(value):
    if value is None or value in ("", "."):
        return "."
    parts = value.split(",")
    return parts[0] if len(parts) >= 1 else "."

def pick_ad_alt(value, alt_i):
    if value is None or value in ("", "."):
        return "."
    parts = value.split(",")
    j = alt_i + 1
    return parts[j] if j < len(parts) else "."

def classify_variant(ref, alt):
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(ref) == len(alt) and len(ref) > 1:
        return "MNP"
    if len(ref) != len(alt):
        return "INDEL"
    return "OTHER"

def as_float(x):
    try:
        if x in (None, "", "."):
            return None
        return float(x)
    except Exception:
        return None

def as_int(x):
    try:
        if x in (None, "", "."):
            return None
        return int(float(x))
    except Exception:
        return None

with open(vcf_list) as f:
    vcfs = [line.strip() for line in f if line.strip()]

variant_cols = [
    "pair", "pair_no",
    "chrom", "pos", "id", "ref", "alt", "variant_type",
    "qual", "filter",
    "tumor_sample", "normal_sample",
    "tumor_GT", "tumor_DP", "tumor_AD_REF", "tumor_AD_ALT", "tumor_AF",
    "normal_GT", "normal_DP", "normal_AD_REF", "normal_AD_ALT", "normal_AF",
    "INFO_TLOD", "INFO_NLOD", "INFO_ECNT", "INFO_POPAF", "INFO_CONTQ",
    "INFO_RAW", "vcf_path"
]

pair_stats = defaultdict(lambda: {
    "total": 0,
    "SNV": 0,
    "INDEL": 0,
    "MNP": 0,
    "OTHER": 0,
    "tumor_af": [],
    "tumor_dp": [],
    "normal_af": [],
    "normal_dp": [],
})

type_stats = Counter()
locus_pairs = defaultdict(set)
rows_written = 0

with gzip.open(long_tsv_gz, "wt", newline="") as out_gz, open(annovar_input, "w", newline="") as ann:
    writer = csv.DictWriter(out_gz, fieldnames=variant_cols, delimiter="\t", extrasaction="ignore")
    writer.writeheader()

    ann_writer = csv.writer(ann, delimiter="\t")
    ann_writer.writerow(["chrom", "start", "end", "ref", "alt", "pair", "variant_type", "tumor_AF", "tumor_DP", "normal_AF", "normal_DP"])

    for vcf in vcfs:
        m = re.search(r"candidate_kenya_pair_(\d{3})", vcf)
        if not m:
            raise RuntimeError(f"Cannot parse pair number from {vcf}")

        pair_no = m.group(1)
        pair = f"candidate_kenya_pair_{pair_no}"
        header_cols = None

        with open_text(vcf) as fh:
            for line in fh:
                line = line.rstrip("\n")

                if line.startswith("#CHROM"):
                    header_cols = line.lstrip("#").split("\t")
                    continue

                if line.startswith("#"):
                    continue

                if not line:
                    continue

                if header_cols is None:
                    raise RuntimeError(f"Missing #CHROM header in {vcf}")

                parts = line.split("\t")
                if len(parts) < 10:
                    continue

                chrom, pos, vid, ref, alts, qual, filt, info_raw, fmt = parts[:9]
                sample_values = parts[9:]
                sample_names = header_cols[9:]

                tumor_idx = None
                normal_idx = None

                for idx, s in enumerate(sample_names):
                    if s.endswith("_T"):
                        tumor_idx = idx
                    elif s.endswith("_N"):
                        normal_idx = idx

                if tumor_idx is None or normal_idx is None:
                    if len(sample_values) >= 2:
                        tumor_idx = 0
                        normal_idx = 1
                    else:
                        raise RuntimeError(f"Cannot identify tumor/normal columns in {vcf}")

                tumor_sample = sample_names[tumor_idx]
                normal_sample = sample_names[normal_idx]

                tumor_fmt = parse_format(fmt, sample_values[tumor_idx])
                normal_fmt = parse_format(fmt, sample_values[normal_idx])
                info = parse_info(info_raw)

                for alt_i, alt in enumerate(alts.split(",")):
                    vtype = classify_variant(ref, alt)

                    tumor_af = pick_alt_value(tumor_fmt.get("AF"), alt_i)
                    normal_af = pick_alt_value(normal_fmt.get("AF"), alt_i)
                    tumor_dp = tumor_fmt.get("DP", ".")
                    normal_dp = normal_fmt.get("DP", ".")

                    row = {
                        "pair": pair,
                        "pair_no": pair_no,
                        "chrom": chrom,
                        "pos": pos,
                        "id": vid,
                        "ref": ref,
                        "alt": alt,
                        "variant_type": vtype,
                        "qual": qual,
                        "filter": filt,
                        "tumor_sample": tumor_sample,
                        "normal_sample": normal_sample,
                        "tumor_GT": tumor_fmt.get("GT", "."),
                        "tumor_DP": tumor_dp,
                        "tumor_AD_REF": pick_ad_ref(tumor_fmt.get("AD")),
                        "tumor_AD_ALT": pick_ad_alt(tumor_fmt.get("AD"), alt_i),
                        "tumor_AF": tumor_af,
                        "normal_GT": normal_fmt.get("GT", "."),
                        "normal_DP": normal_dp,
                        "normal_AD_REF": pick_ad_ref(normal_fmt.get("AD")),
                        "normal_AD_ALT": pick_ad_alt(normal_fmt.get("AD"), alt_i),
                        "normal_AF": normal_af,
                        "INFO_TLOD": pick_alt_value(info.get("TLOD"), alt_i),
                        "INFO_NLOD": pick_alt_value(info.get("NLOD"), alt_i),
                        "INFO_ECNT": info.get("ECNT", "."),
                        "INFO_POPAF": pick_alt_value(info.get("POPAF"), alt_i),
                        "INFO_CONTQ": info.get("CONTQ", "."),
                        "INFO_RAW": info_raw,
                        "vcf_path": vcf,
                    }

                    writer.writerow(row)

                    start = int(pos)
                    end = start + len(ref) - 1
                    ann_writer.writerow([chrom, start, end, ref, alt, pair, vtype, tumor_af, tumor_dp, normal_af, normal_dp])

                    rows_written += 1
                    pair_stats[pair]["total"] += 1
                    pair_stats[pair][vtype] += 1
                    type_stats[vtype] += 1
                    locus_pairs[(chrom, pos, ref, alt)].add(pair)

                    taf = as_float(tumor_af)
                    naf = as_float(normal_af)
                    tdp = as_int(tumor_dp)
                    ndp = as_int(normal_dp)

                    if taf is not None:
                        pair_stats[pair]["tumor_af"].append(taf)
                    if naf is not None:
                        pair_stats[pair]["normal_af"].append(naf)
                    if tdp is not None:
                        pair_stats[pair]["tumor_dp"].append(tdp)
                    if ndp is not None:
                        pair_stats[pair]["normal_dp"].append(ndp)

with open(pair_metrics_tsv, "w", newline="") as f:
    cols = [
        "pair", "pass_variants", "SNV", "INDEL", "MNP", "OTHER",
        "median_tumor_AF", "median_tumor_DP", "median_normal_AF", "median_normal_DP"
    ]
    writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
    writer.writeheader()

    for pair in sorted(pair_stats.keys()):
        st = pair_stats[pair]
        writer.writerow({
            "pair": pair,
            "pass_variants": st["total"],
            "SNV": st["SNV"],
            "INDEL": st["INDEL"],
            "MNP": st["MNP"],
            "OTHER": st["OTHER"],
            "median_tumor_AF": round(median(st["tumor_af"]), 5) if st["tumor_af"] else ".",
            "median_tumor_DP": round(median(st["tumor_dp"]), 1) if st["tumor_dp"] else ".",
            "median_normal_AF": round(median(st["normal_af"]), 5) if st["normal_af"] else ".",
            "median_normal_DP": round(median(st["normal_dp"]), 1) if st["normal_dp"] else ".",
        })

with open(type_summary_tsv, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["variant_type", "count"])
    for k, v in sorted(type_stats.items()):
        writer.writerow([k, v])

with open(recurrent_loci_tsv, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["chrom", "pos", "ref", "alt", "n_pairs", "pairs"])
    for (chrom, pos, ref, alt), pairs in sorted(
        locus_pairs.items(),
        key=lambda x: (-len(x[1]), x[0][0], int(x[0][1]))
    ):
        writer.writerow([chrom, pos, ref, alt, len(pairs), ",".join(sorted(pairs))])

print(f"rows_written={rows_written}")
print(f"long_tsv_gz={long_tsv_gz}")
print(f"pair_metrics_tsv={pair_metrics_tsv}")
print(f"type_summary_tsv={type_summary_tsv}")
print(f"recurrent_loci_tsv={recurrent_loci_tsv}")
print(f"annovar_input={annovar_input}")
PY

echo
echo "=== 5. Generate basic somatic figures ==="

PLOT_ENV="hadza-wes"
if micromamba run -n rnaseq python - <<'PY' >/dev/null 2>&1
import pandas
import matplotlib
PY
then
  PLOT_ENV="rnaseq"
fi

export SOMATIC_TABLES
export SOMATIC_FIGS

micromamba run -n "$PLOT_ENV" python - <<'PY'
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tables = os.environ["SOMATIC_TABLES"]
figs = os.environ["SOMATIC_FIGS"]
os.makedirs(figs, exist_ok=True)

metrics = pd.read_csv(os.path.join(tables, "PRJNA913947_21pairs_somatic_PASS_pair_metrics.tsv"), sep="\t")
types = pd.read_csv(os.path.join(tables, "PRJNA913947_21pairs_somatic_PASS_variant_type_summary.tsv"), sep="\t")

metrics["pair_short"] = metrics["pair"].str.replace("candidate_kenya_pair_", "P", regex=False)

plt.figure(figsize=(12, 5))
plt.bar(metrics["pair_short"], metrics["pass_variants"])
plt.xlabel("Pair")
plt.ylabel("PASS somatic variants")
plt.title("PRJNA913947 candidate Kenyan tumor-normal WES: PASS somatic variant burden")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_pair_PASS_variant_burden.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
bottom = None
for col in ["SNV", "INDEL", "MNP", "OTHER"]:
    if col not in metrics.columns:
        continue
    if bottom is None:
        plt.bar(metrics["pair_short"], metrics[col], label=col)
        bottom = metrics[col].copy()
    else:
        plt.bar(metrics["pair_short"], metrics[col], bottom=bottom, label=col)
        bottom = bottom + metrics[col]
plt.xlabel("Pair")
plt.ylabel("Variant count")
plt.title("PRJNA913947 variant-class composition by pair")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_pair_variant_class_stacked.png"), dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.bar(types["variant_type"], types["count"])
plt.xlabel("Variant type")
plt.ylabel("Count")
plt.title("PRJNA913947 overall PASS variant-type summary")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_overall_variant_type_summary.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(metrics["pair_short"], metrics["median_tumor_AF"])
plt.xlabel("Pair")
plt.ylabel("Median tumor AF")
plt.title("PRJNA913947 median tumor allele fraction by pair")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figs, "PRJNA913947_pair_median_tumor_AF.png"), dpi=300)
plt.close()

print("Figures written to:", figs)
PY

echo
echo "=== 6. Write compact status report ==="

REPORT="$STATUS/PRJNA913947_somatic_stage_00_01_status.txt"

{
  echo "PRJNA913947 somatic WES stage 00-01 status"
  echo "Generated: $(date)"
  echo
  echo "PASS VCF count:"
  wc -l "$VCF_LIST"
  echo
  echo "Output folders:"
  echo "$SOMATIC_QC"
  echo "$SOMATIC_TABLES"
  echo "$SOMATIC_FIGS"
  echo "$SOMATIC_ANNOT_INPUT"
  echo
  echo "Pair metrics:"
  column -t -s $'\t' "$SOMATIC_TABLES/PRJNA913947_21pairs_somatic_PASS_pair_metrics.tsv"
  echo
  echo "Variant type summary:"
  column -t -s $'\t' "$SOMATIC_TABLES/PRJNA913947_21pairs_somatic_PASS_variant_type_summary.tsv"
  echo
  echo "Top recurrent loci:"
  column -t -s $'\t' "$SOMATIC_TABLES/PRJNA913947_21pairs_somatic_PASS_recurrent_loci.tsv" | head -30
} > "$REPORT"

echo
echo "=== OUTPUT TREE ==="
find "$PAPER" -maxdepth 4 -type f | sort

echo
echo "=== PAIR METRICS PREVIEW ==="
column -t -s $'\t' "$SOMATIC_TABLES/PRJNA913947_21pairs_somatic_PASS_pair_metrics.tsv" | head -30

echo
echo "=== VARIANT TYPE SUMMARY ==="
column -t -s $'\t' "$SOMATIC_TABLES/PRJNA913947_21pairs_somatic_PASS_variant_type_summary.tsv"

echo
echo "=== TOP RECURRENT LOCI ==="
column -t -s $'\t' "$SOMATIC_TABLES/PRJNA913947_21pairs_somatic_PASS_recurrent_loci.tsv" | head -30

echo
echo "=== DONE: STAGE 00-01 ==="
date
