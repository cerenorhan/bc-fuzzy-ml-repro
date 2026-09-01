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
OUT="$PAPER/05_local_breast_cancer_tumor_WES"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/79A_audit_local_tumor_WES_inputs_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 79A: AUDIT LOCAL BREAST CANCER TUMOR WES INPUTS ==="
date

META="metadata/local_wes.tsv"

if [[ ! -s "$META" ]]; then
  echo "ERROR: metadata/local_wes.tsv not found"
  exit 1
fi

echo
echo "=== LOCAL TUMOR WES METADATA ==="
micromamba run -n hadza-wes python - <<'PY'
import pandas as pd
from pathlib import Path

meta = pd.read_csv("metadata/local_wes.tsv", sep="\t", dtype=str)

bc = meta[meta["group"].astype(str).str.contains("breast", case=False, na=False)].copy()

cols = [
    "sample_id",
    "group",
    "sex",
    "age_years",
    "participants_origin",
    "bc_pathologic_stage",
    "age_at_bc_diagnosis_years",
    "laterality",
    "er_status",
    "pr_status",
    "her_2_status",
    "molecular_subtype",
    "histological_type",
    "fastq_r1",
    "fastq_r2"
]
cols = [c for c in cols if c in bc.columns]

bc[cols].to_csv(
    "paper_results/05_local_breast_cancer_tumor_WES/local_tumor_wes_metadata.tsv",
    sep="\t",
    index=False
)

print("n_local_tumor_wes_samples\t", len(bc), sep="")
print(bc[cols].to_string(index=False))
PY

echo
echo "=== LOCAL TUMOR FASTQ EXISTENCE ==="
micromamba run -n hadza-wes python - <<'PY'
import pandas as pd
from pathlib import Path

meta = pd.read_csv("metadata/local_wes.tsv", sep="\t", dtype=str)
bc = meta[meta["group"].astype(str).str.contains("breast", case=False, na=False)].copy()

rows = []
sample_sheet_rows = []

for _, r in bc.iterrows():
    sample = r["sample_id"]

    r1_raw = str(r.get("fastq_r1", "") or "")
    r2_raw = str(r.get("fastq_r2", "") or "")

    r1s = [x.strip() for x in r1_raw.replace(",", ";").split(";") if x.strip() and x.strip().lower() != "nan"]
    r2s = [x.strip() for x in r2_raw.replace(",", ";").split(";") if x.strip() and x.strip().lower() != "nan"]

    r1_ok = []
    r2_ok = []

    for p in r1s:
        pp = Path(p).expanduser()
        rows.append({
            "sample_id": sample,
            "mate": "R1",
            "path": str(pp),
            "exists": "YES" if pp.exists() else "NO",
            "size_mb": round(pp.stat().st_size / 1024 / 1024, 2) if pp.exists() else "."
        })
        if pp.exists():
            r1_ok.append(str(pp))

    for p in r2s:
        pp = Path(p).expanduser()
        rows.append({
            "sample_id": sample,
            "mate": "R2",
            "path": str(pp),
            "exists": "YES" if pp.exists() else "NO",
            "size_mb": round(pp.stat().st_size / 1024 / 1024, 2) if pp.exists() else "."
        })
        if pp.exists():
            r2_ok.append(str(pp))

    sample_sheet_rows.append({
        "sample_id": sample,
        "tumor_fastq_r1": ",".join(r1_ok) if r1_ok else ".",
        "tumor_fastq_r2": ",".join(r2_ok) if r2_ok else ".",
        "n_r1_found": len(r1_ok),
        "n_r2_found": len(r2_ok),
        "ready_for_fastq_pipeline": "YES" if len(r1_ok) > 0 and len(r2_ok) > 0 else "NO",
        "stage": r.get("bc_pathologic_stage", "."),
        "molecular_subtype": r.get("molecular_subtype", "."),
        "er_status": r.get("er_status", "."),
        "pr_status": r.get("pr_status", "."),
        "her_2_status": r.get("her_2_status", ".")
    })

exist = pd.DataFrame(rows)
sheet = pd.DataFrame(sample_sheet_rows)

exist.to_csv(
    "paper_results/05_local_breast_cancer_tumor_WES/local_tumor_wes_fastq_existence.tsv",
    sep="\t",
    index=False
)

sheet.to_csv(
    "paper_results/05_local_breast_cancer_tumor_WES/local_tumor_wes_sample_sheet.tsv",
    sep="\t",
    index=False
)

print("FASTQ existence:")
print(exist.to_string(index=False))
print()
print("Tumor WES sample sheet:")
print(sheet.to_string(index=False))
PY

echo
echo "=== EXISTING LOCAL TUMOR BAM/VCF CHECK ==="
SAMPLE_SHEET="paper_results/05_local_breast_cancer_tumor_WES/local_tumor_wes_sample_sheet.tsv"
if [[ -s "$SAMPLE_SHEET" ]]; then
  tail -n +2 "$SAMPLE_SHEET" | cut -f1 | while IFS= read -r s; do
    [[ -z "$s" ]] && continue
    echo "--- $s"
    find results data -type f 2>/dev/null | grep -E "${s}.*(bam|bai|cram|crai|vcf.gz|g.vcf.gz)$" | sort || true
  done
else
  echo "WARNING: Local tumor sample sheet not found: $SAMPLE_SHEET" >&2
fi

echo
echo "=== LOCAL NON-CANCER GERMLINE BACKGROUND VCF CHECK ==="
BG="results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.autosomal.vcf.gz"
if [[ -s "$BG" ]]; then
  echo "background_vcf=$BG"
  echo -n "background_samples="
  micromamba run -n hadza-wes bcftools query -l "$BG" | wc -l
  echo -n "background_variants="
  micromamba run -n hadza-wes bcftools view -H "$BG" | wc -l
else
  echo "background_vcf_missing=$BG"
fi

echo
echo "=== DECISION SUMMARY ==="
micromamba run -n hadza-wes python - <<'PY'
import pandas as pd

sheet = pd.read_csv(
    "paper_results/05_local_breast_cancer_tumor_WES/local_tumor_wes_sample_sheet.tsv",
    sep="\t",
    dtype=str
)

print(sheet[["sample_id", "ready_for_fastq_pipeline", "n_r1_found", "n_r2_found", "stage", "molecular_subtype"]].to_string(index=False))

ready = (sheet["ready_for_fastq_pipeline"] == "YES").sum()
total = len(sheet)

print()
print(f"READY_TUMOR_FASTQ_SAMPLES={ready}/{total}")

if ready == total:
    print("NEXT_STEP=79B_run_local_tumor_WES_FASTQ_to_BAM_and_tumor_only_Mutect2")
else:
    print("NEXT_STEP=resolve_missing_FASTQ_paths_before_79B")
PY

REPORT="$STATUS/79A_local_tumor_WES_input_audit_status.txt"

{
  echo "79A local breast cancer tumor WES input audit"
  echo "Generated: $(date)"
  echo
  echo "Tumor WES metadata:"
  cat "$OUT/local_tumor_wes_metadata.tsv"
  echo
  echo "Tumor WES sample sheet:"
  cat "$OUT/local_tumor_wes_sample_sheet.tsv"
  echo
  echo "FASTQ existence:"
  cat "$OUT/local_tumor_wes_fastq_existence.tsv"
} > "$REPORT"

echo
echo "=== OUTPUT TABLES ==="
find "$OUT" -maxdepth 1 -type f -name "*.tsv" -printf "%f\t%k KB\n" | sort

echo
echo "=== REPORT ==="
cat "$REPORT" | head -120

echo
echo "=== DONE 79A ==="
date
