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
OUT="$PAPER/05_local_germline_WES"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/78A_audit_local_germline_WES_publication_inputs_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 78A:LOGDIR/78A_audit_local_germline_WES_publication_inputs_$(date +%Y%m%d_%H%M%S). AUDIT LOCAL GERMLINE WES PUBLICATION INPUTS ==="
date

echo
echo "=== METADATA ==="
for f in \
  metadata/local_wes.tsv \
  metadata/local_wes.csv \
  metadata/*.tsv \
  metadata/*.csv
do
  if [[ -s "$f" ]]; then
    echo "--- $f"
    ls -lh "$f"
    head -5 "$f" || true
    echo
  fi
done

echo
echo "=== GERMLINE RESULT TREE ==="
find results/germline -maxdepth 4 -type f 2>/dev/null | sort | sed 's#^#/#' | head -300

echo
echo "=== KEY VCF / PLINK OUTPUTS ==="
find results/germline -type f \( \
  -name "*.vcf.gz" -o \
  -name "*.vcf.gz.tbi" -o \
  -name "*.bcf" -o \
  -name "*.pgen" -o \
  -name "*.pvar" -o \
  -name "*.psam" -o \
  -name "*.eigenvec" -o \
  -name "*.eigenval" -o \
  -name "*pca*" -o \
  -name "*PCA*" \
\) 2>/dev/null | sort

echo
echo "=== SUMMARY FILES ==="
find results/germline -type f \( \
  -name "*.tsv" -o \
  -name "*.csv" -o \
  -name "*.txt" \
\) 2>/dev/null | sort | head -300

echo
echo "=== VARIANT COUNTS IF VCF EXISTS ==="
for vcf in \
  results/germline/filtered/local_wes_germline_only.pass.vcf.gz \
  results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.vcf.gz \
  results/germline/filtered/local_wes_germline_only.pass.biallelic_snps.autosomal.vcf.gz \
  results/germline/joint/local_wes_germline_only.raw.vcf.gz
do
  if [[ -s "$vcf" ]]; then
    echo "--- $vcf"
    echo -n "samples: "
    micromamba run -n hadza-wes bcftools query -l "$vcf" | wc -l
    echo -n "variants: "
    micromamba run -n hadza-wes bcftools view -H "$vcf" | wc -l
  fi
done

echo
echo "=== PLINK SAMPLE PREVIEW ==="
for psam in results/germline/plink/*.psam; do
  if [[ -s "$psam" ]]; then
    echo "--- $psam"
    head -10 "$psam"
    echo
  fi
done

REPORT="$STATUS/78A_local_germline_WES_publication_input_audit_status.txt"

{
  echo "78A local germline WES publication input audit"
  echo "Generated: $(date)"
  echo
  echo "Metadata files:"
  find metadata -maxdepth 1 -type f \( -name "*.tsv" -o -name "*.csv" \) 2>/dev/null | sort
  echo
  echo "Germline key outputs:"
  find results/germline -type f \( -name "*.vcf.gz" -o -name "*.pgen" -o -name "*.pvar" -o -name "*.psam" -o -name "*pca*" -o -name "*PCA*" \) 2>/dev/null | sort
} > "$REPORT"

echo
echo "=== REPORT ==="
cat "$REPORT"

echo
echo "=== DONE 78A ==="
date
