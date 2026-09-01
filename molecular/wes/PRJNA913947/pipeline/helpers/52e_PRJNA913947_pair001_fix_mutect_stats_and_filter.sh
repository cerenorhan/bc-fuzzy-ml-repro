#!/usr/bin/env bash

# Load portable molecular-analysis paths.
_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${_REPO_ROOT}" ]]; then
    echo "ERROR: Run this script from within the bc-fuzzy-ml-repro repository." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${_REPO_ROOT}/molecular/config/load_config.sh"

set -Eeuo pipefail

cd "$PROJECT_ROOT"

BASE="results/EA_BC_AI_MultiOmics/PRJNA913947_candidate_kenya_wes"
PAIR="candidate_kenya_pair_001"
PAIR_DIR="$BASE/mutect2/$PAIR"
SHARD_DIR="$PAIR_DIR/shards"
TMP="$BASE/tmp"

UNFILTERED="$PAIR_DIR/${PAIR}.mutect2.unfiltered.vcf.gz"
FILTERED="$PAIR_DIR/${PAIR}.mutect2.filtered.vcf.gz"
PASS="$PAIR_DIR/${PAIR}.PASS.vcf.gz"

CONTAM="$PAIR_DIR/${PAIR}.contamination.table"
SEGMENTS="$PAIR_DIR/${PAIR}.segments.table"
ORIENTATION="$PAIR_DIR/${PAIR}.read_orientation_model.tar.gz"
CONTIGS="$PAIR_DIR/primary_contigs.txt"

LOG="logs/public/PRJNA913947_pair001_fix_stats_filter_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/public "$TMP"

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 pair001 fix Mutect stats + FilterMutectCalls"
date --iso-8601=seconds

echo
echo "=== Input check ==="
[[ -s "$UNFILTERED" ]] || { echo "ERROR: missing $UNFILTERED"; exit 1; }
[[ -s "$UNFILTERED.tbi" ]] || { echo "ERROR: missing $UNFILTERED.tbi"; exit 1; }
[[ -s "$CONTAM" ]] || { echo "ERROR: missing $CONTAM"; exit 1; }
[[ -s "$SEGMENTS" ]] || { echo "ERROR: missing $SEGMENTS"; exit 1; }
[[ -s "$ORIENTATION" ]] || { echo "ERROR: missing $ORIENTATION"; exit 1; }
[[ -s "$CONTIGS" ]] || { echo "ERROR: missing $CONTIGS"; exit 1; }

STATS_OUT="${UNFILTERED}.stats"
STATS_ARGS=()

missing=0
while read -r chr
do
    s="$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz.stats"
    if [[ ! -s "$s" ]]; then
        echo "MISSING shard stats: $s"
        missing=1
    fi
    STATS_ARGS+=("-stats" "$s")
done < "$CONTIGS"

[[ "$missing" -eq 0 ]] || { echo "ERROR: missing shard stats"; exit 1; }

echo
echo "=== MergeMutectStats ==="
rm -f "$STATS_OUT"

gatk --java-options "-Xmx8g -Djava.io.tmpdir=$TMP" MergeMutectStats \
  "${STATS_ARGS[@]}" \
  -O "$STATS_OUT"

ls -lh "$STATS_OUT"

echo
echo "=== Remove failed/incomplete filtered outputs ==="
rm -f "$FILTERED" "$FILTERED.tbi" "$PASS" "$PASS.tbi" \
      "$PAIR_DIR/${PAIR}.filtered.bcftools.stats.txt" \
      "$PAIR_DIR/${PAIR}.PASS.bcftools.stats.txt"

echo
echo "=== FilterMutectCalls ==="
gatk --java-options "-Xmx32g -Djava.io.tmpdir=$TMP" FilterMutectCalls \
  -R "$REF_FASTA" \
  -V "$UNFILTERED" \
  --contamination-table "$CONTAM" \
  --tumor-segmentation "$SEGMENTS" \
  --ob-priors "$ORIENTATION" \
  -O "$FILTERED"

tabix -f -p vcf "$FILTERED"

echo
echo "=== PASS VCF ==="
bcftools view -f PASS -Oz -o "$PASS" "$FILTERED"
tabix -f -p vcf "$PASS"

bcftools stats "$FILTERED" > "$PAIR_DIR/${PAIR}.filtered.bcftools.stats.txt"
bcftools stats "$PASS" > "$PAIR_DIR/${PAIR}.PASS.bcftools.stats.txt"

echo
echo "=== Counts ==="
echo -n "Unfiltered variants: "
bcftools view -H "$UNFILTERED" | wc -l

echo -n "Filtered variants: "
bcftools view -H "$FILTERED" | wc -l

echo -n "PASS variants: "
bcftools view -H "$PASS" | wc -l

echo
echo "=== Outputs ==="
ls -lh "$PAIR_DIR" | sed -n '1,80p'

echo
echo "DONE"
date --iso-8601=seconds
