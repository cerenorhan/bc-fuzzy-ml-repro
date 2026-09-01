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
BAM_BASE="$BASE/bam"

echo "=== Verify BQSR BAMs before cleanup ==="

for pair in candidate_kenya_pair_001 candidate_kenya_pair_002 candidate_kenya_pair_003
do
  for suffix in T N
  do
    sample="${pair}_${suffix}"
    bam="$BAM_BASE/$sample/${sample}.bqsr.bam"
    idx1="${bam}.bai"
    idx2="${bam%.bam}.bai"
    table="$BAM_BASE/$sample/${sample}.bqsr.table"

    echo
    echo "Checking $sample"
    [[ -s "$bam" ]] || { echo "ERROR: missing $bam"; exit 1; }
    [[ -s "$idx1" || -s "$idx2" ]] || { echo "ERROR: missing BQSR index for $sample"; exit 1; }
    [[ -s "$table" ]] || { echo "ERROR: missing $table"; exit 1; }

    ls -lh "$bam" "$table"
  done
done

echo
echo "=== Intermediate BAMs to delete ==="
find "$BAM_BASE" -path "*candidate_kenya_pair_00[123]_*" -type f \
  \( -name "*.sorted.bam" -o -name "*.sorted.bam.bai" -o -name "*.marked.bam" -o -name "*.marked.bai" -o -name "*.marked.bam.bai" \) \
  -printf "%p\t%s bytes\n" | sort

echo
echo "=== Delete intermediate BAMs ==="
find "$BAM_BASE" -path "*candidate_kenya_pair_00[123]_*" -type f \
  \( -name "*.sorted.bam" -o -name "*.sorted.bam.bai" -o -name "*.marked.bam" -o -name "*.marked.bai" -o -name "*.marked.bam.bai" \) \
  -delete

echo
echo "=== Remaining final BQSR BAMs ==="
find "$BAM_BASE" -path "*candidate_kenya_pair_00[123]_*" -type f \
  \( -name "*.bqsr.bam" -o -name "*.bqsr.bam.bai" -o -name "*.bqsr.bai" -o -name "*.bqsr.table" \) \
  -printf "%p\t%s bytes\n" | sort

echo
df -h $PROJECT_ROOT
