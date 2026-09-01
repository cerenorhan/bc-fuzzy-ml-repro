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

cd "$PROJECT_ROOT"

LANES="metadata/local_wes/local_wes.trimmed_lanes.tsv"

if [[ ! -s "$LANES" ]]; then
    echo "ERROR: Missing trimmed lane manifest: $LANES" >&2
    exit 1
fi

if [[ ! -s "$REF_FASTA" ]]; then
    echo "ERROR: Missing reference FASTA: $REF_FASTA" >&2
    exit 1
fi

if [[ ! -s "${REF_FASTA}.bwt.2bit.64" ]]; then
    echo "ERROR: Missing BWA-MEM2 index for: $REF_FASTA" >&2
    exit 1
fi

mkdir -p "$BAM_DIR/lanes"
mkdir -p "$LOG_DIR/bwa"
mkdir -p "$TMP_DIR/bwa"

JOBS="${JOBS:-1}"
BWA_THREADS="${BWA_THREADS:-16}"
SAMTOOLS_SORT_THREADS="${SAMTOOLS_SORT_THREADS:-4}"
MAX_LANES="${MAX_LANES:-0}"

run_lane() {
    local sample_id="$1"
    local group="$2"
    local sex="$3"
    local lane_id="$4"
    local library="$5"
    local flowcell="$6"
    local lane="$7"
    local r1="$8"
    local r2="$9"

    local outdir="$BAM_DIR/lanes/$sample_id"
    local outbam="$outdir/${lane_id}.sorted.bam"
    local log="$LOG_DIR/bwa/${lane_id}.bwa.log"
    local rg

    mkdir -p "$outdir"

    rg="@RG\tID:${lane_id}\tSM:${sample_id}\tLB:${library}\tPL:ILLUMINA\tPU:${flowcell}.${lane}"

    if [[ -s "$outbam" && -s "${outbam}.bai" ]]; then
        echo "SKIP existing: $lane_id"
        return 0
    fi

    echo "RUN alignment: $lane_id"

    {
        echo "sample_id=$sample_id"
        echo "lane_id=$lane_id"
        echo "r1=$r1"
        echo "r2=$r2"
        echo "outbam=$outbam"
        echo "started=$(date --iso-8601=seconds)"
    } > "$log"

    nice -n 10 ionice -c2 -n7 \
    bwa-mem2 mem \
        -t "$BWA_THREADS" \
        -R "$rg" \
        "$REF_FASTA" \
        "$r1" \
        "$r2" \
    2>> "$log" \
    | nice -n 10 ionice -c2 -n7 \
      samtools sort \
        -@ "$SAMTOOLS_SORT_THREADS" \
        -m 2G \
        -T "$TMP_DIR/bwa/${lane_id}" \
        -o "$outbam" \
        - \
        >> "$log" 2>&1

    samtools index -@ "$SAMTOOLS_SORT_THREADS" "$outbam" >> "$log" 2>&1

    echo "finished=$(date --iso-8601=seconds)" >> "$log"
    echo "DONE alignment: $lane_id"
}

echo "Running BWA-MEM2 alignment with:"
echo "  JOBS=$JOBS"
echo "  BWA_THREADS=$BWA_THREADS"
echo "  SAMTOOLS_SORT_THREADS=$SAMTOOLS_SORT_THREADS"
echo "  MAX_LANES=$MAX_LANES"
echo

count=0

{
    read -r header

    while IFS=$'\t' read -r sample_id group sex lane_id library flowcell lane trimmed_r1 trimmed_r2 trimmed_r1_exists trimmed_r2_exists
    do
        [[ -z "${sample_id:-}" ]] && continue

        count=$((count + 1))

        if [[ "$MAX_LANES" -gt 0 && "$count" -gt "$MAX_LANES" ]]; then
            break
        fi

        run_lane "$sample_id" "$group" "$sex" "$lane_id" "$library" "$flowcell" "$lane" "$trimmed_r1" "$trimmed_r2" &

        while [[ $(jobs -rp | wc -l) -ge "$JOBS" ]]
        do
            sleep 10
        done

    done

    wait
} < "$LANES"

echo
echo "BWA-MEM2 lane alignment completed."
date --iso-8601=seconds
