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

LANES="metadata/local_wes/local_wes.lanes.tsv"

if [[ ! -s "$LANES" ]]; then
    echo "ERROR: Missing lane manifest: $LANES" >&2
    exit 1
fi

if ! command -v fastp >/dev/null 2>&1; then
    echo "ERROR: fastp not found in PATH" >&2
    exit 1
fi

mkdir -p "$TRIMMED_LOCAL"
mkdir -p "$QC_DIR/fastp/local_wes"
mkdir -p "$LOG_DIR/fastp"

JOBS="${JOBS:-2}"
THREADS_PER_JOB="${THREADS_PER_JOB:-8}"

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

    local outdir="$TRIMMED_LOCAL/$sample_id"
    local qcdir="$QC_DIR/fastp/local_wes/$sample_id"

    local out_r1="$outdir/${lane_id}.R1.trimmed.fq.gz"
    local out_r2="$outdir/${lane_id}.R2.trimmed.fq.gz"

    local html="$qcdir/${lane_id}.fastp.html"
    local json="$qcdir/${lane_id}.fastp.json"
    local log="$LOG_DIR/fastp/${lane_id}.fastp.log"

    mkdir -p "$outdir" "$qcdir"

    if [[ -s "$out_r1" && -s "$out_r2" && -s "$json" ]]; then
        echo "SKIP existing: $lane_id"
        return 0
    fi

    echo "RUN fastp: $lane_id"

    nice -n 10 ionice -c2 -n7 fastp \
        --in1 "$r1" \
        --in2 "$r2" \
        --out1 "$out_r1" \
        --out2 "$out_r2" \
        --thread "$THREADS_PER_JOB" \
        --detect_adapter_for_pe \
        --html "$html" \
        --json "$json" \
        --report_title "$lane_id" \
        > "$log" 2>&1

    echo "DONE fastp: $lane_id"
}

echo "Running fastp with:"
echo "  JOBS=$JOBS"
echo "  THREADS_PER_JOB=$THREADS_PER_JOB"
echo

{
    read -r header

    while IFS=$'\t' read -r sample_id group sex lane_id library flowcell lane r1 r2
    do
        [[ -z "${sample_id:-}" ]] && continue

        run_lane "$sample_id" "$group" "$sex" "$lane_id" "$library" "$flowcell" "$lane" "$r1" "$r2" &

        while [[ $(jobs -rp | wc -l) -ge "$JOBS" ]]
        do
            sleep 5
        done

    done

    wait
} < "$LANES"

echo
echo "fastp completed."
date --iso-8601=seconds
