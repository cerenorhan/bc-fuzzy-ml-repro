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

mkdir -p logs/local_wes
mkdir -p metadata/local_wes
mkdir -p "$BAM_DIR/markdup"
mkdir -p "$BAM_DIR/markdup_metrics"
mkdir -p "$QC_DIR/samtools/markdup"
mkdir -p "$QC_DIR/multiqc/local_wes_bam"
mkdir -p "$TMP_DIR/gatk"

AUTO_LOG="logs/local_wes/auto_after_bwa_local_wes_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$AUTO_LOG") 2>&1

send_alert() {
    local subject="$1"
    local body_file="$2"

    if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
        bash "$NOTIFICATION_HELPER" "$subject" "$body_file" || true
    fi
}

fail_alert() {
    local exit_code="$?"
    local body
    body="$(mktemp)"

    {
        echo "Hadza WES automatic post-BWA pipeline FAILED"
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Host: $(hostname)"
        echo "Exit code: $exit_code"
        echo "Working directory: $(pwd)"
        echo "Log: $AUTO_LOG"
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
        echo
        echo "Counts:"
        echo -n "Lane BAM: "
        find results/germline/bam/lanes -name "*.sorted.bam" 2>/dev/null | wc -l || true
        echo -n "Lane BAI: "
        find results/germline/bam/lanes -name "*.sorted.bam.bai" 2>/dev/null | wc -l || true
        echo -n "Marked BAM: "
        find results/germline/bam/markdup -name "*.marked.bam" 2>/dev/null | wc -l || true
        echo -n "Marked BAI: "
        find results/germline/bam/markdup -name "*.marked.bam.bai" 2>/dev/null | wc -l || true
        echo
        echo "Active processes:"
        pgrep -af "bwa-mem2|samtools sort|MarkDuplicates|aria2c|PRJNA913947|42_download_rnaseq" || true
        echo
        echo "Last 180 lines of auto log:"
        tail -n 180 "$AUTO_LOG" || true
    } > "$body"

    send_alert "FAILED post-BWA pipeline" "$body"
    rm -f "$body"
}

trap fail_alert ERR

echo "Automatic post-BWA local WES pipeline started"
date --iso-8601=seconds
echo "Log: $AUTO_LOG"

LANES="metadata/local_wes/local_wes.trimmed_lanes.tsv"

if [[ ! -s "$LANES" ]]; then
    echo "ERROR: Missing trimmed lane manifest: $LANES" >&2
    exit 1
fi

EXPECTED_LANES=$(( $(wc -l < "$LANES") - 1 ))
EXPECTED_SAMPLES=$(tail -n +2 "$LANES" | cut -f1 | sort -u | wc -l)

echo
echo "Expected lanes: $EXPECTED_LANES"
echo "Expected samples: $EXPECTED_SAMPLES"

echo
echo "Waiting for BWA lane alignment to finish..."

while true
do
    BAM_COUNT=$(find results/germline/bam/lanes -name "*.sorted.bam" 2>/dev/null | wc -l)
    BAI_COUNT=$(find results/germline/bam/lanes -name "*.sorted.bam.bai" 2>/dev/null | wc -l)

    echo "$(date --iso-8601=seconds) lane BAM=$BAM_COUNT/$EXPECTED_LANES lane BAI=$BAI_COUNT/$EXPECTED_LANES"

    if [[ "$BAM_COUNT" -eq "$EXPECTED_LANES" && "$BAI_COUNT" -eq "$EXPECTED_LANES" ]]; then
        echo "All expected lane BAM/BAI files are present."
        break
    fi

    if ! pgrep -af "bwa-mem2 mem|samtools sort|17_align_local_wes_lanes_bwa.sh" >/dev/null 2>&1; then
        echo "No active BWA process detected. Rechecking in 120 seconds..."
        sleep 120

        BAM_COUNT2=$(find results/germline/bam/lanes -name "*.sorted.bam" 2>/dev/null | wc -l)
        BAI_COUNT2=$(find results/germline/bam/lanes -name "*.sorted.bam.bai" 2>/dev/null | wc -l)

        if [[ "$BAM_COUNT2" -lt "$EXPECTED_LANES" || "$BAI_COUNT2" -lt "$EXPECTED_LANES" ]]; then
            echo "ERROR: BWA appears stopped before expected lane BAM count."
            echo "BAM=$BAM_COUNT2/$EXPECTED_LANES BAI=$BAI_COUNT2/$EXPECTED_LANES"
            exit 1
        fi
    fi

    sleep 300
done

echo
echo "Creating lane BAM list..."
find results/germline/bam/lanes -name "*.sorted.bam" \
  | sort \
  > metadata/local_wes/local_wes.lane_bams.list

echo "Lane BAM list count:"
wc -l metadata/local_wes/local_wes.lane_bams.list

echo
echo "Running samtools quickcheck on all lane BAMs..."
samtools quickcheck -v $(cat metadata/local_wes/local_wes.lane_bams.list)

echo "All lane BAM quickcheck OK"

echo
echo "Running MarkDuplicates per sample..."

SAMPLE_LIST="metadata/local_wes/local_wes.samples_for_markdup.txt"

tail -n +2 "$LANES" \
  | cut -f1 \
  | sort -u \
  > "$SAMPLE_LIST"

while read -r sample_id
do
    [[ -z "$sample_id" ]] && continue

    echo
    echo "========================================"
    echo "Sample: $sample_id"
    echo "========================================"

    outbam="$BAM_DIR/markdup/${sample_id}.marked.bam"
    metrics="$BAM_DIR/markdup_metrics/${sample_id}.markdup.metrics.txt"
    flagstat="$QC_DIR/samtools/markdup/${sample_id}.marked.flagstat.txt"

    if [[ -s "$outbam" && -s "${outbam}.bai" && -s "$metrics" ]]; then
        echo "SKIP existing markdup BAM: $sample_id"
    else
        mapfile -t lane_ids < <(
            awk -F'\t' -v s="$sample_id" 'NR > 1 && $1 == s {print $4}' "$LANES"
        )

        if [[ "${#lane_ids[@]}" -eq 0 ]]; then
            echo "ERROR: No lane IDs found for sample: $sample_id" >&2
            exit 1
        fi

        inputs=()

        for lane_id in "${lane_ids[@]}"
        do
            lane_bam="$BAM_DIR/lanes/$sample_id/${lane_id}.sorted.bam"

            if [[ ! -s "$lane_bam" ]]; then
                echo "ERROR: Missing lane BAM: $lane_bam" >&2
                exit 1
            fi

            inputs+=("-I" "$lane_bam")
        done

        echo "Input lane BAM count for $sample_id: $(( ${#inputs[@]} / 2 ))"
        echo "Output: $outbam"

        nice -n 10 ionice -c2 -n7 \
        gatk --java-options "-Xmx32g -Djava.io.tmpdir=$TMP_DIR/gatk" \
            MarkDuplicates \
            "${inputs[@]}" \
            -O "$outbam" \
            -M "$metrics" \
            --CREATE_INDEX false \
            --VALIDATION_STRINGENCY SILENT \
            --ASSUME_SORT_ORDER coordinate \
            --TMP_DIR "$TMP_DIR/gatk"

        samtools index -@ 8 "$outbam"
    fi

    samtools quickcheck -v "$outbam"
    samtools flagstat -@ 8 "$outbam" > "$flagstat"

    echo "DONE MarkDuplicates: $sample_id"

done < "$SAMPLE_LIST"

echo
echo "Creating markdup BAM list..."
find "$BAM_DIR/markdup" -name "*.marked.bam" \
  | sort \
  > metadata/local_wes/local_wes.markdup_bams.list

echo "Markdup BAM count:"
wc -l metadata/local_wes/local_wes.markdup_bams.list

MARKDUP_COUNT=$(wc -l < metadata/local_wes/local_wes.markdup_bams.list)

if [[ "$MARKDUP_COUNT" -ne "$EXPECTED_SAMPLES" ]]; then
    echo "ERROR: Markdup BAM count does not match expected sample count."
    echo "MARKDUP_COUNT=$MARKDUP_COUNT EXPECTED_SAMPLES=$EXPECTED_SAMPLES"
    exit 1
fi

echo
echo "Running samtools quickcheck on all markdup BAMs..."
samtools quickcheck -v $(cat metadata/local_wes/local_wes.markdup_bams.list)

echo
echo "Running MultiQC for local WES BAM QC..."
multiqc \
  "$QC_DIR/samtools/markdup" \
  "$BAM_DIR/markdup_metrics" \
  -o "$QC_DIR/multiqc/local_wes_bam" \
  -n local_wes_bam_markdup_multiqc.html

echo
echo "Final disk status:"
df -h "$PROJECT_ROOT"

body="$(mktemp)"

{
    echo "Hadza WES automatic post-BWA pipeline completed successfully."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Log: $AUTO_LOG"
    echo
    echo "Lane BAM count:"
    wc -l metadata/local_wes/local_wes.lane_bams.list
    echo
    echo "Markdup BAM count:"
    wc -l metadata/local_wes/local_wes.markdup_bams.list
    echo
    echo "MultiQC report:"
    echo "$QC_DIR/multiqc/local_wes_bam/local_wes_bam_markdup_multiqc.html"
    echo
    echo "Disk:"
    df -h "$PROJECT_ROOT"
    echo
    echo "Downloads still running:"
    pgrep -af "aria2c|PRJNA913947|42_download_rnaseq" || true
} > "$body"

send_alert "SUCCESS post-BWA pipeline" "$body"
rm -f "$body"

echo
echo "Automatic post-BWA local WES pipeline completed successfully."
date --iso-8601=seconds
