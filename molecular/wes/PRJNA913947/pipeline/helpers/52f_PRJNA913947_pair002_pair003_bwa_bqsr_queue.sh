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

PROJECT_TAG="EA_BC_AI_MultiOmics"
BASE="results/${PROJECT_TAG}/PRJNA913947_candidate_kenya_wes"
MANIFEST="$BASE/candidate_kenya_batch_MAX3.tsv"
FASTQ_ROOT="data/public/PRJNA913947_ALL"

THREADS="${RUN_THREADS:-10}"
SORT_THREADS="${SORT_THREADS_PRJNA:-6}"
JAVA_MEM_LOCAL="${JAVA_MEM_PRJNA:-32g}"

LOG="logs/public/PRJNA913947_pair002_pair003_bwa_bqsr_queue_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/public "$BASE/bam" "$BASE/tmp"

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 pair002/pair003 BWA + MarkDuplicates + BQSR queue started"
date --iso-8601=seconds
echo "THREADS=$THREADS"
echo "SORT_THREADS=$SORT_THREADS"
echo "JAVA_MEM_LOCAL=$JAVA_MEM_LOCAL"
echo "MANIFEST=$MANIFEST"

[[ -s "$MANIFEST" ]] || { echo "ERROR: missing manifest: $MANIFEST"; exit 1; }
[[ -s "$REF_FASTA" ]] || { echo "ERROR: missing REF_FASTA: $REF_FASTA"; exit 1; }
[[ -s "$DBSNP" ]] || { echo "ERROR: missing DBSNP: $DBSNP"; exit 1; }
[[ -s "$MILLS" ]] || { echo "ERROR: missing MILLS: $MILLS"; exit 1; }
[[ -s "$KNOWN_SNPS" ]] || { echo "ERROR: missing KNOWN_SNPS: $KNOWN_SNPS"; exit 1; }

resolve_fastq() {
    local x="$1"
    local b
    b=$(basename "$x")

    if [[ -s "$x" ]]; then
        readlink -f "$x"
        return 0
    fi

    if [[ -s "$FASTQ_ROOT/$b" ]]; then
        readlink -f "$FASTQ_ROOT/$b"
        return 0
    fi

    local hit
    hit=$(find "$FASTQ_ROOT" -maxdepth 2 -type f -name "$b" -print -quit 2>/dev/null || true)
    if [[ -n "$hit" && -s "$hit" ]]; then
        readlink -f "$hit"
        return 0
    fi

    echo "ERROR_FASTQ_NOT_FOUND:$x"
    return 1
}

process_sample() {
    local pair="$1"
    local suffix="$2"
    local run="$3"
    local r1_in="$4"
    local r2_in="$5"

    local sample="${pair}_${suffix}"
    local outdir="$BASE/bam/$sample"
    mkdir -p "$outdir"

    local r1 r2
    r1=$(resolve_fastq "$r1_in")
    r2=$(resolve_fastq "$r2_in")

    local sorted="$outdir/${sample}.sorted.bam"
    local marked="$outdir/${sample}.marked.bam"
    local metrics="$outdir/${sample}.marked.metrics.txt"
    local recal="$outdir/${sample}.bqsr.table"
    local bqsr="$outdir/${sample}.bqsr.bam"

    echo
    echo "=============================================="
    echo "Sample: $sample"
    echo "Run: $run"
    echo "R1: $r1"
    echo "R2: $r2"
    echo "Output: $outdir"
    echo "=============================================="

    if [[ -s "$bqsr" && -s "${bqsr}.bai" ]]; then
        echo "SKIP $sample: BQSR BAM exists"
        return 0
    fi

    if [[ ! -s "$sorted" ]]; then
        echo "=== [$sample] BWA-MEM2 alignment + samtools sort ==="
        bwa-mem2 mem \
            -t "$THREADS" \
            -R "@RG\tID:${sample}\tSM:${sample}\tPL:ILLUMINA\tLB:${sample}\tPU:${run}" \
            "$REF_FASTA" \
            "$r1" "$r2" \
        | samtools sort -@ "$SORT_THREADS" -m 3G -o "$sorted" -
    else
        echo "SKIP alignment: sorted BAM exists"
    fi

    if [[ ! -s "${sorted}.bai" ]]; then
        samtools index -@ "$SORT_THREADS" "$sorted"
    fi

    if [[ ! -s "$marked" ]]; then
        echo "=== [$sample] MarkDuplicates ==="
        gatk --java-options "-Xmx${JAVA_MEM_LOCAL} -Djava.io.tmpdir=$BASE/tmp" MarkDuplicates \
            -I "$sorted" \
            -O "$marked" \
            -M "$metrics" \
            --CREATE_INDEX true
    else
        echo "SKIP MarkDuplicates: marked BAM exists"
    fi

    if [[ ! -s "$recal" ]]; then
        echo "=== [$sample] BaseRecalibrator ==="
        gatk --java-options "-Xmx${JAVA_MEM_LOCAL} -Djava.io.tmpdir=$BASE/tmp" BaseRecalibrator \
            -R "$REF_FASTA" \
            -I "$marked" \
            --known-sites "$DBSNP" \
            --known-sites "$MILLS" \
            --known-sites "$KNOWN_SNPS" \
            -O "$recal"
    else
        echo "SKIP BaseRecalibrator: recal table exists"
    fi

    if [[ ! -s "$bqsr" ]]; then
        echo "=== [$sample] ApplyBQSR ==="
        gatk --java-options "-Xmx${JAVA_MEM_LOCAL} -Djava.io.tmpdir=$BASE/tmp" ApplyBQSR \
            -R "$REF_FASTA" \
            -I "$marked" \
            --bqsr-recal-file "$recal" \
            -O "$bqsr"
    else
        echo "SKIP ApplyBQSR: BQSR BAM exists"
    fi

    if [[ ! -s "${bqsr}.bai" ]]; then
        samtools index -@ "$SORT_THREADS" "$bqsr"
    fi

    echo "DONE sample $sample"
    ls -lh "$outdir" | sed -n '1,80p'
}

echo
echo "=== Selected manifest rows: pair002 and pair003 ==="
awk -F'\t' 'NR==1 || $1=="candidate_kenya_pair_002" || $1=="candidate_kenya_pair_003"' "$MANIFEST"

echo
echo "=== Start queue ==="

awk -F'\t' 'NR>1 && ($1=="candidate_kenya_pair_002" || $1=="candidate_kenya_pair_003")' "$MANIFEST" \
| while IFS=$'\t' read -r analysis_pair_id pair_id pair_rule tumor_run normal_run tumor_sample normal_sample tumor_numeric normal_numeric absdiff tumor_R1 tumor_R2 normal_R1 normal_R2 tumor_size_gb normal_size_gb
do
    echo
    echo "##############################################"
    echo "PAIR: $analysis_pair_id"
    echo "Tumor:  $tumor_run / $tumor_sample"
    echo "Normal: $normal_run / $normal_sample"
    echo "##############################################"

    process_sample "$analysis_pair_id" "T" "$tumor_run" "$tumor_R1" "$tumor_R2"
    process_sample "$analysis_pair_id" "N" "$normal_run" "$normal_R1" "$normal_R2"

    echo "DONE pair alignment/BQSR: $analysis_pair_id"
done

echo
echo "=== Final BQSR BAM count ==="
find "$BASE/bam" -path "*candidate_kenya_pair_00[23]_*/*.bqsr.bam" -printf "%p\t%s bytes\n" | sort

echo
echo "DONE PRJNA pair002/pair003 BWA+BQSR queue"
date --iso-8601=seconds
