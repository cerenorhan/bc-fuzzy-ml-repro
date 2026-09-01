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

PAIR_TSV="metadata/public/PRJNA913947/pairing_inference/PRJNA913947_likely_kenya_low_numeric_le46_pairs.tsv"
FASTQ_DIR="data/public/PRJNA913947_ALL"

MAX_PAIRS="${MAX_PAIRS:-3}"

OUT_BASE="results/${PROJECT_TAG}/PRJNA913947_candidate_kenya_wes"
TRIM_DIR="$OUT_BASE/trimmed"
BAM_DIR="$OUT_BASE/bam"
VCF_DIR="$OUT_BASE/mutect2"
QC_DIR="$OUT_BASE/qc"
TMP_DIR="$OUT_BASE/tmp"
LOG_DIR="logs/public"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/PRJNA913947_candidate_kenya_mutect2_batch_${RUN_ID}.log"

mkdir -p "$TRIM_DIR" "$BAM_DIR" "$VCF_DIR" "$QC_DIR" "$TMP_DIR" "$LOG_DIR"

exec > >(tee -a "$LOG") 2>&1

send_alert() {
    local subject="$1"
    local body_file="$2"
    if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
        bash "$NOTIFICATION_HELPER" "$subject" "$body_file" || true
    fi
}

fail_alert() {
    local code="$?"
    local body
    body="$(mktemp)"
    {
        echo "PRJNA913947 candidate Kenya Mutect2 batch FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Exit code: $code"
        echo "Log: $LOG"
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
        echo
        tail -n 120 "$LOG" || true
    } > "$body"
    send_alert "FAILED PRJNA913947 candidate Kenya Mutect2 batch" "$body"
    rm -f "$body"
}
trap fail_alert ERR

echo "PRJNA913947 candidate Kenya Mutect2 batch started"
date --iso-8601=seconds
echo "PROJECT_TAG=$PROJECT_TAG"
echo "PAIR_TSV=$PAIR_TSV"
echo "FASTQ_DIR=$FASTQ_DIR"
echo "MAX_PAIRS=$MAX_PAIRS"
echo "LOG=$LOG"

[[ -s "$PAIR_TSV" ]] || { echo "ERROR: missing $PAIR_TSV"; exit 1; }
[[ -d "$FASTQ_DIR" ]] || { echo "ERROR: missing $FASTQ_DIR"; exit 1; }

for f in "$REF_FASTA" "$DBSNP" "$MILLS" "$KNOWN_SNPS" "$GNOMAD_AF" "$SMALL_EXAC_COMMON"
do
    [[ -s "$f" ]] || { echo "ERROR: missing reference/resource $f"; exit 1; }
done

echo
echo "=== Disk before run ==="
df -h "$PROJECT_ROOT"

echo
echo "=== Software versions ==="
bwa-mem2 version || true
samtools --version | head -n 2 || true
gatk --version || true
fastp --version || true
bcftools --version | head -n 2 || true

echo
echo "=== Create batch manifest ==="

BATCH_MANIFEST="$OUT_BASE/candidate_kenya_batch_MAX${MAX_PAIRS}.tsv"

python - "$PAIR_TSV" "$BATCH_MANIFEST" "$MAX_PAIRS" <<'PY'
import sys
import pandas as pd
from pathlib import Path

inp = Path(sys.argv[1])
outp = Path(sys.argv[2])
max_pairs = int(sys.argv[3])

df = pd.read_csv(inp, sep="\t", dtype=str).fillna("")
df = df.head(max_pairs).copy()

df["analysis_pair_id"] = [f"candidate_kenya_pair_{i+1:03d}" for i in range(len(df))]

cols = ["analysis_pair_id"] + [c for c in df.columns if c != "analysis_pair_id"]
df = df[cols]
df.to_csv(outp, sep="\t", index=False)

print(df[["analysis_pair_id", "tumor_run", "normal_run", "tumor_sample", "normal_sample"]].to_string(index=False))
PY

echo
echo "Batch manifest:"
cat "$BATCH_MANIFEST"

COMPLETED="$OUT_BASE/completed_pairs.tsv"
FAILED="$OUT_BASE/failed_pairs.tsv"

touch "$COMPLETED" "$FAILED"

process_sample() {
    local pair_id="$1"
    local role="$2"
    local run="$3"
    local r1_name="$4"
    local r2_name="$5"

    local sample_id="${pair_id}_${role}"
    local sample_trim="$TRIM_DIR/$sample_id"
    local sample_bam="$BAM_DIR/$sample_id"
    local sample_qc="$QC_DIR/$sample_id"

    mkdir -p "$sample_trim" "$sample_bam" "$sample_qc"

    local r1="$FASTQ_DIR/$r1_name"
    local r2="$FASTQ_DIR/$r2_name"

    [[ -s "$r1" ]] || { echo "ERROR: missing $r1"; exit 1; }
    [[ -s "$r2" ]] || { echo "ERROR: missing $r2"; exit 1; }

    local trim_r1="$sample_trim/${sample_id}_R1.trim.fastq.gz"
    local trim_r2="$sample_trim/${sample_id}_R2.trim.fastq.gz"

    local sorted_bam="$sample_bam/${sample_id}.sorted.bam"
    local marked_bam="$sample_bam/${sample_id}.marked.bam"
    local bqsr_table="$sample_bam/${sample_id}.bqsr.table"
    local bqsr_bam="$sample_bam/${sample_id}.bqsr.bam"

    echo
    echo "=== [$sample_id] fastp ==="

    if [[ ! -s "$trim_r1" || ! -s "$trim_r2" ]]; then
        fastp \
            -i "$r1" \
            -I "$r2" \
            -o "$trim_r1" \
            -O "$trim_r2" \
            --thread 8 \
            --detect_adapter_for_pe \
            --json "$sample_qc/${sample_id}.fastp.json" \
            --html "$sample_qc/${sample_id}.fastp.html"
    else
        echo "SKIP fastp: trimmed FASTQ exists"
    fi

    echo
    echo "=== [$sample_id] BWA-MEM2 alignment ==="

    if [[ ! -s "$sorted_bam" ]]; then
        bwa-mem2 mem \
            -t "${ALIGN_THREADS:-24}" \
            -R "@RG\tID:${run}\tSM:${sample_id}\tPL:ILLUMINA\tLB:${run}\tPU:${run}" \
            "$REF_FASTA" \
            "$trim_r1" \
            "$trim_r2" \
        | samtools sort \
            -@ "${SORT_THREADS:-8}" \
            -m 3G \
            -o "$sorted_bam" -
    else
        echo "SKIP alignment: sorted BAM exists"
    fi

    samtools quickcheck -v "$sorted_bam"

    echo
    echo "=== [$sample_id] MarkDuplicates ==="

    if [[ ! -s "$marked_bam" ]]; then
        gatk --java-options "-Xmx${JAVA_MEM:-64g} -Djava.io.tmpdir=$TMP_DIR" MarkDuplicates \
            -I "$sorted_bam" \
            -O "$marked_bam" \
            -M "$sample_qc/${sample_id}.markdup.metrics.txt" \
            --CREATE_INDEX true
    else
        echo "SKIP MarkDuplicates: marked BAM exists"
    fi

    samtools quickcheck -v "$marked_bam"

    echo
    echo "=== [$sample_id] BQSR ==="

    if [[ ! -s "$bqsr_table" ]]; then
        gatk --java-options "-Xmx${JAVA_MEM:-64g} -Djava.io.tmpdir=$TMP_DIR" BaseRecalibrator \
            -R "$REF_FASTA" \
            -I "$marked_bam" \
            --known-sites "$DBSNP" \
            --known-sites "$MILLS" \
            --known-sites "$KNOWN_SNPS" \
            -O "$bqsr_table"
    else
        echo "SKIP BaseRecalibrator: table exists"
    fi

    if [[ ! -s "$bqsr_bam" ]]; then
        gatk --java-options "-Xmx${JAVA_MEM:-64g} -Djava.io.tmpdir=$TMP_DIR" ApplyBQSR \
            -R "$REF_FASTA" \
            -I "$marked_bam" \
            --bqsr-recal-file "$bqsr_table" \
            -O "$bqsr_bam"
        samtools index -@ "${SORT_THREADS:-8}" "$bqsr_bam"
    else
        echo "SKIP ApplyBQSR: BQSR BAM exists"
    fi

    samtools quickcheck -v "$bqsr_bam"

    echo "$bqsr_bam"
}

echo
echo "=== Start pair processing ==="

tail -n +2 "$BATCH_MANIFEST" | while IFS=$'\t' read -r analysis_pair_id pair_id pair_rule tumor_run normal_run tumor_sample normal_sample tumor_numeric normal_numeric absdiff tumor_R1 tumor_R2 normal_R1 normal_R2 tumor_size_gb normal_size_gb
do
    echo
    echo "############################################################"
    echo "PAIR: $analysis_pair_id"
    echo "Tumor: $tumor_run / $tumor_sample"
    echo "Normal: $normal_run / $normal_sample"
    echo "############################################################"

    PAIR_DIR="$VCF_DIR/$analysis_pair_id"
    mkdir -p "$PAIR_DIR"

    TUMOR_SM="${analysis_pair_id}_T"
    NORMAL_SM="${analysis_pair_id}_N"

    TUMOR_BQSR="$BAM_DIR/${TUMOR_SM}/${TUMOR_SM}.bqsr.bam"
    NORMAL_BQSR="$BAM_DIR/${NORMAL_SM}/${NORMAL_SM}.bqsr.bam"

    process_sample "$analysis_pair_id" "T" "$tumor_run" "$tumor_R1" "$tumor_R2"
    process_sample "$analysis_pair_id" "N" "$normal_run" "$normal_R1" "$normal_R2"

    echo
    echo "=== [$analysis_pair_id] Mutect2 paired tumor-normal ==="

    UNFILTERED="$PAIR_DIR/${analysis_pair_id}.mutect2.unfiltered.vcf.gz"
    F1R2="$PAIR_DIR/${analysis_pair_id}.f1r2.tar.gz"
    ORIENTATION_MODEL="$PAIR_DIR/${analysis_pair_id}.read_orientation_model.tar.gz"
    TUMOR_PILEUPS="$PAIR_DIR/${analysis_pair_id}.tumor.pileups.table"
    NORMAL_PILEUPS="$PAIR_DIR/${analysis_pair_id}.normal.pileups.table"
    CONTAM="$PAIR_DIR/${analysis_pair_id}.contamination.table"
    SEGMENTS="$PAIR_DIR/${analysis_pair_id}.segments.table"
    FILTERED="$PAIR_DIR/${analysis_pair_id}.mutect2.filtered.vcf.gz"
    PASS_VCF="$PAIR_DIR/${analysis_pair_id}.mutect2.PASS.vcf.gz"
    STATS="$PAIR_DIR/${analysis_pair_id}.mutect2.PASS.stats.txt"

    if [[ ! -s "$UNFILTERED" ]]; then
        gatk --java-options "-Xmx${JAVA_MEM:-64g} -Djava.io.tmpdir=$TMP_DIR" Mutect2 \
            -R "$REF_FASTA" \
            -I "$TUMOR_BQSR" \
            -I "$NORMAL_BQSR" \
            -normal "$NORMAL_SM" \
            --germline-resource "$GNOMAD_AF" \
            --native-pair-hmm-threads "${GATK_THREADS:-8}" \
            --f1r2-tar-gz "$F1R2" \
            -O "$UNFILTERED"
    else
        echo "SKIP Mutect2: unfiltered VCF exists"
    fi

    if [[ ! -s "$ORIENTATION_MODEL" ]]; then
        gatk --java-options "-Xmx16g -Djava.io.tmpdir=$TMP_DIR" LearnReadOrientationModel \
            -I "$F1R2" \
            -O "$ORIENTATION_MODEL"
    fi

    echo
    echo "=== [$analysis_pair_id] Contamination estimates ==="

    if [[ ! -s "$TUMOR_PILEUPS" ]]; then
        gatk --java-options "-Xmx16g -Djava.io.tmpdir=$TMP_DIR" GetPileupSummaries \
            -I "$TUMOR_BQSR" \
            -V "$SMALL_EXAC_COMMON" \
            -L "$SMALL_EXAC_COMMON" \
            -O "$TUMOR_PILEUPS"
    fi

    if [[ ! -s "$NORMAL_PILEUPS" ]]; then
        gatk --java-options "-Xmx16g -Djava.io.tmpdir=$TMP_DIR" GetPileupSummaries \
            -I "$NORMAL_BQSR" \
            -V "$SMALL_EXAC_COMMON" \
            -L "$SMALL_EXAC_COMMON" \
            -O "$NORMAL_PILEUPS"
    fi

    if [[ ! -s "$CONTAM" ]]; then
        gatk --java-options "-Xmx16g -Djava.io.tmpdir=$TMP_DIR" CalculateContamination \
            -I "$TUMOR_PILEUPS" \
            -matched "$NORMAL_PILEUPS" \
            -O "$CONTAM" \
            --tumor-segmentation "$SEGMENTS"
    fi

    echo
    echo "=== [$analysis_pair_id] FilterMutectCalls ==="

    if [[ ! -s "$FILTERED" ]]; then
        gatk --java-options "-Xmx${JAVA_MEM:-64g} -Djava.io.tmpdir=$TMP_DIR" FilterMutectCalls \
            -R "$REF_FASTA" \
            -V "$UNFILTERED" \
            --contamination-table "$CONTAM" \
            --tumor-segmentation "$SEGMENTS" \
            --ob-priors "$ORIENTATION_MODEL" \
            -O "$FILTERED"
    fi

    if [[ ! -s "$PASS_VCF" ]]; then
        bcftools view -f PASS -Oz -o "$PASS_VCF" "$FILTERED"
        tabix -p vcf "$PASS_VCF"
    fi

    {
        echo -e "metric\tvalue"
        echo -ne "unfiltered_records\t"; bcftools view -H "$UNFILTERED" | wc -l
        echo -ne "filtered_records\t"; bcftools view -H "$FILTERED" | wc -l
        echo -ne "pass_records\t"; bcftools view -H "$PASS_VCF" | wc -l
    } > "$STATS"

    echo
    echo "=== [$analysis_pair_id] stats ==="
    cat "$STATS"

    echo -e "${analysis_pair_id}\t${tumor_run}\t${normal_run}\t${PASS_VCF}\t$(date --iso-8601=seconds)" >> "$COMPLETED"

    echo
    echo "=== [$analysis_pair_id] cleanup large intermediates ==="

    # Raw FASTQ remains untouched. Final VCF/QC remains.
    rm -rf "$TRIM_DIR/${TUMOR_SM}" "$TRIM_DIR/${NORMAL_SM}"
    rm -f "$BAM_DIR/${TUMOR_SM}/${TUMOR_SM}.sorted.bam" "$BAM_DIR/${TUMOR_SM}/${TUMOR_SM}.marked.bam" "$BAM_DIR/${TUMOR_SM}/${TUMOR_SM}.marked.bai" "$BAM_DIR/${TUMOR_SM}/${TUMOR_SM}.bqsr.bam" "$BAM_DIR/${TUMOR_SM}/${TUMOR_SM}.bqsr.bam.bai"
    rm -f "$BAM_DIR/${NORMAL_SM}/${NORMAL_SM}.sorted.bam" "$BAM_DIR/${NORMAL_SM}/${NORMAL_SM}.marked.bam" "$BAM_DIR/${NORMAL_SM}/${NORMAL_SM}.marked.bai" "$BAM_DIR/${NORMAL_SM}/${NORMAL_SM}.bqsr.bam" "$BAM_DIR/${NORMAL_SM}/${NORMAL_SM}.bqsr.bam.bai"

    echo
    echo "Disk after $analysis_pair_id:"
    df -h "$PROJECT_ROOT"
done

echo
echo "=== Batch completed ==="

SUMMARY="$OUT_BASE/batch_summary_MAX${MAX_PAIRS}.txt"
{
    echo "PRJNA913947 candidate Kenya Mutect2 batch completed"
    echo "Time: $(date --iso-8601=seconds)"
    echo "MAX_PAIRS=$MAX_PAIRS"
    echo "Log: $LOG"
    echo
    echo "Completed pairs:"
    cat "$COMPLETED"
    echo
    echo "Output directory:"
    echo "$OUT_BASE"
    echo
    echo "PASS VCF count:"
    find "$VCF_DIR" -name "*.mutect2.PASS.vcf.gz" | wc -l
    echo
    echo "Disk:"
    df -h "$PROJECT_ROOT"
} > "$SUMMARY"

cat "$SUMMARY"

BODY="$(mktemp)"
cp "$SUMMARY" "$BODY"
send_alert "SUCCESS PRJNA913947 candidate Kenya Mutect2 batch" "$BODY"
rm -f "$BODY"

echo
echo "PRJNA913947 candidate Kenya Mutect2 batch finished"
date --iso-8601=seconds
