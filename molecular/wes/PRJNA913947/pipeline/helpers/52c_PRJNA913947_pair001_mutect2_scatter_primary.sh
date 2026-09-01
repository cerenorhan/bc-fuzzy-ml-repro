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

TUMOR_SM="${PAIR}_T"
NORMAL_SM="${PAIR}_N"

TUMOR_BAM="$BASE/bam/${TUMOR_SM}/${TUMOR_SM}.bqsr.bam"
NORMAL_BAM="$BASE/bam/${NORMAL_SM}/${NORMAL_SM}.bqsr.bam"

PAIR_DIR="$BASE/mutect2/$PAIR"
SHARD_DIR="$PAIR_DIR/shards"
TMP="$BASE/tmp"

CHR_JOBS="${CHR_JOBS:-4}"
THREADS_PER_CHR="${THREADS_PER_CHR:-2}"
JAVA_MEM_PER_CHR="${JAVA_MEM_PER_CHR:-24g}"

LOG="logs/public/PRJNA913947_${PAIR}_mutect2_scatter_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PAIR_DIR" "$SHARD_DIR" "$TMP" logs/public

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 $PAIR scatter Mutect2 started"
date --iso-8601=seconds
echo "CHR_JOBS=$CHR_JOBS"
echo "THREADS_PER_CHR=$THREADS_PER_CHR"
echo "JAVA_MEM_PER_CHR=$JAVA_MEM_PER_CHR"

[[ -s "$TUMOR_BAM" ]] || { echo "ERROR: missing $TUMOR_BAM"; exit 1; }
[[ -s "$NORMAL_BAM" ]] || { echo "ERROR: missing $NORMAL_BAM"; exit 1; }
[[ -s "$TUMOR_BAM.bai" || -s "${TUMOR_BAM%.bam}.bai" ]] || samtools index -@ 8 "$TUMOR_BAM"
[[ -s "$NORMAL_BAM.bai" || -s "${NORMAL_BAM%.bam}.bai" ]] || samtools index -@ 8 "$NORMAL_BAM"

for f in "$REF_FASTA" "$GNOMAD_AF" "$SMALL_EXAC_COMMON"
do
    [[ -s "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

cat > "$PAIR_DIR/primary_contigs.txt" <<'EOC'
chr1
chr2
chr3
chr4
chr5
chr6
chr7
chr8
chr9
chr10
chr11
chr12
chr13
chr14
chr15
chr16
chr17
chr18
chr19
chr20
chr21
chr22
chrX
EOC

run_chr() {
    local chr="$1"
    local outvcf="$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz"
    local f1r2="$SHARD_DIR/${PAIR}.${chr}.f1r2.tar.gz"

    if [[ -s "$outvcf" && -s "$outvcf.tbi" ]]; then
        echo "SKIP $chr: shard exists"
        return 0
    fi

    echo
    echo "=== Mutect2 shard $chr started ==="
    date --iso-8601=seconds

    gatk --java-options "-Xmx${JAVA_MEM_PER_CHR} -Djava.io.tmpdir=$TMP" Mutect2 \
        -R "$REF_FASTA" \
        -L "$chr" \
        -I "$TUMOR_BAM" \
        -tumor "$TUMOR_SM" \
        -I "$NORMAL_BAM" \
        -normal "$NORMAL_SM" \
        --germline-resource "$GNOMAD_AF" \
        --native-pair-hmm-threads "$THREADS_PER_CHR" \
        --f1r2-tar-gz "$f1r2" \
        -O "$outvcf"

    echo "=== Mutect2 shard $chr done ==="
    date --iso-8601=seconds
}

export -f run_chr
export PAIR SHARD_DIR TUMOR_BAM NORMAL_BAM TUMOR_SM NORMAL_SM REF_FASTA GNOMAD_AF TMP JAVA_MEM_PER_CHR THREADS_PER_CHR

echo
echo "=== Running chromosome shards ==="
cat "$PAIR_DIR/primary_contigs.txt" | xargs -I{} -P "$CHR_JOBS" bash -c 'run_chr "$@"' _ {}

echo
echo "=== Check shards ==="
while read -r chr
do
    [[ -s "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz" ]] || { echo "ERROR: missing shard $chr"; exit 1; }
    [[ -s "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz.tbi" ]] || { echo "ERROR: missing shard index $chr"; exit 1; }
done < "$PAIR_DIR/primary_contigs.txt"

echo
echo "=== Merge VCF shards ==="

MERGED="$PAIR_DIR/${PAIR}.mutect2.unfiltered.vcf.gz"
FILTERED="$PAIR_DIR/${PAIR}.mutect2.filtered.vcf.gz"
PASS_VCF="$PAIR_DIR/${PAIR}.mutect2.PASS.vcf.gz"
ORIENTATION_MODEL="$PAIR_DIR/${PAIR}.read_orientation_model.tar.gz"
TUMOR_PILEUPS="$PAIR_DIR/${PAIR}.tumor.pileups.table"
NORMAL_PILEUPS="$PAIR_DIR/${PAIR}.normal.pileups.table"
CONTAM="$PAIR_DIR/${PAIR}.contamination.table"
SEGMENTS="$PAIR_DIR/${PAIR}.segments.table"
STATS="$PAIR_DIR/${PAIR}.mutect2.PASS.stats.txt"

MERGE_ARGS=()
while read -r chr
do
    MERGE_ARGS+=("-I" "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz")
done < "$PAIR_DIR/primary_contigs.txt"

gatk --java-options "-Xmx32g -Djava.io.tmpdir=$TMP" MergeVcfs \
    "${MERGE_ARGS[@]}" \
    -O "$MERGED"

tabix -f -p vcf "$MERGED"

echo
echo "=== LearnReadOrientationModel ==="

F1R2_ARGS=()
while read -r chr
do
    F1R2_ARGS+=("-I" "$SHARD_DIR/${PAIR}.${chr}.f1r2.tar.gz")
done < "$PAIR_DIR/primary_contigs.txt"

gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" LearnReadOrientationModel \
    "${F1R2_ARGS[@]}" \
    -O "$ORIENTATION_MODEL"

echo
echo "=== Contamination ==="

gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" GetPileupSummaries \
    -I "$TUMOR_BAM" \
    -V "$SMALL_EXAC_COMMON" \
    -L "$SMALL_EXAC_COMMON" \
    -O "$TUMOR_PILEUPS"

gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" GetPileupSummaries \
    -I "$NORMAL_BAM" \
    -V "$SMALL_EXAC_COMMON" \
    -L "$SMALL_EXAC_COMMON" \
    -O "$NORMAL_PILEUPS"

gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" CalculateContamination \
    -I "$TUMOR_PILEUPS" \
    -matched "$NORMAL_PILEUPS" \
    -O "$CONTAM" \
    --tumor-segmentation "$SEGMENTS"

echo
echo "=== FilterMutectCalls ==="

gatk --java-options "-Xmx32g -Djava.io.tmpdir=$TMP" FilterMutectCalls \
    -R "$REF_FASTA" \
    -V "$MERGED" \
    --contamination-table "$CONTAM" \
    --tumor-segmentation "$SEGMENTS" \
    --ob-priors "$ORIENTATION_MODEL" \
    -O "$FILTERED"

bcftools view -f PASS -Oz -o "$PASS_VCF" "$FILTERED"
tabix -f -p vcf "$PASS_VCF"

{
    echo -e "metric\tvalue"
    echo -ne "unfiltered_records\t"; bcftools view -H "$MERGED" | wc -l
    echo -ne "filtered_records\t"; bcftools view -H "$FILTERED" | wc -l
    echo -ne "pass_records\t"; bcftools view -H "$PASS_VCF" | wc -l
} > "$STATS"

echo
echo "=== STATS ==="
cat "$STATS"

echo -e "${PAIR}\t${TUMOR_SM}\t${NORMAL_SM}\t${PASS_VCF}\t$(date --iso-8601=seconds)" >> "$BASE/completed_pairs.scatter.tsv"

echo
echo "DONE"
date --iso-8601=seconds
echo "Output: $PAIR_DIR"
