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
VCF_BASE="$BASE/mutect2"
SUMMARY_DIR="$BASE/summary"

CHR_JOBS="${CHR_JOBS:-2}"
THREADS_PER_CHR="${THREADS_PER_CHR:-2}"
JAVA_MEM_PER_CHR="${JAVA_MEM_PER_CHR:-10g}"

mkdir -p "$VCF_BASE" "$SUMMARY_DIR" "$BASE/tmp" logs/public

LOG="logs/public/PRJNA913947_pair002_pair003_mutect2_scatter_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 pair002/pair003 Mutect2 scatter started"
date --iso-8601=seconds
echo "CHR_JOBS=$CHR_JOBS"
echo "THREADS_PER_CHR=$THREADS_PER_CHR"
echo "JAVA_MEM_PER_CHR=$JAVA_MEM_PER_CHR"

PRIMARY_CONTIGS="$BASE/primary_contigs_for_mutect2.txt"
cat > "$PRIMARY_CONTIGS" <<'EOC'
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

    if [[ -s "$outvcf" && -s "${outvcf}.tbi" && -s "${outvcf}.stats" && -s "$f1r2" ]]; then
        echo "SKIP shard $PAIR $chr: exists"
        return 0
    fi

    echo
    echo "=== Mutect2 shard $PAIR $chr started ==="
    date --iso-8601=seconds

    gatk --java-options "-Xmx${JAVA_MEM_PER_CHR} -Djava.io.tmpdir=$TMP" Mutect2 \
        -R "$REF_FASTA" \
        -I "$TUMOR_BAM" \
        -tumor "$TUMOR_SM" \
        -I "$NORMAL_BAM" \
        -normal "$NORMAL_SM" \
        --germline-resource "$GNOMAD_AF" \
        -L "$chr" \
        --native-pair-hmm-threads "$THREADS_PER_CHR" \
        -O "$outvcf" \
        --f1r2-tar-gz "$f1r2"

    echo "=== Mutect2 shard $PAIR $chr done ==="
    date --iso-8601=seconds
}

export -f run_chr

process_pair() {
    local pair="$1"

    PAIR="$pair"
    TUMOR_SM="${PAIR}_T"
    NORMAL_SM="${PAIR}_N"

    TUMOR_BAM="$BAM_BASE/$TUMOR_SM/${TUMOR_SM}.bqsr.bam"
    NORMAL_BAM="$BAM_BASE/$NORMAL_SM/${NORMAL_SM}.bqsr.bam"

    PAIR_DIR="$VCF_BASE/$PAIR"
    SHARD_DIR="$PAIR_DIR/shards"
    TMP="$BASE/tmp/$PAIR"

    mkdir -p "$PAIR_DIR" "$SHARD_DIR" "$TMP"

    UNFILTERED="$PAIR_DIR/${PAIR}.mutect2.unfiltered.vcf.gz"
    MERGED_STATS="$PAIR_DIR/${PAIR}.mutect2.unfiltered.vcf.gz.stats"
    FILTERED="$PAIR_DIR/${PAIR}.mutect2.filtered.vcf.gz"
    PASS="$PAIR_DIR/${PAIR}.PASS.vcf.gz"
    ORIENTATION="$PAIR_DIR/${PAIR}.read_orientation_model.tar.gz"
    TUMOR_PILEUPS="$PAIR_DIR/${PAIR}.tumor.pileups.table"
    NORMAL_PILEUPS="$PAIR_DIR/${PAIR}.normal.pileups.table"
    CONTAM="$PAIR_DIR/${PAIR}.contamination.table"
    SEGMENTS="$PAIR_DIR/${PAIR}.segments.table"

    echo
    echo "##################################################"
    echo "PAIR: $PAIR"
    echo "TUMOR_BAM=$TUMOR_BAM"
    echo "NORMAL_BAM=$NORMAL_BAM"
    echo "PAIR_DIR=$PAIR_DIR"
    echo "##################################################"

    [[ -s "$TUMOR_BAM" ]] || { echo "ERROR: missing tumor BAM $TUMOR_BAM"; exit 1; }
    [[ -s "$NORMAL_BAM" ]] || { echo "ERROR: missing normal BAM $NORMAL_BAM"; exit 1; }

    if [[ -s "$PASS" && -s "${PASS}.tbi" ]]; then
        echo "SKIP $PAIR: PASS VCF already exists"
        return 0
    fi

    export PAIR SHARD_DIR TUMOR_BAM NORMAL_BAM TUMOR_SM NORMAL_SM REF_FASTA GNOMAD_AF TMP JAVA_MEM_PER_CHR THREADS_PER_CHR

    echo
    echo "=== Scatter Mutect2 shards for $PAIR ==="
    cat "$PRIMARY_CONTIGS" | xargs -I{} -P "$CHR_JOBS" bash -c 'run_chr "$@"' _ {}

    echo
    echo "=== Verify shards for $PAIR ==="
    while read -r chr
    do
        [[ -s "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz" ]] || { echo "ERROR: missing shard $chr"; exit 1; }
        [[ -s "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz.tbi" ]] || { echo "ERROR: missing shard index $chr"; exit 1; }
        [[ -s "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz.stats" ]] || { echo "ERROR: missing shard stats $chr"; exit 1; }
        [[ -s "$SHARD_DIR/${PAIR}.${chr}.f1r2.tar.gz" ]] || { echo "ERROR: missing f1r2 $chr"; exit 1; }
    done < "$PRIMARY_CONTIGS"

    echo
    echo "=== Merge VCF shards for $PAIR ==="
    MERGE_ARGS=()
    while read -r chr
    do
        MERGE_ARGS+=("-I" "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz")
    done < "$PRIMARY_CONTIGS"

    gatk --java-options "-Xmx64g -Djava.io.tmpdir=$TMP" MergeVcfs \
        "${MERGE_ARGS[@]}" \
        -O "$UNFILTERED"

    echo
    echo "=== Merge Mutect stats for $PAIR ==="
    STAT_ARGS=()
    while read -r chr
    do
        STAT_ARGS+=("-stats" "$SHARD_DIR/${PAIR}.${chr}.unfiltered.vcf.gz.stats")
    done < "$PRIMARY_CONTIGS"

    gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" MergeMutectStats \
        "${STAT_ARGS[@]}" \
        -O "$MERGED_STATS"

    echo
    echo "=== LearnReadOrientationModel for $PAIR ==="
    F1R2_ARGS=()
    while read -r chr
    do
        F1R2_ARGS+=("-I" "$SHARD_DIR/${PAIR}.${chr}.f1r2.tar.gz")
    done < "$PRIMARY_CONTIGS"

    gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" LearnReadOrientationModel \
        "${F1R2_ARGS[@]}" \
        -O "$ORIENTATION"

    echo
    echo "=== GetPileupSummaries for $PAIR ==="
    gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" GetPileupSummaries \
        -I "$TUMOR_BAM" \
        -V "$GNOMAD_AF" \
        -L "$GNOMAD_AF" \
        -O "$TUMOR_PILEUPS"

    gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" GetPileupSummaries \
        -I "$NORMAL_BAM" \
        -V "$GNOMAD_AF" \
        -L "$GNOMAD_AF" \
        -O "$NORMAL_PILEUPS"

    echo
    echo "=== CalculateContamination for $PAIR ==="
    gatk --java-options "-Xmx24g -Djava.io.tmpdir=$TMP" CalculateContamination \
        -I "$TUMOR_PILEUPS" \
        -matched "$NORMAL_PILEUPS" \
        -O "$CONTAM" \
        --tumor-segmentation "$SEGMENTS"

    echo
    echo "=== FilterMutectCalls for $PAIR ==="
    gatk --java-options "-Xmx48g -Djava.io.tmpdir=$TMP" FilterMutectCalls \
        -R "$REF_FASTA" \
        -V "$UNFILTERED" \
        --stats "$MERGED_STATS" \
        --contamination-table "$CONTAM" \
        --tumor-segmentation "$SEGMENTS" \
        --ob-priors "$ORIENTATION" \
        -O "$FILTERED"

    echo
    echo "=== Extract PASS for $PAIR ==="
    bcftools view -f PASS -Oz -o "$PASS" "$FILTERED"
    tabix -f -p vcf "$PASS"

    bcftools stats "$FILTERED" > "$PAIR_DIR/${PAIR}.filtered.bcftools.stats.txt"
    bcftools stats "$PASS" > "$PAIR_DIR/${PAIR}.PASS.bcftools.stats.txt"

    SUMMARY="$SUMMARY_DIR/${PAIR}_PASS_variant_summary.tsv"
    {
        echo -e "pair\tpass_variants\tsnvs\tindels\tmnps\tothers"
        echo -e "${PAIR}\t$(bcftools view -H "$PASS" | wc -l)\t$(bcftools view -v snps -H "$PASS" | wc -l)\t$(bcftools view -v indels -H "$PASS" | wc -l)\t$(bcftools view -v mnps -H "$PASS" | wc -l)\t$(bcftools view -v other -H "$PASS" | wc -l)"
    } > "$SUMMARY"

    echo
    echo "=== $PAIR summary ==="
    cat "$SUMMARY"

    echo
    echo "DONE pair $PAIR"
    date --iso-8601=seconds
}

process_pair candidate_kenya_pair_002
process_pair candidate_kenya_pair_003

echo
echo "=== All PASS VCFs ==="
find "$VCF_BASE" -name "*.PASS.vcf.gz" -printf "%p\t%s bytes\n" | sort

echo
echo "DONE PRJNA pair002/pair003 Mutect2 scatter"
date --iso-8601=seconds
