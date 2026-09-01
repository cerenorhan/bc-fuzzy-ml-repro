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

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/local_wes/resume_germline_after_bqsr_${RUN_ID}.log"

mkdir -p logs/local_wes metadata/local_wes "$TMP_DIR" "$QC_DIR"

exec > >(tee -a "$LOG") 2>&1

LOCK="$TMP_DIR/resume_germline_after_bqsr.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "ERROR: Another resume germline pipeline is already running."
    exit 1
fi

send_alert() {
    local subject="$1"
    local body_file="$2"

    if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
        bash "$NOTIFICATION_HELPER" "$subject" "$body_file" || true
    fi
}

stage_mail() {
    local stage="$1"
    local body
    body="$(mktemp)"

    {
        echo "Hadza germline-only WES resume pipeline stage completed."
        echo
        echo "Stage: $stage"
        echo "Time: $(date --iso-8601=seconds)"
        echo "Host: $(hostname)"
        echo "Log: $LOG"
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
        echo
        echo "Counts:"
        echo -n "BQSR BAM: "
        find results/germline/bam/bqsr -name "*.bqsr.bam" 2>/dev/null | wc -l || true
        echo -n "gVCF: "
        find results/germline/gvcf -name "*.g.vcf.gz" 2>/dev/null | wc -l || true
        echo -n "Filtered VCF: "
        ls results/germline/filtered/*.vcf.gz 2>/dev/null | wc -l || true
    } > "$body"

    send_alert "STAGE DONE: $stage" "$body"
    rm -f "$body"
}

fail_alert() {
    local exit_code="$?"
    local body
    body="$(mktemp)"

    {
        echo "Hadza germline-only WES resume pipeline FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Host: $(hostname)"
        echo "Exit code: $exit_code"
        echo "Log: $LOG"
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
        echo
        echo "Active processes:"
        pgrep -af "gatk|HaplotypeCaller|GenomicsDBImport|GenotypeGVCFs|VariantFiltration|plink2|aria2c|PRJNA|GSE142" || true
        echo
        echo "Counts:"
        echo -n "BQSR BAM: "
        find results/germline/bam/bqsr -name "*.bqsr.bam" 2>/dev/null | wc -l || true
        echo -n "gVCF: "
        find results/germline/gvcf -name "*.g.vcf.gz" 2>/dev/null | wc -l || true
        echo -n "Filtered VCF: "
        ls results/germline/filtered/*.vcf.gz 2>/dev/null | wc -l || true
        echo
        echo "Last 220 log lines:"
        tail -n 220 "$LOG" || true
    } > "$body"

    send_alert "FAILED germline-only WES resume pipeline" "$body"
    rm -f "$body"
}

trap fail_alert ERR

echo "Hadza germline-only WES resume-after-BQSR pipeline started"
date --iso-8601=seconds
echo "Log: $LOG"

GATK_XMX="${GATK_XMX:-32g}"
HC_THREADS="${HC_THREADS:-4}"
DELETE_GERMLINE_MARKDUP="${DELETE_GERMLINE_MARKDUP:-1}"
DELETE_GERMLINE_BQSR_AFTER_GVCF="${DELETE_GERMLINE_BQSR_AFTER_GVCF:-1}"

MARKDUP_DIR="$BAM_DIR/markdup"
BQSR_BAM_DIR="$BAM_DIR/bqsr"
CALLABLE_DIR="$GERMLINE_DIR/callable"
GVCF_DIR="$GERMLINE_DIR/gvcf"
JOINT_DIR="$GERMLINE_DIR/joint"
FILTER_DIR="$GERMLINE_DIR/filtered"
PLINK_DIR="$GERMLINE_DIR/plink"

mkdir -p \
    "$GVCF_DIR" \
    "$JOINT_DIR" \
    "$FILTER_DIR" \
    "$PLINK_DIR" \
    "$QC_DIR/bcftools" \
    "$QC_DIR/plink" \
    "$QC_DIR/multiqc/local_wes_variants" \
    "$TMP_DIR/gatk"

echo
echo "=== TOOL CHECK ==="
for tool in gatk samtools bcftools python plink2 multiqc
do
    echo -n "$tool: "
    command -v "$tool"
done

echo
echo "=== FIND CALLABLE BED ==="
mapfile -t callable_beds < <(
    ls -t "$CALLABLE_DIR"/local_wes.germline_only.empirical_callable.depth*.bed 2>/dev/null || true
)

if [[ "${#callable_beds[@]}" -eq 0 ]]; then
    echo "ERROR: Germline-only callable BED not found."
    exit 1
fi

COMMON_BED="${callable_beds[0]}"
COMMON_SUMMARY="$CALLABLE_DIR/local_wes.germline_only.empirical_callable.summary.tsv"

echo "COMMON_BED=$COMMON_BED"

if [[ ! -s "$COMMON_BED" ]]; then
    echo "ERROR: COMMON_BED is missing or empty."
    exit 1
fi

echo
echo "=== VERIFY BQSR BAMs ==="
find "$BQSR_BAM_DIR" -maxdepth 1 -name "*.bqsr.bam" | sort | grep -v '/BC' \
    > metadata/local_wes/local_wes.germline_bqsr_bams.list

BQSR_COUNT=$(wc -l < metadata/local_wes/local_wes.germline_bqsr_bams.list)

echo "BQSR BAM count: $BQSR_COUNT"

if [[ "$BQSR_COUNT" -ne 31 ]]; then
    echo "ERROR: Expected 31 germline BQSR BAMs, found $BQSR_COUNT"
    exit 1
fi

samtools quickcheck -v $(cat metadata/local_wes/local_wes.germline_bqsr_bams.list)
echo "All germline BQSR BAM quickcheck OK"

echo
echo "=== VERIFY BC TUMOR BAMs ARE PRESENT ==="
find "$MARKDUP_DIR" -maxdepth 1 -name "BC*.marked.bam" | sort \
    > metadata/local_wes/local_wes.tumor_markdup_bams.list

TUMOR_COUNT=$(wc -l < metadata/local_wes/local_wes.tumor_markdup_bams.list)

echo "Tumor BC marked BAM count: $TUMOR_COUNT"
cat metadata/local_wes/local_wes.tumor_markdup_bams.list

if [[ "$TUMOR_COUNT" -ne 6 ]]; then
    echo "ERROR: Expected 6 BC tumor marked BAMs, found $TUMOR_COUNT"
    exit 1
fi

stage_mail "10 resume validation after BQSR"

echo
echo "=== STAGE 11: delete ONLY germline MarkDuplicates BAMs ==="

if [[ "$DELETE_GERMLINE_MARKDUP" -eq 1 ]]; then
    find "$MARKDUP_DIR" -maxdepth 1 -type f \
        \( -name "C*.marked.bam" -o -name "C*.marked.bam.bai" -o -name "C*.marked.bai" -o \
           -name "H*.marked.bam" -o -name "H*.marked.bam.bai" -o -name "H*.marked.bai" \) \
        | sort \
        > "metadata/local_wes/germline_markdup_files_deleted_by_resume_${RUN_ID}.txt"

    echo "Germline MarkDuplicates files to delete:"
    wc -l "metadata/local_wes/germline_markdup_files_deleted_by_resume_${RUN_ID}.txt"

    echo "Approx size to delete:"
    if [[ -s "metadata/local_wes/germline_markdup_files_deleted_by_resume_${RUN_ID}.txt" ]]; then
        du -ch $(cat "metadata/local_wes/germline_markdup_files_deleted_by_resume_${RUN_ID}.txt") 2>/dev/null | tail -1 || true
        xargs -r rm -f < "metadata/local_wes/germline_markdup_files_deleted_by_resume_${RUN_ID}.txt"
    fi

    echo "Remaining germline MarkDuplicates BAMs:"
    find "$MARKDUP_DIR" -maxdepth 1 -name "[CH]*.marked.bam" | wc -l

    echo "Preserved BC tumor BAMs:"
    find "$MARKDUP_DIR" -maxdepth 1 -name "BC*.marked.bam" | sort
    find "$MARKDUP_DIR" -maxdepth 1 -name "BC*.marked.bam" | wc -l

    echo "Disk after germline MarkDuplicates cleanup:"
    df -h "$PROJECT_ROOT"
else
    echo "Skipping germline MarkDuplicates cleanup."
fi

FREE_MB=$(df -Pm "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
echo "Free MB after cleanup: $FREE_MB"

if [[ "$FREE_MB" -lt 120000 ]]; then
    echo "ERROR: Free disk is below 120 GB after cleanup; stopping before HaplotypeCaller."
    exit 1
fi

stage_mail "11 germline markdup cleanup"

echo
echo "=== STAGE 12: HaplotypeCaller gVCF ==="

while read -r bam
do
    sample_id="$(basename "$bam" .bqsr.bam)"
    gvcf="$GVCF_DIR/${sample_id}.g.vcf.gz"

    echo
    echo "Sample: $sample_id"

    if [[ -s "$gvcf" && ( -s "${gvcf}.tbi" || -s "${gvcf}.idx" ) ]]; then
        echo "SKIP existing gVCF: $sample_id"
        continue
    fi

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        HaplotypeCaller \
        -R "$REF_FASTA" \
        -I "$bam" \
        -O "$gvcf" \
        -ERC GVCF \
        -L "$COMMON_BED" \
        --native-pair-hmm-threads "$HC_THREADS"

    if [[ ! -s "${gvcf}.tbi" && ! -s "${gvcf}.idx" ]]; then
        gatk IndexFeatureFile -I "$gvcf"
    fi

    bcftools view -h "$gvcf" >/dev/null
    echo "DONE gVCF: $sample_id"

done < metadata/local_wes/local_wes.germline_bqsr_bams.list

find "$GVCF_DIR" -name "*.g.vcf.gz" | sort | grep -v '/BC' \
    > metadata/local_wes/local_wes.germline_gvcfs.list

GVCF_COUNT=$(wc -l < metadata/local_wes/local_wes.germline_gvcfs.list)

echo "gVCF count: $GVCF_COUNT"

if [[ "$GVCF_COUNT" -ne 31 ]]; then
    echo "ERROR: Expected 31 germline gVCFs, found $GVCF_COUNT"
    exit 1
fi

while read -r gvcf
do
    bcftools view -h "$gvcf" >/dev/null
done < metadata/local_wes/local_wes.germline_gvcfs.list

stage_mail "12 HaplotypeCaller gVCFs"

echo
echo "=== STAGE 13: delete germline BQSR BAMs after verified gVCFs ==="

if [[ "$DELETE_GERMLINE_BQSR_AFTER_GVCF" -eq 1 ]]; then
    {
        while read -r bam
        do
            [[ -z "$bam" ]] && continue
            echo "$bam"
            [[ -e "${bam}.bai" ]] && echo "${bam}.bai"
            [[ -e "${bam%.bam}.bai" ]] && echo "${bam%.bam}.bai"
        done < metadata/local_wes/local_wes.germline_bqsr_bams.list
    } | sort -u > "metadata/local_wes/germline_bqsr_files_deleted_after_gvcf_${RUN_ID}.txt"

    echo "BQSR files to delete:"
    wc -l "metadata/local_wes/germline_bqsr_files_deleted_after_gvcf_${RUN_ID}.txt"

    if [[ -s "metadata/local_wes/germline_bqsr_files_deleted_after_gvcf_${RUN_ID}.txt" ]]; then
        du -ch $(cat "metadata/local_wes/germline_bqsr_files_deleted_after_gvcf_${RUN_ID}.txt") 2>/dev/null | tail -1 || true
        xargs -r rm -f < "metadata/local_wes/germline_bqsr_files_deleted_after_gvcf_${RUN_ID}.txt"
    fi

    echo "Disk after BQSR cleanup:"
    df -h "$PROJECT_ROOT"
else
    echo "Skipping BQSR cleanup."
fi

stage_mail "13 BQSR cleanup after gVCF"

echo
echo "=== STAGE 14: GenomicsDBImport ==="

SAMPLE_MAP="metadata/local_wes/local_wes.germline_gvcf.sample_map.tsv"
GDB="$JOINT_DIR/local_wes_germline_only_genomicsdb"
GDB_DONE="$JOINT_DIR/local_wes_germline_only_genomicsdb.done"

awk -F'/' '{
    file=$NF
    sample=file
    sub(/\.g\.vcf\.gz$/, "", sample)
    print sample "\t" $0
}' metadata/local_wes/local_wes.germline_gvcfs.list > "$SAMPLE_MAP"

if [[ -s "$GDB_DONE" && -d "$GDB" ]]; then
    echo "SKIP existing GenomicsDB workspace"
else
    rm -rf "$GDB"

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        GenomicsDBImport \
        --genomicsdb-workspace-path "$GDB" \
        --sample-name-map "$SAMPLE_MAP" \
        -L "$COMMON_BED" \
        --reader-threads 4 \
        --batch-size 50

    date --iso-8601=seconds > "$GDB_DONE"
fi

stage_mail "14 GenomicsDBImport"

echo
echo "=== STAGE 15: GenotypeGVCFs ==="

RAW_VCF="$JOINT_DIR/local_wes_germline_only.raw.vcf.gz"

if [[ -s "$RAW_VCF" && -s "${RAW_VCF}.tbi" ]]; then
    echo "SKIP existing raw cohort VCF"
else
    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        GenotypeGVCFs \
        -R "$REF_FASTA" \
        -V "gendb://$GDB" \
        -O "$RAW_VCF"

    if [[ ! -s "${RAW_VCF}.tbi" ]]; then
        gatk IndexFeatureFile -I "$RAW_VCF"
    fi
fi

bcftools view -h "$RAW_VCF" >/dev/null
bcftools stats "$RAW_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.raw.bcftools.stats.txt"

stage_mail "15 GenotypeGVCFs"

echo
echo "=== STAGE 16: hard filtering and PASS VCF ==="

FILTERED_VCF="$FILTER_DIR/local_wes_germline_only.hard_filtered.vcf.gz"
PASS_VCF="$FILTER_DIR/local_wes_germline_only.pass.vcf.gz"
PASS_TAGGED_VCF="$FILTER_DIR/local_wes_germline_only.pass.filltags.vcf.gz"
BIALLELIC_SNPS_VCF="$FILTER_DIR/local_wes_germline_only.pass.biallelic_snps.vcf.gz"

if [[ -s "$FILTERED_VCF" && -s "${FILTERED_VCF}.tbi" ]]; then
    echo "SKIP existing hard-filtered VCF"
else
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        VariantFiltration \
        -R "$REF_FASTA" \
        -V "$RAW_VCF" \
        -O "$FILTERED_VCF" \
        --filter-name "SNP_HARD_FILTER" \
        --filter-expression "vc.isSNP() && (QD < 2.0 || FS > 60.0 || MQ < 40.0 || SOR > 3.0 || MQRankSum < -12.5 || ReadPosRankSum < -8.0)" \
        --filter-name "INDEL_HARD_FILTER" \
        --filter-expression "vc.isIndel() && (QD < 2.0 || FS > 200.0 || SOR > 10.0 || ReadPosRankSum < -20.0)"

    if [[ ! -s "${FILTERED_VCF}.tbi" ]]; then
        gatk IndexFeatureFile -I "$FILTERED_VCF"
    fi
fi

if [[ -s "$PASS_VCF" && -s "${PASS_VCF}.tbi" ]]; then
    echo "SKIP existing PASS VCF"
else
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        SelectVariants \
        -R "$REF_FASTA" \
        -V "$FILTERED_VCF" \
        --exclude-filtered \
        -O "$PASS_VCF"

    if [[ ! -s "${PASS_VCF}.tbi" ]]; then
        gatk IndexFeatureFile -I "$PASS_VCF"
    fi
fi

bcftools stats "$FILTERED_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.hard_filtered.bcftools.stats.txt"
bcftools stats "$PASS_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.pass.bcftools.stats.txt"

if bcftools plugin -l | grep -qx "fill-tags"; then
    bcftools +fill-tags "$PASS_VCF" -Oz -o "$PASS_TAGGED_VCF" -- -t AC,AN,AF,NS
    bcftools index -t "$PASS_TAGGED_VCF"
else
    echo "WARNING: bcftools fill-tags plugin not found; using PASS VCF directly."
    PASS_TAGGED_VCF="$PASS_VCF"
fi

bcftools view \
    -m2 -M2 \
    -v snps \
    -f PASS \
    "$PASS_TAGGED_VCF" \
    -Oz \
    -o "$BIALLELIC_SNPS_VCF"

bcftools index -t "$BIALLELIC_SNPS_VCF"
bcftools stats "$BIALLELIC_SNPS_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.pass.biallelic_snps.bcftools.stats.txt"

stage_mail "16 hard filtering"

echo
echo "=== STAGE 17: PLINK2 QC and PCA ==="

PLINK_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps"
PLINK_QC_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.qc"
PLINK_FILTERED_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered"
PLINK_PRUNE_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered.prune"
PLINK_PCA_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered.pca"

if [[ -s "${PLINK_PREFIX}.pgen" ]]; then
    echo "SKIP existing PLINK pgen"
else
    plink2 \
        --vcf "$BIALLELIC_SNPS_VCF" \
        --double-id \
        --allow-extra-chr \
        --set-missing-var-ids @:#:\$r:\$a \
        --make-pgen \
        --out "$PLINK_PREFIX"
fi

plink2 \
    --pfile "$PLINK_PREFIX" \
    --allow-extra-chr \
    --freq \
    --missing \
    --hardy \
    --out "$PLINK_QC_PREFIX"

plink2 \
    --pfile "$PLINK_PREFIX" \
    --allow-extra-chr \
    --maf 0.01 \
    --geno 0.10 \
    --mind 0.10 \
    --hwe 1e-6 midp \
    --make-pgen \
    --out "$PLINK_FILTERED_PREFIX"

plink2 \
    --pfile "$PLINK_FILTERED_PREFIX" \
    --allow-extra-chr \
    --indep-pairwise 200 50 0.2 \
    --out "$PLINK_PRUNE_PREFIX"

plink2 \
    --pfile "$PLINK_FILTERED_PREFIX" \
    --allow-extra-chr \
    --extract "${PLINK_PRUNE_PREFIX}.prune.in" \
    --pca 20 \
    --out "$PLINK_PCA_PREFIX"

stage_mail "17 PLINK2 QC/PCA"

echo
echo "=== STAGE 18: final summary ==="

multiqc \
    "$QC_DIR/bcftools" \
    "$QC_DIR/plink" \
    "$GERMLINE_DIR/filtered" \
    "$PLINK_DIR" \
    -o "$QC_DIR/multiqc/local_wes_variants" \
    -n local_wes_germline_only_variants_multiqc.html \
    || true

rm -rf "$TMP_DIR/gatk"/* || true

FINAL_BODY="$(mktemp)"

{
    echo "Hadza germline-only WES resume pipeline completed successfully."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Log: $LOG"
    echo
    echo "Callable BED:"
    echo "$COMMON_BED"
    [[ -s "$COMMON_SUMMARY" ]] && cat "$COMMON_SUMMARY"
    echo
    echo "Main outputs:"
    echo "gVCF dir: $GVCF_DIR"
    echo "Raw cohort VCF: $RAW_VCF"
    echo "Filtered VCF: $FILTERED_VCF"
    echo "PASS VCF: $PASS_VCF"
    echo "Biallelic SNP VCF: $BIALLELIC_SNPS_VCF"
    echo "PCA output: ${PLINK_PCA_PREFIX}.eigenvec"
    echo
    echo "BC tumor BAMs preserved:"
    find "$MARKDUP_DIR" -maxdepth 1 -name "BC*.marked.bam" | sort
    echo
    echo "Counts:"
    echo -n "gVCF: "
    find "$GVCF_DIR" -name "*.g.vcf.gz" | grep -v '/BC' | wc -l
    echo -n "Filtered VCF files: "
    ls "$FILTER_DIR"/*.vcf.gz 2>/dev/null | wc -l
    echo
    echo "Disk:"
    df -h "$PROJECT_ROOT"
} > "$FINAL_BODY"

send_alert "SUCCESS germline-only WES resume pipeline" "$FINAL_BODY"
rm -f "$FINAL_BODY"

echo
echo "Hadza germline-only WES resume pipeline completed successfully."
date --iso-8601=seconds
