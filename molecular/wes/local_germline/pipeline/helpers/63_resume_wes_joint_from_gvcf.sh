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
LOG="logs/local_wes/resume_wes_joint_from_gvcf_${RUN_ID}.log"

mkdir -p logs/local_wes metadata/local_wes "$TMP_DIR/gatk" \
    "$GERMLINE_DIR/joint" "$GERMLINE_DIR/filtered" "$GERMLINE_DIR/plink" "$QC_DIR/bcftools"

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
        echo "WES joint-from-gVCF pipeline FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Exit code: $code"
        echo "Log: $LOG"
        echo
        echo "Last 160 log lines:"
        tail -n 160 "$LOG" || true
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
    } > "$body"
    send_alert "FAILED WES joint from gVCF" "$body"
    rm -f "$body"
}
trap fail_alert ERR

echo "WES joint-from-gVCF pipeline started"
date --iso-8601=seconds
echo "Log: $LOG"

GATK_XMX="${GATK_XMX:-32g}"

COMMON_BED=$(ls -t results/germline/callable/local_wes.germline_only.empirical_callable.depth*.bed | head -1)

echo "COMMON_BED=$COMMON_BED"

find results/germline/gvcf -maxdepth 1 -name "*.g.vcf.gz" | sort | grep -v '/BC' \
    > metadata/local_wes/local_wes.germline_gvcfs.list

GVCF_COUNT=$(wc -l < metadata/local_wes/local_wes.germline_gvcfs.list)

echo "gVCF count: $GVCF_COUNT"

if [[ "$GVCF_COUNT" -ne 31 ]]; then
    echo "ERROR: expected 31 gVCFs"
    exit 1
fi

while read -r gvcf
do
    bcftools view -h "$gvcf" >/dev/null
done < metadata/local_wes/local_wes.germline_gvcfs.list

echo "All gVCF headers OK"

JOINT_DIR="$GERMLINE_DIR/joint"
FILTER_DIR="$GERMLINE_DIR/filtered"
PLINK_DIR="$GERMLINE_DIR/plink"

COMBINED_GVCF="$JOINT_DIR/local_wes_germline_only.combined.g.vcf.gz"
RAW_VCF="$JOINT_DIR/local_wes_germline_only.raw.vcf.gz"

echo
echo "=== CombineGVCFs ==="

if [[ -s "$COMBINED_GVCF" && ( -s "${COMBINED_GVCF}.tbi" || -s "${COMBINED_GVCF}.idx" ) ]]; then
    echo "SKIP existing combined gVCF"
else
    rm -f "$COMBINED_GVCF" "${COMBINED_GVCF}.tbi" "${COMBINED_GVCF}.idx"

    GVCF_ARGS=()
    while read -r gvcf
    do
        GVCF_ARGS+=("-V" "$gvcf")
    done < metadata/local_wes/local_wes.germline_gvcfs.list

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        CombineGVCFs \
        -R "$REF_FASTA" \
        "${GVCF_ARGS[@]}" \
        -L "$COMMON_BED" \
        -O "$COMBINED_GVCF"

    gatk IndexFeatureFile -I "$COMBINED_GVCF" || true
fi

echo
echo "=== GenotypeGVCFs ==="

if [[ -s "$RAW_VCF" && ( -s "${RAW_VCF}.tbi" || -s "${RAW_VCF}.idx" ) ]]; then
    echo "SKIP existing raw VCF"
else
    rm -f "$RAW_VCF" "${RAW_VCF}.tbi" "${RAW_VCF}.idx"

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        GenotypeGVCFs \
        -R "$REF_FASTA" \
        -V "$COMBINED_GVCF" \
        -O "$RAW_VCF"

    gatk IndexFeatureFile -I "$RAW_VCF" || true
fi

bcftools view -h "$RAW_VCF" >/dev/null
bcftools stats "$RAW_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.raw.bcftools.stats.txt"

echo
echo "Deleting combined gVCF to save disk after raw VCF validation..."
rm -f "$COMBINED_GVCF" "${COMBINED_GVCF}.tbi" "${COMBINED_GVCF}.idx"

echo
echo "=== Hard filtering ==="

FILTERED_VCF="$FILTER_DIR/local_wes_germline_only.hard_filtered.vcf.gz"
PASS_VCF="$FILTER_DIR/local_wes_germline_only.pass.vcf.gz"
PASS_TAGGED_VCF="$FILTER_DIR/local_wes_germline_only.pass.filltags.vcf.gz"
BIALLELIC_SNPS_VCF="$FILTER_DIR/local_wes_germline_only.pass.biallelic_snps.vcf.gz"

gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
    VariantFiltration \
    -R "$REF_FASTA" \
    -V "$RAW_VCF" \
    -O "$FILTERED_VCF" \
    --filter-name "SNP_HARD_FILTER" \
    --filter-expression "vc.isSNP() && (QD < 2.0 || FS > 60.0 || MQ < 40.0 || SOR > 3.0 || MQRankSum < -12.5 || ReadPosRankSum < -8.0)" \
    --filter-name "INDEL_HARD_FILTER" \
    --filter-expression "vc.isIndel() && (QD < 2.0 || FS > 200.0 || SOR > 10.0 || ReadPosRankSum < -20.0)"

gatk IndexFeatureFile -I "$FILTERED_VCF" || true

gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
    SelectVariants \
    -R "$REF_FASTA" \
    -V "$FILTERED_VCF" \
    --exclude-filtered \
    -O "$PASS_VCF"

gatk IndexFeatureFile -I "$PASS_VCF" || true

bcftools stats "$FILTERED_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.hard_filtered.bcftools.stats.txt"
bcftools stats "$PASS_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.pass.bcftools.stats.txt"

if bcftools plugin -l | grep -qx "fill-tags"; then
    bcftools +fill-tags "$PASS_VCF" -Oz -o "$PASS_TAGGED_VCF" -- -t AC,AN,AF,NS
    bcftools index -t "$PASS_TAGGED_VCF"
else
    PASS_TAGGED_VCF="$PASS_VCF"
fi

bcftools view -m2 -M2 -v snps -f PASS "$PASS_TAGGED_VCF" \
    -Oz -o "$BIALLELIC_SNPS_VCF"

bcftools index -t "$BIALLELIC_SNPS_VCF"
bcftools stats "$BIALLELIC_SNPS_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.pass.biallelic_snps.bcftools.stats.txt"

echo
echo "=== PLINK2 PCA ==="

PLINK_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps"
PLINK_QC_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.qc"
PLINK_FILTERED_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered"
PLINK_PRUNE_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered.prune"
PLINK_PCA_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered.pca"

plink2 \
    --vcf "$BIALLELIC_SNPS_VCF" \
    --double-id \
    --allow-extra-chr \
    --set-missing-var-ids @:#:\$r:\$a \
    --make-pgen \
    --out "$PLINK_PREFIX"

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

echo
echo "=== MultiQC ==="
export PATH="$HOME/.local/bin:$PATH"
multiqc "$QC_DIR/bcftools" "$PLINK_DIR" \
    -o "$QC_DIR/multiqc/local_wes_variants" \
    -n local_wes_germline_only_variants_multiqc.html \
    || true

BODY="$(mktemp)"
{
    echo "WES joint-from-gVCF pipeline completed."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Log: $LOG"
    echo
    echo "Outputs:"
    echo "$RAW_VCF"
    echo "$FILTERED_VCF"
    echo "$PASS_VCF"
    echo "$BIALLELIC_SNPS_VCF"
    echo "${PLINK_PCA_PREFIX}.eigenvec"
    echo
    echo "Counts:"
    echo -n "Raw variants: "
    bcftools view -H "$RAW_VCF" | wc -l
    echo -n "PASS variants: "
    bcftools view -H "$PASS_VCF" | wc -l
    echo -n "PASS biallelic SNPs: "
    bcftools view -H "$BIALLELIC_SNPS_VCF" | wc -l
    echo
    echo "Disk:"
    df -h "$PROJECT_ROOT"
} > "$BODY"

send_alert "SUCCESS WES joint from gVCF" "$BODY"
rm -f "$BODY"

echo
echo "WES joint-from-gVCF pipeline completed successfully."
date --iso-8601=seconds
