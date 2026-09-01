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
LOG="logs/local_wes/finish_wes_autosomal_plink_pca_${RUN_ID}.log"

mkdir -p logs/local_wes "$GERMLINE_DIR/plink" "$QC_DIR/bcftools"

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
        echo "WES autosomal PLINK/PCA FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Exit code: $code"
        echo "Log: $LOG"
        echo
        echo "Last 120 log lines:"
        tail -n 120 "$LOG" || true
    } > "$body"
    send_alert "FAILED WES autosomal PLINK PCA" "$body"
    rm -f "$body"
}
trap fail_alert ERR

echo "WES autosomal PLINK/PCA started"
date --iso-8601=seconds
echo "Log: $LOG"

BIALLELIC_SNPS_VCF="$GERMLINE_DIR/filtered/local_wes_germline_only.pass.biallelic_snps.vcf.gz"
AUTOSOMAL_VCF="$GERMLINE_DIR/filtered/local_wes_germline_only.pass.biallelic_snps.autosomal.vcf.gz"

[[ -s "$BIALLELIC_SNPS_VCF" ]] || { echo "ERROR: missing $BIALLELIC_SNPS_VCF"; exit 1; }

echo
echo "=== Create autosomal VCF chr1-chr22 ==="

CHRS="$(printf "chr%s," {1..22} | sed 's/,$//')"

bcftools view \
    -r "$CHRS" \
    "$BIALLELIC_SNPS_VCF" \
    -Oz \
    -o "$AUTOSOMAL_VCF"

bcftools index -t "$AUTOSOMAL_VCF"

bcftools stats "$AUTOSOMAL_VCF" \
    > "$QC_DIR/bcftools/local_wes_germline_only.pass.biallelic_snps.autosomal.bcftools.stats.txt"

echo -n "Autosomal biallelic SNP count: "
bcftools view -H "$AUTOSOMAL_VCF" | wc -l

echo
echo "=== Clean previous incomplete PLINK temp files ==="

rm -f "$GERMLINE_DIR"/plink/local_wes_germline_only.pass.biallelic_snps-temporary.*

PLINK_DIR="$GERMLINE_DIR/plink"

PLINK_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.autosomal"
PLINK_QC_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.autosomal.qc"
PLINK_FILTERED_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered"
PLINK_PRUNE_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.prune"
PLINK_PCA_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.pca"

echo
echo "=== PLINK import ==="

plink2 \
    --vcf "$AUTOSOMAL_VCF" \
    --double-id \
    --allow-extra-chr \
    --set-missing-var-ids @:#:\$r:\$a \
    --make-pgen \
    --out "$PLINK_PREFIX"

echo
echo "=== PLINK QC metrics ==="

plink2 \
    --pfile "$PLINK_PREFIX" \
    --allow-extra-chr \
    --freq \
    --missing \
    --hardy \
    --out "$PLINK_QC_PREFIX"

echo
echo "=== PLINK variant/sample filters ==="

plink2 \
    --pfile "$PLINK_PREFIX" \
    --allow-extra-chr \
    --maf 0.01 \
    --geno 0.10 \
    --mind 0.10 \
    --hwe 1e-6 midp \
    --make-pgen \
    --out "$PLINK_FILTERED_PREFIX"

echo
echo "=== LD pruning ==="

plink2 \
    --pfile "$PLINK_FILTERED_PREFIX" \
    --allow-extra-chr \
    --indep-pairwise 200 50 0.2 \
    --out "$PLINK_PRUNE_PREFIX"

echo
echo "=== PCA ==="

plink2 \
    --pfile "$PLINK_FILTERED_PREFIX" \
    --allow-extra-chr \
    --extract "${PLINK_PRUNE_PREFIX}.prune.in" \
    --pca 20 \
    --out "$PLINK_PCA_PREFIX"

echo
echo "=== MultiQC optional ==="

export PATH="$HOME/.local/bin:$PATH"

multiqc "$QC_DIR/bcftools" "$PLINK_DIR" \
    -o "$QC_DIR/multiqc/local_wes_variants" \
    -n local_wes_germline_only_variants_multiqc.html \
    || true

BODY="$(mktemp)"
{
    echo "WES autosomal PLINK/PCA completed."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Log: $LOG"
    echo
    echo "Outputs:"
    echo "$AUTOSOMAL_VCF"
    echo "${PLINK_PCA_PREFIX}.eigenvec"
    echo "${PLINK_PCA_PREFIX}.eigenval"
    echo "$QC_DIR/multiqc/local_wes_variants/local_wes_germline_only_variants_multiqc.html"
    echo
    echo "Counts:"
    echo -n "Autosomal biallelic SNPs: "
    bcftools view -H "$AUTOSOMAL_VCF" | wc -l
    echo -n "PCA samples: "
    tail -n +2 "${PLINK_PCA_PREFIX}.eigenvec" | wc -l
    echo
    echo "Disk:"
    df -h "$PROJECT_ROOT"
} > "$BODY"

send_alert "SUCCESS WES autosomal PLINK PCA" "$BODY"
rm -f "$BODY"

echo
echo "WES autosomal PLINK/PCA completed successfully."
date --iso-8601=seconds
