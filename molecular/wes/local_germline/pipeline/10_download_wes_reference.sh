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


mkdir -p "$REF_DIR"
cd "$REF_DIR"

BROAD_HG38="https://storage.googleapis.com/gcp-public-data--broad-references/hg38/v0"
GATK_SOMATIC_HG38="https://storage.googleapis.com/gatk-best-practices/somatic-hg38"

download_if_missing() {
    local url="$1"
    local file="$2"

    if [[ -s "$file" ]]; then
        echo "Already exists: $file"
    else
        echo "Downloading: $file"
        curl -fL -C - "$url" -o "$file"
    fi
}

download_if_missing "$BROAD_HG38/Homo_sapiens_assembly38.fasta" \
    "Homo_sapiens_assembly38.fasta"

download_if_missing "$BROAD_HG38/Homo_sapiens_assembly38.fasta.fai" \
    "Homo_sapiens_assembly38.fasta.fai"

download_if_missing "$BROAD_HG38/Homo_sapiens_assembly38.dict" \
    "Homo_sapiens_assembly38.dict"

download_if_missing "$BROAD_HG38/Homo_sapiens_assembly38.dbsnp138.vcf" \
    "Homo_sapiens_assembly38.dbsnp138.vcf"

download_if_missing "$BROAD_HG38/Homo_sapiens_assembly38.dbsnp138.vcf.idx" \
    "Homo_sapiens_assembly38.dbsnp138.vcf.idx"

download_if_missing "$BROAD_HG38/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz" \
    "Mills_and_1000G_gold_standard.indels.hg38.vcf.gz"

download_if_missing "$BROAD_HG38/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz.tbi" \
    "Mills_and_1000G_gold_standard.indels.hg38.vcf.gz.tbi"

download_if_missing "$BROAD_HG38/1000G_phase1.snps.high_confidence.hg38.vcf.gz" \
    "1000G_phase1.snps.high_confidence.hg38.vcf.gz"

download_if_missing "$BROAD_HG38/1000G_phase1.snps.high_confidence.hg38.vcf.gz.tbi" \
    "1000G_phase1.snps.high_confidence.hg38.vcf.gz.tbi"

download_if_missing "$GATK_SOMATIC_HG38/af-only-gnomad.hg38.vcf.gz" \
    "af-only-gnomad.hg38.vcf.gz"

download_if_missing "$GATK_SOMATIC_HG38/af-only-gnomad.hg38.vcf.gz.tbi" \
    "af-only-gnomad.hg38.vcf.gz.tbi"

download_if_missing "$GATK_SOMATIC_HG38/small_exac_common_3.hg38.vcf.gz" \
    "small_exac_common_3.hg38.vcf.gz"

download_if_missing "$GATK_SOMATIC_HG38/small_exac_common_3.hg38.vcf.gz.tbi" \
    "small_exac_common_3.hg38.vcf.gz.tbi"

if [[ ! -s "${REF_FASTA}.0123" && ! -s "${REF_FASTA}.bwt.2bit.64" ]]; then
    echo "Indexing reference with bwa-mem2..."
    bwa-mem2 index "$REF_FASTA"
else
    echo "BWA-MEM2 index seems to exist."
fi

echo "Reference download and indexing completed."
