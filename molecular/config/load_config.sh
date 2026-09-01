#!/usr/bin/env bash

# Shared configuration loader for the molecular-analysis workflows.
#
# Priority:
#   1. Existing environment variables
#   2. BC_MOLECULAR_CONFIG, if explicitly supplied
#   3. molecular/config/config.local.sh, if present
#
# config.local.sh is intentionally excluded from Git.

_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${BC_MOLECULAR_CONFIG:-}" && -f "${BC_MOLECULAR_CONFIG}" ]]; then
    # shellcheck source=/dev/null
    source "${BC_MOLECULAR_CONFIG}"
elif [[ -f "${_CONFIG_DIR}/config.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "${_CONFIG_DIR}/config.local.sh"
fi

: "${PROJECT_ROOT:?PROJECT_ROOT is not configured. See molecular/config/config.example.sh}"

# Backward-compatible alias used by legacy analysis scripts.
PROJECT="${PROJECT:-${PROJECT_ROOT}}"
export PROJECT

LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${PROJECT_ROOT}/data}"
PUBLIC_DATA_ROOT="${PUBLIC_DATA_ROOT:-${PROJECT_ROOT}/data/public}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${PROJECT_ROOT}/results}"
SECONDARY_SCRATCH_ROOT="${SECONDARY_SCRATCH_ROOT:-${SCRATCH_ROOT}}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${PROJECT_ROOT}/reference}"
PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-${PROJECT_ROOT}/paper_results}"
ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-0}"
NOTIFICATION_HELPER="${NOTIFICATION_HELPER:-}"

export PROJECT_ROOT
export LOCAL_DATA_ROOT
export PUBLIC_DATA_ROOT
export SCRATCH_ROOT
export SECONDARY_SCRATCH_ROOT
export REFERENCE_ROOT
export PAPER_RESULTS_ROOT
export ENABLE_NOTIFICATIONS
export NOTIFICATION_HELPER

# ------------------------------------------------------------------
# Computational resources
# ------------------------------------------------------------------

THREADS="${THREADS:-32}"
ALIGN_THREADS="${ALIGN_THREADS:-24}"
SORT_THREADS="${SORT_THREADS:-8}"
GATK_THREADS="${GATK_THREADS:-8}"
JAVA_MEM="${JAVA_MEM:-64g}"

# ------------------------------------------------------------------
# WES reference resources
# ------------------------------------------------------------------

WES_REFERENCE_ROOT="${WES_REFERENCE_ROOT:-${REFERENCE_ROOT}/wes}"

REF_DIR="${REF_DIR:-${WES_REFERENCE_ROOT}}"
REF_FASTA="${REF_FASTA:-${REF_DIR}/Homo_sapiens_assembly38.fasta}"
REF_DICT="${REF_DICT:-${REF_DIR}/Homo_sapiens_assembly38.dict}"

DBSNP="${DBSNP:-${REF_DIR}/Homo_sapiens_assembly38.dbsnp138.vcf}"
MILLS="${MILLS:-${REF_DIR}/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz}"
KNOWN_SNPS="${KNOWN_SNPS:-${REF_DIR}/1000G_phase1.snps.high_confidence.hg38.vcf.gz}"

GNOMAD_AF="${GNOMAD_AF:-${REF_DIR}/af-only-gnomad.hg38.vcf.gz}"
SMALL_EXAC_COMMON="${SMALL_EXAC_COMMON:-${REF_DIR}/small_exac_common_3.hg38.vcf.gz}"

CAPTURE_BED_RAW="${CAPTURE_BED_RAW:-${REF_DIR}/local_capture.original.bed}"
CAPTURE_BED="${CAPTURE_BED:-${REF_DIR}/local_capture.sorted.merged.bed}"
CAPTURE_INTERVALS="${CAPTURE_INTERVALS:-${REF_DIR}/local_capture.interval_list}"

# ------------------------------------------------------------------
# Local WES workspace
# ------------------------------------------------------------------

LOCAL_WES_METADATA="${LOCAL_WES_METADATA:-${PROJECT_ROOT}/metadata/local_wes.tsv}"

LOCAL_FASTQ="${LOCAL_FASTQ:-${LOCAL_DATA_ROOT}/local_wes}"
TRIMMED_LOCAL="${TRIMMED_LOCAL:-${PROJECT_ROOT}/data/trimmed/local_wes}"

GERMLINE_DIR="${GERMLINE_DIR:-${PROJECT_ROOT}/results/germline}"
BAM_DIR="${BAM_DIR:-${GERMLINE_DIR}/bam}"
GVCF_DIR="${GVCF_DIR:-${GERMLINE_DIR}/gvcf}"
JOINT_DIR="${JOINT_DIR:-${GERMLINE_DIR}/joint}"

QC_DIR="${QC_DIR:-${PROJECT_ROOT}/results/qc}"
TMP_DIR="${TMP_DIR:-${PROJECT_ROOT}/tmp}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

export THREADS ALIGN_THREADS SORT_THREADS GATK_THREADS JAVA_MEM
export WES_REFERENCE_ROOT REF_DIR REF_FASTA REF_DICT
export DBSNP MILLS KNOWN_SNPS GNOMAD_AF SMALL_EXAC_COMMON
export CAPTURE_BED_RAW CAPTURE_BED CAPTURE_INTERVALS
export LOCAL_WES_METADATA LOCAL_FASTQ TRIMMED_LOCAL
export GERMLINE_DIR BAM_DIR GVCF_DIR JOINT_DIR QC_DIR TMP_DIR LOG_DIR
