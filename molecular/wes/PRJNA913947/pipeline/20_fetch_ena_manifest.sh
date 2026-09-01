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

ACCESSION="${1:?Usage: bash scripts/20_fetch_ena_manifest.sh ACCESSION OUTPUT.tsv}"
OUTPUT="${2:?Usage: bash scripts/20_fetch_ena_manifest.sh ACCESSION OUTPUT.tsv}"

mkdir -p "$(dirname "$OUTPUT")"

BASE_URL="https://www.ebi.ac.uk/ena/portal/api/filereport"

FIELDS="run_accession,study_accession,sample_accession,secondary_sample_accession,experiment_accession,scientific_name,library_name,library_strategy,library_source,library_selection,library_layout,instrument_platform,instrument_model,read_count,base_count,fastq_ftp,fastq_md5,fastq_bytes"

curl -fL \
  "${BASE_URL}?accession=${ACCESSION}&result=read_run&fields=${FIELDS}&format=tsv&download=true&limit=0" \
  -o "$OUTPUT"

if [[ $(wc -l < "$OUTPUT") -le 1 ]]; then
    echo "ERROR: No ENA records returned for ${ACCESSION}" >&2
    exit 1
fi

echo "Saved: $OUTPUT"
echo "Runs: $(( $(wc -l < "$OUTPUT") - 1 ))"
