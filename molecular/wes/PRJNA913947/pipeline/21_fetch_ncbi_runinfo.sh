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

QUERY="${1:?Usage: bash scripts/21_fetch_ncbi_runinfo.sh QUERY OUTPUT.csv}"
OUTPUT="${2:?Usage: bash scripts/21_fetch_ncbi_runinfo.sh QUERY OUTPUT.csv}"

mkdir -p "$(dirname "$OUTPUT")"

esearch -db sra -query "$QUERY" \
| efetch -format runinfo \
> "$OUTPUT"

if [[ $(wc -l < "$OUTPUT") -le 1 ]]; then
    echo "ERROR: No NCBI SRA records returned for ${QUERY}" >&2
    exit 1
fi

echo "Saved: $OUTPUT"
echo "Runs: $(( $(wc -l < "$OUTPUT") - 1 ))"
