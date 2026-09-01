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

INPUT="${1:?Usage: bash scripts/29_fetch_biosample_xml_from_ena.sh biosamples.txt outdir}"
OUTDIR="${2:?Usage: bash scripts/29_fetch_biosample_xml_from_ena.sh biosamples.txt outdir}"

mkdir -p "$OUTDIR"

cat "$INPUT" \
| parallel \
    --jobs 6 \
    --halt soon,fail=1 \
    '
    ACC={}
    OUT="'"$OUTDIR"'/${ACC}.xml"

    if [[ -s "$OUT" ]]; then
        echo "Already exists: $ACC"
    else
        echo "Downloading: $ACC"
        curl -fL \
          --retry 10 \
          --retry-delay 5 \
          --retry-all-errors \
          --connect-timeout 30 \
          "https://www.ebi.ac.uk/ena/browser/api/xml/${ACC}" \
          -o "$OUT"
    fi
    '
