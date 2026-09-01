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

cd "$PROJECT_ROOT/data/public/PRJNA913947_ALL"

LOG="$PROJECT_ROOT/logs/downloads/PRJNA913947_md5_verify_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT_ROOT/logs/downloads"

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 MD5 verification started"
date --iso-8601=seconds
echo "Directory: $PWD"
echo "Log: $LOG"

echo
echo "FASTQ count:"
find . -name "*.fastq.gz" | wc -l

echo
echo "aria2 partial count:"
find . -name "*.aria2" | wc -l

echo
echo "Running md5sum -c md5.txt..."
md5sum -c md5.txt > "${LOG}.md5raw" 2>&1

echo
echo "MD5 summary:"
grep -c ': OK$' "${LOG}.md5raw" || true
grep -n 'FAILED' "${LOG}.md5raw" || true

echo
echo "PRJNA913947 MD5 verification completed successfully."
date --iso-8601=seconds
