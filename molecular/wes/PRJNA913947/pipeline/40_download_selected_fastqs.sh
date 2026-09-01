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

cd "$PROJECT_ROOT"

mkdir -p logs/downloads

ARIA_INPUT="data/public/selected_fastqs_now.aria2.txt"
SESSION_FILE="data/public/selected_fastqs_now.session"

rm -f "$ARIA_INPUT"

make_aria2_entries() {
    local dataset="$1"
    local urls="data/public/${dataset}/urls.txt"
    local outdir="$PUBLIC_DATA_ROOT/${dataset}"

    if [[ ! -s "$urls" ]]; then
        echo "ERROR: Missing $urls" >&2
        exit 1
    fi

    while IFS= read -r url
    do
        [[ -z "$url" ]] && continue
        echo "$url" >> "$ARIA_INPUT"
        echo "  dir=${outdir}" >> "$ARIA_INPUT"
    done < "$urls"
}

make_aria2_entries "GSE142258"
make_aria2_entries "PRJNA913947_ALL"

echo "Total FASTQ URLs:"
grep -c '^https://' "$ARIA_INPUT"

echo
echo "Disk status before:"
df -h "$PROJECT_ROOT"
df -h "$PUBLIC_DATA_ROOT" || true

echo
echo "Starting download:"
date --iso-8601=seconds

aria2c \
  --continue=true \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --max-concurrent-downloads=8 \
  --max-connection-per-server=4 \
  --split=4 \
  --min-split-size=20M \
  --summary-interval=60 \
  --download-result=full \
  --save-session="$SESSION_FILE" \
  --save-session-interval=60 \
  --input-file="$ARIA_INPUT"

echo
echo "Download command completed:"
date --iso-8601=seconds

echo
echo "Disk status after:"
df -h "$PROJECT_ROOT"
df -h "$PUBLIC_DATA_ROOT" || true

echo
echo "MD5 check: GSE142258"
(
    cd "$PUBLIC_DATA_ROOT/GSE142258"
    md5sum -c "$PROJECT_ROOT/data/public/GSE142258/md5.txt"
)

echo

echo
echo "MD5 check: PRJNA913947_ALL"
(
    cd "$PUBLIC_DATA_ROOT/PRJNA913947_ALL"
    md5sum -c "$PROJECT_ROOT/data/public/PRJNA913947_ALL/md5.txt"
)

echo
echo "All selected FASTQ downloads and MD5 checks completed."
date --iso-8601=seconds
