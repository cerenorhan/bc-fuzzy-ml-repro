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

DATASET="PRJNA913947_ALL"
DIR="$PROJECT_ROOT/data/public/${DATASET}"
SESSION_FILE="$DIR/${DATASET}.aria2.session"

if [[ ! -s "$DIR/urls.txt" ]]; then
    echo "ERROR: Missing $DIR/urls.txt"
    exit 1
fi

echo "Dataset: $DATASET"
echo "Directory: $DIR"
echo "URLs:"
wc -l "$DIR/urls.txt"

echo
echo "Disk status before:"
df -h "$PROJECT_ROOT"

echo
echo "Starting PRJNA913947 internal-only download:"
date --iso-8601=seconds

cd "$DIR"

aria2c \
  --continue=true \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --max-concurrent-downloads=4 \
  --max-connection-per-server=2 \
  --split=2 \
  --min-split-size=20M \
  --disk-cache=64M \
  --summary-interval=60 \
  --download-result=full \
  --console-log-level=notice \
  --save-session="$SESSION_FILE" \
  --save-session-interval=60 \
  --input-file=urls.txt

echo
echo "Download command completed:"
date --iso-8601=seconds

echo
echo "Disk status after:"
df -h "$PROJECT_ROOT"

echo
echo "MD5 check: $DATASET"
md5sum -c md5.txt

echo
echo "PRJNA913947 download and MD5 check completed."
date --iso-8601=seconds
