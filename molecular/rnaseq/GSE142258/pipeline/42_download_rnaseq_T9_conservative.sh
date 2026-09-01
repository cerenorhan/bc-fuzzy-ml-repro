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

T9ROOT="${PUBLIC_DATA_ROOT}"

DATASETS=(
  "GSE142258"
)

echo "RNA-seq download to T9 started:"
date --iso-8601=seconds

echo
echo "T9 disk status before:"
df -h "$PUBLIC_DATA_ROOT"

for DATASET in "${DATASETS[@]}"
do
    DIR="$T9ROOT/$DATASET"
    SESSION_FILE="$DIR/${DATASET}.aria2.session"

    echo
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "Directory: $DIR"
    echo "========================================"

    if [[ ! -s "$DIR/urls.txt" ]]; then
        echo "ERROR: Missing $DIR/urls.txt" >&2
        exit 1
    fi

    echo "URL count:"
    wc -l "$DIR/urls.txt"

    cd "$DIR"

    echo
    echo "Starting aria2c for $DATASET"
    date --iso-8601=seconds

    aria2c \
      --continue=true \
      --file-allocation=none \
      --auto-file-renaming=false \
      --allow-overwrite=false \
      --max-concurrent-downloads=2 \
      --max-connection-per-server=1 \
      --split=1 \
      --min-split-size=50M \
      --disk-cache=16M \
      --summary-interval=60 \
      --download-result=full \
      --console-log-level=notice \
      --save-session="$SESSION_FILE" \
      --save-session-interval=60 \
      --retry-wait=30 \
      --timeout=60 \
      --max-tries=0 \
      --input-file=urls.txt

    echo
    echo "Download completed for $DATASET:"
    date --iso-8601=seconds

    if [[ -s md5.txt ]]; then
        echo
        echo "MD5 check for $DATASET"
        md5sum -c md5.txt
    else
        echo "WARNING: md5.txt not found for $DATASET"
    fi

    echo
    echo "T9 disk status after $DATASET:"
    df -h "$PUBLIC_DATA_ROOT"

    cd "$PROJECT_ROOT"
done

echo
echo "RNA-seq T9 downloads completed:"
date --iso-8601=seconds
