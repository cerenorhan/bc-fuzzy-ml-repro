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

PROJECT="$PROJECT_ROOT"
DIR="$PROJECT/data/public/PRJNA913947_ALL"
LOG="$PROJECT/logs/downloads/PRJNA913947_retry_failed_md5_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PROJECT/logs/downloads"

exec > >(tee -a "$LOG") 2>&1

cd "$DIR"

echo "PRJNA failed-MD5 retry started"
date --iso-8601=seconds
echo "Directory: $PWD"
echo "Log: $LOG"

cat > failed_md5_files.txt <<'LIST'
SRR22868113_2.fastq.gz
SRR22868284_2.fastq.gz
SRR22868086_2.fastq.gz
SRR22868119_1.fastq.gz
SRR22868182_2.fastq.gz
SRR22868396_2.fastq.gz
LIST

echo
echo "Failed files:"
cat failed_md5_files.txt

echo
echo "Creating retry URL list..."
grep -F -f failed_md5_files.txt urls.txt > urls.failed_md5.retry.txt

echo "Retry URL count:"
wc -l urls.failed_md5.retry.txt

if [[ "$(wc -l < urls.failed_md5.retry.txt)" -ne 6 ]]; then
    echo "ERROR: expected 6 retry URLs"
    exit 1
fi

echo
echo "Deleting failed local files before re-download..."
while read -r f
do
    rm -f "$f" "$f.aria2"
done < failed_md5_files.txt

echo
echo "Re-downloading failed files..."
aria2c \
  --continue=true \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --max-concurrent-downloads=3 \
  --max-connection-per-server=2 \
  --split=2 \
  --min-split-size=20M \
  --disk-cache=64M \
  --summary-interval=60 \
  --download-result=full \
  --console-log-level=notice \
  --input-file=urls.failed_md5.retry.txt

echo
echo "Checking MD5 only for retried files..."
grep -F -f failed_md5_files.txt md5.txt > md5.failed_md5.retry.txt

md5sum -c md5.failed_md5.retry.txt | tee md5.failed_md5.retry.result.txt

if grep -q "FAILED" md5.failed_md5.retry.result.txt; then
    echo "ERROR: retried MD5 still failed"
    exit 1
fi

echo
echo "Retry MD5 OK for all 6 files."

BODY="$(mktemp)"
{
    echo "PRJNA913947 failed MD5 retry completed."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Log: $LOG"
    echo
    echo "Retried files:"
    cat failed_md5_files.txt
    echo
    echo "Disk:"
    df -h "$PROJECT"
} > "$BODY"

if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
    bash "$NOTIFICATION_HELPER" "SUCCESS PRJNA failed MD5 retry" "$BODY" || true
fi

rm -f "$BODY"

echo
echo "PRJNA failed-MD5 retry completed successfully."
date --iso-8601=seconds
