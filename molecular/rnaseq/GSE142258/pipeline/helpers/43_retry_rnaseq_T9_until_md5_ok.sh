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

cd "$PROJECT_ROOT"

mkdir -p logs/downloads

LOG="logs/downloads/RNASEQ_T9_RETRY_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

send_alert() {
    local subject="$1"
    local body_file="$2"

    if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
        bash "$NOTIFICATION_HELPER" "$subject" "$body_file" || true
    fi
}

fail_alert() {
    local exit_code="$?"
    local body
    body="$(mktemp)"

    {
        echo "RNA-seq T9 retry download FAILED"
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Host: $(hostname)"
        echo "Exit code: $exit_code"
        echo "Log: $LOG"
        echo
        echo "Disk T9:"
        df -h "$PUBLIC_DATA_ROOT" || true
        echo
        echo "Counts:"
        echo -n "GSE142258 FASTQ: "
        find ${PUBLIC_DATA_ROOT}/GSE142258 -name "*.fastq.gz" 2>/dev/null | wc -l || true
        echo
        echo "Last 180 log lines:"
        tail -n 180 "$LOG" || true
    } > "$body"

    send_alert "FAILED RNA-seq T9 retry download" "$body"
    rm -f "$body"
}

trap fail_alert ERR

T9ROOT="${PUBLIC_DATA_ROOT}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"

DATASETS=(
  "GSE142258"
)

echo "RNA-seq T9 retry-until-MD5-OK started"
date --iso-8601=seconds
echo "Log: $LOG"
echo
df -h "$PUBLIC_DATA_ROOT"

for DATASET in "${DATASETS[@]}"
do
    DIR="$T9ROOT/$DATASET"

    echo
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "Directory: $DIR"
    echo "========================================"

    if [[ ! -d "$DIR" ]]; then
        echo "ERROR: Missing directory: $DIR" >&2
        exit 1
    fi

    if [[ ! -s "$DIR/urls.txt" ]]; then
        echo "ERROR: Missing urls.txt: $DIR/urls.txt" >&2
        exit 1
    fi

    if [[ ! -s "$DIR/md5.txt" ]]; then
        echo "ERROR: Missing md5.txt: $DIR/md5.txt" >&2
        exit 1
    fi

    cd "$DIR"

    round=1

    while [[ "$round" -le "$MAX_ROUNDS" ]]
    do
        echo
        echo "---- $DATASET round $round / $MAX_ROUNDS ----"
        date --iso-8601=seconds

        echo "Running MD5 precheck..."
        md5sum -c md5.txt > "md5.precheck.round${round}.txt" 2>&1 || true

        awk -F': ' '/FAILED/ {print $1}' "md5.precheck.round${round}.txt" \
          | sed 's#^\./##' \
          | sort -u \
          > bad_files.txt

        BAD_COUNT=$(wc -l < bad_files.txt)

        if [[ "$BAD_COUNT" -eq 0 ]]; then
            echo "$DATASET MD5 already OK."
            break
        fi

        echo "$DATASET bad/missing FASTQ count: $BAD_COUNT"
        echo "First bad/missing files:"
        head -20 bad_files.txt

        python - <<'PY'
from pathlib import Path

bad = {x.strip() for x in Path("bad_files.txt").read_text().splitlines() if x.strip()}
urls = []

for line in Path("urls.txt").read_text().splitlines():
    u = line.strip()
    if not u:
        continue
    name = u.rstrip("/").split("/")[-1]
    if name in bad:
        urls.append(u)

Path("urls.todo.txt").write_text("\n".join(urls) + ("\n" if urls else ""))

print(f"bad_files={len(bad)}")
print(f"todo_urls={len(urls)}")

missing_url = sorted(bad - {u.rstrip('/').split('/')[-1] for u in urls})
if missing_url:
    print("WARNING: bad files without matching URL:")
    for x in missing_url[:20]:
        print(x)
PY

        TODO_COUNT=$(wc -l < urls.todo.txt)

        if [[ "$TODO_COUNT" -eq 0 ]]; then
            echo "ERROR: MD5 failed but urls.todo.txt is empty." >&2
            exit 1
        fi

        echo "Downloading/retrying $TODO_COUNT FASTQ files for $DATASET"

        aria2c \
          --continue=true \
          --file-allocation=none \
          --auto-file-renaming=false \
          --allow-overwrite=true \
          --max-concurrent-downloads=2 \
          --max-connection-per-server=1 \
          --split=1 \
          --min-split-size=50M \
          --disk-cache=16M \
          --summary-interval=60 \
          --download-result=full \
          --console-log-level=notice \
          --save-session="${DATASET}.retry.aria2.session" \
          --save-session-interval=60 \
          --retry-wait=30 \
          --timeout=60 \
          --max-tries=0 \
          --input-file=urls.todo.txt || true

        echo "Round $round completed for $DATASET; will re-check MD5."
        round=$((round + 1))
    done

    echo
    echo "Final MD5 check for $DATASET"
    md5sum -c md5.txt > md5.final.txt 2>&1

    echo "$DATASET final MD5 OK"

    echo
    echo "$DATASET size/count:"
    du -sh "$DIR"
    echo -n "FASTQ: "
    find "$DIR" -name "*.fastq.gz" | wc -l
    echo -n "aria2 partial: "
    find "$DIR" -name "*.aria2" | wc -l

    cd "$PROJECT_ROOT"
done

body="$(mktemp)"

{
    echo "RNA-seq T9 retry download completed successfully."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Log: $LOG"
    echo
    echo "GSE142258:"
    du -sh ${PUBLIC_DATA_ROOT}/GSE142258
    echo -n "FASTQ: "
    find ${PUBLIC_DATA_ROOT}/GSE142258 -name "*.fastq.gz" | wc -l
    echo
    echo -n "FASTQ: "
    echo
    echo "T9 disk:"
    df -h "$PUBLIC_DATA_ROOT"
} > "$body"

send_alert "SUCCESS RNA-seq T9 download" "$body"
rm -f "$body"

echo
echo "RNA-seq T9 retry-until-MD5-OK completed successfully."
date --iso-8601=seconds
