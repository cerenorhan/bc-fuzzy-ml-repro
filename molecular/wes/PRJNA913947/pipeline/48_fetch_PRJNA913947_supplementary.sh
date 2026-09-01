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

OUT="metadata/public/PRJNA913947/supplementary"
LOG="logs/public/PRJNA913947_fetch_supplementary_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT" logs/public

exec > >(tee -a "$LOG") 2>&1

echo "Fetching PRJNA913947 / PMC10629394 supplementary metadata"
date --iso-8601=seconds
echo "OUT=$OUT"
echo "LOG=$LOG"

cd "$OUT"

echo
echo "=== Fetch OA package XML ==="
curl -L "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC10629394" \
  -o PMC10629394_oa.xml

echo
echo "=== Extract package/download links ==="
grep -oE 'href="[^"]+"' PMC10629394_oa.xml \
  | sed 's/^href="//; s/"$//' \
  | tee PMC10629394_links.txt

echo
echo "=== Try downloading linked OA package files ==="

while read -r url
do
    [[ -z "$url" ]] && continue

    case "$url" in
        ftp://*|https://*|http://*)
            fname=$(basename "$url")
            echo "Downloading: $url"
            curl -L "$url" -o "$fname" || true
            ;;
    esac
done < PMC10629394_links.txt

echo
echo "=== Extract archives if any ==="
shopt -s nullglob
for f in *.tar.gz *.tgz *.zip
do
    [[ -e "$f" ]] || continue
    echo "Extracting $f"
    case "$f" in
        *.zip) unzip -o "$f" -d extracted_"$f" || true ;;
        *.tar.gz|*.tgz) mkdir -p extracted_"$f"; tar -xzf "$f" -C extracted_"$f" || true ;;
    esac
done

echo
echo "=== Find supplementary tables ==="
find "$OUT" -type f \( \
    -iname "*.xlsx" -o \
    -iname "*.xls" -o \
    -iname "*.csv" -o \
    -iname "*.tsv" -o \
    -iname "*.txt" -o \
    -iname "*.docx" -o \
    -iname "*.pdf" \
\) -printf "%p\t%s\n" | sort > PRJNA913947_supplementary_file_inventory.tsv

cat PRJNA913947_supplementary_file_inventory.tsv

echo
echo "Completed."
date --iso-8601=seconds
