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

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/local_wes/finish_wes_pca_smallN_${RUN_ID}.log"

mkdir -p logs/local_wes "$GERMLINE_DIR/plink" "$GERMLINE_DIR/summary"

exec > >(tee -a "$LOG") 2>&1

send_alert() {
    local subject="$1"
    local body_file="$2"
    if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
        bash "$NOTIFICATION_HELPER" "$subject" "$body_file" || true
    fi
}

fail_alert() {
    local code="$?"
    local body
    body="$(mktemp)"
    {
        echo "WES small-N PCA FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Exit code: $code"
        echo "Log: $LOG"
        echo
        tail -n 120 "$LOG" || true
    } > "$body"
    send_alert "FAILED WES small-N PCA" "$body"
    rm -f "$body"
}
trap fail_alert ERR

echo "WES small-N PCA started"
date --iso-8601=seconds
echo "Log: $LOG"

IN_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered"

[[ -s "${IN_PREFIX}.pgen" ]] || { echo "ERROR: missing ${IN_PREFIX}.pgen"; exit 1; }
[[ -s "${IN_PREFIX}.pvar" ]] || { echo "ERROR: missing ${IN_PREFIX}.pvar"; exit 1; }
[[ -s "${IN_PREFIX}.psam" ]] || { echo "ERROR: missing ${IN_PREFIX}.psam"; exit 1; }

PRUNE_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.smallN_badld_prune"
PCA_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.smallN_badld_pruned.pca"
NOPRUNE_PCA_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.no_prune.pca"

echo
echo "=== Small-N LD pruning with --bad-ld ==="

plink2 \
    --pfile "$IN_PREFIX" \
    --allow-extra-chr \
    --indep-pairwise 200 50 0.2 \
    --bad-ld \
    --out "$PRUNE_PREFIX"

echo -n "Pruned-in variants: "
wc -l < "${PRUNE_PREFIX}.prune.in"

echo
echo "=== PCA on small-N LD-pruned variants ==="

plink2 \
    --pfile "$IN_PREFIX" \
    --allow-extra-chr \
    --extract "${PRUNE_PREFIX}.prune.in" \
    --pca 20 \
    --out "$PCA_PREFIX"

echo
echo "=== Backup PCA without LD pruning ==="

plink2 \
    --pfile "$IN_PREFIX" \
    --allow-extra-chr \
    --pca 20 \
    --out "$NOPRUNE_PCA_PREFIX"

echo
echo "=== Create annotated PCA TSV files ==="

python - "$PCA_PREFIX.eigenvec" "$GERMLINE_DIR/summary/wes_pca_smallN_badld_pruned.tsv" <<'PY'
import sys

inp, outp = sys.argv[1], sys.argv[2]

with open(inp) as f:
    lines = [line.strip().split() for line in f if line.strip()]

header = lines[0]
header[0] = header[0].lstrip("#")
pc_cols = header[2:]

with open(outp, "w") as out:
    out.write("\t".join(["sample_id", "group", "sex_from_id"] + pc_cols) + "\n")
    for row in lines[1:]:
        sid = row[1]
        group = "Hadza" if sid.startswith("H") else "Tanzanian_control" if sid.startswith("C") else "Unknown"
        sex = "Female" if sid.endswith("F") else "Male" if sid.endswith("M") else "Unknown"
        out.write("\t".join([sid, group, sex] + row[2:]) + "\n")
PY

python - "$NOPRUNE_PCA_PREFIX.eigenvec" "$GERMLINE_DIR/summary/wes_pca_no_prune.tsv" <<'PY'
import sys

inp, outp = sys.argv[1], sys.argv[2]

with open(inp) as f:
    lines = [line.strip().split() for line in f if line.strip()]

header = lines[0]
header[0] = header[0].lstrip("#")
pc_cols = header[2:]

with open(outp, "w") as out:
    out.write("\t".join(["sample_id", "group", "sex_from_id"] + pc_cols) + "\n")
    for row in lines[1:]:
        sid = row[1]
        group = "Hadza" if sid.startswith("H") else "Tanzanian_control" if sid.startswith("C") else "Unknown"
        sex = "Female" if sid.endswith("F") else "Male" if sid.endswith("M") else "Unknown"
        out.write("\t".join([sid, group, sex] + row[2:]) + "\n")
PY

BODY="$(mktemp)"
{
    echo "WES small-N PCA completed."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Log: $LOG"
    echo
    echo "Main PCA outputs:"
    echo "${PCA_PREFIX}.eigenvec"
    echo "${PCA_PREFIX}.eigenval"
    echo "$GERMLINE_DIR/summary/wes_pca_smallN_badld_pruned.tsv"
    echo
    echo "Backup PCA outputs:"
    echo "${NOPRUNE_PCA_PREFIX}.eigenvec"
    echo "${NOPRUNE_PCA_PREFIX}.eigenval"
    echo "$GERMLINE_DIR/summary/wes_pca_no_prune.tsv"
    echo
    echo "Counts:"
    echo -n "PCA sample rows: "
    tail -n +2 "${PCA_PREFIX}.eigenvec" | wc -l
    echo -n "No-prune PCA sample rows: "
    tail -n +2 "${NOPRUNE_PCA_PREFIX}.eigenvec" | wc -l
} > "$BODY"

send_alert "SUCCESS WES small-N PCA" "$BODY"
rm -f "$BODY"

echo
echo "WES small-N PCA completed successfully."
date --iso-8601=seconds
