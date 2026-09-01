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
LOG="logs/local_wes/finish_wes_pca_readfreq_${RUN_ID}.log"

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
        echo "WES read-freq PCA FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Exit code: $code"
        echo "Log: $LOG"
        echo
        tail -n 120 "$LOG" || true
    } > "$body"
    send_alert "FAILED WES read-freq PCA" "$body"
    rm -f "$body"
}
trap fail_alert ERR

echo "WES read-freq PCA started"
date --iso-8601=seconds
echo "Log: $LOG"

IN_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered"
FREQ_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.freq_for_pca"
PRUNE_IN="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.smallN_badld_prune.prune.in"

PCA_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.smallN_readfreq_pruned.pca"
NOPRUNE_PCA_PREFIX="$GERMLINE_DIR/plink/local_wes_germline_only.pass.biallelic_snps.autosomal.filtered.readfreq_no_prune.pca"

[[ -s "${IN_PREFIX}.pgen" ]] || { echo "ERROR: missing ${IN_PREFIX}.pgen"; exit 1; }
[[ -s "${IN_PREFIX}.pvar" ]] || { echo "ERROR: missing ${IN_PREFIX}.pvar"; exit 1; }
[[ -s "${IN_PREFIX}.psam" ]] || { echo "ERROR: missing ${IN_PREFIX}.psam"; exit 1; }
[[ -s "$PRUNE_IN" ]] || { echo "ERROR: missing $PRUNE_IN"; exit 1; }

echo
echo "=== Generate allele frequency file for PCA ==="

plink2 \
    --pfile "$IN_PREFIX" \
    --allow-extra-chr \
    --freq \
    --out "$FREQ_PREFIX"

[[ -s "${FREQ_PREFIX}.afreq" ]] || { echo "ERROR: missing ${FREQ_PREFIX}.afreq"; exit 1; }

echo
echo "=== PCA with LD-pruned variants + read-freq ==="

plink2 \
    --pfile "$IN_PREFIX" \
    --allow-extra-chr \
    --extract "$PRUNE_IN" \
    --read-freq "${FREQ_PREFIX}.afreq" \
    --pca 20 \
    --out "$PCA_PREFIX"

echo
echo "=== Backup PCA without pruning + read-freq ==="

plink2 \
    --pfile "$IN_PREFIX" \
    --allow-extra-chr \
    --read-freq "${FREQ_PREFIX}.afreq" \
    --pca 20 \
    --out "$NOPRUNE_PCA_PREFIX"

echo
echo "=== Create annotated PCA tables ==="

python - "$PCA_PREFIX.eigenvec" "$GERMLINE_DIR/summary/wes_pca_smallN_readfreq_pruned.tsv" <<'PY'
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

python - "$NOPRUNE_PCA_PREFIX.eigenvec" "$GERMLINE_DIR/summary/wes_pca_readfreq_no_prune.tsv" <<'PY'
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
    echo "WES read-freq PCA completed."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Log: $LOG"
    echo
    echo "Main PCA:"
    echo "${PCA_PREFIX}.eigenvec"
    echo "${PCA_PREFIX}.eigenval"
    echo "$GERMLINE_DIR/summary/wes_pca_smallN_readfreq_pruned.tsv"
    echo
    echo "Backup PCA:"
    echo "${NOPRUNE_PCA_PREFIX}.eigenvec"
    echo "${NOPRUNE_PCA_PREFIX}.eigenval"
    echo "$GERMLINE_DIR/summary/wes_pca_readfreq_no_prune.tsv"
    echo
    echo "Counts:"
    echo -n "Pruned PCA rows: "
    tail -n +2 "${PCA_PREFIX}.eigenvec" | wc -l
    echo -n "No-prune PCA rows: "
    tail -n +2 "${NOPRUNE_PCA_PREFIX}.eigenvec" | wc -l
} > "$BODY"

send_alert "SUCCESS WES read-freq PCA" "$BODY"
rm -f "$BODY"

echo
echo "WES read-freq PCA completed successfully."
date --iso-8601=seconds
