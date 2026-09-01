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

OUT="results/EA_BC_AI_MultiOmics/rnaseq_geo_metadata"
LOG="logs/public/rnaseq_geo_metadata_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT/raw" logs/public

exec > >(tee -a "$LOG") 2>&1

echo "Fetching GEO RNA-seq metadata"
date --iso-8601=seconds

for GSE in GSE142258
do
    PREFIX="${GSE:0:6}nnn"
    URL="https://ftp.ncbi.nlm.nih.gov/geo/series/${PREFIX}/${GSE}/matrix/${GSE}_series_matrix.txt.gz"
    echo
    echo "Downloading $GSE"
    curl -L --retry 5 --retry-delay 10 "$URL" -o "$OUT/raw/${GSE}_series_matrix.txt.gz"
done

micromamba run -n rnaseq python - "$OUT" <<'PY'
import sys, gzip, re
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
raw = out / "raw"

all_rows = []

for gz in sorted(raw.glob("*_series_matrix.txt.gz")):
    gse = gz.name.replace("_series_matrix.txt.gz", "")
    meta = {}

    with gzip.open(gz, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0].replace("!", "")
            vals = [x.strip().strip('"') for x in parts[1:]]
            meta[key] = vals

    accessions = meta.get("Sample_geo_accession", [])
    titles = meta.get("Sample_title", [""] * len(accessions))

    rows = []
    for i, gsm in enumerate(accessions):
        row = {
            "dataset": gse,
            "GSM": gsm,
            "title": titles[i] if i < len(titles) else ""
        }

        for key, vals in meta.items():
            if key in ["Sample_geo_accession", "Sample_title"]:
                continue
            if i < len(vals):
                val = vals[i]
                if key == "Sample_characteristics_ch1":
                    # Multiple characteristics lines collapse imperfectly in series matrix;
                    # keep raw text as searchable field.
                    row.setdefault("characteristics", "")
                    row["characteristics"] += " | " + val
                else:
                    row[key] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{gse}_sample_metadata.tsv", sep="\t", index=False)
    all_rows.extend(rows)

pd.DataFrame(all_rows).to_csv(out / "all_geo_sample_metadata.tsv", sep="\t", index=False)

print("Wrote metadata for:")
for p in sorted(out.glob("*_sample_metadata.tsv")):
    print(p)
PY

echo
echo "DONE"
ls -lh "$OUT"
