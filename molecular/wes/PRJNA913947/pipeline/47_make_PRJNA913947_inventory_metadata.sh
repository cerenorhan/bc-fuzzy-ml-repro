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

PRJNA_DIR="$PROJECT_ROOT/data/public/PRJNA913947_ALL"
OUT_DIR="$PROJECT_ROOT/metadata/public/PRJNA913947"
LOG="$PROJECT_ROOT/logs/public/PRJNA913947_inventory_metadata_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/logs/public"

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 inventory/metadata started"
date --iso-8601=seconds
echo "PRJNA_DIR=$PRJNA_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"

echo
echo "=== FASTQ inventory ==="

find "$PRJNA_DIR" -maxdepth 1 -name "SRR*.fastq.gz" -printf "%f\t%s\n" \
  | sort \
  > "$OUT_DIR/PRJNA913947_fastq_files.tsv"

python - "$OUT_DIR/PRJNA913947_fastq_files.tsv" "$OUT_DIR/PRJNA913947_run_fastq_inventory.tsv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

inp = Path(sys.argv[1])
outp = Path(sys.argv[2])

df = pd.read_csv(inp, sep="\t", names=["filename", "size_bytes"])
df["run"] = df["filename"].str.replace(r"_[12]\.fastq\.gz$", "", regex=True)
df["mate"] = df["filename"].str.extract(r"_(1|2)\.fastq\.gz$")[0]
df["size_gb"] = df["size_bytes"] / (1024**3)

wide = df.pivot_table(
    index="run",
    columns="mate",
    values=["filename", "size_bytes", "size_gb"],
    aggfunc="first"
)

wide.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in wide.columns]
wide = wide.reset_index()

for col in ["filename_1", "filename_2", "size_bytes_1", "size_bytes_2", "size_gb_1", "size_gb_2"]:
    if col not in wide.columns:
        wide[col] = pd.NA

wide["has_R1"] = wide["filename_1"].notna()
wide["has_R2"] = wide["filename_2"].notna()
wide["paired_complete"] = wide["has_R1"] & wide["has_R2"]
wide["total_size_gb"] = wide[["size_gb_1", "size_gb_2"]].fillna(0).sum(axis=1)

wide = wide[[
    "run", "paired_complete",
    "filename_1", "filename_2",
    "size_gb_1", "size_gb_2", "total_size_gb"
]]

wide.to_csv(outp, sep="\t", index=False)

print("FASTQ files:", len(df))
print("Runs:", len(wide))
print("Complete paired runs:", int(wide["paired_complete"].sum()))
print("Total FASTQ GiB:", round(wide["total_size_gb"].sum(), 2))
PY

echo
echo "=== SRA RunInfo metadata ==="

RUNINFO="$OUT_DIR/PRJNA913947_SraRunInfo.csv"

if [[ ! -s "$RUNINFO" ]]; then
    echo "Downloading SRA RunInfo..."
    curl -L \
      "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=PRJNA913947" \
      -o "$RUNINFO"
fi

echo "RunInfo lines:"
wc -l "$RUNINFO"

echo
echo "=== Merge FASTQ inventory with SRA RunInfo ==="

python - "$OUT_DIR/PRJNA913947_run_fastq_inventory.tsv" "$RUNINFO" "$OUT_DIR/PRJNA913947_run_inventory_with_metadata.tsv" "$OUT_DIR/PRJNA913947_pairing_candidate_summary.tsv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

fastq_tsv = Path(sys.argv[1])
runinfo_csv = Path(sys.argv[2])
merged_out = Path(sys.argv[3])
pair_out = Path(sys.argv[4])

fq = pd.read_csv(fastq_tsv, sep="\t")
ri = pd.read_csv(runinfo_csv)

if "Run" not in ri.columns:
    raise SystemExit("ERROR: Run column not found in SraRunInfo")

merged = fq.merge(ri, left_on="run", right_on="Run", how="left")

merged.to_csv(merged_out, sep="\t", index=False)

candidate_cols = [
    c for c in [
        "BioProject", "BioSample", "SampleName", "LibraryName",
        "tissue", "source_name", "sample_type", "tumor", "disease",
        "geo_loc_name", "population", "ethnicity", "LibraryStrategy",
        "LibrarySource", "LibrarySelection", "Platform", "Model"
    ]
    if c in merged.columns
]

print("Merged rows:", len(merged))
print("Metadata columns available:")
for c in candidate_cols:
    print(" -", c)

with open(pair_out, "w") as out:
    out.write("field\tunique_values_or_count\n")
    for c in candidate_cols:
        vals = merged[c].dropna().astype(str).unique()
        if len(vals) <= 30:
            out.write(f"{c}\t" + " | ".join(sorted(vals)) + "\n")
        else:
            out.write(f"{c}\t{len(vals)} unique values\n")

# Potential pairing overview by BioSample/SampleName/LibraryName if present
for c in ["BioSample", "SampleName", "LibraryName"]:
    if c in merged.columns:
        tmp = merged.groupby(c)["run"].nunique().sort_values(ascending=False)
        tmp.to_csv(pair_out.with_name(f"PRJNA913947_runs_per_{c}.tsv"), sep="\t", header=["n_runs"])
PY

echo
echo "=== Output files ==="
ls -lh "$OUT_DIR"

echo
echo "PRJNA913947 inventory/metadata completed."
date --iso-8601=seconds
