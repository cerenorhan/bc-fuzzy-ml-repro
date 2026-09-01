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

BASE="metadata/public/PRJNA913947"
IN="$BASE/kenya_resolution/PRJNA913947_all_descriptors_merged.tsv"
OUT="$BASE/pairing_inference"
LOG="logs/public/PRJNA913947_pairing_inference_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT" logs/public

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 tumor-normal pairing inference started"
date --iso-8601=seconds
echo "IN=$IN"
echo "OUT=$OUT"
echo "LOG=$LOG"

[[ -s "$IN" ]] || { echo "ERROR: Missing $IN"; exit 1; }

micromamba run -n rnaseq python - "$IN" "$OUT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
import re

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp, sep="\t", dtype=str).fillna("")

if "run" not in df.columns and "Run" in df.columns:
    df["run"] = df["Run"]

def get_text(row):
    preferred = [
        "SRA_EXPERIMENT_TITLE",
        "SRA_SAMPLE_TITLE",
        "SRA_SAMPLE_DESCRIPTION",
        "SRA_ATTR_isolate",
        "BioSample_ATTR_isolate",
        "BioSample_Title",
    ]
    vals = []
    for c in preferred:
        if c in row.index:
            vals.append(str(row[c]))
    return " ".join(vals).lower()

def classify_tissue(row):
    text = get_text(row)
    if "normal breast tissue" in text or "adjacent" in text or "non-cancer" in text or "non cancer" in text:
        return "normal"
    if "tumor breast tissue" in text or "tumour breast tissue" in text:
        return "tumor"
    return "unknown"

def sample_numeric(s):
    m = re.search(r"Sample_([0-9]+)$", str(s))
    return int(m.group(1)) if m else None

def sample_numeric_str(s):
    m = re.search(r"Sample_([0-9]+)$", str(s))
    return m.group(1) if m else ""

df["tissue_class"] = df.apply(classify_tissue, axis=1)
df["sample_numeric"] = df["SampleName"].apply(sample_numeric) if "SampleName" in df.columns else None
df["sample_numeric_str"] = df["SampleName"].apply(sample_numeric_str) if "SampleName" in df.columns else ""

keep = [c for c in [
    "run", "filename_1", "filename_2", "total_size_gb",
    "BioSample", "SampleName", "LibraryName", "Sex",
    "tissue_class", "sample_numeric", "sample_numeric_str",
    "SRA_EXPERIMENT_TITLE", "SRA_ATTR_isolate", "BioSample_ATTR_isolate"
] if c in df.columns]

classified = df[keep].copy()
classified = classified.sort_values(["sample_numeric", "SampleName", "run"], na_position="last")
classified.to_csv(out / "PRJNA913947_all_runs_tissue_classified.tsv", sep="\t", index=False)

tumors = classified[classified["tissue_class"] == "tumor"].copy()
normals = classified[classified["tissue_class"] == "normal"].copy()

# All candidate pairs with numeric difference 1.
pairs = []
for _, t in tumors.iterrows():
    tn = t.get("sample_numeric")
    if pd.isna(tn):
        continue
    for _, n in normals.iterrows():
        nn = n.get("sample_numeric")
        if pd.isna(nn):
            continue
        diff = abs(int(tn) - int(nn))
        if diff == 1:
            pairs.append({
                "pair_rule": "numeric_absdiff_1",
                "tumor_run": t["run"],
                "normal_run": n["run"],
                "tumor_sample": t.get("SampleName", ""),
                "normal_sample": n.get("SampleName", ""),
                "tumor_numeric": int(tn),
                "normal_numeric": int(nn),
                "absdiff": diff,
                "tumor_R1": t.get("filename_1", ""),
                "tumor_R2": t.get("filename_2", ""),
                "normal_R1": n.get("filename_1", ""),
                "normal_R2": n.get("filename_2", ""),
                "tumor_size_gb": t.get("total_size_gb", ""),
                "normal_size_gb": n.get("total_size_gb", ""),
            })

cand = pd.DataFrame(pairs)
cand.to_csv(out / "PRJNA913947_numeric_absdiff1_pair_candidates.tsv", sep="\t", index=False)

# Greedy one-to-one pair set.
greedy = []
used_t = set()
used_n = set()

if not cand.empty:
    cand2 = cand.sort_values(["tumor_numeric", "normal_numeric"])
    for _, r in cand2.iterrows():
        tr = r["tumor_run"]
        nr = r["normal_run"]
        if tr in used_t or nr in used_n:
            continue
        used_t.add(tr)
        used_n.add(nr)
        rr = r.to_dict()
        rr["pair_id"] = f"PRJNA_pair_{len(greedy)+1:03d}"
        greedy.append(rr)

greedy_df = pd.DataFrame(greedy)
if not greedy_df.empty:
    cols = ["pair_id"] + [c for c in greedy_df.columns if c != "pair_id"]
    greedy_df = greedy_df[cols]
greedy_df.to_csv(out / "PRJNA913947_greedy_tumor_normal_pairs.tsv", sep="\t", index=False)

# Low numeric block inspection: likely place where Kenyan sample IDs may live, but this is NOT final.
low = classified[
    classified["sample_numeric"].notna() &
    (classified["sample_numeric"].astype(int) <= 100)
].copy()
low.to_csv(out / "PRJNA913947_low_numeric_sample_runs_le100.tsv", sep="\t", index=False)

low46 = classified[
    classified["sample_numeric"].notna() &
    (classified["sample_numeric"].astype(int) <= 46)
].copy()
low46.to_csv(out / "PRJNA913947_low_numeric_sample_runs_le46.tsv", sep="\t", index=False)

if not greedy_df.empty:
    g = greedy_df.copy()
    g["tumor_numeric"] = g["tumor_numeric"].astype(int)
    g["normal_numeric"] = g["normal_numeric"].astype(int)

    likely_kenya_pairs = g[
        (g["tumor_numeric"] <= 46) &
        (g["normal_numeric"] <= 46)
    ].copy()
else:
    likely_kenya_pairs = pd.DataFrame()

likely_kenya_pairs.to_csv(out / "PRJNA913947_likely_kenya_low_numeric_le46_pairs.tsv", sep="\t", index=False)

with open(out / "PRJNA913947_pairing_inference_summary.txt", "w") as f:
    f.write("PRJNA913947 pairing inference summary\n\n")
    f.write(f"Total runs: {len(classified)}\n")
    f.write(f"Tumor-classified runs: {len(tumors)}\n")
    f.write(f"Normal-classified runs: {len(normals)}\n")
    f.write(f"Unknown tissue runs: {(classified['tissue_class'] == 'unknown').sum()}\n")
    f.write(f"Numeric absdiff=1 pair candidates: {len(cand)}\n")
    f.write(f"Greedy one-to-one pairs: {len(greedy_df)}\n")
    f.write(f"Low numeric <=100 runs: {len(low)}\n")
    f.write(f"Low numeric <=46 runs: {len(low46)}\n")
    f.write(f"Likely Kenya low-numeric <=46 greedy pairs: {len(likely_kenya_pairs)}\n\n")
    f.write("Important note:\n")
    f.write("The low-numeric Kenya label is heuristic only. It must be validated against Supplementary Table S2 or official sample descriptors before final somatic analysis claims.\n")

print(open(out / "PRJNA913947_pairing_inference_summary.txt").read())

print("Wrote:")
for name in [
    "PRJNA913947_all_runs_tissue_classified.tsv",
    "PRJNA913947_numeric_absdiff1_pair_candidates.tsv",
    "PRJNA913947_greedy_tumor_normal_pairs.tsv",
    "PRJNA913947_low_numeric_sample_runs_le100.tsv",
    "PRJNA913947_low_numeric_sample_runs_le46.tsv",
    "PRJNA913947_likely_kenya_low_numeric_le46_pairs.tsv",
    "PRJNA913947_pairing_inference_summary.txt",
]:
    print(out / name)
PY

echo
echo "Completed."
date --iso-8601=seconds
