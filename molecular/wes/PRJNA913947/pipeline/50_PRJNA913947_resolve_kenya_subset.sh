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

PROJECT_TAG="EA_BC_AI_MultiOmics"
BASE="metadata/public/PRJNA913947"
SUPP="$BASE/supplementary"
XML="$BASE/xml_descriptors"
OUT="$BASE/kenya_resolution"
LOG="logs/public/PRJNA913947_kenya_resolver_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$SUPP" "$XML" "$OUT" logs/public "results/$PROJECT_TAG"

exec > >(tee -a "$LOG") 2>&1

echo "PRJNA913947 Kenya resolver started"
date --iso-8601=seconds
echo "PROJECT_TAG=$PROJECT_TAG"
echo "LOG=$LOG"

RUNINV="$BASE/PRJNA913947_run_inventory_with_metadata.tsv"
FASTQINV="$BASE/PRJNA913947_run_fastq_inventory.tsv"

[[ -s "$RUNINV" ]] || { echo "ERROR: Missing $RUNINV"; exit 1; }
[[ -s "$FASTQINV" ]] || { echo "ERROR: Missing $FASTQINV"; exit 1; }

echo
echo "=== Existing PRJNA inventory ==="
echo -n "Run inventory rows including header: "
wc -l < "$RUNINV"
echo -n "FASTQ inventory rows including header: "
wc -l < "$FASTQINV"

echo
echo "=== Try PMC OA package through HTTPS ==="

cd "$SUPP"

curl -L --retry 5 --retry-delay 5 \
  "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC10629394" \
  -o PMC10629394_oa.xml || true

grep -oE 'ftp://[^"]+PMC10629394\.tar\.gz' PMC10629394_oa.xml 2>/dev/null \
  | sed 's#ftp://ftp.ncbi.nlm.nih.gov#https://ftp.ncbi.nlm.nih.gov#' \
  | sort -u \
  > PMC10629394_package_urls_https.txt || true

cat PMC10629394_package_urls_https.txt || true

while read -r url
do
    [[ -z "$url" ]] && continue
    fname="$(basename "$url")"
    echo "Downloading $url"
    curl -L --retry 5 --retry-delay 10 "$url" -o "$fname" || true
done < PMC10629394_package_urls_https.txt

shopt -s nullglob

for f in *.tar.gz *.tgz
do
    echo "Extracting $f"
    mkdir -p "extracted_${f}"
    tar -xzf "$f" -C "extracted_${f}" || true
done

for f in *.zip
do
    echo "Extracting $f"
    mkdir -p "extracted_${f}"
    unzip -o "$f" -d "extracted_${f}" || true
done

cd "$PROJECT_ROOT"

find "$SUPP" -type f \( \
    -iname "*.xlsx" -o \
    -iname "*.xls" -o \
    -iname "*.csv" -o \
    -iname "*.tsv" -o \
    -iname "*.txt" -o \
    -iname "*.docx" -o \
    -iname "*.pdf" -o \
    -iname "*.xml" -o \
    -iname "*.nxml" \
\) -printf "%p\t%s\n" \
  | sort \
  > "$OUT/supplementary_file_inventory.tsv"

echo
echo "=== Supplementary/data file inventory ==="
cat "$OUT/supplementary_file_inventory.tsv"

echo
echo "=== Prepare run and BioSample lists ==="

cut -f1 "$FASTQINV" | tail -n +2 | sort -u > "$OUT/runs.txt"

python - "$RUNINV" "$OUT/biosamples.txt" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
if "BioSample" not in df.columns:
    raise SystemExit("BioSample column not found")
df["BioSample"].replace("", pd.NA).dropna().drop_duplicates().sort_values().to_csv(
    sys.argv[2], index=False, header=False
)
PY

echo -n "Runs: "
wc -l < "$OUT/runs.txt"
echo -n "BioSamples: "
wc -l < "$OUT/biosamples.txt"

echo
echo "=== Download SRA XML descriptors in chunks ==="

split -l 40 "$OUT/runs.txt" "$XML/runs_chunk_"

i=0
for chunk in "$XML"/runs_chunk_*
do
    [[ -s "$chunk" ]] || continue
    i=$((i+1))
    ids="$(paste -sd, "$chunk")"
    xml="$XML/sra_chunk_${i}.xml"

    if [[ -s "$xml" ]]; then
        echo "SKIP existing $xml"
    else
        echo "Downloading SRA XML chunk $i"
        curl -L --retry 5 --retry-delay 5 \
          "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${ids}&retmode=xml" \
          -o "$xml" || true
        sleep 1
    fi
done

echo
echo "=== Download BioSample XML descriptors in chunks ==="

split -l 40 "$OUT/biosamples.txt" "$XML/biosamples_chunk_"

j=0
for chunk in "$XML"/biosamples_chunk_*
do
    [[ -s "$chunk" ]] || continue
    j=$((j+1))
    ids="$(paste -sd, "$chunk")"
    xml="$XML/biosample_chunk_${j}.xml"

    if [[ -s "$xml" ]]; then
        echo "SKIP existing $xml"
    else
        echo "Downloading BioSample XML chunk $j"
        curl -L --retry 5 --retry-delay 5 \
          "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=biosample&id=${ids}&retmode=xml" \
          -o "$xml" || true
        sleep 1
    fi
done

echo
echo "=== Parse descriptors and resolve Kenya candidates ==="

micromamba run -n rnaseq python - "$RUNINV" "$OUT" "$XML" <<'PY'
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import re

runinv = Path(sys.argv[1])
out = Path(sys.argv[2])
xml_dir = Path(sys.argv[3])

out.mkdir(parents=True, exist_ok=True)

base = pd.read_csv(runinv, sep="\t", dtype=str).fillna("")
if "run" not in base.columns and "Run" in base.columns:
    base["run"] = base["Run"]

# ---------- Parse SRA XML ----------
sra_rows = []
sra_long = []

for xp in sorted(xml_dir.glob("sra_chunk_*.xml")):
    try:
        root = ET.parse(xp).getroot()
    except Exception as e:
        print(f"WARNING: could not parse {xp}: {e}")
        continue

    for pkg in root.findall(".//EXPERIMENT_PACKAGE"):
        runs = [r.attrib.get("accession", "") for r in pkg.findall(".//RUN_SET/RUN")]
        exp = pkg.find(".//EXPERIMENT")
        sample = pkg.find(".//SAMPLE")

        attrs = {}

        if exp is not None:
            attrs["SRA_EXPERIMENT_accession"] = exp.attrib.get("accession", "")
            title = exp.findtext("TITLE")
            if title:
                attrs["SRA_EXPERIMENT_TITLE"] = title.strip()

        if sample is not None:
            attrs["SRA_SAMPLE_accession"] = sample.attrib.get("accession", "")

            title = sample.findtext("TITLE")
            if title:
                attrs["SRA_SAMPLE_TITLE"] = title.strip()

            desc = sample.findtext("DESCRIPTION")
            if desc:
                attrs["SRA_SAMPLE_DESCRIPTION"] = desc.strip()

            for ident in sample.findall(".//IDENTIFIERS/*"):
                val = (ident.text or "").strip()
                if val:
                    attrs[f"SRA_IDENTIFIER_{ident.tag}"] = val

            for a in sample.findall(".//SAMPLE_ATTRIBUTE"):
                tag = (a.findtext("TAG") or "").strip()
                val = (a.findtext("VALUE") or "").strip()
                if tag:
                    key = f"SRA_ATTR_{tag}"
                    attrs[key] = val
                    for run in runs:
                        sra_long.append({"run": run, "source": "sra", "attribute": tag, "value": val})

        for run in runs:
            if run:
                row = {"run": run}
                row.update(attrs)
                sra_rows.append(row)

sra = pd.DataFrame(sra_rows).drop_duplicates() if sra_rows else pd.DataFrame(columns=["run"])
sra.to_csv(out / "PRJNA913947_sra_descriptors_wide.tsv", sep="\t", index=False)
pd.DataFrame(sra_long).to_csv(out / "PRJNA913947_sra_descriptors_long.tsv", sep="\t", index=False)

# ---------- Parse BioSample XML ----------
bio_rows = []
bio_long = []

for xp in sorted(xml_dir.glob("biosample_chunk_*.xml")):
    try:
        root = ET.parse(xp).getroot()
    except Exception as e:
        print(f"WARNING: could not parse {xp}: {e}")
        continue

    for bs in root.findall(".//BioSample"):
        acc = bs.attrib.get("accession", "")
        row = {"BioSample": acc}

        title = bs.findtext("Description/Title")
        if title:
            row["BioSample_Title"] = title.strip()

        organism = bs.findtext("Description/Organism/OrganismName")
        if organism:
            row["BioSample_Organism"] = organism.strip()

        for attr in bs.findall(".//Attributes/Attribute"):
            name = attr.attrib.get("attribute_name") or attr.attrib.get("harmonized_name") or ""
            val = (attr.text or "").strip()
            if name:
                key = f"BioSample_ATTR_{name}"
                row[key] = val
                bio_long.append({"BioSample": acc, "source": "biosample", "attribute": name, "value": val})

        bio_rows.append(row)

bio = pd.DataFrame(bio_rows).drop_duplicates() if bio_rows else pd.DataFrame(columns=["BioSample"])
bio.to_csv(out / "PRJNA913947_biosample_descriptors_wide.tsv", sep="\t", index=False)
pd.DataFrame(bio_long).to_csv(out / "PRJNA913947_biosample_descriptors_long.tsv", sep="\t", index=False)

# ---------- Merge ----------
merged = base.merge(sra, on="run", how="left")
if "BioSample" in merged.columns and "BioSample" in bio.columns:
    merged = merged.merge(bio, on="BioSample", how="left")

merged.to_csv(out / "PRJNA913947_all_descriptors_merged.tsv", sep="\t", index=False)

# ---------- Search text ----------
keywords = re.compile(
    r"kenya|kenyan|nairobi|kijabe|aga\s*khan|aic\s*kijabe|tumou?r|normal|adjacent|non[- ]?cancer",
    re.I
)

rows = []
for _, r in merged.iterrows():
    run = r.get("run", "")
    biosample = r.get("BioSample", "")
    sample = r.get("SampleName", "")
    lib = r.get("LibraryName", "")

    hits = []
    for col, val in r.items():
        val = "" if pd.isna(val) else str(val)
        if val and keywords.search(val):
            hits.append(f"{col}={val}")

    if hits:
        rows.append({
            "run": run,
            "BioSample": biosample,
            "SampleName": sample,
            "LibraryName": lib,
            "hits": " || ".join(hits)
        })

hits_df = pd.DataFrame(rows)
hits_df.to_csv(out / "PRJNA913947_keyword_hits_kenya_tumor_normal.tsv", sep="\t", index=False)

kenya_mask = merged.apply(
    lambda row: any(
        re.search(r"kenya|kenyan|nairobi|kijabe|aga\s*khan|aic\s*kijabe", str(v), re.I)
        for v in row.values
    ),
    axis=1
)

kenya = merged.loc[kenya_mask].copy()
kenya.to_csv(out / "PRJNA913947_kenya_candidate_runs.tsv", sep="\t", index=False)

# Tumor/normal indicators among Kenya candidates, if available
def classify_sample(row):
    text = " ".join(str(v) for v in row.values)
    if re.search(r"adjacent|normal|non[- ]?cancer", text, re.I):
        return "normal_or_adjacent"
    if re.search(r"tumou?r|tumor", text, re.I):
        return "tumor"
    return "unknown"

if len(kenya):
    kenya["tumor_normal_guess"] = kenya.apply(classify_sample, axis=1)
    keep_cols = [c for c in [
        "run", "filename_1", "filename_2", "total_size_gb",
        "BioSample", "SampleName", "LibraryName", "Sex",
        "tumor_normal_guess"
    ] if c in kenya.columns]
    kenya[keep_cols].to_csv(out / "PRJNA913947_kenya_candidate_runs_compact.tsv", sep="\t", index=False)
else:
    pd.DataFrame(columns=[
        "run", "filename_1", "filename_2", "total_size_gb",
        "BioSample", "SampleName", "LibraryName", "Sex", "tumor_normal_guess"
    ]).to_csv(out / "PRJNA913947_kenya_candidate_runs_compact.tsv", sep="\t", index=False)

# ---------- Summary ----------
with open(out / "PRJNA913947_kenya_resolution_summary.txt", "w") as f:
    f.write("PRJNA913947 Kenya resolution summary\n\n")
    f.write(f"Total runs in inventory: {len(base)}\n")
    f.write(f"SRA descriptor rows: {len(sra)}\n")
    f.write(f"BioSample descriptor rows: {len(bio)}\n")
    f.write(f"Keyword-hit runs: {len(hits_df)}\n")
    f.write(f"Kenya candidate runs: {len(kenya)}\n\n")
    if len(kenya):
        f.write("Kenya candidate compact file:\n")
        f.write(str(out / "PRJNA913947_kenya_candidate_runs_compact.tsv") + "\n")
    else:
        f.write("No Kenya-specific descriptor was found in parsed SRA/BioSample XML.\n")
        f.write("Pairing/sample descriptor metadata must be obtained from article supplementary files, SRA sample descriptors, or corresponding-author dataset metadata.\n")

print("Total runs:", len(base))
print("SRA descriptor rows:", len(sra))
print("BioSample descriptor rows:", len(bio))
print("Keyword-hit runs:", len(hits_df))
print("Kenya candidate runs:", len(kenya))
print()
print("Wrote:")
for p in [
    "PRJNA913947_all_descriptors_merged.tsv",
    "PRJNA913947_keyword_hits_kenya_tumor_normal.tsv",
    "PRJNA913947_kenya_candidate_runs.tsv",
    "PRJNA913947_kenya_candidate_runs_compact.tsv",
    "PRJNA913947_kenya_resolution_summary.txt",
]:
    print(out / p)
PY

echo
echo "=== Kenya resolver summary ==="
cat "$OUT/PRJNA913947_kenya_resolution_summary.txt"

echo
echo "=== Output files ==="
ls -lh "$OUT"

echo
echo "PRJNA913947 Kenya resolver completed"
date --iso-8601=seconds
