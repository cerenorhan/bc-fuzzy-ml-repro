#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd


INPUT = "metadata/public/PRJNA913947.merged_metadata.tsv"
OUTPUT = "metadata/public/PRJNA913947.run_level_tumor_normal.tsv"


df = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")

required = [
    "Run",
    "SampleName",
    "LibraryName",
    "BioSample",
    "isolate",
    "Sex",
    "size_MB",
    "LibraryStrategy",
    "LibraryLayout",
    "Model",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

out = df[required].copy()

out["sample_num"] = (
    out["SampleName"]
    .str.extract(r"(\d+)", expand=False)
    .astype("Int64")
)

out["sample_type"] = (
    out["isolate"]
    .str.lower()
    .map(
        {
            "normal breast tissue": "normal",
            "tumor breast tissue": "tumor",
        }
    )
)

out = out.sort_values(["sample_num", "sample_type"]).copy()

out.to_csv(OUTPUT, sep="\t", index=False)

print("Rows:", len(out))
print("Sample type counts:")
print(out["sample_type"].value_counts(dropna=False))

print("\nSaved:", OUTPUT)

print("\nFirst 30 rows:")
print(out.head(30).to_string(index=False))
