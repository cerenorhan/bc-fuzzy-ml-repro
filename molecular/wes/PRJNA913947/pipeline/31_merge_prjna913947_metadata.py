#!/usr/bin/env python3

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runinfo", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run = pd.read_csv(args.runinfo, dtype=str).fillna("")
    attr = pd.read_csv(args.attributes, sep="\t", dtype=str).fillna("")

    if "biosample" not in attr.columns:
        raise ValueError("biosample column missing from attributes table.")

    merged = run.merge(
        attr,
        left_on="BioSample",
        right_on="biosample",
        how="left",
        suffixes=("", "_attr"),
    )

    merged.to_csv(args.output, sep="\t", index=False)

    print("RunInfo rows:", len(run))
    print("Attribute rows:", len(attr))
    print("Merged rows:", len(merged))
    print("Saved:", args.output)

    keywords = [
        "kenya", "kenyan",
        "african", "european", "asian",
        "tumor", "tumour", "normal", "adjacent",
        "race", "ethnic", "population",
        "tissue", "sample", "subject",
        "patient", "case"
    ]

    print("\nColumns containing useful keywords:")
    for col in merged.columns:
        text = " ".join(merged[col].astype(str).tolist()).lower()
        if any(k in text for k in keywords):
            print("\n---", col, "---")
            print(merged[col].value_counts(dropna=False).head(80).to_string())


if __name__ == "__main__":
    main()
