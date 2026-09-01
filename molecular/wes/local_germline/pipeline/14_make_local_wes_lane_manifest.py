#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT = Path(".")
FASTQ_ROOT = PROJECT / "data/local_wes/fastq_extracted/RawData"
SAMPLE_METADATA = PROJECT / "metadata/local_wes.tsv"

OUT_LANES = PROJECT / "metadata/local_wes/local_wes.lanes.tsv"


def parse_fastq_name(path: Path) -> dict[str, str]:
    filename = path.name
    sample_id = path.parent.name

    pattern = re.compile(
        r"^(?P<sample>.+?)_"
        r"(?P<library>TKHS[^_]+)_"
        r"(?P<flowcell>[^_]+)_"
        r"(?P<lane>L\d+)_"
        r"(?P<read>[12])\.fq\.gz$"
    )

    match = pattern.match(filename)

    if not match:
        raise ValueError(f"Could not parse FASTQ filename: {filename}")

    d = match.groupdict()

    return {
        "sample_id": sample_id,
        "sample_from_filename": d["sample"],
        "library": d["library"],
        "flowcell": d["flowcell"],
        "lane": d["lane"],
        "read": d["read"],
        "fastq": str(path),
    }


def main() -> None:
    sample_meta = pd.read_csv(SAMPLE_METADATA, sep="\t", dtype=str).fillna("")

    fastq_paths = sorted(FASTQ_ROOT.glob("*/*.fq.gz"))

    rows = [parse_fastq_name(p) for p in fastq_paths]
    df = pd.DataFrame(rows)

    df["lane_id"] = (
        df["sample_id"]
        + "__"
        + df["library"]
        + "__"
        + df["flowcell"]
        + "__"
        + df["lane"]
    )

    pair_rows = []

    for lane_id, sub in df.groupby("lane_id", sort=True):
        sample_id = sub["sample_id"].iloc[0]
        library = sub["library"].iloc[0]
        flowcell = sub["flowcell"].iloc[0]
        lane = sub["lane"].iloc[0]

        r1 = sub.loc[sub["read"].eq("1"), "fastq"].tolist()
        r2 = sub.loc[sub["read"].eq("2"), "fastq"].tolist()

        if len(r1) != 1 or len(r2) != 1:
            raise ValueError(
                f"Pairing problem for {lane_id}: "
                f"R1={len(r1)}, R2={len(r2)}"
            )

        pair_rows.append(
            {
                "sample_id": sample_id,
                "lane_id": lane_id,
                "library": library,
                "flowcell": flowcell,
                "lane": lane,
                "r1": r1[0],
                "r2": r2[0],
            }
        )

    lanes = pd.DataFrame(pair_rows)

    lanes = lanes.merge(
        sample_meta[["sample_id", "group", "sex"]],
        on="sample_id",
        how="left",
    )

    lanes = lanes[
        [
            "sample_id",
            "group",
            "sex",
            "lane_id",
            "library",
            "flowcell",
            "lane",
            "r1",
            "r2",
        ]
    ]

    lanes.to_csv(OUT_LANES, sep="\t", index=False)

    print("FASTQ files:", len(df))
    print("Lane pairs:", len(lanes))
    print("Samples:", lanes["sample_id"].nunique())
    print()
    print("Lane pairs per sample:")
    print(lanes.groupby("sample_id").size().sort_values().to_string())
    print()
    print("Output:", OUT_LANES)


if __name__ == "__main__":
    main()
