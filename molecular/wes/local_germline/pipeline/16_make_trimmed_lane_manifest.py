#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd


IN_LANES = Path("metadata/local_wes/local_wes.lanes.tsv")
OUT = Path("metadata/local_wes/local_wes.trimmed_lanes.tsv")


def main() -> None:
    lanes = pd.read_csv(IN_LANES, sep="\t", dtype=str).fillna("")

    rows = []

    for _, row in lanes.iterrows():
        sample_id = row["sample_id"]
        lane_id = row["lane_id"]

        r1 = Path(f"data/trimmed/local_wes/{sample_id}/{lane_id}.R1.trimmed.fq.gz")
        r2 = Path(f"data/trimmed/local_wes/{sample_id}/{lane_id}.R2.trimmed.fq.gz")

        rows.append({
            "sample_id": sample_id,
            "group": row["group"],
            "sex": row["sex"],
            "lane_id": lane_id,
            "library": row["library"],
            "flowcell": row["flowcell"],
            "lane": row["lane"],
            "trimmed_r1": str(r1),
            "trimmed_r2": str(r2),
            "trimmed_r1_exists": r1.exists() and r1.stat().st_size > 0,
            "trimmed_r2_exists": r2.exists() and r2.stat().st_size > 0,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)

    print("Rows:", len(out))
    print("Samples:", out["sample_id"].nunique())
    print("Missing R1:", (~out["trimmed_r1_exists"]).sum())
    print("Missing R2:", (~out["trimmed_r2_exists"]).sum())
    print("Output:", OUT)


if __name__ == "__main__":
    main()
