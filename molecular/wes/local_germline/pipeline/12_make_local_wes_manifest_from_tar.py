#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT = Path("metadata/local_wes/local_fastq_paths_in_tar.txt")
OUTPUT_DETAIL = Path("metadata/local_wes/local_fastq_files_from_tar.tsv")
OUTPUT_SAMPLE = Path("metadata/local_wes/local_wes.samples_from_tar.tsv")
OUTPUT_PROJECT_ENV = Path("metadata/local_wes/local_wes.tsv")


def infer_mate(filename: str) -> str:
    stem = filename

    if re.search(r"_1\.f(ast)?q\.gz$", stem, re.IGNORECASE):
        return "R1"

    if re.search(r"_2\.f(ast)?q\.gz$", stem, re.IGNORECASE):
        return "R2"

    return ""


def infer_sample_from_path(path: str) -> str:
    parts = Path(path).parts

    if len(parts) >= 2 and parts[0] == "RawData":
        return parts[1]

    filename = Path(path).name
    return filename.split("_", 1)[0]


def infer_group(sample_id: str) -> str:
    if sample_id.startswith("BC"):
        return "breast_cancer"

    if sample_id.startswith("H"):
        return "hadza"

    if sample_id.startswith("C"):
        return "tanzanian_control"

    return "unknown"


def infer_sex(sample_id: str) -> str:
    if sample_id.endswith("F"):
        return "female"

    if sample_id.endswith("M"):
        return "male"

    return "unknown"


def main() -> None:
    rows = []

    for path in INPUT.read_text().splitlines():
        path = path.strip()

        if not path:
            continue

        filename = Path(path).name
        sample_id = infer_sample_from_path(path)
        mate = infer_mate(filename)

        extracted_path = f"data/local_wes/fastq_extracted/{path}"

        rows.append(
            {
                "sample_id": sample_id,
                "group": infer_group(sample_id),
                "sex": infer_sex(sample_id),
                "mate": mate,
                "filename": filename,
                "path_in_tar": path,
                "extracted_path": extracted_path,
            }
        )

    detail = pd.DataFrame(rows)

    if detail.empty:
        raise SystemExit("No FASTQ rows found.")

    detail = detail.sort_values(["sample_id", "mate", "filename"])
    detail.to_csv(OUTPUT_DETAIL, sep="\t", index=False)

    sample_rows = []

    for sample_id, sub in detail.groupby("sample_id", sort=True):
        r1 = sub.loc[sub["mate"].eq("R1"), "extracted_path"].tolist()
        r2 = sub.loc[sub["mate"].eq("R2"), "extracted_path"].tolist()

        sample_rows.append(
            {
                "sample_id": sample_id,
                "group": infer_group(sample_id),
                "sex": infer_sex(sample_id),
                "n_fastq_total": len(sub),
                "n_r1": len(r1),
                "n_r2": len(r2),
                "fastq_r1": ",".join(r1),
                "fastq_r2": ",".join(r2),
            }
        )

    samples = pd.DataFrame(sample_rows)

    samples.to_csv(OUTPUT_SAMPLE, sep="\t", index=False)

    # Bu dosya config/config.local.sh içinde LOCAL_WES_METADATA olarak tanımlı.
    samples.to_csv(OUTPUT_PROJECT_ENV, sep="\t", index=False)

    print("FASTQ files:", len(detail))
    print("Samples:", len(samples))
    print()
    print("Group counts:")
    print(samples["group"].value_counts().to_string())
    print()
    print("Sex counts:")
    print(samples["sex"].value_counts().to_string())
    print()
    print("FASTQ pairing counts:")
    print(samples[["n_fastq_total", "n_r1", "n_r2"]].value_counts().sort_index().to_string())
    print()
    print("Samples with non-2 FASTQ files:")
    print(samples.loc[samples["n_fastq_total"].ne(2)].to_string(index=False))


if __name__ == "__main__":
    main()
