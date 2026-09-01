#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_colname(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_").lower()


def sex_code_from_value(value: str) -> str:
    value = str(value).strip().lower()

    if value.startswith("f"):
        return "F"

    if value.startswith("m"):
        return "M"

    return ""


def normalize_sample_id(lab_id: str, sex: str) -> str:
    lab_id = str(lab_id).strip()
    lab_id = re.sub(r"\s+", "", lab_id)
    lab_id = lab_id.upper()

    sex_code = sex_code_from_value(sex)

    # Excel: HD002 -> FASTQ: H002F / H002M
    if lab_id.startswith("HD"):
        return "H" + lab_id[2:] + sex_code

    # Example mapping: BC<number> -> BC<number>F
    if lab_id.startswith("BC"):
        if re.search(r"[FM]$", lab_id):
            return lab_id
        return lab_id + sex_code

    # Control identifiers are expected to follow the configured local naming convention
    if lab_id.startswith("C"):
        if re.search(r"[FM]$", lab_id):
            return lab_id
        return lab_id + sex_code

    return lab_id


def infer_group_from_lab_id(lab_id: str) -> str:
    lab_id = str(lab_id).strip().upper()

    if lab_id.startswith("BC"):
        return "breast_cancer"

    if lab_id.startswith("HD"):
        return "hadza"

    if lab_id.startswith("C"):
        return "tanzanian_control"

    return "unknown"


def normalize_sex(value: str) -> str:
    value = str(value).strip().lower()

    if value.startswith("f"):
        return "female"

    if value.startswith("m"):
        return "male"

    return value if value else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized local WES metadata by merging an institutional metadata spreadsheet with the FASTQ manifest."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to the institutional Excel metadata file.",
    )
    parser.add_argument(
        "--fastq-manifest",
        type=Path,
        default=Path("metadata/local_wes/local_wes.samples_from_tar.tsv"),
        help="FASTQ sample manifest TSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/local_wes"),
        help="Directory for normalized and merged metadata outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    excel_path = args.metadata
    fastq_manifest = args.fastq_manifest
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_excel_norm = output_dir / "local_wes.excel_metadata.normalized.tsv"
    out_merged = output_dir / "local_wes.metadata_merged.tsv"
    out_pipeline = output_dir / "local_wes.tsv"
    out_missing_fastq = output_dir / "local_wes.excel_samples_missing_fastq.tsv"
    out_fastq_not_excel = output_dir / "local_wes.fastq_samples_not_in_excel.tsv"

    fastq = pd.read_csv(fastq_manifest, sep="\t", dtype=str).fillna("")

    # FASTQ manifestindeki sex/group kolonlarını merge sonrası net ayır.
    fastq = fastq.rename(
        columns={
            "sex": "sex_fastq",
            "group": "group_fastq",
        }
    )

    excel_raw = pd.read_excel(excel_path, sheet_name=0, dtype=str).fillna("")
    excel_raw.columns = [clean_colname(c) for c in excel_raw.columns]

    if "lab_id" not in excel_raw.columns:
        raise ValueError(f"lab_id column not found. Columns: {excel_raw.columns.tolist()}")

    if "sex" not in excel_raw.columns:
        raise ValueError(f"sex column not found. Columns: {excel_raw.columns.tolist()}")

    excel = excel_raw.copy()
    excel["lab_id_original"] = excel["lab_id"]
    excel["sample_id"] = [
        normalize_sample_id(lab_id, sex)
        for lab_id, sex in zip(excel["lab_id"], excel["sex"])
    ]

    excel["group_excel"] = excel["lab_id_original"].map(infer_group_from_lab_id)
    excel["sex_excel"] = excel["sex"].map(normalize_sex)

    excel.to_csv(out_excel_norm, sep="\t", index=False)

    merged = fastq.merge(
        excel,
        on="sample_id",
        how="outer",
        indicator=True,
    )

    merged.to_csv(out_merged, sep="\t", index=False)

    missing_fastq = merged.loc[merged["_merge"].eq("right_only")].copy()
    fastq_not_excel = merged.loc[merged["_merge"].eq("left_only")].copy()

    missing_fastq.to_csv(out_missing_fastq, sep="\t", index=False)
    fastq_not_excel.to_csv(out_fastq_not_excel, sep="\t", index=False)

    # Ana pipeline metadata: sadece FASTQ bulunan örnekler.
    pipeline = merged.loc[merged["_merge"].isin(["both", "left_only"])].copy()

    pipeline["group"] = pipeline["group_fastq"].where(
        pipeline["group_fastq"].ne(""),
        pipeline["group_excel"],
    )

    pipeline["sex"] = pipeline["sex_fastq"].where(
        pipeline["sex_fastq"].ne(""),
        pipeline["sex_excel"],
    )

    keep_first = [
        "sample_id",
        "group",
        "sex",
        "n_fastq_total",
        "n_r1",
        "n_r2",
        "fastq_r1",
        "fastq_r2",
        "lab_id_original",
        "age_years",
        "participants_origin",
        "bc_pathologic_stage",
        "age_at_bc_diagnosis_years",
        "laterality",
        "er_status",
        "pr_status",
        "her_2_status",
        "molecular_subtype",
        "histological_type",
        "_merge",
    ]

    existing_keep = [c for c in keep_first if c in pipeline.columns]
    other_cols = [c for c in pipeline.columns if c not in existing_keep]

    pipeline = pipeline[existing_keep + other_cols]
    pipeline.to_csv(out_pipeline, sep="\t", index=False)

    print("Excel normalized samples:", excel["sample_id"].nunique())
    print("FASTQ samples:", fastq["sample_id"].nunique())
    print("Merged rows:", len(merged))
    print()

    print("Excel group counts:")
    print(excel["group_excel"].value_counts().to_string())
    print()

    print("FASTQ group counts:")
    print(fastq["group_fastq"].value_counts().to_string())
    print()

    print("Samples in Excel but missing FASTQ:")
    if missing_fastq.empty:
        print("None")
    else:
        cols = ["sample_id", "lab_id_original", "group_excel", "sex_excel"]
        print(missing_fastq[cols].to_string(index=False))

    print()

    print("Samples in FASTQ but missing Excel:")
    if fastq_not_excel.empty:
        print("None")
    else:
        cols = ["sample_id", "group_fastq", "sex_fastq"]
        print(fastq_not_excel[cols].to_string(index=False))

    print()
    print("Output pipeline metadata:")
    print(out_pipeline)


if __name__ == "__main__":
    main()
