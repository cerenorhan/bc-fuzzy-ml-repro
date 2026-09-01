#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ena", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run-list")
    args = parser.parse_args()

    ena = pd.read_csv(args.ena, sep="\t", dtype=str).fillna("")

    if args.run_list:
        selected = {
            line.strip()
            for line in Path(args.run_list).read_text().splitlines()
            if line.strip()
        }
        ena = ena.loc[ena["run_accession"].isin(selected)].copy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    urls = []
    md5_lines = []

    for _, row in ena.iterrows():
        run = row["run_accession"]

        ftp_values = row["fastq_ftp"].split(";") if row["fastq_ftp"] else []
        md5_values = row["fastq_md5"].split(";") if row["fastq_md5"] else []
        byte_values = row["fastq_bytes"].split(";") if row["fastq_bytes"] else []

        if len(ftp_values) != len(md5_values):
            raise ValueError(f"URL/MD5 mismatch for {run}")

        for i, ftp in enumerate(ftp_values):
            if not ftp:
                continue

            url = "https://" + ftp
            filename = Path(ftp).name
            md5 = md5_values[i]
            size = byte_values[i] if i < len(byte_values) else ""

            urls.append(url)
            md5_lines.append(f"{md5}  {filename}")

            rows.append(
                {
                    "run_accession": run,
                    "sample_accession": row.get("sample_accession", ""),
                    "study_accession": row.get("study_accession", ""),
                    "library_strategy": row.get("library_strategy", ""),
                    "library_layout": row.get("library_layout", ""),
                    "instrument_model": row.get("instrument_model", ""),
                    "fastq_url": url,
                    "filename": filename,
                    "md5": md5,
                    "bytes": size,
                }
            )

    manifest = pd.DataFrame(rows)

    manifest.to_csv(
        outdir / "fastq_download_manifest.tsv",
        sep="\t",
        index=False,
    )

    (outdir / "urls.txt").write_text("\n".join(urls) + "\n")
    (outdir / "md5.txt").write_text("\n".join(md5_lines) + "\n")

    total_bytes = pd.to_numeric(manifest["bytes"], errors="coerce").sum()

    print(f"Runs selected: {ena['run_accession'].nunique()}")
    print(f"FASTQ files: {len(manifest)}")
    print(f"Compressed FASTQ size: {total_bytes / 1024**3:.2f} GiB")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
