#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


def clean_key(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
    )


def parse_one_xml(path: Path) -> dict[str, str]:
    row: dict[str, str] = {
        "biosample": path.stem,
        "xml_file": str(path),
    }

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        row["parse_error"] = str(exc)
        return row

    root = tree.getroot()

    if root.attrib.get("accession"):
        row["xml_accession"] = root.attrib.get("accession", "")

    for element in root.iter():
        tag = element.tag.split("}")[-1]

        if tag in {"TITLE", "Title"} and element.text:
            row.setdefault("title", element.text.strip())

        if tag in {"DESCRIPTION", "Description"} and element.text:
            row.setdefault("description", element.text.strip())

        # ENA SAMPLE_ATTRIBUTE structure:
        # <SAMPLE_ATTRIBUTE><TAG>sex</TAG><VALUE>female</VALUE></SAMPLE_ATTRIBUTE>
        if tag == "SAMPLE_ATTRIBUTE":
            attr_tag = ""
            attr_value = ""

            for child in element:
                child_tag = child.tag.split("}")[-1]
                if child_tag == "TAG" and child.text:
                    attr_tag = child.text.strip()
                elif child_tag == "VALUE" and child.text:
                    attr_value = child.text.strip()

            if attr_tag:
                row[clean_key(attr_tag)] = attr_value

        # NCBI BioSample structure:
        # <Attribute attribute_name="sex">female</Attribute>
        if tag == "Attribute":
            key = (
                element.attrib.get("attribute_name")
                or element.attrib.get("harmonized_name")
                or element.attrib.get("display_name")
                or ""
            )
            value = element.text.strip() if element.text else ""

            if key:
                row[clean_key(key)] = value

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    rows = []

    for path in sorted(xml_dir.glob("*.xml")):
        rows.append(parse_one_xml(path))

    df = pd.DataFrame(rows)
    df.to_csv(args.output, sep="\t", index=False)

    print("Parsed XML files:", len(df))
    print("Columns:")
    for col in df.columns:
        print(col)

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
