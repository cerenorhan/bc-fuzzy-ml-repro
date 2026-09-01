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

RNA_BASE="results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2"
GTF="reference/rnaseq/gencode.v44.annotation.gtf"
IN="$RNA_BASE/DESeq2/GSE142258_DESeq2_late_vs_early_results.tsv"
OUTDIR="$RNA_BASE/DESeq2/annotated"
mkdir -p "$OUTDIR"

OUT="$OUTDIR/GSE142258_DESeq2_late_vs_early_results.annotated.tsv"
SIG005="$OUTDIR/GSE142258_DESeq2_late_vs_early_results.annotated.padj005.tsv"
SIG010="$OUTDIR/GSE142258_DESeq2_late_vs_early_results.annotated.padj010.tsv"
TOP="$OUTDIR/GSE142258_DESeq2_late_vs_early_top50.annotated.tsv"

python3 - <<PY
import re
import pandas as pd
from pathlib import Path

gtf = Path("$GTF")
inp = Path("$IN")
out = Path("$OUT")
sig005 = Path("$SIG005")
sig010 = Path("$SIG010")
top = Path("$TOP")

if not gtf.exists():
    raise SystemExit(f"Missing GTF: {gtf}")
if not inp.exists():
    raise SystemExit(f"Missing DESeq2 result: {inp}")

rows = []
with gtf.open() as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\\n").split("\\t")
        if len(parts) < 9:
            continue
        if parts[2] != "gene":
            continue
        attrs = parts[8]
        def get_attr(key):
            m = re.search(rf'{key} "([^"]+)"', attrs)
            return m.group(1) if m else ""
        gene_id = get_attr("gene_id")
        gene_name = get_attr("gene_name")
        gene_type = get_attr("gene_type") or get_attr("gene_biotype")
        rows.append({
            "Geneid": gene_id,
            "Geneid_no_version": gene_id.split(".")[0],
            "gene_name": gene_name,
            "gene_type": gene_type,
            "chr": parts[0],
            "start": int(parts[3]),
            "end": int(parts[4]),
            "strand": parts[6],
        })

ann = pd.DataFrame(rows).drop_duplicates("Geneid_no_version")

res = pd.read_csv(inp, sep="\\t")
res["Geneid_no_version"] = res["Geneid"].astype(str).str.replace(r"\\.\\d+$", "", regex=True)

merged = res.merge(
    ann.drop(columns=["Geneid"]),
    on="Geneid_no_version",
    how="left"
)

# kolon sırası
front = ["Geneid", "Geneid_no_version", "gene_name", "gene_type", "chr", "start", "end", "strand"]
other = [c for c in merged.columns if c not in front]
merged = merged[front + other]

# padj NA güvenliği
merged["padj_numeric"] = pd.to_numeric(merged["padj"], errors="coerce")
merged["pvalue_numeric"] = pd.to_numeric(merged["pvalue"], errors="coerce")
merged["abs_log2FC"] = pd.to_numeric(merged["log2FoldChange"], errors="coerce").abs()

merged = merged.sort_values(["padj_numeric", "pvalue_numeric"], na_position="last")
merged.to_csv(out, sep="\\t", index=False)

merged[merged["padj_numeric"] < 0.05].to_csv(sig005, sep="\\t", index=False)
merged[merged["padj_numeric"] < 0.10].to_csv(sig010, sep="\\t", index=False)

merged.sort_values(["padj_numeric", "abs_log2FC"], ascending=[True, False], na_position="last").head(50).to_csv(top, sep="\\t", index=False)

print("Annotated DESeq2 written:")
print(out)
print()
print("Significant padj < 0.05:", int((merged["padj_numeric"] < 0.05).sum()))
print("Suggestive padj < 0.10:", int((merged["padj_numeric"] < 0.10).sum()))
print()
print("Top annotated rows:")
print(merged[["Geneid","gene_name","gene_type","baseMean","log2FoldChange","pvalue","padj"]].head(20).to_string(index=False))
PY

echo
echo "=== Output files ==="
ls -lh "$OUTDIR"
