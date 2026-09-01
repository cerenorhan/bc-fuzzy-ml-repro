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

THREADS="${RUN_THREADS:-12}"

BASE="results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2"

RAWLIST="$BASE/metadata/GSE142258_FASTQ_files.tsv"
MANIFEST="$BASE/metadata/GSE142258_FASTQ_manifest.tsv"
META_GEO="results/EA_BC_AI_MultiOmics/rnaseq_geo_metadata/GSE142258_sample_metadata.tsv"

REF="$REF_FASTA"
GTF_GZ="reference/rnaseq/gencode.v44.annotation.gtf.gz"
GTF="reference/rnaseq/gencode.v44.annotation.gtf"

BOWTIE2_INDEX_DIR="reference/rnaseq/bowtie2_GRCh38"
BOWTIE2_INDEX_BASE="$BOWTIE2_INDEX_DIR/GRCh38"

TRIM_DIR="$BASE/trimmomatic"
FASTQC_RAW_DIR="$BASE/fastqc_raw"
FASTQC_TRIM_DIR="$BASE/fastqc_trimmed"
TOPHAT_DIR="$BASE/tophat"
BAM_DIR="$BASE/bam"
COUNT_DIR="$BASE/featureCounts"
DESEQ_DIR="$BASE/DESeq2"
QC_DIR="$BASE/multiqc"

LOG="logs/public/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2_$(date +%Y%m%d_%H%M%S).log"

mkdir -p \
  "$BASE/metadata" \
  "$TRIM_DIR" \
  "$FASTQC_RAW_DIR" \
  "$FASTQC_TRIM_DIR" \
  "$TOPHAT_DIR" \
  "$BAM_DIR" \
  "$COUNT_DIR" \
  "$DESEQ_DIR" \
  "$QC_DIR" \
  logs/public

exec > >(tee -a "$LOG") 2>&1

echo "GSE142258 Trimmomatic + TopHat2 + featureCounts + DESeq2 started"
date --iso-8601=seconds
echo "THREADS=$THREADS"

echo
echo "=== Input check ==="
[[ -s "$REF" ]] || { echo "ERROR: missing REF_FASTA: $REF"; exit 1; }
[[ -s "$META_GEO" ]] || { echo "ERROR: missing metadata: $META_GEO"; exit 1; }

if [[ ! -s "$GTF" ]]; then
    [[ -s "$GTF_GZ" ]] || { echo "ERROR: missing GTF: $GTF_GZ"; exit 1; }
    gzip -dc "$GTF_GZ" > "$GTF"
fi

echo "REF=$REF"
echo "GTF=$GTF"

echo
echo "=== Locate GSE142258 FASTQs ==="
find "$PUBLIC_DATA_ROOT" "$PROJECT_ROOT" \
  -type f \( -name "SRR107298*.fastq.gz" -o -name "SRR107298*.fq.gz" \) \
  -printf "%p\t%s\n" 2>/dev/null \
  | sort > "$RAWLIST"

cat "$RAWLIST"

N_FASTQ=$(wc -l < "$RAWLIST")
echo "FASTQ count: $N_FASTQ"

if [[ "$N_FASTQ" -ne 30 ]]; then
    echo "ERROR: expected 30 FASTQ files for 15 paired-end samples, found $N_FASTQ"
    exit 1
fi

echo
echo "=== Build sample manifest and stage metadata ==="

micromamba run -n rnaseq python - "$RAWLIST" "$META_GEO" "$MANIFEST" "$BASE/metadata/GSE142258_DESeq2_metadata.tsv" <<'PY'
import sys
import re
import subprocess
from pathlib import Path
import pandas as pd

rawlist = Path(sys.argv[1])
geo_path = Path(sys.argv[2])
manifest_out = Path(sys.argv[3])
meta_out = Path(sys.argv[4])

records = []
for line in rawlist.read_text().splitlines():
    if not line.strip():
        continue
    path, size = line.split("\t")
    fn = Path(path).name
    srr = re.search(r"(SRR[0-9]+)", fn).group(1)

    if re.search(r"(_1|_R1)(\.|_|fastq|fq)", fn):
        mate = "R1"
    elif re.search(r"(_2|_R2)(\.|_|fastq|fq)", fn):
        mate = "R2"
    else:
        raise SystemExit(f"Cannot infer mate from {fn}")

    records.append({"sample_id": srr, "mate": mate, "fastq": path, "size_bytes": int(size)})

fq = pd.DataFrame(records)
manifest = fq.pivot_table(index="sample_id", columns="mate", values="fastq", aggfunc="first").reset_index()
manifest = manifest[["sample_id", "R1", "R2"]]

geo = pd.read_csv(geo_path, sep="\t", dtype=str).fillna("")

def extract_srx(row):
    text = " ".join(str(v) for v in row.values)
    m = re.search(r"SRX[0-9]+", text)
    return m.group(0) if m else ""

def infer_stage(row):
    text = " ".join(str(v) for v in row.values).lower()
    patterns = [
        ("Stage IIIC", "late"),
        ("Stage IIIB", "late"),
        ("Stage IIIA", "late"),
        ("Stage IIB", "early"),
        ("Stage IIA", "early"),
        ("Stage I", "early"),
    ]
    for stage, group in patterns:
        if stage.lower() in text:
            return stage, group
    return "", "unknown"

geo["Experiment"] = geo.apply(extract_srx, axis=1)
tmp = geo.apply(infer_stage, axis=1)
geo["stage"] = [x[0] for x in tmp]
geo["stage_group"] = [x[1] for x in tmp]

srx = sorted([x for x in geo["Experiment"].unique() if x.startswith("SRX")])
runinfo_files = []
tmpdir = manifest_out.parent

for i in range(0, len(srx), 80):
    chunk = srx[i:i+80]
    f = tmpdir / f"GSE142258_SraRunInfo_chunk_{i//80+1}.csv"
    if not f.exists() or f.stat().st_size == 0:
        url = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=" + ",".join(chunk)
        subprocess.run(["curl", "-L", "--retry", "5", "--retry-delay", "5", url, "-o", str(f)], check=True)
    runinfo_files.append(f)

runinfos = []
for f in runinfo_files:
    x = pd.read_csv(f, dtype=str).fillna("")
    if "Run" in x.columns:
        runinfos.append(x)

runinfo = pd.concat(runinfos, ignore_index=True).drop_duplicates()

stage_meta = runinfo.merge(
    geo[["GSM", "title", "Experiment", "stage", "stage_group"]],
    on="Experiment",
    how="left"
).rename(columns={"Run": "sample_id"})

stage_meta = stage_meta[["sample_id", "Experiment", "GSM", "title", "stage", "stage_group"]].drop_duplicates()

manifest = manifest.merge(stage_meta, on="sample_id", how="left")

if manifest["R1"].isna().any() or manifest["R2"].isna().any():
    raise SystemExit("Missing R1/R2")
if (manifest["stage_group"] == "unknown").any() or manifest["stage_group"].isna().any():
    raise SystemExit("Unknown stage group in manifest")

manifest.to_csv(manifest_out, sep="\t", index=False)

meta = manifest[["sample_id", "GSM", "title", "stage", "stage_group"]].copy()
meta.to_csv(meta_out, sep="\t", index=False)

print("Manifest:")
print(manifest.to_string(index=False))
print()
print("Stage summary:")
print(meta.groupby(["stage", "stage_group"]).size().reset_index(name="n").to_string(index=False))
PY

cat "$MANIFEST"

echo
echo "=== Locate Trimmomatic adapter ==="
ADAPTER="$(micromamba run -n rnaseq_tophat bash -lc 'ls $CONDA_PREFIX/share/trimmomatic/adapters/TruSeq3-PE*.fa 2>/dev/null | head -1')"

if [[ -z "$ADAPTER" || ! -s "$ADAPTER" ]]; then
    echo "ERROR: Trimmomatic adapter not found"
    exit 1
fi

echo "Adapter: $ADAPTER"

echo
echo "=== Raw FastQC ==="
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r sample_id R1 R2 Experiment GSM title stage stage_group
do
    if [[ ! -s "$FASTQC_RAW_DIR/${sample_id}_1_fastqc.html" && ! -s "$FASTQC_RAW_DIR/${sample_id}_R1_fastqc.html" ]]; then
        fastqc -t 2 -o "$FASTQC_RAW_DIR" "$R1" "$R2" || true
    fi
done

echo
echo "=== Trimmomatic PE ==="
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r sample_id R1 R2 Experiment GSM title stage stage_group
do
    echo
    echo "### Trimming $sample_id ###"

    P1="$TRIM_DIR/${sample_id}_R1.paired.fastq.gz"
    U1="$TRIM_DIR/${sample_id}_R1.unpaired.fastq.gz"
    P2="$TRIM_DIR/${sample_id}_R2.paired.fastq.gz"
    U2="$TRIM_DIR/${sample_id}_R2.unpaired.fastq.gz"

    if [[ -s "$P1" && -s "$P2" ]]; then
        echo "Trimmed paired FASTQs exist, skipping"
        continue
    fi

    env _JAVA_OPTIONS="-Xmx16g" trimmomatic PE \
      -threads "$THREADS" \
      -phred33 \
      "$R1" "$R2" \
      "$P1" "$U1" \
      "$P2" "$U2" \
      ILLUMINACLIP:"$ADAPTER":2:30:10 \
      LEADING:20 \
      TRAILING:20 \
      SLIDINGWINDOW:4:20 \
      MINLEN:35
done

echo
echo "=== Trimmed FastQC ==="
find "$TRIM_DIR" -name "*.paired.fastq.gz" | sort | xargs -r fastqc -t "$THREADS" -o "$FASTQC_TRIM_DIR" || true

echo
echo "=== Build Bowtie2 genome index for TopHat2 if needed ==="
if [[ ! -s "${BOWTIE2_INDEX_BASE}.1.bt2" && ! -s "${BOWTIE2_INDEX_BASE}.1.bt2l" ]]; then
    mkdir -p "$BOWTIE2_INDEX_DIR"
    bowtie2-build --threads "$THREADS" "$REF" "$BOWTIE2_INDEX_BASE"
else
    echo "Bowtie2 index exists: $BOWTIE2_INDEX_BASE"
fi

echo
echo "=== TopHat2 alignment ==="
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r sample_id R1 R2 Experiment GSM title stage stage_group
do
    echo
    echo "### TopHat2 $sample_id ###"

    P1="$TRIM_DIR/${sample_id}_R1.paired.fastq.gz"
    P2="$TRIM_DIR/${sample_id}_R2.paired.fastq.gz"

    SAMPLE_OUT="$TOPHAT_DIR/$sample_id"
    ACCEPTED="$SAMPLE_OUT/accepted_hits.bam"
    SORTED="$BAM_DIR/${sample_id}.tophat.accepted_hits.sorted.bam"

    if [[ -s "$SORTED" ]]; then
        echo "Sorted BAM exists, skipping TopHat2"
        continue
    fi

    rm -rf "$SAMPLE_OUT"
    mkdir -p "$SAMPLE_OUT"

    tophat2 \
      -p "$THREADS" \
      -G "$GTF" \
      -o "$SAMPLE_OUT" \
      --library-type fr-unstranded \
      "$BOWTIE2_INDEX_BASE" \
      "$P1" "$P2"

    [[ -s "$ACCEPTED" ]] || { echo "ERROR: missing accepted_hits.bam for $sample_id"; exit 1; }

    samtools sort -@ "$THREADS" -o "$SORTED" "$ACCEPTED"
    samtools index -@ "$THREADS" "$SORTED"
done

echo
echo "=== TopHat alignment summary ==="
for f in "$TOPHAT_DIR"/*/align_summary.txt
do
    [[ -s "$f" ]] || continue
    sample=$(basename "$(dirname "$f")")
    echo "### $sample ###"
    cat "$f"
    echo
done > "$QC_DIR/GSE142258_TopHat_align_summary_all.txt"

cat "$QC_DIR/GSE142258_TopHat_align_summary_all.txt"

echo
echo "=== featureCounts raw gene counts, strandedness test ==="
mapfile -t BAMS < <(find "$BAM_DIR" -name "*.tophat.accepted_hits.sorted.bam" | sort)

for STRAND in 0 1 2
do
    OUT_COUNTS="$COUNT_DIR/GSE142258.featureCounts.s${STRAND}.txt"

    if [[ -s "$OUT_COUNTS" ]]; then
        echo "featureCounts s${STRAND} exists, skipping"
        continue
    fi

    featureCounts \
      -T "$THREADS" \
      -p \
      --countReadPairs \
      -B \
      -C \
      -s "$STRAND" \
      -t exon \
      -g gene_id \
      -a "$GTF" \
      -o "$OUT_COUNTS" \
      "${BAMS[@]}"
done

echo
echo "=== MultiQC ==="
micromamba run -n rnaseq multiqc "$BASE" -o "$QC_DIR" --force || true

echo
echo "=== Prepare DESeq2 input ==="

micromamba run -n rnaseq python - "$COUNT_DIR" "$DESEQ_DIR" "$BASE/metadata/GSE142258_DESeq2_metadata.tsv" <<'PY'
import sys
import re
from pathlib import Path
import pandas as pd

count_dir = Path(sys.argv[1])
deseq_dir = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
deseq_dir.mkdir(parents=True, exist_ok=True)

summaries = []
for strand in [0, 1, 2]:
    sf = count_dir / f"GSE142258.featureCounts.s{strand}.txt.summary"
    if not sf.exists():
        continue
    s = pd.read_csv(sf, sep="\t")
    sample_cols = [c for c in s.columns if c != "Status"]
    assigned = int(s.loc[s["Status"] == "Assigned", sample_cols].sum(axis=1).iloc[0])
    total = int(s[sample_cols].sum().sum())
    summaries.append({
        "strand": strand,
        "assigned_total": assigned,
        "total_counted_categories": total,
        "assigned_percent": 100 * assigned / total if total else 0
    })

strand_df = pd.DataFrame(summaries).sort_values("assigned_percent", ascending=False)
strand_df.to_csv(deseq_dir / "featureCounts_strandedness_summary.tsv", sep="\t", index=False)

best = int(strand_df.iloc[0]["strand"])
print("Selected strandedness:")
print(strand_df.to_string(index=False))

cf = count_dir / f"GSE142258.featureCounts.s{best}.txt"
counts = pd.read_csv(cf, sep="\t", comment="#")

sample_cols = counts.columns[6:].tolist()

ren = {}
for c in sample_cols:
    m = re.search(r"(SRR[0-9]+)", c)
    if m:
        ren[c] = m.group(1)

mat = counts[["Geneid"] + sample_cols].rename(columns=ren)
mat = mat.rename(columns={"Geneid": "gene_id"})
mat = mat.groupby("gene_id", as_index=False).sum(numeric_only=True)

meta = pd.read_csv(meta_path, sep="\t", dtype=str)
keep_cols = ["gene_id"] + meta["sample_id"].tolist()
mat = mat[keep_cols]

mat.to_csv(deseq_dir / "GSE142258_featureCounts_raw_gene_counts.tsv", sep="\t", index=False)
meta.to_csv(deseq_dir / "GSE142258_DESeq2_metadata.tsv", sep="\t", index=False)
PY

echo
echo "=== Run DESeq2 ==="

cat > "$DESEQ_DIR/run_DESeq2_GSE142258_stage.R" <<'RSCRIPT'
suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
})

args <- commandArgs(trailingOnly=TRUE)
count_file <- args[1]
meta_file <- args[2]
outdir <- args[3]

dir.create(outdir, recursive=TRUE, showWarnings=FALSE)

counts_df <- read.delim(count_file, check.names=FALSE)
meta <- read.delim(meta_file, check.names=FALSE)

rownames(counts_df) <- counts_df$gene_id
counts_df$gene_id <- NULL

counts_mat <- round(as.matrix(counts_df))

meta <- meta[match(colnames(counts_mat), meta$sample_id), ]
stopifnot(all(meta$sample_id == colnames(counts_mat)))

meta$stage_group <- factor(meta$stage_group, levels=c("early", "late"))

dds <- DESeqDataSetFromMatrix(
  countData = counts_mat,
  colData = meta,
  design = ~ stage_group
)

keep <- rowSums(counts(dds) >= 10) >= min(table(meta$stage_group))
dds <- dds[keep, ]

dds <- DESeq(dds)

res <- results(dds, contrast=c("stage_group", "late", "early"))
res_df <- as.data.frame(res)
res_df$gene_id <- rownames(res_df)
res_df <- res_df[, c("gene_id", setdiff(colnames(res_df), "gene_id"))]
res_df <- res_df[order(res_df$padj, res_df$pvalue), ]

write.table(
  res_df,
  file=file.path(outdir, "GSE142258_DESeq2_late_vs_early_results.tsv"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE
)

sig <- subset(res_df, !is.na(padj) & padj < 0.05 & abs(log2FoldChange) >= 1)

write.table(
  sig,
  file=file.path(outdir, "GSE142258_DESeq2_late_vs_early_FDR005_absLFC1.tsv"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE
)

norm_counts <- counts(dds, normalized=TRUE)
norm_counts <- data.frame(gene_id=rownames(norm_counts), norm_counts, check.names=FALSE)

write.table(
  norm_counts,
  file=file.path(outdir, "GSE142258_DESeq2_normalized_counts.tsv"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE
)

rank_df <- res_df[!is.na(res_df$stat), c("gene_id", "stat")]
rank_df <- rank_df[order(rank_df$stat, decreasing=TRUE), ]

write.table(
  rank_df,
  file=file.path(outdir, "GSE142258_DESeq2_late_vs_early_STAT_preranked.rnk"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE,
  col.names=FALSE
)

vsd <- vst(dds, blind=FALSE)
pca <- plotPCA(vsd, intgroup=c("stage_group"), returnData=TRUE)
percentVar <- round(100 * attr(pca, "percentVar"))

write.table(
  pca,
  file=file.path(outdir, "GSE142258_DESeq2_VST_PCA.tsv"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE
)

png(file.path(outdir, "GSE142258_DESeq2_VST_PCA.png"), width=1600, height=1200, res=200)
print(
  ggplot(pca, aes(PC1, PC2, color=stage_group, label=name)) +
    geom_point(size=3) +
    geom_text(vjust=-0.7, size=3) +
    xlab(paste0("PC1: ", percentVar[1], "% variance")) +
    ylab(paste0("PC2: ", percentVar[2], "% variance")) +
    theme_bw()
)
dev.off()

png(file.path(outdir, "GSE142258_DESeq2_MAplot.png"), width=1600, height=1200, res=200)
plotMA(res, ylim=c(-5,5), main="GSE142258 late vs early")
dev.off()

summary_df <- data.frame(
  metric=c(
    "samples_total",
    "early_n",
    "late_n",
    "genes_after_count_filter",
    "DESeq2_FDR005_absLFC1"
  ),
  value=c(
    nrow(meta),
    sum(meta$stage_group == "early"),
    sum(meta$stage_group == "late"),
    nrow(dds),
    nrow(sig)
  )
)

write.table(
  summary_df,
  file=file.path(outdir, "GSE142258_DESeq2_summary.tsv"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE
)
RSCRIPT

micromamba run -n rnaseq Rscript "$DESEQ_DIR/run_DESeq2_GSE142258_stage.R" \
  "$DESEQ_DIR/GSE142258_featureCounts_raw_gene_counts.tsv" \
  "$DESEQ_DIR/GSE142258_DESeq2_metadata.tsv" \
  "$DESEQ_DIR"

echo
echo "=== DONE ==="
date --iso-8601=seconds

echo
echo "=== featureCounts strandedness summary ==="
cat "$DESEQ_DIR/featureCounts_strandedness_summary.tsv"

echo
echo "=== DESeq2 summary ==="
cat "$DESEQ_DIR/GSE142258_DESeq2_summary.tsv"

echo
echo "Output:"
echo "$BASE"
