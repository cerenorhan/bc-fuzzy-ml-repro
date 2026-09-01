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
TRIM_DIR="$BASE/trimmomatic"
TOPHAT_DIR="$BASE/tophat"
BAM_DIR="$BASE/bam"
FC_DIR="$BASE/featureCounts"
DE_DIR="$BASE/DESeq2"
QC_DIR="$BASE/multiqc"
META="$BASE/metadata/GSE142258_DESeq2_metadata.tsv"

GTF="reference/rnaseq/gencode.v44.annotation.gtf"
BOWTIE_DIR="reference/rnaseq/bowtie2_GRCh38"
BOWTIE_INDEX="$BOWTIE_DIR/GRCh38"

LOG="logs/public/GSE142258_resume_from_trimmed_TopHat_featureCounts_DESeq2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/public "$TOPHAT_DIR" "$BAM_DIR" "$FC_DIR" "$DE_DIR" "$QC_DIR"

exec > >(tee -a "$LOG") 2>&1

echo "GSE142258 resume from trimmed FASTQ started"
date --iso-8601=seconds
echo "THREADS=$THREADS"

echo
echo "=== Input checks ==="

[[ -s "$META" ]] || { echo "ERROR: missing metadata: $META"; exit 1; }
[[ -s "$GTF" ]] || { echo "ERROR: missing GTF: $GTF"; exit 1; }
[[ -s "${BOWTIE_INDEX}.1.bt2" ]] || { echo "ERROR: missing Bowtie2 index: ${BOWTIE_INDEX}.1.bt2"; exit 1; }

ln -sf "$REF_FASTA" "$BOWTIE_DIR/GRCh38.fa"

echo "FASTA:"
readlink -f "$BOWTIE_DIR/GRCh38.fa"
ls -lhL "$BOWTIE_DIR/GRCh38.fa"

echo
echo "=== Discover trimmed paired samples ==="
mapfile -t SAMPLES < <(
  find "$TRIM_DIR" -maxdepth 1 -name "SRR*_R1.paired.fastq.gz" \
    -printf "%f\n" \
  | sed 's/_R1\.paired\.fastq\.gz$//' \
  | sort
)

echo "Samples found: ${#SAMPLES[@]}"
printf '%s\n' "${SAMPLES[@]}"

if [[ "${#SAMPLES[@]}" -ne 15 ]]; then
  echo "ERROR: expected 15 samples from trimmed paired FASTQ, found ${#SAMPLES[@]}"
  exit 1
fi

echo
echo "=== TopHat2 alignment from trimmed paired FASTQ ==="

TOPHAT_BIN="$(command -v tophat2 || command -v tophat || true)"
[[ -n "$TOPHAT_BIN" ]] || { echo "ERROR: tophat/tophat2 not found"; exit 1; }

for sample in "${SAMPLES[@]}"
do
  R1="$TRIM_DIR/${sample}_R1.paired.fastq.gz"
  R2="$TRIM_DIR/${sample}_R2.paired.fastq.gz"
  OUT="$TOPHAT_DIR/$sample"
  SORTED="$BAM_DIR/${sample}.sorted.bam"

  [[ -s "$R1" ]] || { echo "ERROR: missing $R1"; exit 1; }
  [[ -s "$R2" ]] || { echo "ERROR: missing $R2"; exit 1; }

  if [[ -s "$SORTED" && -s "${SORTED}.bai" ]]; then
    echo "SKIP $sample: sorted BAM exists"
    continue
  fi

  if [[ -s "$OUT/accepted_hits.bam" ]]; then
    echo "SKIP TopHat $sample: accepted_hits.bam exists"
  else
    echo
    echo "### TopHat2 $sample ###"
    rm -rf "$OUT"
    "$TOPHAT_BIN" \
      -p "$THREADS" \
      -G "$GTF" \
      -o "$OUT" \
      "$BOWTIE_INDEX" \
      "$R1" "$R2"

    [[ -s "$OUT/accepted_hits.bam" ]] || { echo "ERROR: TopHat failed for $sample"; exit 1; }
  fi

  echo "### samtools sort/index $sample ###"
  samtools sort -@ "$THREADS" -o "$SORTED" "$OUT/accepted_hits.bam"
  samtools index -@ "$THREADS" "$SORTED"
done

echo
echo "=== BAM count check ==="
BAM_N=$(find "$BAM_DIR" -maxdepth 1 -name "SRR*.sorted.bam" | wc -l)
echo "Sorted BAM count: $BAM_N"
[[ "$BAM_N" -eq 15 ]] || { echo "ERROR: expected 15 sorted BAMs, found $BAM_N"; exit 1; }

echo
echo "=== featureCounts ==="
FC_OUT="$FC_DIR/GSE142258.featureCounts.txt"

mapfile -t BAMS < <(find "$BAM_DIR" -maxdepth 1 -name "SRR*.sorted.bam" | sort)

featureCounts \
  -T "$THREADS" \
  -p -B -C \
  -s 0 \
  -a "$GTF" \
  -o "$FC_OUT" \
  "${BAMS[@]}"

echo
echo "=== DESeq2 ==="
R_SCRIPT="$DE_DIR/run_GSE142258_DESeq2_late_vs_early.R"

cat > "$R_SCRIPT" <<'RS'
suppressPackageStartupMessages({
  library(DESeq2)
})

base <- "results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2"
fc_file <- file.path(base, "featureCounts", "GSE142258.featureCounts.txt")
meta_file <- file.path(base, "metadata", "GSE142258_DESeq2_metadata.tsv")
outdir <- file.path(base, "DESeq2")
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)

fc <- read.delim(fc_file, comment.char="#", check.names=FALSE)
meta <- read.delim(meta_file, check.names=FALSE)

count_cols <- 7:ncol(fc)
counts <- as.matrix(fc[, count_cols])
rownames(counts) <- fc$Geneid

sample_names <- basename(colnames(counts))
sample_names <- sub("\\.sorted\\.bam$", "", sample_names)
sample_names <- sub("^.*/", "", sample_names)
colnames(counts) <- sample_names

sample_col <- intersect(c("sample_id","sample","run","Run","SRR","accession"), colnames(meta))[1]
if (is.na(sample_col)) {
  sample_col <- colnames(meta)[1]
}

group_col <- intersect(c("stage_group","early_late","stage_binary","group","StageGroup"), colnames(meta))[1]

if (is.na(group_col)) {
  stage_col <- intersect(c("stage","Stage","pathological_stage","PATHOLOGICAL_STAGE"), colnames(meta))[1]
  if (is.na(stage_col)) stop("No group/stage column found in metadata.")
  st <- as.character(meta[[stage_col]])
  meta$stage_group <- ifelse(grepl("III|IV|late", st, ignore.case=TRUE), "late", "early")
  group_col <- "stage_group"
}

meta[[sample_col]] <- as.character(meta[[sample_col]])
rownames(meta) <- meta[[sample_col]]

keep <- intersect(colnames(counts), rownames(meta))
counts <- counts[, keep, drop=FALSE]
meta <- meta[keep, , drop=FALSE]

meta[[group_col]] <- factor(meta[[group_col]])
if ("early" %in% levels(meta[[group_col]])) {
  meta[[group_col]] <- relevel(meta[[group_col]], "early")
}

dds <- DESeqDataSetFromMatrix(
  countData = round(counts),
  colData = meta,
  design = as.formula(paste0("~", group_col))
)

dds <- dds[rowSums(counts(dds)) >= 10, ]
dds <- DESeq(dds)

lev <- levels(colData(dds)[[group_col]])
if (all(c("early","late") %in% lev)) {
  res <- results(dds, contrast=c(group_col, "late", "early"))
  contrast_name <- "late_vs_early"
} else {
  res <- results(dds)
  contrast_name <- paste0(lev[length(lev)], "_vs_", lev[1])
}

res_df <- as.data.frame(res)
res_df$Geneid <- rownames(res_df)
res_df <- res_df[, c("Geneid", setdiff(colnames(res_df), "Geneid"))]
res_df <- res_df[order(res_df$padj, na.last=TRUE), ]

write.table(res_df,
            file=file.path(outdir, paste0("GSE142258_DESeq2_", contrast_name, "_results.tsv")),
            sep="\t", quote=FALSE, row.names=FALSE)

norm_counts <- counts(dds, normalized=TRUE)
write.table(as.data.frame(norm_counts),
            file=file.path(outdir, "GSE142258_DESeq2_normalized_counts.tsv"),
            sep="\t", quote=FALSE, row.names=TRUE)

sink(file.path(outdir, "GSE142258_DESeq2_summary.txt"))
cat("Samples:", ncol(counts), "\n")
cat("Genes after filter:", nrow(dds), "\n")
cat("Group column:", group_col, "\n")
print(table(colData(dds)[[group_col]]))
print(summary(res))
sink()
RS

micromamba run -n rnaseq Rscript "$R_SCRIPT"

echo
echo "=== MultiQC ==="
micromamba run -n rnaseq multiqc "$BASE" -o "$QC_DIR" --force || true

echo
echo "=== Final counts ==="
echo -n "TopHat accepted BAM: "
find "$TOPHAT_DIR" -name "accepted_hits.bam" | wc -l

echo -n "Sorted BAM: "
find "$BAM_DIR" -name "SRR*.sorted.bam" | wc -l

echo -n "featureCounts outputs: "
find "$FC_DIR" -type f | wc -l

echo -n "DESeq2 result files: "
find "$DE_DIR" -name "*DESeq2*results.tsv" | wc -l

echo
echo "DONE"
date --iso-8601=seconds
