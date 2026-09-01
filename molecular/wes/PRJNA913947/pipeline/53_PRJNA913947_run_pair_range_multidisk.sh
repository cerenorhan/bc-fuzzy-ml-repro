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

PAIR_START="${PAIR_START:?Set PAIR_START, e.g. 4}"
PAIR_END="${PAIR_END:?Set PAIR_END, e.g. 6}"
QUEUE_NAME="${QUEUE_NAME:-queue}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT output root}"

FASTQ_DIR="${FASTQ_DIR:-data/public/PRJNA913947_ALL}"
PAIR_TSV="${PAIR_TSV:-metadata/public/PRJNA913947/pairing_inference/PRJNA913947_likely_kenya_low_numeric_le46_pairs.tsv}"

ALIGN_THREADS="${ALIGN_THREADS_PRJNA:-8}"
SORT_THREADS="${SORT_THREADS_PRJNA:-4}"
JAVA_MEM="${JAVA_MEM_PRJNA:-24g}"

CHR_JOBS="${CHR_JOBS:-1}"
THREADS_PER_CHR="${THREADS_PER_CHR:-2}"
JAVA_MEM_PER_CHR="${JAVA_MEM_PER_CHR:-10g}"

CENTRAL_ROOT="results/EA_BC_AI_MultiOmics/PRJNA913947_candidate_kenya_wes"

mkdir -p "$WORK_ROOT"/{bam,mutect2,tmp,summary,logs}
mkdir -p "$CENTRAL_ROOT"/{bam,mutect2,summary}
mkdir -p logs/public

LOG="logs/public/PRJNA913947_${QUEUE_NAME}_pairs_${PAIR_START}_${PAIR_END}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== PRJNA913947 multidisk queue started ==="
date
echo "QUEUE_NAME=$QUEUE_NAME"
echo "PAIR_START=$PAIR_START"
echo "PAIR_END=$PAIR_END"
echo "WORK_ROOT=$WORK_ROOT"
echo "FASTQ_DIR=$FASTQ_DIR"
echo "PAIR_TSV=$PAIR_TSV"
echo "ALIGN_THREADS=$ALIGN_THREADS"
echo "SORT_THREADS=$SORT_THREADS"
echo "JAVA_MEM=$JAVA_MEM"
echo "CHR_JOBS=$CHR_JOBS"
echo "THREADS_PER_CHR=$THREADS_PER_CHR"
echo "JAVA_MEM_PER_CHR=$JAVA_MEM_PER_CHR"
echo "LOG=$LOG"
echo

[[ -s "$PAIR_TSV" ]] || { echo "ERROR: missing PAIR_TSV=$PAIR_TSV"; exit 1; }
[[ -d "$FASTQ_DIR" ]] || { echo "ERROR: missing FASTQ_DIR=$FASTQ_DIR"; exit 1; }
[[ -s "$REF_FASTA" ]] || { echo "ERROR: missing REF_FASTA=$REF_FASTA"; exit 1; }
[[ -s "$GNOMAD_AF" ]] || { echo "ERROR: missing GNOMAD_AF=$GNOMAD_AF"; exit 1; }
[[ -s "$SMALL_EXAC_COMMON" ]] || { echo "ERROR: missing SMALL_EXAC_COMMON=$SMALL_EXAC_COMMON"; exit 1; }

QUEUE_TSV="$WORK_ROOT/summary/${QUEUE_NAME}_pairs_${PAIR_START}_${PAIR_END}.tsv"

awk -F'\t' -v OFS='\t' -v start="$PAIR_START" -v end="$PAIR_END" '
NR==1 { print; next }
{
  n=$1
  sub(/^PRJNA_pair_/, "", n)
  if ((n+0) >= start && (n+0) <= end) print
}
' "$PAIR_TSV" > "$QUEUE_TSV"

N_PAIRS=$(( $(wc -l < "$QUEUE_TSV") - 1 ))
echo "Pairs in this queue: $N_PAIRS"
cat "$QUEUE_TSV"
echo

if [[ "$N_PAIRS" -le 0 ]]; then
  echo "ERROR: no pairs selected"
  exit 1
fi

safe_link_dir() {
  local target="$1"
  local link="$2"
  mkdir -p "$target"
  mkdir -p "$(dirname "$link")"

  local target_abs link_abs
  target_abs="$(readlink -m "$target")"
  link_abs="$(readlink -m "$link")"

  if [[ "$target_abs" == "$link_abs" ]]; then
    return 0
  fi

  if [[ -L "$link" ]]; then
    rm -f "$link"
  elif [[ -e "$link" ]]; then
    echo "ERROR: $link exists and is not a symlink. Not overwriting."
    exit 1
  fi

  ln -s "$target_abs" "$link"
}

run_bqsr_sample() {
  local sample="$1"
  local run="$2"
  local r1="$3"
  local r2="$4"
  local outdir="$5"
  local tmp="$6"

  mkdir -p "$outdir" "$tmp"

  local bqsr_bam="$outdir/${sample}.bqsr.bam"
  local bqsr_bai="$outdir/${sample}.bqsr.bam.bai"

  if [[ -s "$bqsr_bam" && -s "$bqsr_bai" ]]; then
    echo "SKIP BQSR $sample: exists"
    return 0
  fi

  [[ -s "$r1" ]] || { echo "ERROR: missing R1=$r1"; exit 1; }
  [[ -s "$r2" ]] || { echo "ERROR: missing R2=$r2"; exit 1; }

  local sorted="$outdir/${sample}.sorted.bam"
  local marked="$outdir/${sample}.marked.bam"
  local metrics="$outdir/${sample}.markdup.metrics.txt"
  local recal="$outdir/${sample}.bqsr.table"
  local rg="@RG\tID:${sample}\tSM:${sample}\tPL:ILLUMINA\tLB:WES\tPU:${run}"

  echo
  echo "=== BWA-MEM2 + sort $sample ==="
  bwa-mem2 mem -t "$ALIGN_THREADS" -R "$rg" "$REF_FASTA" "$r1" "$r2" \
    | samtools sort -@ "$SORT_THREADS" -m 2G -T "$tmp/${sample}.sort" -o "$sorted" -

  samtools index "$sorted"

  echo
  echo "=== MarkDuplicates $sample ==="
  gatk --java-options "-Xmx${JAVA_MEM} -Djava.io.tmpdir=$tmp" MarkDuplicates \
    -I "$sorted" \
    -O "$marked" \
    -M "$metrics" \
    --CREATE_INDEX true \
    --TMP_DIR "$tmp"

  echo
  echo "=== BaseRecalibrator $sample ==="
  gatk --java-options "-Xmx${JAVA_MEM} -Djava.io.tmpdir=$tmp" BaseRecalibrator \
    -R "$REF_FASTA" \
    -I "$marked" \
    --known-sites "$DBSNP" \
    --known-sites "$MILLS" \
    --known-sites "$KNOWN_SNPS" \
    -O "$recal"

  echo
  echo "=== ApplyBQSR $sample ==="
  gatk --java-options "-Xmx${JAVA_MEM} -Djava.io.tmpdir=$tmp" ApplyBQSR \
    -R "$REF_FASTA" \
    -I "$marked" \
    --bqsr-recal-file "$recal" \
    -O "$bqsr_bam"

  samtools index "$bqsr_bam"

  echo
  echo "=== Cleanup alignment intermediates $sample ==="
  rm -f "$sorted" "$sorted.bai" "$marked" "$marked.bai"

  echo "DONE BQSR $sample"
}

run_pair_mutect2() {
  local pair="$1"
  local tumor_bam="$2"
  local normal_bam="$3"
  local outdir="$4"
  local tmp="$5"

  mkdir -p "$outdir/shards" "$tmp"

  local pass_vcf="$outdir/${pair}.PASS.vcf.gz"
  if [[ -s "$pass_vcf" && -s "${pass_vcf}.tbi" ]]; then
    echo "SKIP Mutect2 $pair: PASS exists"
    return 0
  fi

  local contigs=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)

  run_shard() {
    local chr="$1"
    local shard="$outdir/shards/${pair}.${chr}.unfiltered.vcf.gz"
    local f1r2="$outdir/shards/${pair}.${chr}.f1r2.tar.gz"

    if [[ -s "$shard" && -s "${shard}.tbi" && -s "${shard}.stats" && -s "$f1r2" ]]; then
      echo "SKIP shard $pair $chr: exists"
      return 0
    fi

    echo "=== Mutect2 shard $pair $chr started ==="
    gatk --java-options "-Xmx${JAVA_MEM_PER_CHR} -Djava.io.tmpdir=$tmp" Mutect2 \
      -R "$REF_FASTA" \
      -I "$tumor_bam" -tumor "${pair}_T" \
      -I "$normal_bam" -normal "${pair}_N" \
      --germline-resource "$GNOMAD_AF" \
      -L "$chr" \
      --native-pair-hmm-threads "$THREADS_PER_CHR" \
      -O "$shard" \
      --f1r2-tar-gz "$f1r2"
    echo "=== Mutect2 shard $pair $chr done ==="
  }

  echo
  echo "=== Mutect2 scatter $pair ==="
  for chr in "${contigs[@]}"
  do
    run_shard "$chr" &
    while [[ "$(jobs -rp | wc -l)" -ge "$CHR_JOBS" ]]
    do
      wait -n
    done
  done
  wait

  echo
  echo "=== Verify shards $pair ==="
  local n_vcf n_tbi n_stats n_f1r2
  n_vcf=$(find "$outdir/shards" -name "${pair}.*.unfiltered.vcf.gz" | wc -l)
  n_tbi=$(find "$outdir/shards" -name "${pair}.*.unfiltered.vcf.gz.tbi" | wc -l)
  n_stats=$(find "$outdir/shards" -name "${pair}.*.unfiltered.vcf.gz.stats" | wc -l)
  n_f1r2=$(find "$outdir/shards" -name "${pair}.*.f1r2.tar.gz" | wc -l)
  echo "VCF=$n_vcf TBI=$n_tbi STATS=$n_stats F1R2=$n_f1r2"
  [[ "$n_vcf" -eq 23 && "$n_tbi" -eq 23 && "$n_stats" -eq 23 && "$n_f1r2" -eq 23 ]] || {
    echo "ERROR: incomplete shards for $pair"
    exit 1
  }

  local unfiltered="$outdir/${pair}.mutect2.unfiltered.vcf.gz"
  local filtered="$outdir/${pair}.mutect2.filtered.vcf.gz"
  local orientation="$outdir/${pair}.read_orientation_model.tar.gz"
  local tumor_pileups="$outdir/${pair}.tumor.pileups.table"
  local normal_pileups="$outdir/${pair}.normal.pileups.table"
  local contam="$outdir/${pair}.contamination.table"
  local segments="$outdir/${pair}.segments.table"

  local merge_args=()
  local stat_args=()
  local f1r2_args=()

  for chr in "${contigs[@]}"
  do
    merge_args+=( -I "$outdir/shards/${pair}.${chr}.unfiltered.vcf.gz" )
    stat_args+=( -stats "$outdir/shards/${pair}.${chr}.unfiltered.vcf.gz.stats" )
    f1r2_args+=( -I "$outdir/shards/${pair}.${chr}.f1r2.tar.gz" )
  done

  echo
  echo "=== Merge VCF shards $pair ==="
  gatk --java-options "-Xmx64g -Djava.io.tmpdir=$tmp" MergeVcfs \
    "${merge_args[@]}" \
    -O "$unfiltered"

  echo
  echo "=== Merge Mutect stats $pair ==="
  gatk --java-options "-Xmx24g -Djava.io.tmpdir=$tmp" MergeMutectStats \
    "${stat_args[@]}" \
    -O "${unfiltered}.stats"

  echo
  echo "=== LearnReadOrientationModel $pair ==="
  gatk --java-options "-Xmx24g -Djava.io.tmpdir=$tmp" LearnReadOrientationModel \
    "${f1r2_args[@]}" \
    -O "$orientation"

  echo
  echo "=== GetPileupSummaries tumor $pair ==="
  gatk --java-options "-Xmx24g -Djava.io.tmpdir=$tmp" GetPileupSummaries \
    -I "$tumor_bam" \
    -V "$SMALL_EXAC_COMMON" \
    -L "$SMALL_EXAC_COMMON" \
    -O "$tumor_pileups"

  echo
  echo "=== GetPileupSummaries normal $pair ==="
  gatk --java-options "-Xmx24g -Djava.io.tmpdir=$tmp" GetPileupSummaries \
    -I "$normal_bam" \
    -V "$SMALL_EXAC_COMMON" \
    -L "$SMALL_EXAC_COMMON" \
    -O "$normal_pileups"

  echo
  echo "=== CalculateContamination $pair ==="
  gatk --java-options "-Xmx24g -Djava.io.tmpdir=$tmp" CalculateContamination \
    -I "$tumor_pileups" \
    -matched "$normal_pileups" \
    -O "$contam" \
    --tumor-segmentation "$segments"

  echo
  echo "=== FilterMutectCalls $pair ==="
  gatk --java-options "-Xmx48g -Djava.io.tmpdir=$tmp" FilterMutectCalls \
    -R "$REF_FASTA" \
    -V "$unfiltered" \
    --stats "${unfiltered}.stats" \
    --contamination-table "$contam" \
    --tumor-segmentation "$segments" \
    --ob-priors "$orientation" \
    -O "$filtered"

  echo
  echo "=== Extract PASS $pair ==="
  bcftools view -f PASS -Oz -o "$pass_vcf" "$filtered"
  bcftools index -f -t "$pass_vcf"

  local total snvs indels mnps others
  total=$(bcftools view -H "$pass_vcf" | wc -l | tr -d ' ')
  snvs=$(bcftools view -v snps -H "$pass_vcf" | wc -l | tr -d ' ')
  indels=$(bcftools view -v indels -H "$pass_vcf" | wc -l | tr -d ' ')
  mnps=$(bcftools view -v mnps -H "$pass_vcf" | wc -l | tr -d ' ')
  others=$(bcftools view -v other -H "$pass_vcf" | wc -l | tr -d ' ')

  {
    echo -e "pair\tpass_variants\tsnvs\tindels\tmnps\tothers\tpass_vcf"
    echo -e "${pair}\t${total}\t${snvs}\t${indels}\t${mnps}\t${others}\t${pass_vcf}"
  } > "$outdir/${pair}.PASS_variant_summary.tsv"

  echo "PASS summary $pair:"
  cat "$outdir/${pair}.PASS_variant_summary.tsv"

  rm -rf "$tmp"

  echo "DONE Mutect2 $pair"
}

while IFS=$'\t' read -r pair_id pair_rule tumor_run normal_run tumor_sample normal_sample tumor_numeric normal_numeric absdiff tumor_R1 tumor_R2 normal_R1 normal_R2 tumor_size_gb normal_size_gb
do
  [[ "$pair_id" == "pair_id" ]] && continue

  idx_raw="${pair_id#PRJNA_pair_}"
  idx_num=$((10#$idx_raw))
  idx3=$(printf "%03d" "$idx_num")
  pair="candidate_kenya_pair_${idx3}"

  echo
  echo "================================================================"
  echo "PAIR $pair from $pair_id"
  echo "Tumor:  $tumor_run $tumor_sample"
  echo "Normal: $normal_run $normal_sample"
  echo "================================================================"

  pair_tmp="$WORK_ROOT/tmp/$pair"
  tumor_out="$WORK_ROOT/bam/${pair}_T"
  normal_out="$WORK_ROOT/bam/${pair}_N"
  mutect_out="$WORK_ROOT/mutect2/$pair"

  mkdir -p "$pair_tmp" "$tumor_out" "$normal_out" "$mutect_out"

  safe_link_dir "$tumor_out" "$CENTRAL_ROOT/bam/${pair}_T"
  safe_link_dir "$normal_out" "$CENTRAL_ROOT/bam/${pair}_N"
  safe_link_dir "$mutect_out" "$CENTRAL_ROOT/mutect2/$pair"

  tumor_r1="$FASTQ_DIR/$tumor_R1"
  tumor_r2="$FASTQ_DIR/$tumor_R2"
  normal_r1="$FASTQ_DIR/$normal_R1"
  normal_r2="$FASTQ_DIR/$normal_R2"

  run_bqsr_sample "${pair}_T" "$tumor_run" "$tumor_r1" "$tumor_r2" "$tumor_out" "$pair_tmp/${pair}_T"
  run_bqsr_sample "${pair}_N" "$normal_run" "$normal_r1" "$normal_r2" "$normal_out" "$pair_tmp/${pair}_N"

  tumor_bam="$tumor_out/${pair}_T.bqsr.bam"
  normal_bam="$normal_out/${pair}_N.bqsr.bam"

  run_pair_mutect2 "$pair" "$tumor_bam" "$normal_bam" "$mutect_out" "$pair_tmp/mutect2"

  echo
  echo "Disk after $pair:"
  df -h "$WORK_ROOT" "$PROJECT_ROOT" || true

done < "$QUEUE_TSV"

echo
echo "=== Queue completed ==="
date
echo "QUEUE_NAME=$QUEUE_NAME"
echo "PAIR_START=$PAIR_START"
echo "PAIR_END=$PAIR_END"
echo "WORK_ROOT=$WORK_ROOT"
echo "LOG=$LOG"
