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

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/local_wes/auto_local_wes_germline_only_no_bed_${RUN_ID}.log"

mkdir -p logs/local_wes metadata/local_wes "$TMP_DIR" "$GERMLINE_DIR" "$QC_DIR"

exec > >(tee -a "$LOG") 2>&1

LOCK="$TMP_DIR/auto_local_wes_germline_only_no_bed.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "ERROR: Another germline-only no-BED WES pipeline is already running."
    exit 1
fi

send_alert() {
    local subject="$1"
    local body_file="$2"

    if [[ "${ENABLE_NOTIFICATIONS:-0}" == "1" && -n "${NOTIFICATION_HELPER:-}" && -x "$NOTIFICATION_HELPER" ]]; then
        bash "$NOTIFICATION_HELPER" "$subject" "$body_file" || true
    fi
}

stage_mail() {
    local stage="$1"
    local body
    body="$(mktemp)"

    {
        echo "Hadza germline-only local WES pipeline stage completed."
        echo
        echo "Stage: $stage"
        echo "Time: $(date --iso-8601=seconds)"
        echo "Host: $(hostname)"
        echo "Log: $LOG"
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
        echo
        echo "Counts:"
        echo -n "Germline BQSR BAM: "
        find results/germline/bam/bqsr -name "*.bqsr.bam" 2>/dev/null | wc -l || true
        echo -n "gVCF: "
        find results/germline/gvcf -name "*.g.vcf.gz" 2>/dev/null | wc -l || true
    } > "$body"

    send_alert "STAGE DONE: $stage" "$body"
    rm -f "$body"
}

fail_alert() {
    local exit_code="$?"
    local body
    body="$(mktemp)"

    {
        echo "Hadza germline-only local WES pipeline FAILED."
        echo
        echo "Time: $(date --iso-8601=seconds)"
        echo "Host: $(hostname)"
        echo "Exit code: $exit_code"
        echo "Log: $LOG"
        echo
        echo "Disk:"
        df -h "$PROJECT_ROOT" || true
        echo
        echo "Active processes:"
        pgrep -af "gatk|samtools|bcftools|plink2|aria2c|PRJNA|GSE142" || true
        echo
        echo "Output counts:"
        echo -n "Germline BQSR BAM: "
        find results/germline/bam/bqsr -name "*.bqsr.bam" 2>/dev/null | wc -l || true
        echo -n "gVCF: "
        find results/germline/gvcf -name "*.g.vcf.gz" 2>/dev/null | wc -l || true
        echo
        echo "Last 220 log lines:"
        tail -n 220 "$LOG" || true
    } > "$body"

    send_alert "FAILED germline-only local WES pipeline" "$body"
    rm -f "$body"
}

trap fail_alert ERR

echo "Hadza germline-only local WES no-BED pipeline started"
date --iso-8601=seconds
echo "Log: $LOG"

GATK_XMX="${GATK_XMX:-32g}"
HC_THREADS="${HC_THREADS:-4}"

MIN_DEPTH="${MIN_DEPTH:-10}"
MIN_MAPQ="${MIN_MAPQ:-20}"
MIN_BASEQ="${MIN_BASEQ:-20}"
MIN_SAMPLE_FRACTION="${MIN_SAMPLE_FRACTION:-0.80}"
MIN_INTERVAL_LEN="${MIN_INTERVAL_LEN:-50}"
MERGE_GAP="${MERGE_GAP:-20}"

DELETE_GERMLINE_MARKDUP_AFTER_BQSR="${DELETE_GERMLINE_MARKDUP_AFTER_BQSR:-1}"

MARKDUP_DIR="$BAM_DIR/markdup"
BQSR_BAM_DIR="$BAM_DIR/bqsr"
BQSR_TABLE_DIR="$GERMLINE_DIR/bqsr/recal_tables"
CALLABLE_DIR="$GERMLINE_DIR/callable"
PER_SAMPLE_BED_DIR="$CALLABLE_DIR/per_sample_depth${MIN_DEPTH}_mq${MIN_MAPQ}_bq${MIN_BASEQ}"
GVCF_DIR="$GERMLINE_DIR/gvcf"
JOINT_DIR="$GERMLINE_DIR/joint"
FILTER_DIR="$GERMLINE_DIR/filtered"
PLINK_DIR="$GERMLINE_DIR/plink"

mkdir -p \
    "$BQSR_BAM_DIR" \
    "$BQSR_TABLE_DIR" \
    "$CALLABLE_DIR" \
    "$PER_SAMPLE_BED_DIR" \
    "$GVCF_DIR" \
    "$JOINT_DIR" \
    "$FILTER_DIR" \
    "$PLINK_DIR" \
    "$QC_DIR/bcftools" \
    "$QC_DIR/plink" \
    "$QC_DIR/multiqc/local_wes_variants" \
    "$TMP_DIR/gatk"

echo
echo "=== TOOL CHECK ==="
for tool in gatk samtools bcftools python plink2 multiqc
do
    echo -n "$tool: "
    command -v "$tool"
done

echo
echo "=== INPUT CHECK ==="
[[ -s "$REF_FASTA" ]] || { echo "ERROR missing REF_FASTA"; exit 1; }
[[ -s "${REF_FASTA}.fai" ]] || { echo "ERROR missing REF_FASTA.fai"; exit 1; }
[[ -s "$DBSNP" ]] || { echo "ERROR missing DBSNP"; exit 1; }
[[ -s "$MILLS" ]] || { echo "ERROR missing MILLS"; exit 1; }
[[ -s "$KNOWN_SNPS" ]] || { echo "ERROR missing KNOWN_SNPS"; exit 1; }

find "$MARKDUP_DIR" -name "*.marked.bam" | sort \
    > metadata/local_wes/local_wes.all_markdup_bams.list

grep -v '/BC[^/]*\.marked\.bam$' metadata/local_wes/local_wes.all_markdup_bams.list \
    > metadata/local_wes/local_wes.germline_markdup_bams.list

grep '/BC[^/]*\.marked\.bam$' metadata/local_wes/local_wes.all_markdup_bams.list \
    > metadata/local_wes/local_wes.tumor_markdup_bams.list || true

ALL_COUNT=$(wc -l < metadata/local_wes/local_wes.all_markdup_bams.list)
GERMLINE_COUNT=$(wc -l < metadata/local_wes/local_wes.germline_markdup_bams.list)
TUMOR_COUNT=$(wc -l < metadata/local_wes/local_wes.tumor_markdup_bams.list)

echo "All marked BAM count: $ALL_COUNT"
echo "Germline marked BAM count: $GERMLINE_COUNT"
echo "Tumor marked BAM count: $TUMOR_COUNT"

if [[ "$ALL_COUNT" -ne 37 ]]; then
    echo "ERROR: Expected 37 total marked BAMs, found $ALL_COUNT"
    exit 1
fi

if [[ "$GERMLINE_COUNT" -ne 31 ]]; then
    echo "ERROR: Expected 31 germline marked BAMs, found $GERMLINE_COUNT"
    exit 1
fi

if [[ "$TUMOR_COUNT" -ne 6 ]]; then
    echo "ERROR: Expected 6 BC tumor marked BAMs, found $TUMOR_COUNT"
    exit 1
fi

echo
echo "Tumor BAMs will be preserved and excluded from germline analysis:"
cat metadata/local_wes/local_wes.tumor_markdup_bams.list

echo
echo "Running quickcheck on germline marked BAMs..."
samtools quickcheck -v $(cat metadata/local_wes/local_wes.germline_markdup_bams.list)
echo "Germline marked BAM quickcheck OK"

stage_mail "01 germline marked BAM validation"

echo
echo "=== STAGE 02: empirical callable BED from 31 germline BAMs ==="

EXPECTED_SAMPLES="$GERMLINE_COUNT"

MIN_SAMPLES=$(python - <<PY
import math
n = int("$EXPECTED_SAMPLES")
f = float("$MIN_SAMPLE_FRACTION")
print(math.ceil(n * f))
PY
)

COMMON_BED="$CALLABLE_DIR/local_wes.germline_only.empirical_callable.depth${MIN_DEPTH}.mq${MIN_MAPQ}.bq${MIN_BASEQ}.min${MIN_SAMPLES}of${EXPECTED_SAMPLES}.bed"
COMMON_SUMMARY="$CALLABLE_DIR/local_wes.germline_only.empirical_callable.summary.tsv"

echo "MIN_DEPTH=$MIN_DEPTH"
echo "MIN_MAPQ=$MIN_MAPQ"
echo "MIN_BASEQ=$MIN_BASEQ"
echo "MIN_SAMPLE_FRACTION=$MIN_SAMPLE_FRACTION"
echo "MIN_SAMPLES=$MIN_SAMPLES / $EXPECTED_SAMPLES"
echo "COMMON_BED=$COMMON_BED"

while read -r bam
do
    sample_id="$(basename "$bam" .marked.bam)"
    bed="$PER_SAMPLE_BED_DIR/${sample_id}.depth${MIN_DEPTH}.bed"

    if [[ -s "$bed" ]]; then
        echo "SKIP existing per-sample callable BED: $sample_id"
        continue
    fi

    echo "Creating per-sample callable BED: $sample_id"

    nice -n 10 ionice -c2 -n7 \
    samtools depth \
        -q "$MIN_MAPQ" \
        -Q "$MIN_BASEQ" \
        -G 3844 \
        "$bam" \
    | awk -v OFS='\t' -v d="$MIN_DEPTH" '
        function is_primary_chr(c) {
            return (c ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y|M)$/)
        }
        $3 >= d && is_primary_chr($1) {
            s = $2 - 1
            e = $2
            if (chr == $1 && s == prev_end) {
                prev_end = e
            } else {
                if (chr != "") {
                    print chr, run_start, prev_end
                }
                chr = $1
                run_start = s
                prev_end = e
            }
        }
        END {
            if (chr != "") {
                print chr, run_start, prev_end
            }
        }
    ' > "$bed"

    if [[ ! -s "$bed" ]]; then
        echo "ERROR: Empty callable BED for $sample_id"
        exit 1
    fi

done < metadata/local_wes/local_wes.germline_markdup_bams.list

echo
echo "Merging per-sample callable BEDs..."

python - "$PER_SAMPLE_BED_DIR" "$COMMON_BED" "$COMMON_SUMMARY" "$MIN_SAMPLES" "$MERGE_GAP" "$MIN_INTERVAL_LEN" "$REF_FASTA" <<'PY'
import sys
from pathlib import Path
from collections import defaultdict
from itertools import groupby

bed_dir = Path(sys.argv[1])
out_bed = Path(sys.argv[2])
summary = Path(sys.argv[3])
min_samples = int(sys.argv[4])
merge_gap = int(sys.argv[5])
min_len = int(sys.argv[6])
ref = sys.argv[7]

fai = Path(ref + ".fai")

primary = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
chrom_order = []

with fai.open() as f:
    for line in f:
        chrom = line.split("\t")[0]
        if chrom in primary:
            chrom_order.append(chrom)

events = defaultdict(list)
sample_files = sorted(bed_dir.glob("*.bed"))

if len(sample_files) < min_samples:
    raise SystemExit(f"Too few per-sample BED files: {len(sample_files)}")

for bed in sample_files:
    with bed.open() as f:
        for line in f:
            if not line.strip():
                continue
            chrom, s, e = line.rstrip("\n").split("\t")[:3]
            s = int(s)
            e = int(e)
            if e <= s:
                continue
            events[chrom].append((s, 1))
            events[chrom].append((e, -1))

raw = []

for chrom in chrom_order:
    ev = events.get(chrom, [])
    if not ev:
        continue

    ev.sort()
    cur = 0
    last = None

    for pos, group in groupby(ev, key=lambda x: x[0]):
        delta = sum(x[1] for x in group)

        if last is not None and pos > last and cur >= min_samples:
            raw.append((chrom, last, pos))

        cur += delta
        last = pos

merged = []

for chrom, group in groupby(raw, key=lambda x: x[0]):
    group = list(group)
    if not group:
        continue

    cs, ce = group[0][1], group[0][2]

    for _, s, e in group[1:]:
        if s - ce <= merge_gap:
            ce = max(ce, e)
        else:
            if ce - cs >= min_len:
                merged.append((chrom, cs, ce))
            cs, ce = s, e

    if ce - cs >= min_len:
        merged.append((chrom, cs, ce))

out_bed.parent.mkdir(parents=True, exist_ok=True)

with out_bed.open("w") as out:
    for chrom, s, e in merged:
        out.write(f"{chrom}\t{s}\t{e}\n")

total_bp = sum(e - s for _, s, e in merged)

with summary.open("w") as out:
    out.write("metric\tvalue\n")
    out.write(f"sample_beds\t{len(sample_files)}\n")
    out.write(f"min_samples\t{min_samples}\n")
    out.write(f"merge_gap_bp\t{merge_gap}\n")
    out.write(f"min_interval_len_bp\t{min_len}\n")
    out.write(f"intervals\t{len(merged)}\n")
    out.write(f"total_bp\t{total_bp}\n")
    out.write(f"total_Mb\t{total_bp/1_000_000:.3f}\n")

print(f"sample_beds={len(sample_files)}")
print(f"intervals={len(merged)}")
print(f"total_bp={total_bp}")
print(f"total_Mb={total_bp/1_000_000:.3f}")

if not merged or total_bp < 1_000_000:
    raise SystemExit("Common callable BED looks too small; stopping")
PY

echo
echo "Callable BED summary:"
cat "$COMMON_SUMMARY"

stage_mail "02 empirical callable BED"

echo
echo "=== STAGE 03: BQSR on germline-only BAMs ==="

while read -r bam
do
    sample_id="$(basename "$bam" .marked.bam)"
    recal="$BQSR_TABLE_DIR/${sample_id}.recal.table"
    outbam="$BQSR_BAM_DIR/${sample_id}.bqsr.bam"

    echo
    echo "Sample: $sample_id"

    if [[ -s "$outbam" && -s "${outbam}.bai" && -s "$recal" ]]; then
        echo "SKIP existing BQSR BAM: $sample_id"
        continue
    fi

    echo "Running BaseRecalibrator: $sample_id"

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        BaseRecalibrator \
        -R "$REF_FASTA" \
        -I "$bam" \
        --known-sites "$DBSNP" \
        --known-sites "$MILLS" \
        --known-sites "$KNOWN_SNPS" \
        -L "$COMMON_BED" \
        -O "$recal"

    echo "Running ApplyBQSR: $sample_id"

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        ApplyBQSR \
        -R "$REF_FASTA" \
        -I "$bam" \
        --bqsr-recal-file "$recal" \
        -L "$COMMON_BED" \
        -O "$outbam"

    samtools index -@ 8 "$outbam"
    samtools quickcheck -v "$outbam"

    echo "DONE BQSR: $sample_id"

done < metadata/local_wes/local_wes.germline_markdup_bams.list

find "$BQSR_BAM_DIR" -name "*.bqsr.bam" | sort > metadata/local_wes/local_wes.germline_bqsr_bams.list
BQSR_COUNT=$(wc -l < metadata/local_wes/local_wes.germline_bqsr_bams.list)

echo "BQSR BAM count: $BQSR_COUNT"

if [[ "$BQSR_COUNT" -ne "$EXPECTED_SAMPLES" ]]; then
    echo "ERROR: Expected $EXPECTED_SAMPLES BQSR BAMs, found $BQSR_COUNT"
    exit 1
fi

samtools quickcheck -v $(cat metadata/local_wes/local_wes.germline_bqsr_bams.list)
echo "All germline BQSR BAM quickcheck OK"

stage_mail "03 BQSR BAMs"

echo
echo "=== STAGE 04: delete only germline MarkDuplicates BAMs after BQSR ==="

if [[ "$DELETE_GERMLINE_MARKDUP_AFTER_BQSR" -eq 1 ]]; then
    {
        while read -r bam
        do
            [[ -z "$bam" ]] && continue
            echo "$bam"
            [[ -e "${bam}.bai" ]] && echo "${bam}.bai"
            [[ -e "${bam%.bam}.bai" ]] && echo "${bam%.bam}.bai"
        done < metadata/local_wes/local_wes.germline_markdup_bams.list
    } | sort -u > "metadata/local_wes/germline_markdup_files_deleted_after_bqsr_${RUN_ID}.txt"

    echo "Files to delete:"
    wc -l "metadata/local_wes/germline_markdup_files_deleted_after_bqsr_${RUN_ID}.txt"

    echo "Tumor BC marked BAMs preserved:"
    cat metadata/local_wes/local_wes.tumor_markdup_bams.list

    while read -r f
    do
        [[ -z "$f" ]] && continue
        rm -f "$f"
    done < "metadata/local_wes/germline_markdup_files_deleted_after_bqsr_${RUN_ID}.txt"

    echo "Disk after deleting germline markdup BAMs:"
    df -h "$PROJECT_ROOT"
else
    echo "Skipping germline markdup deletion."
fi

stage_mail "04 germline markdup cleanup"

echo
echo "=== STAGE 05: HaplotypeCaller gVCF ==="

while read -r bam
do
    sample_id="$(basename "$bam" .bqsr.bam)"
    gvcf="$GVCF_DIR/${sample_id}.g.vcf.gz"

    echo
    echo "Sample: $sample_id"

    if [[ -s "$gvcf" && ( -s "${gvcf}.tbi" || -s "${gvcf}.idx" ) ]]; then
        echo "SKIP existing gVCF: $sample_id"
        continue
    fi

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        HaplotypeCaller \
        -R "$REF_FASTA" \
        -I "$bam" \
        -O "$gvcf" \
        -ERC GVCF \
        -L "$COMMON_BED" \
        --native-pair-hmm-threads "$HC_THREADS"

    if [[ ! -s "${gvcf}.tbi" && ! -s "${gvcf}.idx" ]]; then
        gatk IndexFeatureFile -I "$gvcf"
    fi

    bcftools view -h "$gvcf" >/dev/null
    echo "DONE gVCF: $sample_id"

done < metadata/local_wes/local_wes.germline_bqsr_bams.list

find "$GVCF_DIR" -name "*.g.vcf.gz" | sort > metadata/local_wes/local_wes.germline_gvcfs.list
GVCF_COUNT=$(wc -l < metadata/local_wes/local_wes.germline_gvcfs.list)

echo "gVCF count: $GVCF_COUNT"

if [[ "$GVCF_COUNT" -ne "$EXPECTED_SAMPLES" ]]; then
    echo "ERROR: Expected $EXPECTED_SAMPLES gVCFs, found $GVCF_COUNT"
    exit 1
fi

stage_mail "05 HaplotypeCaller gVCFs"

echo
echo "=== STAGE 06: GenomicsDBImport ==="

SAMPLE_MAP="metadata/local_wes/local_wes.germline_gvcf.sample_map.tsv"
GDB="$JOINT_DIR/local_wes_germline_only_genomicsdb"
GDB_DONE="$JOINT_DIR/local_wes_germline_only_genomicsdb.done"

awk -F'/' '{
    file=$NF
    sample=file
    sub(/\.g\.vcf\.gz$/, "", sample)
    print sample "\t" $0
}' metadata/local_wes/local_wes.germline_gvcfs.list > "$SAMPLE_MAP"

if [[ -s "$GDB_DONE" && -d "$GDB" ]]; then
    echo "SKIP existing GenomicsDB workspace"
else
    rm -rf "$GDB"

    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        GenomicsDBImport \
        --genomicsdb-workspace-path "$GDB" \
        --sample-name-map "$SAMPLE_MAP" \
        -L "$COMMON_BED" \
        --reader-threads 4 \
        --batch-size 50

    date --iso-8601=seconds > "$GDB_DONE"
fi

stage_mail "06 GenomicsDBImport"

echo
echo "=== STAGE 07: GenotypeGVCFs ==="

RAW_VCF="$JOINT_DIR/local_wes_germline_only.raw.vcf.gz"

if [[ -s "$RAW_VCF" && -s "${RAW_VCF}.tbi" ]]; then
    echo "SKIP existing raw cohort VCF"
else
    nice -n 10 ionice -c2 -n7 \
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        GenotypeGVCFs \
        -R "$REF_FASTA" \
        -V "gendb://$GDB" \
        -O "$RAW_VCF"

    if [[ ! -s "${RAW_VCF}.tbi" ]]; then
        gatk IndexFeatureFile -I "$RAW_VCF"
    fi
fi

bcftools view -h "$RAW_VCF" >/dev/null
bcftools stats "$RAW_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.raw.bcftools.stats.txt"

stage_mail "07 GenotypeGVCFs"

echo
echo "=== STAGE 08: hard filtering and PASS VCF ==="

FILTERED_VCF="$FILTER_DIR/local_wes_germline_only.hard_filtered.vcf.gz"
PASS_VCF="$FILTER_DIR/local_wes_germline_only.pass.vcf.gz"
PASS_TAGGED_VCF="$FILTER_DIR/local_wes_germline_only.pass.filltags.vcf.gz"
BIALLELIC_SNPS_VCF="$FILTER_DIR/local_wes_germline_only.pass.biallelic_snps.vcf.gz"

if [[ -s "$FILTERED_VCF" && -s "${FILTERED_VCF}.tbi" ]]; then
    echo "SKIP existing hard-filtered VCF"
else
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        VariantFiltration \
        -R "$REF_FASTA" \
        -V "$RAW_VCF" \
        -O "$FILTERED_VCF" \
        --filter-name "SNP_HARD_FILTER" \
        --filter-expression "vc.isSNP() && (QD < 2.0 || FS > 60.0 || MQ < 40.0 || SOR > 3.0 || MQRankSum < -12.5 || ReadPosRankSum < -8.0)" \
        --filter-name "INDEL_HARD_FILTER" \
        --filter-expression "vc.isIndel() && (QD < 2.0 || FS > 200.0 || SOR > 10.0 || ReadPosRankSum < -20.0)"

    if [[ ! -s "${FILTERED_VCF}.tbi" ]]; then
        gatk IndexFeatureFile -I "$FILTERED_VCF"
    fi
fi

if [[ -s "$PASS_VCF" && -s "${PASS_VCF}.tbi" ]]; then
    echo "SKIP existing PASS VCF"
else
    gatk --java-options "-Xmx${GATK_XMX} -Djava.io.tmpdir=$TMP_DIR/gatk" \
        SelectVariants \
        -R "$REF_FASTA" \
        -V "$FILTERED_VCF" \
        --exclude-filtered \
        -O "$PASS_VCF"

    if [[ ! -s "${PASS_VCF}.tbi" ]]; then
        gatk IndexFeatureFile -I "$PASS_VCF"
    fi
fi

bcftools stats "$FILTERED_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.hard_filtered.bcftools.stats.txt"
bcftools stats "$PASS_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.pass.bcftools.stats.txt"

if bcftools plugin -l | grep -qx "fill-tags"; then
    bcftools +fill-tags "$PASS_VCF" -Oz -o "$PASS_TAGGED_VCF" -- -t AC,AN,AF,NS
    bcftools index -t "$PASS_TAGGED_VCF"
else
    echo "WARNING: bcftools fill-tags plugin not found; using PASS VCF without fill-tags."
    PASS_TAGGED_VCF="$PASS_VCF"
fi

bcftools view \
    -m2 -M2 \
    -v snps \
    -f PASS \
    "$PASS_TAGGED_VCF" \
    -Oz \
    -o "$BIALLELIC_SNPS_VCF"

bcftools index -t "$BIALLELIC_SNPS_VCF"
bcftools stats "$BIALLELIC_SNPS_VCF" > "$QC_DIR/bcftools/local_wes_germline_only.pass.biallelic_snps.bcftools.stats.txt"

stage_mail "08 hard filtering"

echo
echo "=== STAGE 09: PLINK2 QC and PCA ==="

PLINK_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps"
PLINK_QC_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.qc"
PLINK_FILTERED_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered"
PLINK_PRUNE_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered.prune"
PLINK_PCA_PREFIX="$PLINK_DIR/local_wes_germline_only.pass.biallelic_snps.filtered.pca"

if [[ -s "${PLINK_PREFIX}.pgen" ]]; then
    echo "SKIP existing PLINK pgen"
else
    plink2 \
        --vcf "$BIALLELIC_SNPS_VCF" \
        --double-id \
        --allow-extra-chr \
        --set-missing-var-ids @:#:\$r:\$a \
        --make-pgen \
        --out "$PLINK_PREFIX"
fi

plink2 \
    --pfile "$PLINK_PREFIX" \
    --allow-extra-chr \
    --freq \
    --missing \
    --hardy \
    --out "$PLINK_QC_PREFIX"

plink2 \
    --pfile "$PLINK_PREFIX" \
    --allow-extra-chr \
    --maf 0.01 \
    --geno 0.10 \
    --mind 0.10 \
    --hwe 1e-6 midp \
    --make-pgen \
    --out "$PLINK_FILTERED_PREFIX"

plink2 \
    --pfile "$PLINK_FILTERED_PREFIX" \
    --allow-extra-chr \
    --indep-pairwise 200 50 0.2 \
    --out "$PLINK_PRUNE_PREFIX"

plink2 \
    --pfile "$PLINK_FILTERED_PREFIX" \
    --allow-extra-chr \
    --extract "${PLINK_PRUNE_PREFIX}.prune.in" \
    --pca 20 \
    --out "$PLINK_PCA_PREFIX"

stage_mail "09 PLINK2 QC/PCA"

echo
echo "=== STAGE 10: final summary ==="

multiqc \
    "$QC_DIR/bcftools" \
    "$QC_DIR/plink" \
    "$GERMLINE_DIR/bqsr" \
    "$GERMLINE_DIR/filtered" \
    "$PLINK_DIR" \
    -o "$QC_DIR/multiqc/local_wes_variants" \
    -n local_wes_germline_only_variants_multiqc.html \
    || true

rm -rf "$TMP_DIR/gatk"/* || true

FINAL_BODY="$(mktemp)"

{
    echo "Hadza germline-only local WES no-BED pipeline completed successfully."
    echo
    echo "Time: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Log: $LOG"
    echo
    echo "Callable BED:"
    echo "$COMMON_BED"
    cat "$COMMON_SUMMARY"
    echo
    echo "Main outputs:"
    echo "BQSR BAM dir: $BQSR_BAM_DIR"
    echo "gVCF dir: $GVCF_DIR"
    echo "Raw cohort VCF: $RAW_VCF"
    echo "Filtered VCF: $FILTERED_VCF"
    echo "PASS VCF: $PASS_VCF"
    echo "Biallelic SNP VCF: $BIALLELIC_SNPS_VCF"
    echo "PLINK prefix: $PLINK_PREFIX"
    echo "PCA output: ${PLINK_PCA_PREFIX}.eigenvec"
    echo
    echo "Tumor BC BAMs preserved:"
    cat metadata/local_wes/local_wes.tumor_markdup_bams.list
    echo
    echo "Counts:"
    echo -n "BQSR BAM: "
    find "$BQSR_BAM_DIR" -name "*.bqsr.bam" | wc -l
    echo -n "gVCF: "
    find "$GVCF_DIR" -name "*.g.vcf.gz" | wc -l
    echo
    echo "Disk:"
    df -h "$PROJECT_ROOT"
    echo
    echo "Downloads:"
    pgrep -af "aria2c|41_download_PRJNA|43_retry_rnaseq|GSE142" || true
} > "$FINAL_BODY"

send_alert "SUCCESS germline-only local WES no-BED pipeline" "$FINAL_BODY"
rm -f "$FINAL_BODY"

echo
echo "Hadza germline-only local WES no-BED pipeline completed successfully."
date --iso-8601=seconds
