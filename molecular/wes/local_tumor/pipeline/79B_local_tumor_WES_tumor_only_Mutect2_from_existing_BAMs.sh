#!/usr/bin/env bash

# Load portable molecular-analysis paths.
_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${_REPO_ROOT}" ]]; then
    echo "ERROR: Run this script from within the bc-fuzzy-ml-repro repository." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${_REPO_ROOT}/molecular/config/load_config.sh"

set -euo pipefail

cd $PROJECT_ROOT


PROJECT="$PROJECT_ROOT"
PAPER="$PAPER_RESULTS_ROOT"

SHEET="$PAPER/05_local_breast_cancer_tumor_WES/local_tumor_wes_sample_sheet.tsv"

OUT="$PAPER/05_local_breast_cancer_tumor_WES/01_tumor_only_mutect2"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"
TMP="$PROJECT/tmp/local_tumor_mutect2"

mkdir -p "$OUT" "$STATUS" "$LOGDIR" "$TMP"

LOG="$LOGDIR/79B_local_tumor_WES_tumor_only_Mutect2_from_existing_BAMs_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 79B: LOCAL TUMOR WES TUMOR-ONLY MUTECT2 FROM EXISTING BAMS ==="
date

if [[ ! -s "$SHEET" ]]; then
  echo "ERROR: sample sheet not found:"
  echo "$SHEET"
  exit 1
fi

for f in "$REF_FASTA" "$GNOMAD_AF" "$SMALL_EXAC_COMMON"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing required resource:"
    echo "$f"
    exit 1
  fi
done

TARGET_BED="results/germline/callable/local_wes.germline_only.empirical_callable.depth10.mq20.bq20.min25of31.bed"

if [[ -s "$TARGET_BED" ]]; then
  TARGET_ARG=(-L "$TARGET_BED")
  echo "Using empirical local WES callable target:"
  echo "$TARGET_BED"
else
  TARGET_ARG=()
  echo "WARNING: empirical callable BED not found; Mutect2 will run without -L target restriction."
fi

echo
echo "Reference:"
echo "$REF_FASTA"
echo
echo "Germline resource:"
echo "$GNOMAD_AF"
echo
echo "Common variants for contamination:"
echo "$SMALL_EXAC_COMMON"

SUMMARY="$OUT/local_tumor_WES_tumor_only_Mutect2_PASS_summary.tsv"
MANIFEST="$OUT/local_tumor_WES_tumor_only_Mutect2_manifest.tsv"

echo -e "sample_id\tbam\tunfiltered_vcf\tfiltered_vcf\tpass_vcf\tpass_snpeff_vcf\tstatus" > "$MANIFEST"

tail -n +2 "$SHEET" | while IFS=$'\t' read -r sample r1 r2 n_r1 n_r2 ready stage subtype er pr her2; do
  echo
  echo "============================================================"
  echo "Sample: $sample"
  echo "Stage: $stage | Subtype: $subtype"
  echo "============================================================"

  SAMPLE_OUT="$OUT/$sample"
  mkdir -p "$SAMPLE_OUT"

  BAM="results/germline/bam/markdup/${sample}.marked.bam"
  BAI="${BAM}.bai"

  if [[ ! -s "$BAM" ]]; then
    echo "ERROR: BAM missing for $sample:"
    echo "$BAM"
    echo -e "$sample\t$BAM\t.\t.\t.\t.\tBAM_MISSING" >> "$MANIFEST"
    continue
  fi

  if [[ ! -s "$BAI" ]]; then
    echo "BAM index missing; indexing:"
    micromamba run -n hadza-wes samtools index "$BAM"
  fi

  # Read sample name from BAM header. Fall back to sample_id.
  BAM_SM=$(
    micromamba run -n hadza-wes samtools view -H "$BAM" \
      | awk -F'\t' '/^@RG/ {for(i=1;i<=NF;i++){if($i ~ /^SM:/){sub(/^SM:/,"",$i); print $i; exit}}}' \
      || true
  )

  if [[ -z "$BAM_SM" ]]; then
    BAM_SM="$sample"
  fi

  echo "BAM: $BAM"
  echo "BAM SM: $BAM_SM"

  PILEUPS="$SAMPLE_OUT/${sample}.pileups.table"
  CONTAM="$SAMPLE_OUT/${sample}.contamination.table"
  SEGMENTS="$SAMPLE_OUT/${sample}.segments.table"

  UNFILTERED="$SAMPLE_OUT/${sample}.tumor_only.unfiltered.vcf.gz"
  F1R2="$SAMPLE_OUT/${sample}.f1r2.tar.gz"
  ORIENTATION="$SAMPLE_OUT/${sample}.read_orientation_model.tar.gz"
  FILTERED="$SAMPLE_OUT/${sample}.tumor_only.filtered.vcf.gz"
  PASS="$SAMPLE_OUT/${sample}.tumor_only.PASS.vcf.gz"
  PASS_FILLED="$SAMPLE_OUT/${sample}.tumor_only.PASS.filltags.vcf.gz"
  PASS_SNPEFF="$SAMPLE_OUT/${sample}.tumor_only.PASS.snpeff.vcf.gz"

  echo
  echo "--- GetPileupSummaries / CalculateContamination"
  if [[ ! -s "$CONTAM" ]]; then
    micromamba run -n hadza-wes gatk --java-options "-Xmx16g -Djava.io.tmpdir=$TMP" GetPileupSummaries \
      -R "$REF_FASTA" \
      -I "$BAM" \
      -V "$SMALL_EXAC_COMMON" \
      -L "$SMALL_EXAC_COMMON" \
      -O "$PILEUPS"

    micromamba run -n hadza-wes gatk --java-options "-Xmx8g -Djava.io.tmpdir=$TMP" CalculateContamination \
      -I "$PILEUPS" \
      -O "$CONTAM" \
      --tumor-segmentation "$SEGMENTS"
  else
    echo "Existing contamination table found; skipping."
  fi

  echo
  echo "--- Mutect2 tumor-only"
  if [[ ! -s "$UNFILTERED" ]]; then
    micromamba run -n hadza-wes gatk --java-options "-Xmx32g -Djava.io.tmpdir=$TMP" Mutect2 \
      -R "$REF_FASTA" \
      -I "$BAM" \
      -tumor "$BAM_SM" \
      --germline-resource "$GNOMAD_AF" \
      --f1r2-tar-gz "$F1R2" \
      --native-pair-hmm-threads 8 \
      "${TARGET_ARG[@]}" \
      -O "$UNFILTERED"
  else
    echo "Existing unfiltered Mutect2 VCF found; skipping."
  fi

  echo
  echo "--- LearnReadOrientationModel"
  if [[ ! -s "$ORIENTATION" ]]; then
    micromamba run -n hadza-wes gatk --java-options "-Xmx8g -Djava.io.tmpdir=$TMP" LearnReadOrientationModel \
      -I "$F1R2" \
      -O "$ORIENTATION"
  else
    echo "Existing orientation model found; skipping."
  fi

  echo
  echo "--- FilterMutectCalls"
  if [[ ! -s "$FILTERED" ]]; then
    micromamba run -n hadza-wes gatk --java-options "-Xmx16g -Djava.io.tmpdir=$TMP" FilterMutectCalls \
      -R "$REF_FASTA" \
      -V "$UNFILTERED" \
      --contamination-table "$CONTAM" \
      --ob-priors "$ORIENTATION" \
      -O "$FILTERED"
  else
    echo "Existing filtered VCF found; skipping."
  fi

  echo
  echo "--- Extract PASS"
  if [[ ! -s "$PASS" ]]; then
    micromamba run -n hadza-wes bcftools view -f PASS "$FILTERED" -Oz -o "$PASS"
    micromamba run -n hadza-wes bcftools index -t -f "$PASS"
  else
    echo "Existing PASS VCF found; skipping."
  fi

  echo
  echo "--- Fill tags"
  if [[ ! -s "$PASS_FILLED" ]]; then
    micromamba run -n hadza-wes bcftools +fill-tags "$PASS" -Oz -o "$PASS_FILLED" -- -t AC,AN,AF,NS
    micromamba run -n hadza-wes bcftools index -t -f "$PASS_FILLED"
  else
    echo "Existing filled PASS VCF found; skipping."
  fi

  echo
  echo "--- SnpEff annotation"
  if [[ ! -s "$PASS_SNPEFF" ]]; then
    micromamba run -n cancer_anno snpEff -Xmx16g -v GRCh38.99 "$PASS_FILLED" \
      | micromamba run -n hadza-wes bgzip -c > "$PASS_SNPEFF"
    micromamba run -n hadza-wes bcftools index -t -f "$PASS_SNPEFF"
  else
    echo "Existing SnpEff PASS VCF found; skipping."
  fi

  if [[ -s "$PASS_SNPEFF" ]]; then
    echo -e "$sample\t$BAM\t$UNFILTERED\t$FILTERED\t$PASS\t$PASS_SNPEFF\tDONE" >> "$MANIFEST"
  else
    echo -e "$sample\t$BAM\t$UNFILTERED\t$FILTERED\t$PASS\t$PASS_SNPEFF\tFAILED" >> "$MANIFEST"
  fi

done

echo
echo "=== BUILD SUMMARY TABLE ==="

micromamba run -n hadza-wes python - <<'PY'
import os
import gzip
import subprocess
import pandas as pd
from pathlib import Path

out = Path("paper_results/05_local_breast_cancer_tumor_WES/01_tumor_only_mutect2")
manifest = out / "local_tumor_WES_tumor_only_Mutect2_manifest.tsv"
summary = out / "local_tumor_WES_tumor_only_Mutect2_PASS_summary.tsv"

man = pd.read_csv(manifest, sep="\t")

rows = []
for _, r in man.iterrows():
    sample = r["sample_id"]
    pass_vcf = str(r["pass_vcf"])
    snpeff_vcf = str(r["pass_snpeff_vcf"])

    if not Path(pass_vcf).exists():
        rows.append({
            "sample_id": sample,
            "status": r["status"],
            "n_PASS": 0,
            "n_SNV": 0,
            "n_INDEL": 0,
            "n_MNP": 0,
            "n_HIGH": 0,
            "n_MODERATE": 0,
            "n_LOW": 0,
            "n_MODIFIER": 0
        })
        continue

    # Variant-type counts via bcftools.
    def count_expr(expr):
        try:
            cmd = ["bash", "-lc", f"micromamba run -n hadza-wes bcftools view {expr} {pass_vcf} -H | wc -l"]
            return int(subprocess.check_output(cmd, text=True).strip())
        except Exception:
            return 0

    n_pass = count_expr("")
    n_snv = count_expr("-v snps")
    n_indel = count_expr("-v indels")
    n_mnp = count_expr("-v mnps")

    impacts = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "MODIFIER": 0}

    if Path(snpeff_vcf).exists():
        opener = gzip.open if snpeff_vcf.endswith(".gz") else open
        with opener(snpeff_vcf, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                info = fields[7] if len(fields) > 7 else ""
                ann = None
                for item in info.split(";"):
                    if item.startswith("ANN="):
                        ann = item[4:]
                        break
                if not ann:
                    continue
                best = "MODIFIER"
                score = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
                for rec in ann.split(","):
                    parts = rec.split("|")
                    if len(parts) > 2:
                        imp = parts[2]
                        if score.get(imp, 0) > score.get(best, 0):
                            best = imp
                if best in impacts:
                    impacts[best] += 1

    rows.append({
        "sample_id": sample,
        "status": r["status"],
        "n_PASS": n_pass,
        "n_SNV": n_snv,
        "n_INDEL": n_indel,
        "n_MNP": n_mnp,
        "n_HIGH": impacts["HIGH"],
        "n_MODERATE": impacts["MODERATE"],
        "n_LOW": impacts["LOW"],
        "n_MODIFIER": impacts["MODIFIER"]
    })

df = pd.DataFrame(rows)
df.to_csv(summary, sep="\t", index=False)
print(df.to_string(index=False))
PY

echo
echo "=== MANIFEST ==="
column -t -s $'\t' "$MANIFEST"

echo
echo "=== PASS SUMMARY ==="
column -t -s $'\t' "$SUMMARY"

REPORT="$STATUS/79B_local_tumor_WES_tumor_only_Mutect2_status.txt"

{
  echo "79B local tumor WES tumor-only Mutect2"
  echo "Generated: $(date)"
  echo
  echo "Analysis note:"
  echo "Tumor-only Mutect2 candidate calling. No matched normals were available; therefore variants must be reported as candidate tumor variants, not definitive somatic calls."
  echo
  echo "Manifest:"
  cat "$MANIFEST"
  echo
  echo "PASS summary:"
  cat "$SUMMARY"
} > "$REPORT"

echo
echo "=== REPORT ==="
cat "$REPORT"

echo
echo "=== DONE 79B ==="
date
