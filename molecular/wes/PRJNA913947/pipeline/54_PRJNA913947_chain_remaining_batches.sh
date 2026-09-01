#!/usr/bin/env bash

# Load portable molecular-analysis paths.
_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${_REPO_ROOT}" ]]; then
    echo "ERROR: Run this script from within the bc-fuzzy-ml-repro repository." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${_REPO_ROOT}/molecular/config/load_config.sh"

set -u -o pipefail

cd "$PROJECT_ROOT" || exit 1

mkdir -p logs/public tmp

CHAIN_LOG="logs/public/PRJNA913947_CHAIN_remaining_batches_$(date +%Y%m%d_%H%M%S).log"
LOCK_FILE="tmp/PRJNA913947_chain_remaining_batches.lock"

exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=== PRJNA913947 CHAIN SUPERVISOR START ==="
date
echo "CHAIN_LOG=$CHAIN_LOG"
echo

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: Another chain supervisor is already running."
  exit 1
fi

PRJNA_BASE="results/EA_BC_AI_MultiOmics/PRJNA913947_candidate_kenya_wes"
PAIR_RUN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/53_PRJNA913947_run_pair_range_multidisk.sh"

LOCAL_ROOT="$PROJECT_ROOT/results/EA_BC_AI_MultiOmics/PRJNA913947_candidate_kenya_wes"
T9_ROOT="${SCRATCH_ROOT}/PRJNA913947_candidate_kenya_wes"
T7_ROOT="${SECONDARY_SCRATCH_ROOT}/PRJNA913947_candidate_kenya_wes"

MIN_FREE_GB=120
SLEEP_SECONDS=600

pair_name() {
  printf "candidate_kenya_pair_%03d" "$1"
}

log_time() {
  date "+%Y-%m-%d %H:%M:%S"
}

queue_active() {
  local qname="$1"
  pgrep -af "QUEUE_NAME=${qname}" | grep -v "pgrep" >/dev/null 2>&1
}

pair_pass_ok() {
  local pair="$1"
  local vcf

  vcf=$(find -L "$PRJNA_BASE/mutect2/$pair" -maxdepth 1 -name "${pair}.PASS.vcf.gz" -print -quit 2>/dev/null || true)

  [[ -n "$vcf" ]] || return 1
  [[ -s "$vcf" ]] || return 1

  if [[ ! -s "${vcf}.tbi" && ! -s "${vcf}.csi" ]]; then
    return 1
  fi

  micromamba run -n hadza-wes bcftools view -h "$vcf" >/dev/null 2>&1 || return 1
  return 0
}

block_pass_ok() {
  local start="$1"
  local end="$2"
  local i pair

  for i in $(seq "$start" "$end"); do
    pair=$(pair_name "$i")
    pair_pass_ok "$pair" || return 1
  done

  return 0
}

print_block_status() {
  local start="$1"
  local end="$2"
  local i pair bqsr_t bqsr_n stats pass

  for i in $(seq "$start" "$end"); do
    pair=$(pair_name "$i")

    bqsr_t=$(find -L "$PRJNA_BASE/bam/${pair}_T" -maxdepth 1 -name "*.bqsr.bam" 2>/dev/null | wc -l)
    bqsr_n=$(find -L "$PRJNA_BASE/bam/${pair}_N" -maxdepth 1 -name "*.bqsr.bam" 2>/dev/null | wc -l)
    stats=$(find -L "$PRJNA_BASE/mutect2/$pair/shards" -name "*.unfiltered.vcf.gz.stats" 2>/dev/null | wc -l)
    pass=$(find -L "$PRJNA_BASE/mutect2/$pair" -maxdepth 1 -name "*.PASS.vcf.gz" 2>/dev/null | wc -l)

    echo -e "${pair}\tBQSR_T=${bqsr_t}\tBQSR_N=${bqsr_n}\tshard_stats=${stats}\tPASS=${pass}"
  done
}

free_gb_for_root() {
  local root="$1"
  mkdir -p "$root" 2>/dev/null || true
  df -BG "$root" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}'
}

root_ready() {
  local label="$1"
  local root="$2"
  local free_gb

  mkdir -p "$root"/{bam,mutect2,tmp,summary,logs} || {
    echo "ERROR [$label]: cannot create root dirs: $root"
    return 1
  }

  touch "$root/.chain_write_test" 2>/dev/null && rm -f "$root/.chain_write_test" || {
    echo "ERROR [$label]: root is not writable: $root"
    return 1
  }

  free_gb=$(free_gb_for_root "$root")
  if [[ -z "$free_gb" ]]; then
    echo "ERROR [$label]: cannot read free space for $root"
    return 1
  fi

  echo "[$label] free_gb=${free_gb}G root=$root"

  if (( free_gb < MIN_FREE_GB )); then
    echo "ERROR [$label]: free space below threshold ${MIN_FREE_GB}G"
    return 1
  fi

  return 0
}

launch_block() {
  local label="$1"
  local start="$2"
  local end="$3"
  local root="$4"
  local align_threads="$5"
  local sort_threads="$6"
  local java_mem="$7"
  local java_mem_chr="$8"

  if queue_active "$label"; then
    echo "[$(log_time)] SKIP $label: already active"
    return 0
  fi

  if block_pass_ok "$start" "$end"; then
    echo "[$(log_time)] SKIP $label: target block already PASS-complete"
    return 0
  fi

  root_ready "$label" "$root" || return 1

  echo "[$(log_time)] LAUNCH $label pair$(printf "%03d" "$start")-$(printf "%03d" "$end")"
  echo "WORK_ROOT=$root"

  nohup systemd-inhibit \
    --what=sleep:shutdown \
    --why="PRJNA913947 ${label} chained pair$(printf "%03d" "$start")-$(printf "%03d" "$end")" \
    env QUEUE_NAME="$label" PAIR_START="$start" PAIR_END="$end" WORK_ROOT="$root" \
        ALIGN_THREADS_PRJNA="$align_threads" SORT_THREADS_PRJNA="$sort_threads" JAVA_MEM_PRJNA="$java_mem" \
        CHR_JOBS=1 THREADS_PER_CHR=2 JAVA_MEM_PER_CHR="$java_mem_chr" \
    micromamba run -n hadza-wes bash "$PAIR_RUN_SCRIPT" \
    > "logs/public/PRJNA913947_${label}_launcher_$(date +%Y%m%d_%H%M%S).log" \
    2>&1 &

  echo "[$(log_time)] LAUNCHED $label pid=$!"
  return 0
}

declare -A BLOCKED

while true; do
  echo
  echo "================================================================"
  echo "CHAIN CHECK: $(log_time)"
  echo "================================================================"

  echo
  echo "=== PASS VCF COUNT ==="
  find -L "$PRJNA_BASE/mutect2" -maxdepth 2 -name "candidate_kenya_pair_*.PASS.vcf.gz" | wc -l

  echo
  echo "=== COMPLETED PASS VCFs ==="
  find -L "$PRJNA_BASE/mutect2" -maxdepth 2 -name "candidate_kenya_pair_*.PASS.vcf.gz" \
    | sed 's#.*/mutect2/##; s#/.*##' | sort || true

  echo
  echo "=== DISK ==="
  df -h "$PROJECT_ROOT" "$SCRATCH_ROOT" "$SECONDARY_SCRATCH_ROOT" 2>/dev/null || true

  all_done=1

  # LOCAL: 004-006 tamamlanınca 013-015 başlat
  echo
  echo "---- LOCAL 004-006 -> 013-015 ----"
  print_block_status 4 6
  print_block_status 13 15

  if block_pass_ok 13 15; then
    echo "STATUS LOCAL_013_015_CHAIN: DONE"
  else
    all_done=0
    if queue_active "LOCAL_013_015_CHAIN"; then
      echo "STATUS LOCAL_013_015_CHAIN: ACTIVE"
    elif [[ "${BLOCKED[LOCAL_013_015_CHAIN]:-0}" == "1" ]]; then
      echo "STATUS LOCAL_013_015_CHAIN: BLOCKED, manual check needed"
    elif block_pass_ok 4 6; then
      launch_block "LOCAL_013_015_CHAIN" 13 15 "$LOCAL_ROOT" 8 4 24g 10g || BLOCKED[LOCAL_013_015_CHAIN]=1
    elif queue_active "LOCAL_004_006"; then
      echo "STATUS LOCAL_004_006: still active, waiting"
    else
      echo "ERROR LOCAL_004_006 inactive but 004-006 not PASS-complete. Blocking LOCAL_013_015_CHAIN."
      BLOCKED[LOCAL_013_015_CHAIN]=1
    fi
  fi

  # T9: 007-009 tamamlanınca 016-018 başlat
  echo
  echo "---- T9 007-009 -> 016-018 ----"
  print_block_status 7 9
  print_block_status 16 18

  if block_pass_ok 16 18; then
    echo "STATUS T9_016_018_CHAIN: DONE"
  else
    all_done=0
    if queue_active "T9_016_018_CHAIN"; then
      echo "STATUS T9_016_018_CHAIN: ACTIVE"
    elif [[ "${BLOCKED[T9_016_018_CHAIN]:-0}" == "1" ]]; then
      echo "STATUS T9_016_018_CHAIN: BLOCKED, manual check needed"
    elif block_pass_ok 7 9; then
      launch_block "T9_016_018_CHAIN" 16 18 "$T9_ROOT" 8 4 24g 10g || BLOCKED[T9_016_018_CHAIN]=1
    elif queue_active "T9_007_009"; then
      echo "STATUS T9_007_009: still active, waiting"
    else
      echo "ERROR T9_007_009 inactive but 007-009 not PASS-complete. Blocking T9_016_018_CHAIN."
      BLOCKED[T9_016_018_CHAIN]=1
    fi
  fi

  # T7: 010-012 tamamlanınca 019-021 başlat
  echo
  echo "---- T7 010-012 -> 019-021 ----"
  print_block_status 10 12
  print_block_status 19 21

  if block_pass_ok 19 21; then
    echo "STATUS T7_019_021_CHAIN: DONE"
  else
    all_done=0
    if queue_active "T7_019_021_CHAIN"; then
      echo "STATUS T7_019_021_CHAIN: ACTIVE"
    elif [[ "${BLOCKED[T7_019_021_CHAIN]:-0}" == "1" ]]; then
      echo "STATUS T7_019_021_CHAIN: BLOCKED, manual check needed"
    elif block_pass_ok 10 12; then
      launch_block "T7_019_021_CHAIN" 19 21 "$T7_ROOT" 8 4 24g 10g || BLOCKED[T7_019_021_CHAIN]=1
    elif queue_active "T7_010_012_RESTARTED"; then
      echo "STATUS T7_010_012_RESTARTED: still active, waiting"
    else
      echo "ERROR T7_010_012_RESTARTED inactive but 010-012 not PASS-complete. Blocking T7_019_021_CHAIN."
      BLOCKED[T7_019_021_CHAIN]=1
    fi
  fi

  if (( all_done == 1 )); then
    echo
    echo "=== ALL CHAINED BLOCKS DONE: pair013-021 PASS-COMPLETE ==="
    date
    exit 0
  fi

  echo
  echo "Sleeping ${SLEEP_SECONDS}s before next check..."
  sleep "$SLEEP_SECONDS"
done
