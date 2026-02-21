#!/usr/bin/env bash
set -euo pipefail

# Monitor cNMF resume/debug progress for Plan 0.
# Default experiment path points to the current run discussed in chat.

EXP_DIR="/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
INTERVAL=60
TAIL_LINES=30

usage() {
  cat <<'EOF'
Usage:
  watch_cnmf_resume.sh [--exp-dir PATH] [--interval SECONDS] [--tail-lines N] [--once]

Options:
  --exp-dir PATH       Experiment directory containing analysis/plan0 and models/cnmf_plan0.
  --interval SECONDS   Refresh interval for continuous mode (default: 60).
  --tail-lines N       Number of log lines to show (default: 30).
  --once               Print one snapshot and exit.
  -h, --help           Show this help.
EOF
}

ONCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-dir)
      EXP_DIR="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --tail-lines)
      TAIL_LINES="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$EXP_DIR" ]]; then
  echo "Experiment directory not found: $EXP_DIR" >&2
  exit 1
fi

print_snapshot() {
  local analysis_dir="$EXP_DIR/analysis/plan0"
  local cnmf_dir="$analysis_dir/cnmf"

  echo "============================================================"
  date
  echo "EXP_DIR=$EXP_DIR"
  echo

  echo "[Processes]"
  ps -eo pid,etime,%cpu,%mem,state,cmd | awk 'NR==1 || /resume_plan0_cnmf|python -u -/ {print}'
  echo

  echo "[Latest debug log]"
  local latest_log=""
  latest_log="$(ls -1t "$analysis_dir"/resume_plan0_cnmf_debug_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_log" ]]; then
    echo "$latest_log"
    tail -n "$TAIL_LINES" "$latest_log" || true
  else
    echo "No debug resume log found under $analysis_dir"
  fi
  echo

  echo "[Per-K markers]"
  for k in 20 40 60; do
    local kd="$cnmf_dir/k_$k"
    local started="N"
    local done="N"
    local err="N"
    local stats="N"
    [[ -f "$kd/debug_started.json" ]] && started="Y"
    [[ -f "$kd/debug_done.json" ]] && done="Y"
    [[ -f "$kd/debug_error.json" ]] && err="Y"
    [[ -f "$kd/consensus_stats.json" ]] && stats="Y"
    printf 'k=%-2s started=%s done=%s error=%s stats=%s\n' "$k" "$started" "$done" "$err" "$stats"
  done
  echo

  echo "[Consensus artifact counts]"
  local tmp_dir="$EXP_DIR/models/cnmf_plan0/plan0_cnmf/cnmf_tmp"
  local n_consensus_npz=0
  local n_consensus_txt=0
  if [[ -d "$tmp_dir" ]]; then
    n_consensus_npz="$(ls -1 "$tmp_dir"/*consensus*.npz 2>/dev/null | wc -l || true)"
    n_consensus_txt="$(ls -1 "$EXP_DIR/models/cnmf_plan0/plan0_cnmf"/*consensus*.txt 2>/dev/null | wc -l || true)"
  fi
  echo "cnmf_tmp consensus npz: $n_consensus_npz"
  echo "plan0_cnmf consensus txt: $n_consensus_txt"
}

if [[ "$ONCE" -eq 1 ]]; then
  print_snapshot
  exit 0
fi

while true; do
  clear || true
  print_snapshot
  sleep "$INTERVAL"
done
