#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-/egr/research-sprintai/baliahsa/miniconda3/envs/AIGB/bin/python}"
CONFIG="${1:-Configs/treasure_continual_fleet_5shot_aigibench_replay.yaml}"
FORCE="${FORCE:-0}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OUT="$("$PYTHON" -m fleet.train.treasure_continual_5shot --config "$CONFIG" --print-output-dir)"
mkdir -p "$OUT/logs"
# Serialize duplicate checks and PID publication, including concurrent launchers.
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo "Another launcher is active for $OUT" >&2; exit 1; }
PID_FILE="$OUT/experiment.pid"
if [[ -f "$PID_FILE" ]]; then
  read -r PREVIOUS_PID < "$PID_FILE" || true
  if [[ "${PREVIOUS_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$PREVIOUS_PID" 2>/dev/null; then
    if [[ "$FORCE" != 1 ]]; then
      echo "Experiment PID $PREVIOUS_PID is live; refusing duplicate. FORCE=1 bypasses this PID check." >&2
      exit 1
    fi
    echo "FORCE bypasses stale/live PID check; the trainer still locks shared outputs." >&2
  fi
fi
"$PYTHON" -m fleet.train.treasure_continual_5shot --config "$CONFIG" --validate-config > "$OUT/logs/config_validation.log" 2>&1
LOG="$OUT/logs/continual_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON" -m fleet.train.treasure_continual_5shot --config "$CONFIG" > "$LOG" 2>&1 < /dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE.tmp"
mv -- "$PID_FILE.tmp" "$PID_FILE"
printf 'PID: %s\nLog: %s\nOutput: %s\n' "$PID" "$LOG" "$OUT"
printf 'Status: ps -p %q -o pid,etime,cmd\n' "$PID"
printf 'Follow: tail -f %q\n' "$LOG"
