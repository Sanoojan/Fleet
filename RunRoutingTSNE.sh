#!/usr/bin/env bash
# Re-run the 64 Treasure 10-shot adaptations and export matched pre/post routing
# weights. Independent workers see one GPU each; nn.DataParallel is not used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-/egr/research-sprintai/baliahsa/miniconda3/envs/AIGB/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# GPUs 0-3 are currently occupied by another user's jobs. Use the available
# devices by default; override with GPUS_CSV when needed.
GPUS_CSV="${GPUS_CSV:-4,5,6,7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs_routing_tsne}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/Plots}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-50}"
DRY_RUN="${DRY_RUN:-0}"

DINO="${DINO:-${REPO_ROOT}/weights/dinov3-vitl16-pretrain-lvd1689m}"
AIGIBENCH_TRAIN="${AIGIBENCH_TRAIN:-${REPO_ROOT}/data/AIGIBench/train}"
AIGIBENCH_VAL="${AIGIBENCH_VAL:-${REPO_ROOT}/data/AIGIBench/val}"
TREASURE_FAKE="${TREASURE_FAKE:-${REPO_ROOT}/data/Treasure/fake}"
TREASURE_REAL="${TREASURE_REAL:-${REPO_ROOT}/data/Treasure/real}"
REAL_SUPPORT="${REAL_SUPPORT:-${TREASURE_REAL}/cc12m-2mp-realistic}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${REPO_ROOT}/outputs_single_gpu/checkpoints_q128/pretrain_dual_branch_with_attn_loss_and_coverage_no_residual_q128.pth}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-${REPO_ROOT}/outputs_single_gpu/checkpoints_q128/prototypes_dual_branch_with_attn_loss_and_coverage_no_residual_q128_freq}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
LOG_DIR="${OUTPUT_ROOT}/logs"
STATUS_FILE="${OUTPUT_ROOT}/status.tsv"
mkdir -p "$MANIFEST_DIR" "$LOG_DIR" "$OUTPUT_ROOT/routing"
for gpu in "${GPUS[@]}"; do : > "${MANIFEST_DIR}/gpu${gpu}.txt"; done
: > "$STATUS_FILE"

job_count=0
for query_dir in "$TREASURE_FAKE"/*; do
  [ -d "$query_dir" ] || continue
  dataset="$(basename "$query_dir")"
  gpu="${GPUS[$((job_count % ${#GPUS[@]}))]}"
  printf '%s\t%s\n' "$dataset" "$query_dir" >> "${MANIFEST_DIR}/gpu${gpu}.txt"
  job_count=$((job_count + 1))
done
echo "[$(date --iso-8601=seconds)] Prepared $job_count routing jobs on GPUs $GPUS_CSV"
if [ "$job_count" -ne 64 ]; then echo "Error: expected 64 Treasure datasets" >&2; exit 2; fi
if [ "$DRY_RUN" = "1" ]; then exit 0; fi

run_worker() {
  local gpu="$1"
  while IFS=$'\t' read -r dataset query_dir; do
    out_dir="${OUTPUT_ROOT}/routing/${dataset}"
    export_path="${out_dir}/routing_weights.npz"
    mkdir -p "$out_dir"
    if [ -s "$export_path" ]; then
      printf '%s\t%s\tSKIP\t%s\n' "$(date --iso-8601=seconds)" "$gpu" "$dataset" >> "$STATUS_FILE"
      continue
    fi
    printf '%s\t%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$gpu" "$dataset" >> "$STATUS_FILE"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" -m fleet.train.fewshot_experiment \
        --pretrain_checkpoint "$PRETRAIN_CHECKPOINT" --prototype_dir "$PROTOTYPE_DIR" \
        --dinov3_model_path "$DINO" --query_dir "$query_dir" \
        --real_support_dir "$REAL_SUPPORT" --real_query_dir "$TREASURE_REAL" \
        --aigibench_train_dir "$AIGIBENCH_TRAIN" --aigibench_val_dir "$AIGIBENCH_VAL" \
        --dataset_name "$dataset" --output_dir "$out_dir" \
        --n_fake_support 10 --n_real_support 10 --n_fake_memory 500 --n_real_memory 500 \
        --num_epochs 20 --batch_size 32 --distill_weight 10 \
        --num_workers 4 --val_batch_size 128 --max_real_val 1000 --force_train \
        --routing_export_path "$export_path" \
        --routing_export_max_samples "$SAMPLES_PER_DATASET" \
        > "${out_dir}/train.log" 2>&1; then
      state="DONE"
    else
      state="FAILED"
    fi
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$gpu" "$state" "$dataset" >> "$STATUS_FILE"
  done < "${MANIFEST_DIR}/gpu${gpu}.txt"
}

pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" > "${LOG_DIR}/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "Started routing worker gpu=$gpu pid=$!"
done
for pid in "${pids[@]}"; do wait "$pid"; done

completed=$(find "${OUTPUT_ROOT}/routing" -name routing_weights.npz -type f | wc -l)
if [ "$completed" -ne 64 ]; then
  echo "Error: only $completed/64 routing exports completed" >&2
  exit 1
fi
MPLBACKEND=Agg "$PYTHON" -m fleet.train.plot_routing_tsne \
  --input-root "$OUTPUT_ROOT" --output-dir "$PLOT_DIR" > "${LOG_DIR}/tsne.log" 2>&1
touch "${OUTPUT_ROOT}/done.flag"
echo "[$(date --iso-8601=seconds)] Routing t-SNE complete"
