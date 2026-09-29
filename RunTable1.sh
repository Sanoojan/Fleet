#!/usr/bin/env bash
# Reproduce Fleet Table 1 on Treasure and AIGIBench-13. Four independent
# workers use GPUs 0-3; every Python process sees one GPU, never DataParallel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/egr/research-sprintai/baliahsa/miniconda3/envs/AIGB/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
SHOTS_CSV="${SHOTS_CSV:-1,5,10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs_table1}"
DRY_RUN="${DRY_RUN:-0}"

DINO="${DINO:-${REPO_ROOT}/weights/dinov3-vitl16-pretrain-lvd1689m}"
AIGIBENCH_ROOT="${AIGIBENCH_ROOT:-${REPO_ROOT}/data/AIGIBench}"
AIGIBENCH_TRAIN="${AIGIBENCH_TRAIN:-${AIGIBENCH_ROOT}/train}"
AIGIBENCH_VAL="${AIGIBENCH_VAL:-${AIGIBENCH_ROOT}/val}"
AIGIBENCH_TEST="${AIGIBENCH_TEST:-${AIGIBENCH_ROOT}/test}"
TREASURE_ROOT="${TREASURE_ROOT:-${REPO_ROOT}/data/Treasure}"
TREASURE_FAKE="${TREASURE_FAKE:-${TREASURE_ROOT}/fake}"
TREASURE_REAL="${TREASURE_REAL:-${TREASURE_ROOT}/real}"
REAL_SUPPORT="${REAL_SUPPORT:-${TREASURE_REAL}/cc12m-2mp-realistic}"

# Stable single-GPU pretrain from the preceding reproduction. Override these
# to use the authors' official checkpoint or a later exact three-epoch model.
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${REPO_ROOT}/outputs_single_gpu/checkpoints_q128/pretrain_dual_branch_with_attn_loss_and_coverage_no_residual_q128.pth}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-${REPO_ROOT}/outputs_single_gpu/checkpoints_q128/prototypes_dual_branch_with_attn_loss_and_coverage_no_residual_q128_freq}"

NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a SHOTS <<< "$SHOTS_CSV"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "Error: GPUS_CSV is empty" >&2
  exit 2
fi

for required in "$PYTHON" "$PRETRAIN_CHECKPOINT"; do
  [ -f "$required" ] || { echo "Error: required file not found: $required" >&2; exit 2; }
done
for required in "$DINO" "$PROTOTYPE_DIR" "$AIGIBENCH_TRAIN" "$AIGIBENCH_VAL" "$AIGIBENCH_TEST" "$TREASURE_FAKE" "$REAL_SUPPORT"; do
  [ -d "$required" ] || { echo "Error: required directory not found: $required" >&2; exit 2; }
done

MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
LOG_DIR="${OUTPUT_ROOT}/logs"
STATUS_FILE="${OUTPUT_ROOT}/status.tsv"
mkdir -p "$MANIFEST_DIR" "$LOG_DIR"
for gpu in "${GPUS[@]}"; do
  : > "${MANIFEST_DIR}/gpu${gpu}.tsv"
done
: > "$STATUS_FILE"

AIGIBENCH_13=(
  "ProGAN" "R3GAN" "StyleGAN3" "StyleGAN-XL" "StyleSwim" "WFIR"
  "DALLE-3" "FLUX1-dev" "GLIDE" "Imagen3" "Midjourney" "SD3" "SDXL"
)

job_count=0
add_job() {
  local benchmark="$1" shot="$2" dataset="$3" fake_query="$4" real_query="$5"
  local gpu_index=$((job_count % ${#GPUS[@]}))
  local gpu="${GPUS[$gpu_index]}"
  local out_dir="${OUTPUT_ROOT}/runs/${benchmark}/${shot}shot/${dataset}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$benchmark" "$shot" "$dataset" "$fake_query" "$real_query" "$out_dir" \
    >> "${MANIFEST_DIR}/gpu${gpu}.tsv"
  job_count=$((job_count + 1))
}

for shot in "${SHOTS[@]}"; do
  for fake_query in "$TREASURE_FAKE"/*; do
    [ -d "$fake_query" ] || continue
    add_job "Treasure" "$shot" "$(basename "$fake_query")" "$fake_query" "$TREASURE_REAL"
  done
  for dataset in "${AIGIBENCH_13[@]}"; do
    fake_query="${AIGIBENCH_TEST}/${dataset}/1_fake"
    real_query="${AIGIBENCH_TEST}/${dataset}/0_real"
    if [ ! -d "$fake_query" ] || [ ! -d "$real_query" ]; then
      echo "Error: AIGIBench-13 subset is incomplete: $dataset" >&2
      exit 2
    fi
    add_job "AIGIBench-13" "$shot" "$dataset" "$fake_query" "$real_query"
  done
done

echo "[$(date --iso-8601=seconds)] Prepared $job_count jobs across GPUs: $GPUS_CSV"
echo "Checkpoint: $PRETRAIN_CHECKPOINT"
echo "Output: $OUTPUT_ROOT"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1; manifests created, no training started"
  exit 0
fi

run_worker() {
  local gpu="$1"
  local manifest="${MANIFEST_DIR}/gpu${gpu}.tsv"
  while IFS=$'\t' read -r benchmark shot dataset fake_query real_query out_dir; do
    [ -n "$dataset" ] || continue
    mkdir -p "$out_dir"
    if [ -s "${out_dir}/fewshot_results.json" ]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$gpu" "SKIP" "$benchmark" "${shot}shot/$dataset" >> "$STATUS_FILE"
      continue
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$gpu" "START" "$benchmark" "${shot}shot/$dataset" >> "$STATUS_FILE"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" -m fleet.train.fewshot_experiment \
        --pretrain_checkpoint "$PRETRAIN_CHECKPOINT" \
        --prototype_dir "$PROTOTYPE_DIR" \
        --dinov3_model_path "$DINO" \
        --query_dir "$fake_query" \
        --real_support_dir "$REAL_SUPPORT" \
        --real_query_dir "$real_query" \
        --aigibench_train_dir "$AIGIBENCH_TRAIN" \
        --aigibench_val_dir "$AIGIBENCH_VAL" \
        --dataset_name "$dataset" \
        --output_dir "$out_dir" \
        --n_fake_support "$shot" --n_real_support "$shot" \
        --n_fake_memory 500 --n_real_memory 500 \
        --num_epochs "$NUM_EPOCHS" --batch_size "$BATCH_SIZE" \
        --distill_weight 10 --num_workers "$NUM_WORKERS" \
        --val_batch_size "$VAL_BATCH_SIZE" --max_real_val 1000 \
        --force_train \
        > "${out_dir}/train.log" 2>&1; then
      state="DONE"
    else
      state="FAILED"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$gpu" "$state" "$benchmark" "${shot}shot/$dataset" >> "$STATUS_FILE"
  done < "$manifest"
}

worker_pids=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" > "${LOG_DIR}/worker_gpu${gpu}.log" 2>&1 &
  worker_pids+=("$!")
  echo "Started single-GPU worker gpu=$gpu pid=$!"
done

for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

"$PYTHON" -m fleet.train.table1_summary --output_root "$OUTPUT_ROOT" > "${LOG_DIR}/summary.log" 2>&1
touch "${OUTPUT_ROOT}/table1_done.flag"
echo "[$(date --iso-8601=seconds)] Table 1 sweep complete"
cat "${OUTPUT_ROOT}/table1_summary.txt"
