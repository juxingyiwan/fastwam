#!/usr/bin/env bash
# Native single-task training; every supported GPU count retains batch 128.
set -euo pipefail
task="${1:?Usage: bash experiments/b601/train_single_task.sh <drawer|cube|pen> [num_gpus] [hydra_overrides...]}"
shift
num_processes="${1:-4}"
if [[ $# -gt 0 ]]; then shift; fi
case "$task" in drawer|cube|pen) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
case "$num_processes" in 1|2|4|8) ;; *) echo "num_gpus must be 1, 2, 4 or 8" >&2; exit 2 ;; esac
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/fastwam}"
export FASTWAM_RUN_ROOT="${FASTWAM_RUN_ROOT:-$FASTWAM_CACHE_ROOT/runs}"
epochs="${FASTWAM_SINGLE_TASK_EPOCHS:-5}"
case "$epochs" in 5|10) ;; *) echo "FASTWAM_SINGLE_TASK_EPOCHS must be 5 or 10" >&2; exit 2 ;; esac
config_name="b601_${task}_only_${epochs}epoch"
if [[ "$epochs" == 5 && "$num_processes" == 2 && "$task" != drawer ]]; then config_name="${config_name}_2gpu"; fi
[[ -f "$repo_dir/configs/task/$config_name.yaml" ]] || { echo "Missing task configuration: $config_name" >&2; exit 2; }
run_dir="$FASTWAM_RUN_ROOT/${config_name}/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
# Two-rank ZeRO-2 retains more optimizer state per GPU; use a smaller micro-batch.
case "$num_processes" in 1|2) default_micro=4 ;; *) default_micro=16 ;; esac
micro_batch="${FASTWAM_MICRO_BATCH:-$default_micro}"
[[ "$micro_batch" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid micro-batch" >&2; exit 2; }
(( 128 % (micro_batch * num_processes) == 0 )) || { echo "Micro-batch must divide effective batch 128" >&2; exit 2; }
accumulation=$((128 / (micro_batch * num_processes)))
# Four-card ZeRO-2 needs the VAE graph pool released; preserve batch 16 x 4 x 2.
if [[ "$num_processes" == 4 ]]; then
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
exec bash "$repo_dir/experiments/b601/train.sh" "$num_processes" \
  "task=$config_name" "output_dir=$run_dir" \
  "batch_size=$micro_batch" "gradient_accumulation_steps=$accumulation" "$@"
