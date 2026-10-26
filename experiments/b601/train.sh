#!/usr/bin/env bash
# Launch the native trainer after sourcing a host-local paths file.
set -euo pipefail

num_processes="${1:?Usage: bash experiments/b601/train.sh <num_processes> [hydra_overrides...]}"
shift
: "${FASTWAM_DATASET_ROOT:?Set FASTWAM_DATASET_ROOT to the prepared dataset}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH to the base-model cache}"
if [[ ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_processes must be a positive integer" >&2
  exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
export FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/fastwam}"
export FASTWAM_RUN_ROOT="${FASTWAM_RUN_ROOT:-$FASTWAM_CACHE_ROOT/runs}"
export FASTWAM_TEXT_CACHE="${FASTWAM_TEXT_CACHE:-$FASTWAM_CACHE_ROOT/text_embeds/b601}"
export FASTWAM_ACTION_BACKBONE="${FASTWAM_ACTION_BACKBONE:-$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

run_dir="$FASTWAM_RUN_ROOT/b601_ee_delta_v2/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "$num_processes" --main_process_port "${MASTER_PORT:-29500}" \
  scripts/train.py task=b601_uncond_2cam224 "output_dir=$run_dir" "$@"
