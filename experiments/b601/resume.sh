#!/usr/bin/env bash
# Archived continuation recipe; preserves the original full-state LR schedule.
# For a fresh ten-epoch run, use task=b601_uncond_2cam224_10epoch with train.sh.
set -euo pipefail

num_processes="${1:?Usage: bash experiments/b601/resume.sh <num_processes> [hydra_overrides...]}"
shift
: "${FASTWAM_DATASET_ROOT:?Set FASTWAM_DATASET_ROOT to the prepared dataset}"
: "${FASTWAM_RESUME_RUN:?Set FASTWAM_RESUME_RUN to the original run}"
: "${FASTWAM_RESUME_STEP:?Set FASTWAM_RESUME_STEP to the full-state directory name}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH to the base-model cache}"
if [[ ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_processes must be a positive integer" >&2
  exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
export FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/fastwam}"
export FASTWAM_TEXT_CACHE="${FASTWAM_TEXT_CACHE:-$FASTWAM_CACHE_ROOT/text_embeds/b601}"
export FASTWAM_ACTION_BACKBONE="${FASTWAM_ACTION_BACKBONE:-$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "$num_processes" --main_process_port "${MASTER_PORT:-29500}" \
  --module experiments.b601.train_resume task=archive/legacy/b601_resume_10epoch "$@"
