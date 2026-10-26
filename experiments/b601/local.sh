#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${FASTWAM_PYTHON:-python}" -m experiments.b601.local "$@"
