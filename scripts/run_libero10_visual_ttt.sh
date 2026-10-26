#!/usr/bin/env bash
# Run from an already allocated compute node. Launches only local processes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Reuse the runtime used by the completed original LIBERO evaluations.
exec bash cluster/runtime/run.sh python -u experiments/libero/run_visual_ttt.py "$@"
