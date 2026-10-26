# RoboTwin

[中文](robotwin.zh-CN.md) · [Home](../README.md)

Use the [training environment](../environments/README.md) and follow the original [model/data setup](../README.upstream.md). Edit a machine-local copy of the dataset paths or pass Hydra overrides. The native task presets live under `configs/task/archive/simulation/`; the archive name is retained for configuration compatibility.

## Training and evaluation

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/precompute_text_embeds.py task=archive/simulation/robotwin_uncond_3cam_384_1e-4
bash scripts/train_zero1.sh 8 task=archive/simulation/robotwin_uncond_3cam_384_1e-4
python experiments/robotwin/run_robotwin_manager.py \
  task=archive/simulation/robotwin_uncond_3cam_384_1e-4 \
  ckpt=/path/to/compatible_checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

Run training/evaluation on allocated GPUs. Supply a compatible checkpoint, saved normalization statistics, simulator and assets; this repository does not ship those files. Inspect `configs/sim_robotwin.yaml` for suite/task selection, seeds, episode counts and inference settings. The manager can schedule multiple workers, so first select a small task scope appropriate for your allocation.

Install the matching RoboTwin simulator assets and GPU graphics runtime before evaluation. Keep their licenses and source revisions. The policy adapter is in [experiments/robotwin/fastwam_policy](../experiments/robotwin/fastwam_policy).

Record the Git commit, resolved configuration, dataset/model revisions and seeds for each run. No new evaluation scores are claimed by this submission preparation.
