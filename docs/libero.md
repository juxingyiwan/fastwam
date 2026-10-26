# LIBERO

[中文](libero.zh-CN.md) · [Home](../README.md)

Use the [training environment](../environments/README.md) and follow the original [model/data setup](../README.upstream.md). Edit a machine-local copy of the dataset paths or pass Hydra overrides. The native task presets live under `configs/task/archive/simulation/`; the archive name is retained for configuration compatibility.

## Training and evaluation

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/precompute_text_embeds.py task=archive/simulation/libero_uncond_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=archive/simulation/libero_uncond_2cam224_1e-4
python experiments/libero/run_libero_manager.py \
  task=archive/simulation/libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/compatible_checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

Run training/evaluation on allocated GPUs. Supply a compatible checkpoint, saved normalization statistics, simulator and assets; this repository does not ship those files. Inspect `configs/sim_libero.yaml` for suite/task selection, seeds, episode counts and inference settings. The manager can schedule multiple workers, so first select a small task scope appropriate for your allocation.

Optional visual-only TTT is available in [visual_ttt/README.md](../experiments/libero/visual_ttt/README.md). Its scores and collection protocol must be reported separately from native inference.

Record the Git commit, resolved configuration, dataset/model revisions and seeds for each run. No new evaluation scores are claimed by this submission preparation.
