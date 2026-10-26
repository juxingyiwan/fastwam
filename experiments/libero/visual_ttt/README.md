# LIBERO-10 Visual-only TTT

The released Optional IDM checkpoint runs in `idm` mode. Collection uses the
frozen model and real future RGB; adaptation updates video-expert LoRA only.
The action expert, VAE, text encoder and proprio projection remain frozen.
LoRA updates are merged into the frozen video Linear weights for evaluation,
so inference uses the original compiled computation path. The exact original
weights are restored before the next collection or optimization repeat.

`TTT.enabled=true` enables adaptation in the full pipeline. Set
`TTT.enabled=false` to run only the frozen baseline. A `baseline` stage always
uses the frozen model. Evaluation records the actual LoRA activation state and
rejects a mismatch between that state and its result label.

Collection and evaluation both use the project's `seed`, inherited from
`configs/train.yaml` (42 unless overridden). Both start at index zero of the
official task initial states. There is no separate collection seed, task/episode
seed offset, or additional initial-state bank. Each phase creates one environment
per task and carries its RNG stream across episodes, as in the original evaluator.
An incomplete evaluation task restarts from episode zero on resume.
Adaptation uses the same project seed: `TTT.adaptation_seeds: ["${seed}"]`.
Both formal and smoke configs inherit this setting, including the integration
diagnostic. A mismatched adaptation seed is rejected before loading the model.
Change `seed` to change collection, adaptation and evaluation together.

Log in to the allocated compute node manually before launching. The launcher
uses the current node and selected visible GPUs. It does not manage allocations.
The shell launcher activates the same configured `fastwam` runtime as the completed
original evaluations, including its LIBERO resource paths and EGL backend.
Direct Python invocation requires that configured runtime to be active.
Action inference compilation is enabled, matching the historical IDM baseline.

From the FastWAM project root:

```bash
python -m unittest experiments.libero.visual_ttt.test_core experiments.libero.visual_ttt.test_sampling
bash scripts/run_libero10_visual_ttt.sh --gpus 0,1,2,3 --stage preflight --run-id libero10_visual
bash scripts/run_libero10_visual_ttt.sh --gpus 0 --config libero10_idm_visual_ttt_smoke --stage check --run-id visual_check
bash scripts/run_libero10_visual_ttt.sh --gpus 0 --config libero10_idm_visual_ttt_smoke --run-id visual_smoke
bash scripts/run_libero10_visual_ttt.sh --gpus 0,1,2,3 --run-id libero10_visual
bash scripts/run_libero10_visual_ttt.sh --gpus 0,1,2,3 --run-id libero10_frozen TTT.enabled=false
```

Stages: `preflight`, `check`, `baseline`, `collect`, `adapt`, `evaluate`, `ttt`, `full`.
`ttt` runs collection, adaptation and evaluation without rerunning the frozen baseline.
`check` is a synthetic repeated-frame gradient/inference diagnostic, not a
performance evaluation. The smoke config uses one task and one evaluation
episode. Formal defaults use all ten tasks, 50 evaluation episodes per task,
8 collection rollouts with 8 windows each, and one adaptation using the same seed.

Use the same run ID and configuration to resume an interrupted stage. Settings,
source files and collection artifacts are checked before reusing outputs.
Results and per-worker logs are under `evaluate_results/libero10_idm_visual_ttt`.
Full-suite metrics are reported only after every task has completed.
