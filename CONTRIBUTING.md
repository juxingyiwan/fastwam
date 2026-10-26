# Contributing

[中文](CONTRIBUTING.zh-CN.md) · [Home](README.md)

Use a separate checkout and environment. Keep machine configuration, credentials, model weights, datasets and runtime records outside the source tree. Preserve upstream license notices and original documentation.

## Checks

```bash
python scripts/check_submission.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$PWD/src:$PWD" python -m pytest -q \
  experiments/b601/tests experiments/libero/visual_ttt
```

Dependency setup is described in [environments/README.md](environments/README.md). The CPU suite uses fake devices and temporary fixtures. Do not put hardware preflight, live inference, simulator rollouts or training in ordinary CI.

For model or normalization changes, compare fixed recorded inputs with the baseline. For hardware or lifecycle changes, retain CPU failure-path coverage and arrange separate attended validation. A passing unit suite does not establish real-robot task success.

## Prepare a commit

Describe the concrete change, compatibility impact and validation. Keep the English/Chinese scenario guides aligned. Inspect `git diff --check`, the staged diff, `git status` and `scripts/check_submission.py` before committing. Record source/model/data versions for experiments and keep their artifacts outside Git.

This preparation checkout starts from the recorded official upstream baseline. Site-specific intermediate commits are not part of its public branch. Add a destination remote and push only after the intended repository and publication scope are selected.
