# Validation record / 验证记录

Prepared on 2026-10-26 from the source identities in [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json). Checks ran in an isolated submission checkout using an existing Linux environment: Python 3.10.12, PyTorch 2.7.1+cu128 and pytest 9.1.1. CUDA devices were hidden, and the imported `fastwam` module was verified to come from this checkout.

## Results

| Check | Result |
| --- | --- |
| `python scripts/check_submission.py` | 977 files, 158 maintained Python files, 94 audited source hashes and 101 local documentation links; zero errors |
| `bash -n` on maintained shell entry points | All 7 passed |
| B601 and visual TTT regression suite | 432 passed, 1 skipped, 2 subtests passed; pytest time 95.13 seconds |
| Runtime attribution hashes | Match the included runtime source |
| RoboTwin policy alias | Relative symlink resolves inside the checkout |
| Git whitespace check | Passed |

The skipped test is `test_media_can_record_in_fastwam_environment_without_camera`: the existing environment lacks the optional `rerun` package. Recording through Rerun was not validated by this run.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m pytest -q -p no:cacheprovider \
  experiments/b601/tests experiments/libero/visual_ttt
```

## Scope

The checks exercised fake hardware, temporary fixtures, configuration resolution, EE geometry, model-loading contracts, RTC recovery and visual TTT logic. They did not run training, simulator rollouts, a real checkpoint inference or robot movement. The source preparation preserves all 94 recorded core/runtime Python files byte for byte; it does not establish robot task success or benchmark performance.

The CPU workflow and `environments/requirements-cpu.txt` are provided for an isolated CI installation. A fresh dependency installation and GitHub-hosted workflow run were not executed during this preparation; the results above apply to the existing environment only. Its package versions differ from the proposed CPU profile. The original training and robot environments were not modified.

本次在独立提交目录使用现有环境完成无硬件回归：432 项通过、1 项因缺少可选 Rerun 依赖跳过，另有 2 项子测试通过。94 个核心与运行时源码文件的哈希保持一致。未执行真机动作、模型实测、训练或仿真评测；全新环境安装和 GitHub CI 尚未实跑。
