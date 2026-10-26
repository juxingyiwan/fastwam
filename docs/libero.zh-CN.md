# LIBERO

[English](libero.md) · [首页](../README.zh-CN.md)

使用[训练环境](../environments/README.md)，按[上游说明](../README.upstream.zh-CN.md)准备模型和数据。数据路径通过机器本地配置或 Hydra override 设置。原生任务配置保留在 `configs/task/archive/simulation/`，目录名用于兼容既有配置引用。

## 训练与评测

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/precompute_text_embeds.py task=archive/simulation/libero_uncond_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=archive/simulation/libero_uncond_2cam224_1e-4
python experiments/libero/run_libero_manager.py \
  task=archive/simulation/libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/compatible_checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

训练和评测在已分配的 GPU 上运行。必须另行提供兼容 checkpoint、保存的归一化统计、仿真器和资产。套件/任务、种子、回合数及推理选项见 `configs/sim_libero.yaml`。管理器可能调度多个 worker，先选择适合当前分配资源的小任务范围。

可选 visual-only TTT 见 [visual_ttt/README.md](../experiments/libero/visual_ttt/README.md)，其采集协议和分数应与原生推理分别报告。

每次运行记录 Git 提交、有效配置、数据/模型版本及随机种子。本次提交整理不新增仿真评测结果声明。
