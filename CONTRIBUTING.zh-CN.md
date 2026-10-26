# 贡献与提交说明

[English](CONTRIBUTING.md) · [首页](README.zh-CN.md)

使用独立 checkout 和环境。机器配置、凭据、权重、数据及运行记录放在源码树外；保留上游许可和原始文档。

## 检查

```bash
python scripts/check_submission.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$PWD/src:$PWD" python -m pytest -q \
  experiments/b601/tests experiments/libero/visual_ttt
```

依赖准备见 [environments/README.md](environments/README.md)。CPU 回归使用假设备与临时样本；普通 CI 不执行硬件预检、现场推理、仿真 rollout 或训练。模型和归一化修改需另做固定输入对照，硬件生命周期修改需补充失败路径回归及有人看护的现场验证。

## 准备提交

说明具体修改、兼容性影响和验证结果，保持中英文操作指南一致。提交前检查 `git diff --check`、暂存区差异、`git status`，运行 `scripts/check_submission.py`。实验记录源码、模型、数据版本，产物留在 Git 外。

本提交工程从记录的官方上游基线建立，不把机器专用中间提交纳入公开分支。确认目标仓库与发布范围后再添加目标 remote 并推送。
