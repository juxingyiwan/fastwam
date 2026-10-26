# Environment profiles / 环境说明

Training, robot deployment and CPU regression checks are separate profiles. Do not overwrite a working robot environment to match a training environment.

| Profile | Observed Python / PyTorch | Use |
| --- | --- | --- |
| Training | 3.10.12 / 2.7.1+cu128 | Upstream training and simulator model process |
| Robot | 3.12.14 / 2.11.0 | Audited B601 local inference and control |
| CPU tests | Python 3.10; CPU or CUDA torch with GPUs hidden | Fake hardware, geometry, RTC and configuration regressions |

These observations are not complete dependency locks or proof that all versions are interchangeable. The original `pyproject.toml` dependency declarations are retained. In particular, its training pins must not be installed over an already validated robot environment. With an independently provisioned compatible environment, link only the source:

```bash
python -m pip install --no-deps -e .
```

For a fresh upstream training environment, follow [upstream setup](../README.upstream.md#environment-setup). The robot hardware overlay is [requirements-deploy.txt](../experiments/b601/requirements-deploy.txt). The driver source hash contract is part of the adapter; installing its version alone is insufficient if files were patched locally.

For CPU checks, create an isolated environment and install CPU PyTorch before the regression dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/requirements-cpu.txt
python -m pip install --no-deps -e .
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src:$PWD" \
  python -m pytest -q experiments/b601/tests experiments/libero/visual_ttt
```

训练端和真机端已观测到 Python、PyTorch、Transformers、NumPy 的版本差异。不要用训练依赖直接覆盖可工作的真机环境。这里保留原依赖声明，CPU 测试使用独立环境；完整安装重现、GPU 推理和现场运行属于单独验证范围。
