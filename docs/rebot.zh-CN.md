# ReBot / B601 数据、训练与真机部署

[English](rebot.md) · [首页](../README.zh-CN.md) · [环境](../environments/README.md)

流程为：准备数据和环境 → 训练 → 导出模型包 → 实验机安装 → 离线检查 → 现场预检 → 真机运行。此适配器在本机进行 FastWAM 推理；其他策略使用的 SSH 隧道不属于该入口。所有命令从工程根目录运行。

## 数据与训练

[数据协议](../experiments/b601/docs/data.md)说明单位、标定、URDF 检查、训练/验证划分、冻结统计量以及七维 EE-delta 动作。不同动作表示的数据不能混用归一化统计。

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/prepare_ee_delta.py \
  --source /path/to/original_joint_dataset \
  --output /path/to/new_ee_delta_dataset \
  --config experiments/b601/configs/ee_delta.json \
  --urdf /path/to/00-arm-rs_asm-v3.urdf \
  --data-template configs/data/b601_2cam_ee_delta_lerobot_v30.yaml
```

复制 [paths.env.example](../experiments/b601/configs/paths.env.example) 到机器配置目录，编辑数据、模型、文本缓存及输出路径。激活训练环境后 source 编辑好的文件；Wan 和 ActionDiT 的准备见[上游模型说明](../README.upstream.zh-CN.md)。

```bash
source /path/to/edited/paths.env
python scripts/precompute_text_embeds_lerobot_v30.py task=b601_uncond_2cam224
bash experiments/b601/train.sh 8 task=b601_uncond_2cam224
# 独立单任务训练；支持 1、2、4、8 卡。
bash experiments/b601/train_single_task.sh cube 4
```

Slurm 环境中先进入已分配的计算节点再训练，保留现有作业和会话。续训需显式设置 `FASTWAM_RESUME_RUN`、`FASTWAM_RESUME_STEP`，检查 `experiments/b601/resume.sh`，选用完整保存的训练状态。单任务入口通过梯度累积保持有效 batch 128；样本数及最终步数由实际数据决定。

## 导出与安装

```bash
python -m experiments.b601.prepare_deployment \
  --dataset "$FASTWAM_DATASET_ROOT" \
  --run /path/to/training_run \
  --checkpoint /path/to/training_run/checkpoints/weights/step_003235.pt \
  --text-cache "$FASTWAM_TEXT_CACHE" \
  --vae /path/to/Wan2.2_VAE.pth \
  --output /path/to/new_model_bundle \
  --export-fixtures
```

输出目录必须不存在。模型包包含原生 checkpoint、model.yaml、codec、归一化统计、URDF、任务文本特征、VAE，以及可选离线样本。上面的 step 3235 是示例，实际应选择本次训练产生的 checkpoint。

完整传输模型包。在实验机将 [local.example.yml](../experiments/b601/configs/local.example.yml) 复制到 `~/.config/fastwam/deploy.yml`，配置模型名、相机设备、所有客户端共用的机器人锁和输出目录，再安装：

```bash
python -m experiments.b601.install_model \
  --source /path/to/received_bundle --name 5epoch \
  --config "$HOME/.config/fastwam/deploy.yml"
```

安装器校验资产、更新指定模型路径，不改变当前默认模型。权重、数据和运行记录均放在 Git 工程外。

## 检查与运行

激活真机环境；包装脚本会设置仓库源码路径。下面两条命令不连接机器人：

```bash
bash experiments/b601/local.sh models
bash experiments/b601/local.sh config cube --model 5epoch --rtc false
```

准备兼容模型包后，离线 `check` 加载模型与验收样本，但不打开 CAN 或相机：

```bash
bash experiments/b601/local.sh check cube --model 5epoch --rtc false
```

现场有人看护时，确认标定、相机映射、共用锁、急停和完整运动空间，再预检及运行：

```bash
bash experiments/b601/local.sh preflight cube --model 5epoch --rtc false
bash experiments/b601/local.sh run cube --model 5epoch --rtc false --duration 300
```

其他任务别名为 `pen` 和 `drawer`，模型必须支持对应任务。`preflight` 会打开硬件，不用于 CI。只有实际核对 front/side 后才能设置 `camera_mapping_verified: true`。

## RTC、日志与回零

RTC 默认关闭。先用 `config cube --model 5epoch --rtc true` 查看配置，再保持同一选项依次执行离线检查和现场预检。同步推理的 backend 为 `local://fastwam`，RTC 使用独立模型进程并记录 backend 和 PID。预测长度为 32，默认同步执行 16 步；完整默认参数见 [runtime/config.py](../experiments/b601/runtime/config.py)。

每次运行保存有效配置 `config.json`、源码哈希和依赖版本 `manifest.json`、模型身份 `model.json`、运行及退出状态 `report.json`。检查动作拒绝、限位裁剪、相机恢复与回零结果；`aborted` 不是任务成功率结论。

原生回零包含反馈到位判定和夹爪精调，轨迹不保证避障。实现历史见 [NATIVE_HOMING.md](../experiments/b601/NATIVE_HOMING.md)。硬件入口核对安装驱动与 [driver-source.json](../experiments/b601/runtime/driver-source.json) 的文件指纹，升级驱动时应同步评估该契约。
