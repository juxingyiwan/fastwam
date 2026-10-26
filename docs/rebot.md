# ReBot / B601

[中文](rebot.zh-CN.md) · [Home](../README.md) · [Environments](../environments/README.md)

Use the native FastWAM checkpoint with two cameras and a B601 arm. This adapter performs local inference; an unrelated SSH policy tunnel is not its serving backend. Use source-checkout commands from the project root.

## Data and training

The [data guide](../experiments/b601/docs/data.md) defines units, calibration, train/validation splits, URDF checks, frozen normalization and the seven-dimensional EE-delta representation. Do not reuse statistics from a different action representation.

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/prepare_ee_delta.py \
  --source /path/to/original_joint_dataset \
  --output /path/to/new_ee_delta_dataset \
  --config experiments/b601/configs/ee_delta.json \
  --urdf /path/to/00-arm-rs_asm-v3.urdf \
  --data-template configs/data/b601_2cam_ee_delta_lerobot_v30.yaml
```

Copy [paths.env.example](../experiments/b601/configs/paths.env.example) to your machine configuration directory, edit it and source that copy after activating the training environment. It selects dataset, model, text-cache and output paths. Prepare the Wan and ActionDiT assets using the [upstream model instructions](../README.upstream.md#model-preparation).

```bash
source /path/to/edited/paths.env
python scripts/precompute_text_embeds_lerobot_v30.py task=b601_uncond_2cam224
bash experiments/b601/train.sh 8 task=b601_uncond_2cam224
# A separate single-task run; supported GPU counts: 1, 2, 4, 8.
bash experiments/b601/train_single_task.sh cube 4
```

On Slurm, enter your allocated compute node before training. Keep existing jobs and sessions intact. For continuation, explicitly set `FASTWAM_RESUME_RUN` and `FASTWAM_RESUME_STEP` and inspect `experiments/b601/resume.sh`; use a completed full-state checkpoint. `train_single_task.sh` preserves effective batch 128 using accumulation. Dataset sizes and final step counts depend on the supplied data.

## Export and install a model

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

The output directory must be new. The bundle includes `checkpoint.pt`, `model.yaml`, `codec.json`, normalization statistics, URDF, task contexts, VAE and optional offline fixtures. Step 3235 above is an example; select the checkpoint actually produced by your run.

Transfer the complete bundle through your normal file-transfer channel. On the robot host, first copy and edit [local.example.yml](../experiments/b601/configs/local.example.yml) at `~/.config/fastwam/deploy.yml`. Set model paths, camera device IDs, shared robot lock paths and the output root. Then install:

```bash
python -m experiments.b601.install_model \
  --source /path/to/received_bundle --name 5epoch \
  --config "$HOME/.config/fastwam/deploy.yml"
```

The installer verifies files and updates the named checkpoint entry without changing the selected default. Store model bundles and run recordings under your cache/output root, outside Git.

## Inspect, validate and operate

Activate the robot environment. The wrapper sets the repository source path. These commands inspect configuration without connecting to the robot:

```bash
bash experiments/b601/local.sh models
bash experiments/b601/local.sh config cube --model 5epoch --rtc false
```

With a compatible installed bundle, `check` runs offline model/fixture validation. It uses model resources but does not open CAN or cameras:

```bash
bash experiments/b601/local.sh check cube --model 5epoch --rtc false
```

For an attended robot session, verify calibration, camera mapping, shared locks, emergency stop and the arm's entire path, then perform hardware preflight before live execution:

```bash
bash experiments/b601/local.sh preflight cube --model 5epoch --rtc false
bash experiments/b601/local.sh run cube --model 5epoch --rtc false --duration 300
```

The other task aliases are `pen` and `drawer`; the selected checkpoint must support the requested instruction. `preflight` opens hardware, so it is not a CI command. Configure `camera_mapping_verified: true` only after checking the physical front/side assignment.

## RTC and runtime records

RTC defaults to disabled. To inspect the guided mode, use `config cube --model 5epoch --rtc true`, then repeat offline validation and attended preflight with the same option. Synchronous inference uses `local://fastwam`; RTC uses a separate model process and records its backend and PID. Prediction horizon is 32 and the default synchronous execution horizon is 16. Defaults are defined in [runtime/config.py](../experiments/b601/runtime/config.py).

Each run stores effective `config.json`, source hashes and dependency versions in `manifest.json`, model identity in `model.json`, and outcome/shutdown details in `report.json`. Review rejected actions, limit clipping, camera recovery and homing status rather than interpreting an `aborted` run as a task-success result.

Homing performs measured arrival checks and gripper refinement; the native trajectory does not guarantee obstacle clearance. See [NATIVE_HOMING.md](../experiments/b601/NATIVE_HOMING.md). The pinned driver files are checked against [driver-source.json](../experiments/b601/runtime/driver-source.json) before hardware use. Keep the driver and its source fingerprint contract aligned.
