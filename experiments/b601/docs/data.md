# Data protocol and preparation


The original `observation.state` contains measured motor angles in degrees.
The original `action` contains absolute commands before the B601 driver's
direction/scaling map `[1, 1, -1, -1, -1, 1, 6]`. Configuration is explicit in
[coordinate recipe](../configs/ee_delta.json), including the driver revision, URDF hash,
feature order, nominal gripper endpoints and calibration assumptions.

For each recorded row, compute FK for the measured joints and commanded
targets. The seven-dimensional action is:

```text
action[:3]  = p_command - p_measured                     # base frame, metres
action[3:6] = Log(R_command @ R_measured.T)               # axis-angle, radians
action[6]   = (gripper_command - closed) / (open - closed)

p_goal = p_current + action[:3]
R_goal = Exp(action[3:6]) @ R_current
```

Each action uses its own row's measured pose, including future rows sampled
within a chunk. Targets are recorded commands, not next-frame measured motion.
The rotation is a single axis-angle vector, not three Euler-angle differences.
This follows LIBERO's end-effector delta and left-composition convention without
inheriting its controller-specific 0.05 metre / 0.5 radian command scaling.

The new state is `[position(3), absolute_axis_angle(3), measured_gripper(1)]`.
Its gripper opening proxy applies the inverse motor scaling before using the
same nominal endpoints. Absolute orientations use the principal axis-angle
branch; no componentwise angle subtraction or temporal unwrapping is applied.
`EEDeltaCodec.state` is shared by offline conversion and future live observation
encoding. Do not append a fictitious second gripper position to match LIBERO.

The source vectors remain available, bit-for-bit in float32, as
`observation.joint_state` and `action.joint_command`. Videos, tasks, frame IDs and
timestamps retain their original content. No arm, gripper, robot parameter or
normalized value is clipped. Out-of-range recorded values are reported.

## Preparation

Obtain the B601 URDF identified in the recipe from the pinned
[driver revision](https://github.com/Seeed-Projects/lerobot-robot-seeed-b601/tree/0da1a1a5575905d77823227003ffee7672059780).
Run from the FastWAM checkout:

```bash
python scripts/prepare_ee_delta.py \
  --source /path/to/original_joint_dataset \
  --output /path/to/new_ee_delta_dataset \
  --config experiments/b601/configs/ee_delta.json \
  --urdf /path/to/00-arm-rs_asm-v3.urdf \
  --data-template configs/data/b601_2cam_ee_delta_lerobot_v30.yaml
```

The output must be new and outside the source dataset. Files are prepared in
a temporary sibling directory and published after validation. Source files are
hashed before and after conversion; videos are copied unchanged, not linked.
Numeric arrays are currently processed in memory.

```text
dataset/
  data/                    converted LeRobot parquet files
  meta/                    updated feature and episode statistics
  videos/                  independent copies of original videos
  assets/robot.urdf         pinned kinematics asset
  processing/              standalone conversion code and data template
  prepare.py               verify / recompute / rebuild entry point
  requirements.txt         standalone Python dependencies
  codec.json               joint mapping and physical coordinate contract
  pipeline.yml             preprocessing recipe, not a training-run recipe
  episode_split.json       explicit train/validation episode IDs
  norm_stats.json           frozen action AND state normalization
  data_config.yaml         FastWAM Hydra data configuration
  validation.json          reproducible full-dataset offline checks
  manifest.json            output and source SHA-256 fingerprints
  README.md                dataset usage and semantics
```

These operations do not require FastWAM or the original reference pipeline:

```bash
python /path/to/dataset/prepare.py
python /path/to/dataset/prepare.py --check-source
python /path/to/dataset/prepare.py --statistics-output /tmp/new_stats.json
python /path/to/dataset/prepare.py --rebuild /path/to/another_new_dataset
python /path/to/dataset/prepare.py --rebuild /path/to/new_dataset --source /path/to/moved_original
python /path/to/dataset/prepare.py --training-config-output /tmp/data.yaml
```

The default command verifies file fingerprints, recomputes the split and
statistics, and checks every row's encoding and pose reconstruction. Recomputed
statistics and rebuilds refuse to overwrite existing outputs. The optional
source check compares original recording files with their frozen fingerprints.

## Statistics and sampling

The provided recipe holds out one episode per task using seed 10222. This
recording yields 197 training episodes / 82,815 frames and 3 validation episodes
/ 1,127 frames (episodes 41, 136 and 199). Statistics fit training rows only,
once per valid frame. Neither validation data nor expanded/padded windows
contribute to normalization. `meta/stats.json` instead describes all recorded
rows, as required for the LeRobot dataset catalog.

All three tasks share one training dataset. The native epoch sampler shuffles
the complete training index pool, mixing tasks in proportion to their frame
counts. No task-specific filtering or equal-task oversampling is applied.

For these data, full training spans are about 1.2–3.2 times their q01–q99 spans,
so the supplied recipe chooses min/max scaling to cover the complete training
range. Both quantiles and moments are saved for inspection. Lower/span are
frozen in float32; a constant dimension uses unit span:

```text
normalized = 2 * (value - lower) / span - 1
value      = (normalized + 1) * span / 2 + lower
```

`EEAffineNormalizer` performs this map without clipping, including for values
outside the training range. It avoids changing the native normalizer's clamp
behavior. The artifact binds the source numeric data, representation, split
and statistics; train, validation and inference must share it.

At the recorded 30 Hz, the reader supplies 32 actions and 33 state positions.
It subsamples images at stride four for nine video frames. Front and side
cameras are resized to 224×224 and concatenated horizontally; no image flip
is introduced. Padding repeats the final state and gripper command and zeros
the first six action channels **before** normalization. Padding masks remain
available to the existing training path.

Use `configs/data/b601_2cam_ee_delta_lerobot_v30.yaml` with
`FASTWAM_DATASET_ROOT` pointing to the prepared directory, or export its resolved
data configuration using `prepare.py`. Configure model checkpoints, text cache,
optimizer and training schedule separately. This utility does not launch
training. Data from an earlier scaled/binary-gripper conversion or a joint-delta
adapter must not be used with these statistics.


## Validation and physical interpretation


```bash
python -m unittest discover -s experiments/b601/tests -t . -p test_ee_delta.py -v
python -m experiments.b601.check_ee_delta_dataset \
  --dataset /path/to/dataset --output /tmp/reader_check.json
```

The second command additionally requires SciPy for independent FK/rotation
checks. It tests all numeric rows, native train/validation loading, both camera
streams, padding, state/action round trips, out-of-range unclipped values and
preservation of the frozen statistics when the native loader exports them.
Model weights are not needed.

For inference, encode fresh motor feedback with the same codec, normalize it
with the frozen state map, and invert the action normalization using
`EEDeltaProcessor.decode_actions`. Each executed delta composes with that step's
measured pose; do not rebase the whole chunk to a single request-time pose or
add rotation-vector components. The deployment adapter below provides numerical
IK and a synchronous controller. Physical calibration and closed-loop validation
remain deployment work. The preparation code never moves a robot.

The mapping is grounded in the pinned driver and URDF, but physical joint-zero,
axis, TCP and jaw-width calibration have not been measured. Nominal gripper
fractions are not calibrated jaw widths. Recorded pre-driver commands may
exceed today's driver limits; their historical applied targets cannot be
recovered by assuming the current limits. Offline timestamp checks establish
the recording grid, not unrecorded camera/CAN latency or feedback validity.
