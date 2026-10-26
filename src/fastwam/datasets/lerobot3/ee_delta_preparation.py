"""Build, verify and reproduce self-contained LeRobot v3 EE-delta bundles."""

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

if __package__:
    from .ee_delta import (
        ACTION_NAMES,
        REPRESENTATION,
        STATE_NAMES,
        EEDeltaCodec,
        compose_delta,
    )
else:
    from ee_delta import (
        ACTION_NAMES,
        REPRESENTATION,
        STATE_NAMES,
        EEDeltaCodec,
        compose_delta,
    )


VECTOR_KEYS = (
    "action",
    "observation.state",
    "action.joint_command",
    "observation.joint_state",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inventory(root, directories=("data", "meta", "videos")):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for directory in directories
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
    }


def numeric_fingerprints(root):
    root = Path(root)
    paths = [root / "meta/info.json", *sorted((root / "data").rglob("*.parquet"))]
    if len(paths) < 2:
        raise ValueError("No numeric data found")
    return {path.relative_to(root).as_posix(): file_hash(path) for path in paths}


def read_rows(root):
    tables = [
        pq.read_table(path) for path in sorted((Path(root) / "data").rglob("*.parquet"))
    ]
    if not tables:
        raise ValueError("No data parquet files found")
    table = pa.concat_tables(tables)
    names = [key for key in VECTOR_KEYS if key in table.column_names]
    names += ["episode_index", "frame_index", "task_index", "timestamp", "index"]
    return {
        key: np.asarray(
            table[key].to_pylist(),
            dtype=np.float32 if key in VECTOR_KEYS or key == "timestamp" else np.int64,
        )
        for key in names
    }


def validate_rows(rows, info):
    count = len(rows["index"])
    if count != info["total_frames"] or not np.array_equal(
        rows["index"], np.arange(count)
    ):
        raise ValueError("Numeric rows must be complete and in global index order")
    ids = np.unique(rows["episode_index"])
    if not np.array_equal(ids, np.arange(info["total_episodes"])):
        raise ValueError("Numeric episodes disagree with metadata")
    for key in VECTOR_KEYS:
        if key in rows and (
            rows[key].shape != (count, 7) or not np.isfinite(rows[key]).all()
        ):
            raise ValueError(f"Invalid seven-dimensional feature: {key}")
    max_time_error = 0.0
    for episode in ids:
        selected = rows["episode_index"] == episode
        frames, times = rows["frame_index"][selected], rows["timestamp"][selected]
        if (
            not np.array_equal(frames, np.arange(len(frames)))
            or len(np.unique(rows["task_index"][selected])) != 1
        ):
            raise ValueError(f"Invalid frame or task order in episode {episode}")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Nonmonotonic timestamps in episode {episode}")
        error = float(np.max(np.abs(times.astype(np.float64) - frames / info["fps"])))
        max_time_error = max(max_time_error, error)
    if max_time_error > 1e-4:
        raise ValueError("Timestamps do not follow the declared frame rate")
    return max_time_error


def split_episodes(rows, seed, validation_per_task):
    if (
        type(seed) is not int
        or seed < 0
        or type(validation_per_task) is not int
        or validation_per_task < 1
    ):
        raise ValueError("Expected a nonnegative seed and positive validation count")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for task in np.unique(rows["task_index"]):
        ids = np.unique(rows["episode_index"][rows["task_index"] == task])
        if len(ids) <= validation_per_task:
            raise ValueError("Every task must retain at least one training episode")
        ids = rng.permutation(ids).tolist()
        val.extend(ids[:validation_per_task])
        train.extend(ids[validation_per_task:])
    return {
        "seed": seed,
        "validation_episodes_per_task": validation_per_task,
        "train_episodes": sorted(train),
        "validation_episodes": sorted(val),
    }


def descriptive_stats(values):
    values = np.asarray(values, dtype=np.float32)
    low, high = np.quantile(values, [0.01, 0.99], axis=0).astype(np.float32)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0, dtype=np.float64).tolist(),
        "std": values.std(0, dtype=np.float64).tolist(),
        "q01": low.tolist(),
        "q99": high.tolist(),
        "count": [len(values)],
    }


def fit_statistics(values, mode):
    stats = descriptive_stats(values)
    lower, upper = (
        (stats["min"], stats["max"])
        if mode == "min/max"
        else (stats["q01"], stats["q99"])
    )
    if mode not in {"min/max", "q01/q99"}:
        raise ValueError("Unsupported normalization mode")
    lower, upper = np.asarray(lower, np.float32), np.asarray(upper, np.float32)
    varying = upper > lower
    lower = np.where(varying, lower, np.asarray(stats["min"], np.float32))
    span = np.where(
        varying, upper - lower, np.asarray(stats["max"], np.float32) - lower
    )
    span = np.where(span > 0, span, np.float32(1))
    return {
        **{f"global_{key}": value for key, value in stats.items() if key != "count"},
        "global_count": stats["count"][0],
        "global_lower": lower.tolist(),
        "global_span": span.tolist(),
    }


def frozen_statistics(root, rows, config, split):
    selected = np.isin(rows["episode_index"], split["train_episodes"])
    stats = {
        name: {"default": fit_statistics(rows[key][selected], config["normalization"])}
        for name, key in (("action", "action"), ("state", "observation.state"))
    }
    return {
        **stats,
        "ee_delta_contract": {
            "representation": REPRESENTATION,
            "anchor": "same_row_measured_pose",
            "rotation": "R_target = Exp(action[3:6]) @ R_measured",
            "units": ["m", "m", "m", "rad", "rad", "rad", "opening_fraction"],
            "statistics_scope": "training_episodes_valid_rows_once",
            "normalization": config["normalization"],
            "clip": False,
            "horizon": config["horizon"],
            "fps": config["expected_fps"],
            "padding": "zero_six_deltas_repeat_last_gripper_before_normalization",
            "split": split,
            "codec_sha256": digest(config),
            "dataset_files": numeric_fingerprints(root),
            "statistics_sha256": digest(stats),
        },
    }


def validate_frozen_statistics(root, artifact):
    root = Path(root)
    contract = artifact["ee_delta_contract"]
    info = json.loads((root / "meta/info.json").read_text())
    config = json.loads((root / "codec.json").read_text())
    train, val = (
        contract["split"][key] for key in ("train_episodes", "validation_episodes")
    )
    if (
        contract["representation"] != REPRESENTATION
        or contract["statistics_scope"] != "training_episodes_valid_rows_once"
        or contract["normalization"] != config["normalization"]
        or contract["clip"] is not False
        or contract["dataset_files"] != numeric_fingerprints(root)
        or contract["statistics_sha256"]
        != digest({key: artifact[key] for key in ("action", "state")})
        or contract["codec_sha256"] != digest(config)
        or contract["fps"] != info["fps"]
        or contract["fps"] != config["expected_fps"]
        or contract["horizon"] != config["horizon"]
        or contract["anchor"] != "same_row_measured_pose"
        or contract["padding"]
        != "zero_six_deltas_repeat_last_gripper_before_normalization"
        or sorted(train + val) != list(range(info["total_episodes"]))
        or len(set(train + val)) != len(train + val)
        or not train
        or not val
    ):
        raise ValueError("Frozen EE statistics, dataset or split mismatch")
    return contract


def _replace(table, key, values, dtype):
    column = pa.array(values, type=dtype)
    index = table.schema.get_field_index(key)
    return (
        table.append_column(key, column)
        if index < 0
        else table.set_column(index, key, column)
    )


def rewrite_metadata(stage, info, rows, old_features):
    names = {"action": ACTION_NAMES, "observation.state": STATE_NAMES}
    for key, raw_key in (
        ("action", "action.joint_command"),
        ("observation.state", "observation.joint_state"),
    ):
        info["features"][raw_key] = copy.deepcopy(old_features[key])
        info["features"][key] = {"dtype": "float32", "shape": [7], "names": names[key]}
    # LeRobot's catalog range describes all available rows; the adapter enforces
    # the noncontiguous training/validation split in episode_split.json.
    write_json(stage / "meta/info.json", info)
    stats_path = stage / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats.update({key: descriptive_stats(rows[key]) for key in VECTOR_KEYS})
    write_json(stats_path, stats)
    seen = set()
    for path in sorted((stage / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        ids = table["episode_index"].to_pylist()
        episode_stats = {}
        for episode, length in zip(ids, table["length"].to_pylist(), strict=True):
            mask = rows["episode_index"] == episode
            if episode in seen or int(mask.sum()) != length:
                raise ValueError("Invalid episode metadata")
            seen.add(episode)
            episode_stats[episode] = {
                key: descriptive_stats(rows[key][mask]) for key in VECTOR_KEYS
            }
        table = table.drop(
            [
                key
                for key in table.column_names
                if any(key.startswith(f"stats/{field}/") for field in VECTOR_KEYS)
            ]
        )
        for field in VECTOR_KEYS:
            for name in stats[field]:
                dtype = pa.list_(pa.int64() if name == "count" else pa.float64())
                table = _replace(
                    table,
                    f"stats/{field}/{name}",
                    [episode_stats[ep][field][name] for ep in ids],
                    dtype,
                )
        table = table.replace_schema_metadata(None)
        pq.write_table(table, path)
    if seen != set(rows["episode_index"].tolist()):
        raise ValueError("Episode metadata is incomplete")


def audit_rows(rows, info, config, codec, artifact):
    time_error = validate_rows(rows, info)
    expected_state, expected_action = codec.encode(
        rows["observation.joint_state"], rows["action.joint_command"]
    )
    np.testing.assert_array_equal(rows["observation.state"], expected_state)
    np.testing.assert_array_equal(rows["action"], expected_action)
    measured = codec.fk(
        np.deg2rad(rows["observation.joint_state"][:, :6].astype(np.float64))
        + codec.offset
    )
    commanded = rows["action.joint_command"] * codec.scale
    target = codec.fk(np.deg2rad(commanded[:, :6]) + codec.offset)
    position, rotation = compose_delta(
        measured[:, :3, 3], measured[:, :3, :3], rows["action"]
    )
    position_error = float(np.max(np.linalg.norm(position - target[:, :3, 3], axis=-1)))
    rotation_error = float(np.max(np.abs(rotation - target[:, :3, :3])))
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise ValueError("Physical pose reconstruction failed")
    limits = np.asarray(config["driver_motor_limits_deg"])
    distributions = {}
    for field, key in (("action", "action"), ("state", "observation.state")):
        saved = artifact[field]["default"]
        lo, span = (
            np.asarray(saved["global_lower"], np.float32),
            np.asarray(saved["global_span"], np.float32),
        )
        normalized = 2 * (rows[key] - lo) / span - 1
        inverse = (normalized + 1) * span / 2 + lo
        err = float(np.max(np.abs(inverse - rows[key])))
        if err > 1e-6:
            raise ValueError("Normalization round trip failed")
        qspan = np.asarray(saved["global_q99"]) - saved["global_q01"]
        full_span = np.asarray(saved["global_max"]) - saved["global_min"]
        distributions[field] = {
            "range_over_q01_q99_span": np.divide(
                full_span, qspan, out=np.ones_like(qspan), where=qspan > 0
            ).tolist(),
            "max_abs_normalized": np.abs(normalized).max(0).tolist(),
            "outside_training_range_per_dim": (
                (rows[key] < np.asarray(saved["global_min"], np.float32))
                | (rows[key] > np.asarray(saved["global_max"], np.float32))
            )
            .sum(0)
            .tolist(),
            "max_round_trip_error": err,
        }
    task_counts = []
    for task in np.unique(rows["task_index"]):
        mask = rows["task_index"] == task
        task_counts.append(
            {
                "task_index": int(task),
                "episodes": len(np.unique(rows["episode_index"][mask])),
                "frames": int(mask.sum()),
            }
        )
    split = artifact["ee_delta_contract"]["split"]
    return {
        "status": "passed_offline_checks",
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "task_counts": task_counts,
        "train_frames": int(
            np.isin(rows["episode_index"], split["train_episodes"]).sum()
        ),
        "validation_frames": int(
            np.isin(rows["episode_index"], split["validation_episodes"]).sum()
        ),
        "max_timestamp_grid_error_s": time_error,
        "max_position_reconstruction_error_m": position_error,
        "max_rotation_matrix_reconstruction_error": rotation_error,
        "normalization": distributions,
        "command_outside_current_driver_limits_per_joint": (
            (commanded < limits[:, 0]) | (commanded > limits[:, 1])
        )
        .sum(0)
        .tolist(),
        "gripper_action_outside_nominal_fraction": int(
            ((rows["action"][:, 6] < 0) | (rows["action"][:, 6] > 1)).sum()
        ),
        "gripper_state_outside_nominal_fraction": int(
            (
                (rows["observation.state"][:, 6] < 0)
                | (rows["observation.state"][:, 6] > 1)
            ).sum()
        ),
        "absolute_orientation_near_pi_rows": int(
            (
                np.linalg.norm(rows["observation.state"][:, 3:6], axis=-1)
                > np.pi - 0.01
            ).sum()
        ),
        "calibration_status": config["calibration_status"],
        "timing_limit": "Frame timestamps are verified; camera/CAN hardware latency was not recorded and cannot be established offline.",
    }


def build_bundle(source, output, config_path, urdf, data_template):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Output must be a new directory outside the original dataset")
    config = json.loads(Path(config_path).read_text())
    if (
        config["horizon"] != 32
        or config["target_policy"] != "recorded_pre_driver_commands_without_clipping"
    ):
        raise ValueError("This recipe uses H32 and recorded, unclipped target commands")
    if file_hash(urdf) != config["urdf_sha256"]:
        raise ValueError("URDF fingerprint differs from the conversion recipe")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != config["expected_fps"]:
        raise ValueError("Expected LeRobot v3 data at the configured recording rate")
    if any(feature["dtype"] == "image" for feature in info["features"].values()):
        raise ValueError("This recipe handles video-backed datasets")
    for key in ("action", "observation.state"):
        if info["features"][key] != {
            "dtype": "float32",
            "names": config["recorded_feature_names"],
            "shape": [7],
        }:
            raise ValueError(f"Unexpected original feature order or type: {key}")
    if any(key in info["features"] for key in VECTOR_KEYS[2:]):
        raise ValueError("Source is already converted; use the original joint dataset")
    original = inventory(source)
    rows = read_rows(source)
    validate_rows(rows, info)
    split = split_episodes(
        rows, config["split_seed"], config["validation_episodes_per_task"]
    )
    codec = EEDeltaCodec(urdf, config)
    rows["observation.joint_state"], rows["action.joint_command"] = (
        rows["observation.state"],
        rows["action"],
    )
    rows["observation.state"], rows["action"] = codec.encode(
        rows["observation.joint_state"], rows["action.joint_command"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        shutil.copytree(source / "meta", stage / "meta")
        offset = 0
        for path in sorted((source / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            for key in VECTOR_KEYS:
                table = _replace(
                    table,
                    key,
                    rows[key][offset : offset + len(table)].tolist(),
                    pa.list_(pa.float32(), 7),
                )
            offset += len(table)
            target = stage / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.replace_schema_metadata(None), target)
        rewrite_metadata(stage, info, rows, copy.deepcopy(info["features"]))
        shutil.copytree(source / "videos", stage / "videos")
        (stage / "assets").mkdir()
        shutil.copy2(urdf, stage / "assets/robot.urdf")
        write_json(stage / "codec.json", config)
        write_json(stage / "episode_split.json", split)
        artifact = frozen_statistics(stage, rows, config, split)
        write_json(stage / "norm_stats.json", artifact)
        report = audit_rows(rows, info, config, codec, artifact)
        write_json(stage / "validation.json", report)
        _write_bundle_files(stage, source, config, artifact, data_template)
        if inventory(source) != original:
            raise ValueError("Original data changed during preparation")
        # Confirm copied videos and every untransformed numeric column.
        for relative, sha in original.items():
            if relative.startswith("videos/") and file_hash(stage / relative) != sha:
                raise ValueError(f"Video copy differs: {relative}")
            if relative.startswith("data/"):
                before, after = (
                    pq.read_table(source / relative),
                    pq.read_table(stage / relative),
                )
                for key in before.column_names:
                    target_key = {
                        "action": "action.joint_command",
                        "observation.state": "observation.joint_state",
                    }.get(key, key)
                    if before[key].to_pylist() != after[target_key].to_pylist():
                        raise ValueError(f"Original column changed: {key}")
        files = {
            p.relative_to(stage).as_posix(): file_hash(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        write_json(
            stage / "manifest.json",
            {
                "version": 2,
                "source": str(source),
                "source_files": original,
                "files": files,
            },
        )
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return report


def verify_bundle(root, check_source=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or file_hash(path) != expected:
            raise ValueError(f"Bundle fingerprint mismatch: {relative}")
    if check_source and inventory(manifest["source"]) != manifest["source_files"]:
        raise ValueError("Original source fingerprint mismatch")
    artifact = json.loads((root / "norm_stats.json").read_text())
    contract = validate_frozen_statistics(root, artifact)
    rows = read_rows(root)
    config = json.loads((root / "codec.json").read_text())
    if (
        split_episodes(
            rows, config["split_seed"], config["validation_episodes_per_task"]
        )
        != contract["split"]
    ):
        raise ValueError("Episode split cannot be reproduced")
    recomputed = frozen_statistics(root, rows, config, contract["split"])
    if artifact != recomputed:
        raise ValueError("Frozen statistics cannot be reproduced from training rows")
    codec = EEDeltaCodec(root / "assets/robot.urdf", config)
    report = audit_rows(
        rows, json.loads((root / "meta/info.json").read_text()), config, codec, artifact
    )
    if report != json.loads((root / "validation.json").read_text()):
        raise ValueError("Validation report cannot be reproduced")
    return report, recomputed


def _write_bundle_files(stage, source, config, artifact, data_template):
    processing = stage / "processing"
    processing.mkdir()
    for name in ("ee_delta.py", "ee_delta_preparation.py"):
        shutil.copy2(Path(__file__).with_name(name), processing / name)
    shutil.copy2(data_template, processing / "data_template.yaml")
    (stage / "prepare.py").write_text(
        '"""Verify, recompute or rebuild this independent dataset bundle."""\n'
        "import sys\nfrom pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(ROOT / "processing"))\n'
        "from ee_delta_preparation import bundle_main\n\n"
        'if __name__ == "__main__":\n    bundle_main(ROOT)\n',
        encoding="utf-8",
    )
    (stage / "requirements.txt").write_text(
        "numpy==1.26.4\npyarrow==24.0.0\nPyYAML>=6,<7\n", encoding="utf-8"
    )
    recipe = {
        "bundle": {
            "version": REPRESENTATION,
            "entrypoint": "prepare.py",
            "runtime_dependencies": "requirements.txt",
        },
        "data": {
            "source": str(source),
            "format": "LeRobot v3.0",
            "fps": config["expected_fps"],
            "horizon": config["horizon"],
        },
        "semantics": {
            "codec": "codec.json",
            "urdf": "assets/robot.urdf",
            "anchor": "same-row measured pose",
            "translation": "p_target - p_measured, base frame, metres",
            "rotation": "Log(R_target @ R_measured.T), axis-angle radians",
            "state": "base position + absolute axis-angle + measured gripper opening proxy",
            "gripper": "continuous fraction; nominal 0 closed, 1 open; unclipped",
            "target_policy": config["target_policy"],
        },
        "normalization": {
            "file": "norm_stats.json",
            "sha256": file_hash(stage / "norm_stats.json"),
            "mode": config["normalization"],
            "scope": "training episodes, one count per valid row",
            "formula": "z = 2 * (x - lower) / span - 1; inverse = (z + 1) * span / 2 + lower",
            "clip": False,
            "padding_in_fit": False,
            "split": "episode_split.json",
        },
        "images": {
            "order": ["front", "side"],
            "per_camera_training_size": [224, 224],
            "flip": False,
            "copied_unchanged": True,
        },
        "calibration_status": config["calibration_status"],
    }
    (stage / "pipeline.yml").write_text(
        yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8"
    )
    _write_data_config(
        stage,
        stage / "data_config.yaml",
        data_template,
        bundle_root="${oc.env:FASTWAM_DATASET_ROOT}",
    )
    split = artifact["ee_delta_contract"]["split"]
    (stage / "README.md").write_text(_bundle_readme(config, split), encoding="utf-8")


def _write_data_config(root, output, template, bundle_root=None):
    config = yaml.safe_load(Path(template).read_text())
    codec_config = json.loads((Path(root) / "codec.json").read_text())
    for name in ("train", "val"):
        config[name] = copy.deepcopy(config[name])
        config[name]["processor"]["norm_default_mode"] = codec_config["normalization"]
        config[name]["dataset_dirs"] = [bundle_root or str(Path(root).resolve())]
        config[name]["pretrained_norm_stats"] = (
            bundle_root or str(Path(root).resolve())
        ) + "/norm_stats.json"
    Path(output).write_text(
        yaml.safe_dump({"data": config}, sort_keys=False), encoding="utf-8"
    )


def _bundle_readme(config, split):
    return f"""# FastWAM B601 末端增量数据集（v2 / H32）

本目录是一份完整、独立的数据处理包。`processing/` 保存转换与统计代码的冻结副本；
无需 Wall-X、参考数据集目录或 FastWAM 安装即可校验、重新计算统计、从原始数据重建。
运行依赖见 `requirements.txt`；训练适配器在 FastWAM 项目中。

## 数据语义

| 内容 | 原始记录 | 本目录 |
| --- | --- | --- |
| action 前 3 维 | 关节绝对指令的一部分 | 基座系位置增量，米 |
| action 第 4–6 维 | 关节绝对指令的一部分 | 基座系轴角增量，弧度 |
| action 第 7 维 | 夹爪绝对指令 | 连续开度代理 `(command - 0) / 45` |
| observation.state | 7 维实测电机角度（度） | 末端位置 3 + 绝对轴角 3 + 实测开度代理 1 |
| 原始关节量 | action / observation.state | 完整保留在 action.joint_command / observation.joint_state |
| 图像、任务、时间戳 | 原始记录 | 原样保存；不翻转、不改帧率 |

每一帧以本帧实测姿态为锚：`dp = p_cmd - p_meas`，
`dr = Log(R_cmd @ R_meas.T)`；逆变换为 `p_goal = p_meas + dp`、
`R_goal = Exp(dr) @ R_meas`。dr 是一个三维轴角向量，不能当作 Euler 三角差。
这与 LIBERO 的末端增量/左乘旋转含义相同；不沿用模拟器的 0.05 m / 0.5 rad 控制器缩放。
数据文件中保存物理量；训练时仅做下面的统计归一化。

命令先乘 `[1, 1, -1, -1, -1, 1, 6]` 得电机目标；反馈已在电机坐标中，不能再次反号。
FK 输入前六轴的弧度值。URDF、关节顺序、零位假设、驱动版本和哈希都在 `codec.json`。
目标使用记录的命令，不替换为下一帧反馈，也不根据当前驱动限制事后裁剪。
动作、机器人状态与参数均不做 clip，URDF 和配置参数按原值使用。
连续夹爪不二值化、不裁剪；0/45 是驱动名义端点，不是已标定的物理夹爪宽度。

## 统计与采样

- {len(split["train_episodes"]) + len(split["validation_episodes"])} 条轨迹按任务分层：{len(split["train_episodes"])} 训练、{len(split["validation_episodes"])} 验证；验证轨迹 {split["validation_episodes"]}，种子 {split["seed"]}。
- `norm_stats.json` 同时含 action 与 state 的训练集统计，按有效帧等权计算一次，排除验证帧、窗口重复和 padding。
- `meta/stats.json` 与 episode metadata 是全量描述统计，不能替代训练冻结统计。
- 本版使用 `{config["normalization"]}`：训练集全幅相对 q01–q99 跨度没有数量级膨胀，选择保留全部训练取值；量化比较见 `validation.json`。
- 保存 FP32 lower/span，`z = 2*(x-lower)/span-1`；常量维 span=1。训练、验证、推理共用，均不截断超范围值。
- 原始 30 Hz；32 动作、33 状态位置，视频每 4 帧采样得到 9 帧。两路图像 front/side，分别缩至 224×224 后横向拼接。
- 末尾越界动作的前六维在物理空间置零，夹爪重复末值，再归一化。有效末帧动作保留；掩码随样本返回。
- 所有绝对姿态与增量旋转均为轴角弧度；绝对轴角使用 [0, pi] 主值，不能逐分量累加。

## 使用与复现

在任意工作目录运行（只需上述 Python 依赖）：

```bash
python /path/to/this_dataset/prepare.py
python /path/to/this_dataset/prepare.py --check-source
python /path/to/this_dataset/prepare.py --statistics-output /tmp/new_norm_stats.json
python /path/to/this_dataset/prepare.py --rebuild /path/to/new_dataset
python /path/to/this_dataset/prepare.py --rebuild /path/to/new_dataset --source /path/to/original_dataset
python /path/to/this_dataset/prepare.py --training-config-output /tmp/fastwam_data.yaml
```

默认校验全包哈希、逐帧转换与位姿重建、冻结划分、norm_stats 重算一致性。
统计重算只写新文件；重建只写新目录，不覆盖原包或原始数据。`--check-source` 额外验证原始文件未变。
`data_config.yaml` 是 FastWAM Hydra `data` 子配置：先设置 `FASTWAM_DATASET_ROOT` 为本目录绝对路径，
或用上述命令导出带绝对路径的配置。缓存和训练参数由训练配置另行指定。
`processing/data_template.yaml` 可用于重建；`manifest.json` 冻结所有交付文件与原始数据哈希。

## 推理边界

使用同一 `EEDeltaCodec.state` 将实测电机反馈编码为 state，再应用冻结 state normalizer。
模型动作先反归一化得到米/弧度/连续开度。执行 chunk 中的每一步时，用该步的新鲜反馈组合目标位姿；
不能全部加到请求时的单一姿态，也不能对旋转三个分量直接相加。
随后仍需真机 IK、连续解选择、关节与工作空间限制、夹爪驱动转换；本数据包不控制机器人。

离线检查不能证明真机零位、轴向与 TCP 已标定。当前状态：{config["calibration_status"]}
记录只支持校验标称时间网格，不能验证未保存的相机/CAN 物理延迟；驱动反馈缺失时的零值也无法仅凭数据可靠判别。
`validation.json` 如实列出超当前驱动界限、超名义夹爪范围的样本，不静默修改或删除。
"""


def prepare_main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "config", "urdf", "data-template"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = build_bundle(
        args.source, args.output, args.config, args.urdf, args.data_template
    )
    print(json.dumps({"output": str(args.output), **report}, indent=2))


def bundle_main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--statistics-output", type=Path)
    operation.add_argument("--rebuild", type=Path)
    operation.add_argument("--training-config-output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    for destination in (
        args.statistics_output,
        args.rebuild,
        args.training_config_output,
    ):
        if destination is not None and destination.exists():
            parser.error(f"Refusing to overwrite {destination}")
    if args.source is not None and args.rebuild is None:
        parser.error("--source is only used with --rebuild")
    report, artifact = verify_bundle(root, args.check_source)
    if args.statistics_output:
        with args.statistics_output.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    if args.rebuild:
        manifest = json.loads((root / "manifest.json").read_text())
        report = build_bundle(
            args.source or manifest["source"],
            args.rebuild,
            root / "codec.json",
            root / "assets/robot.urdf",
            root / "processing/data_template.yaml",
        )
    if args.training_config_output:
        _write_data_config(
            root, args.training_config_output, root / "processing/data_template.yaml"
        )
    print(json.dumps({"bundle": str(root), "verified": True, **report}, indent=2))
