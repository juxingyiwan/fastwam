"""Package a native checkpoint and held-out observations for offline deployment."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .fastwam_policy.observation import ObservationAdapter, file_sha256
from .fastwam_policy.policy import MODEL_ID, context_path


def decode_frame(path, timestamp, fps):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(
            max(0, int(timestamp / stream.time_base)), stream=stream, backward=True
        )
        best, error = None, float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            distance = abs(time - timestamp)
            if distance < error:
                best, error = frame, distance
            if time >= timestamp:
                break
        if best is None or error > 0.51 / fps:
            raise ValueError(
                f"No aligned video frame at {timestamp}: {path}, error={error}"
            )
        return best.to_ndarray(format="rgb24")


def export_fixtures(dataset, destination, adapter):
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((dataset / "meta/episodes").rglob("*.parquet"))
        ]
    )
    tasks = (
        pd.read_parquet(dataset / "meta/tasks.parquet")
        .reset_index()
        .set_index("task_index")
    )
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "observation.joint_state",
        "observation.state",
        "action",
        "action.joint_command",
    ]
    rows = pd.concat(
        [
            pd.read_parquet(p, columns=columns)
            for p in sorted((dataset / "data").rglob("*.parquet"))
        ]
    )
    records = []
    for episode in adapter.contract["split"]["validation_episodes"]:
        part = rows[rows.episode_index == episode].sort_values("frame_index")
        start = min(len(part) // 2, len(part) - adapter.contract["horizon"])
        window = part.iloc[start : start + adapter.contract["horizon"]]
        if len(window) != adapter.contract["horizon"]:
            raise ValueError(
                "Validation episode is too short for the deployment fixture"
            )
        first = window.iloc[0]
        task = str(tasks.loc[int(first.task_index), "task"])
        meta = episodes[episodes.episode_index == episode].iloc[0]
        arrays = {
            "task": task,
            "episode": int(episode),
            "frame_index": int(first.frame_index),
            "motor_state": np.stack(window["observation.joint_state"]),
            "state": np.stack(window["observation.state"]),
            "action": np.stack(window["action"]),
            "joint_command": np.stack(window["action.joint_command"]),
        }
        for camera in adapter.camera_keys:
            key = f"observation.images.{camera}"
            video = info["video_path"].format(
                video_key=key,
                chunk_index=int(meta[f"videos/{key}/chunk_index"]),
                file_index=int(meta[f"videos/{key}/file_index"]),
            )
            timestamp = float(meta[f"videos/{key}/from_timestamp"]) + float(
                first.timestamp
            )
            arrays[camera] = decode_frame(dataset / video, timestamp, info["fps"])
        name = f"fixtures/episode_{episode:06d}.npz"
        (destination / name).parent.mkdir(exist_ok=True)
        np.savez_compressed(destination / name, **arrays)
        records.append(
            {
                "path": name,
                "episode": int(episode),
                "frame_index": int(first.frame_index),
                "task": task,
            }
        )
    return records


def _write_deployment(args):
    before = args.checkpoint.stat()
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    checkpoint_step = int(payload["step"])
    if not payload.get("mot") or not payload.get("proprio_encoder"):
        raise ValueError("Expected both native experts and the proprio encoder")
    del payload
    completion = []
    for log in args.run.glob("*.log"):
        for line in log.read_text(errors="replace").splitlines():
            if (
                ("max_steps reached" in line or "training finished" in line)
                and f"step={checkpoint_step} " in line
                and str(args.checkpoint) in line
            ):
                completion.append(line.strip())
    if args.expected_final_step is not None and (
        checkpoint_step != args.expected_final_step or not completion
    ):
        raise ValueError("Requested final checkpoint has not completed training/save")
    args.output.mkdir(parents=True, exist_ok=False)
    for source, target in [
        (args.dataset / "codec.json", "codec.json"),
        (args.dataset / "norm_stats.json", "norm_stats.json"),
        (args.dataset / "assets/robot.urdf", "robot.urdf"),
    ]:
        shutil.copy2(source, args.output / target)
    if file_sha256(args.run / "dataset_stats.json") != file_sha256(
        args.dataset / "norm_stats.json"
    ):
        # Training's JSON writer may reformat the same canonical artifact.
        if json.loads((args.run / "dataset_stats.json").read_text()) != json.loads(
            (args.dataset / "norm_stats.json").read_text()
        ):
            raise ValueError("Training statistics differ from the selected dataset")
    config = OmegaConf.load(args.run / "config.yaml")
    model_config = OmegaConf.to_container(config.model, resolve=True)
    if model_config["model_id"] != MODEL_ID:
        raise ValueError(f"Only {MODEL_ID} deployment is supported")
    data = config.data.train
    if (
        [item.key for item in data.shape_meta.images] != ["front", "side"]
        or list(data.video_size) != [224, 448]
        or data.concat_multi_camera != "horizontal"
        or data.processor._target_.split(".")[-1] != "EEDeltaProcessor"
        or data.get("override_instruction") is not None
    ):
        raise ValueError("Unsupported training observation protocol")
    OmegaConf.save(OmegaConf.create(model_config), args.output / "model.yaml")
    adapter = ObservationAdapter(
        args.output / "codec.json",
        args.output / "robot.urdf",
        args.output / "norm_stats.json",
    )
    if args.export_fixtures:
        export_fixtures(args.dataset, args.output, adapter)
    tasks = pd.read_parquet(args.dataset / "meta/tasks.parquet").reset_index()["task"]
    for task in tasks:
        source = context_path(args.text_cache, str(task), model_config["tokenizer_max_len"])
        target = args.output / "contexts" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
    # Link large immutable artifacts on the shared cluster; transfer tools must
    # follow the links when creating an independent copy on the robot computer.
    checkpoint = args.output / "checkpoint.pt"
    checkpoint.symlink_to(args.checkpoint.resolve())
    vae = args.output / "models" / model_config["model_id"] / "Wan2.2_VAE.pth"
    vae.parent.mkdir(parents=True)
    vae.symlink_to(args.vae.resolve())
    after = args.checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(
            "Checkpoint changed while packaging; retry with a completed save"
        )
    return {"checkpoint_step": checkpoint_step, "tasks": tasks.tolist()}



def prepare_deployment(args):
    """Publish a deployment directory; failed packaging leaves no destination behind."""
    args = argparse.Namespace(**vars(args))
    for name in ("dataset", "run", "checkpoint", "text_cache", "vae", "output"):
        setattr(args, name, Path(getattr(args, name)).expanduser().absolute())
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        args.output = Path(temporary) / "model"
        metadata = _write_deployment(args)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        args.output.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-final-step", type=int)
    parser.add_argument(
        "--export-fixtures",
        action="store_true",
        help="Include held-out observations for offline checks",
    )
    args = parser.parse_args()
    result = prepare_deployment(args)
    output = args.output.expanduser().absolute()
    print(
        json.dumps(
            {
                "checkpoint": str(output / "checkpoint.pt"),
                "tasks": result["tasks"],
                "bytes": sum(
                    p.stat().st_size for p in output.rglob("*") if p.is_file()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
