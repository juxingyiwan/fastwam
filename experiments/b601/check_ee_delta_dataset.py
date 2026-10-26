"""Validate a prepared bundle through native FastWAM loading and independent FK.

Run in the FastWAM environment; SciPy is used only as an independent validation
oracle, not by the standalone conversion or statistics pipeline.
"""

import argparse
import json
import socket
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from fastwam.datasets.lerobot.processors.ee_delta_processor import EEAffineNormalizer
from fastwam.datasets.lerobot3.ee_delta_preparation import read_rows, write_json
from fastwam.utils.misc import register_work_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation


def independent_fk(urdf, values, config):
    """Separate URDF traversal using SciPy's rotation implementation."""
    by_child = {
        joint.find("child").get("link"): joint
        for joint in ET.parse(urdf).getroot().findall("joint")
    }
    chain = []
    link = config["tool_link"]
    while link != config["base_link"]:
        joint = by_child[link]
        chain.insert(0, joint)
        link = joint.find("parent").get("link")
    pose = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
    for joint in chain:
        origin = joint.find("origin")
        fixed = np.eye(4)
        if origin is not None:
            fixed[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            fixed[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
        pose = pose @ fixed
        if joint.get("type") != "fixed":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            axis /= np.linalg.norm(axis)
            index = config["joint_names"].index(joint.get("name"))
            pose[:, :3, :3] = (
                pose[:, :3, :3]
                @ Rotation.from_rotvec(values[:, index, None] * axis).as_matrix()
            )
    return pose


def check_geometry(root, rows, config):
    raw_state = rows["observation.joint_state"].astype(np.float64)
    raw_action = rows["action.joint_command"].astype(np.float64)
    q = np.deg2rad(raw_state[:, :6]) + config["motor_to_urdf_offset_rad"]
    command_q = (
        np.deg2rad(raw_action[:, :6] * np.array(config["command_to_motor_scale"][:6]))
        + config["motor_to_urdf_offset_rad"]
    )
    measured = independent_fk(root / "assets/robot.urdf", q, config)
    goal = independent_fk(root / "assets/robot.urdf", command_q, config)
    encoded_state, action = rows["observation.state"], rows["action"]
    delta = Rotation.from_matrix(
        goal[:, :3, :3] @ measured[:, :3, :3].transpose(0, 2, 1)
    ).as_rotvec()
    np.testing.assert_allclose(
        action[:, :3], goal[:, :3, 3] - measured[:, :3, 3], atol=1e-8, rtol=1e-6
    )
    np.testing.assert_allclose(action[:, 3:6], delta, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(
        encoded_state[:, :3], measured[:, :3, 3], atol=3e-8, rtol=1e-6
    )
    state_rotation = Rotation.from_rotvec(
        encoded_state[:, 3:6].astype(np.float64)
    ).as_matrix()
    state_rotation_error = Rotation.from_matrix(
        state_rotation @ measured[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    reconstructed = (
        Rotation.from_rotvec(action[:, 3:6].astype(np.float64)).as_matrix()
        @ measured[:, :3, :3]
    )
    rotation_error = Rotation.from_matrix(
        reconstructed @ goal[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    position_error = np.linalg.norm(
        measured[:, :3, 3] + action[:, :3] - goal[:, :3, 3], axis=1
    )
    assert (
        rotation_error.max() < 1e-6
        and state_rotation_error.max() < 1e-6
        and position_error.max() < 1e-7
    )
    closed, opened = config["gripper_command_endpoints"]
    np.testing.assert_allclose(
        action[:, 6], (raw_action[:, 6] - closed) / (opened - closed), atol=1e-7
    )
    np.testing.assert_allclose(
        encoded_state[:, 6],
        (raw_state[:, 6] / config["command_to_motor_scale"][6] - closed)
        / (opened - closed),
        atol=1e-7,
    )
    return {
        "frames": len(action),
        "max_position_error_m": float(position_error.max()),
        "max_action_rotation_error_rad": float(rotation_error.max()),
        "max_state_rotation_error_rad": float(state_rotation_error.max()),
    }


def check_normalization(rows, stats):
    normalizer = EEAffineNormalizer(stats)
    result = {}
    for field, key in (("action", "action"), ("state", "observation.state")):
        values = torch.from_numpy(rows[key])
        encoded = normalizer.transform(values, field)
        saved = stats[field]["default"]
        lo, span = (
            np.array(saved["global_lower"], np.float32),
            np.array(saved["global_span"], np.float32),
        )
        np.testing.assert_array_equal(encoded.numpy(), 2 * (rows[key] - lo) / span - 1)
        decoded = normalizer.transform(encoded, field, inverse=True)
        torch.testing.assert_close(decoded, values, rtol=0, atol=3e-7)
        # Prove that neither direction truncates values outside the fitted range.
        outside = torch.tensor(lo) + 4 * torch.tensor(span)
        torch.testing.assert_close(
            normalizer.transform(outside, field),
            torch.full((7,), 7.0),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            normalizer.transform(torch.full((7,), 7.0), field, inverse=True),
            outside,
            atol=2e-6,
            rtol=0,
        )
        result[field] = {
            "max_round_trip_error": float((decoded - values).abs().max()),
            "unclipped_probe": 7.0,
        }
    return result


def check_reader(root, output, rows, stats):
    cfg = OmegaConf.load(root / "data_config.yaml")
    for name in ("train", "val"):
        cfg.data[name].dataset_dirs = [str(root)]
        cfg.data[name].pretrained_norm_stats = str(root / "norm_stats.json")
        cfg.data[name].use_text_embed_cache = False
    work_dir = output.parent / (output.stem + "-reader")
    work_dir.mkdir(parents=True, exist_ok=True)
    register_work_dir(work_dir)
    report = []
    for name in ("train", "val"):
        dataset = instantiate(cfg.data[name], _recursive_=False)
        processor = dataset.lerobot_dataset.processor
        base = dataset.lerobot_dataset.multi_dataset._datasets[0]
        ids = base.episodes
        expected_ids = stats["ee_delta_contract"]["split"][
            "train_episodes" if name == "train" else "validation_episodes"
        ]
        assert ids == expected_ids
        assert len(dataset) == int(np.isin(rows["episode_index"], ids).sum())
        chosen = []
        for task in np.unique(rows["task_index"]):
            task_episodes = set(
                rows["episode_index"][rows["task_index"] == task].tolist()
            )
            chosen.append(next(episode for episode in ids if episode in task_episodes))
        starts = base.episode_data_index["from"].tolist()
        ends = base.episode_data_index["to"].tolist()
        for episode in chosen:
            ordinal = ids.index(episode)
            source_rows = np.flatnonzero(rows["episode_index"] == episode)
            for index, source_offset in (
                (starts[ordinal], 0),
                (ends[ordinal] - 1, len(source_rows) - 1),
            ):
                sample = dataset.lerobot_dataset[index]
                assert sample["idx"] == index, (
                    "The reader silently replaced a failed sample"
                )
                decoded = processor.decode_actions(sample["action"]).numpy()
                valid = int((~sample["action_is_pad"]).sum())
                row_ids = source_rows[source_offset : source_offset + valid]
                np.testing.assert_allclose(
                    decoded[:valid], rows["action"][row_ids], atol=3e-7, rtol=0
                )
                if valid < 32:
                    np.testing.assert_allclose(
                        decoded[valid:, :6], 0, atol=3e-7, rtol=0
                    )
                    np.testing.assert_allclose(
                        decoded[valid:, 6],
                        rows["action"][source_rows[-1], 6],
                        atol=3e-7,
                        rtol=0,
                    )
                post = processor.postprocess(
                    {
                        "action": sample["action"][None],
                        "proprio": sample["proprio"][None],
                    }
                )
                assert tuple(post["action"]["default"].shape) == (1, 32, 7)
                torch.testing.assert_close(
                    post["action"]["default"][0],
                    torch.from_numpy(decoded),
                    atol=0,
                    rtol=0,
                )
                state_ids = source_rows[
                    np.minimum(np.arange(33) + source_offset, len(source_rows) - 1)
                ]
                np.testing.assert_allclose(
                    post["state"]["default"][0].numpy(),
                    rows["observation.state"][state_ids],
                    atol=3e-7,
                    rtol=0,
                )
                packed = dataset[index]
                assert tuple(packed["video"].shape) == (3, 9, 224, 448)
                assert (
                    tuple(packed["action"].shape)
                    == tuple(packed["proprio"].shape)
                    == (32, 7)
                )
                assert (
                    torch.isfinite(packed["video"]).all()
                    and torch.isfinite(packed["action"]).all()
                )
                report.append(
                    {
                        "split": name,
                        "episode": episode,
                        "frame": source_offset,
                        "padded_actions": 32 - valid,
                        "video_shape": list(packed["video"].shape),
                        "action_shape": list(packed["action"].shape),
                        "state_shape": list(packed["proprio"].shape),
                        "prompt": packed["prompt"],
                    }
                )
        exported = json.loads((work_dir / "dataset_stats.json").read_text())
        assert exported == stats, "Native loader export changed the frozen artifact"
        print(
            f"{name}: {len(ids)} episodes, {len(dataset)} frames; "
            "reader and padding passed",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root)
    stats = json.loads((root / "norm_stats.json").read_text())
    config = json.loads((root / "codec.json").read_text())
    report = {
        "host": socket.gethostname(),
        "geometry": check_geometry(root, rows, config),
        "normalization": check_normalization(rows, stats),
    }
    print("Independent full-dataset geometry and normalization passed", flush=True)
    report["reader_samples"] = check_reader(root, args.output, rows, stats)
    report["status"] = "passed"
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "geometry": report["geometry"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
