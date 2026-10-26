"""Offline semantic and bundle tests; no robot, model weights or video decoder."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastwam.datasets.lerobot3.ee_delta import (
    EEDeltaCodec,
    URDFKinematics,
    compose_delta,
    rotation_exp,
    rotation_log,
)
from fastwam.datasets.lerobot3.ee_delta_preparation import (
    build_bundle,
    file_hash,
    fit_statistics,
    frozen_statistics,
    read_rows,
    split_episodes,
    validate_rows,
    verify_bundle,
    write_json,
)

REPO = Path(__file__).resolve().parents[3]


def fixture(root):
    config = json.loads((REPO / "experiments/b601/configs/ee_delta.json").read_text())
    links = ["base_link", *[f"link{i}" for i in range(1, 6)], "gripper_end"]
    axes = ["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
    joints = [
        f'<joint name="joint{i + 1}" type="revolute"><parent link="{links[i]}"/>'
        f'<child link="{links[i + 1]}"/><origin xyz="0.1 0 0" rpy="0 0 0"/>'
        f'<axis xyz="{axes[i]}"/></joint>'
        for i in range(6)
    ]
    urdf = root / "robot.urdf"
    urdf.write_text(
        '<robot name="test">'
        + "".join(f'<link name="{name}"/>' for name in links)
        + "".join(joints)
        + "</robot>"
    )
    config["urdf_sha256"] = file_hash(urdf)
    config_path = root / "codec.json"
    write_json(config_path, config)
    return config, config_path, urdf


class RotationTests(unittest.TestCase):
    def test_zero_and_single_axes(self):
        np.testing.assert_array_equal(rotation_exp(np.zeros(3)), np.eye(3))
        np.testing.assert_array_equal(rotation_log(np.eye(3)), np.zeros(3))
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(
            rotation_exp([0, 0, np.pi / 2]), expected, atol=1e-15
        )
        np.testing.assert_allclose(
            rotation_log(expected), [0, 0, np.pi / 2], atol=1e-15
        )

    def test_half_turn_and_compound_round_trip(self):
        axes = np.random.default_rng(4).normal(size=(100, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        for angle in (0, 1e-12, 0.73, np.pi - 1e-9, np.pi, np.pi + 1e-9, 4.8):
            matrix = rotation_exp(axes * angle)
            vectors = rotation_log(matrix)
            np.testing.assert_allclose(rotation_exp(vectors), matrix, atol=2e-14)
            self.assertLessEqual(
                float(np.linalg.norm(vectors, axis=-1).max()), np.pi + 1e-14
            )

    def test_left_composition_is_not_euler_subtraction(self):
        current = rotation_exp([0.4, -0.2, 0.5])
        goal = rotation_exp([-0.3, 0.8, -0.1])
        delta = rotation_log(goal @ current.T)
        position, rotation = compose_delta(
            [1, 2, 3], current, [0.1, 0.2, -0.3, *delta, 0.4]
        )
        np.testing.assert_allclose(position, [1.1, 2.2, 2.7])
        np.testing.assert_allclose(rotation, goal, atol=1e-14)
        self.assertGreater(np.abs(current @ rotation_exp(delta) - goal).max(), 0.1)
        self.assertGreater(
            np.abs(rotation_log(goal) - rotation_log(current) - delta).max(), 0.1
        )

    def test_invalid_rotations_rejected(self):
        for matrix in (np.zeros((3, 3)), np.diag([-1, 1, 1]), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                rotation_log(matrix)


class _CodecFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.config_path, self.urdf = fixture(self.root)
        self.codec = EEDeltaCodec(self.urdf, self.config)

    def tearDown(self):
        self.temporary.cleanup()


class CodecTests(_CodecFixture):
    def test_zero_delta_and_continuous_gripper(self):
        state = np.array(
            [[10, 20, 30, 40, 50, 60, 81], [0, 0, 0, 0, 0, 0, -3]], np.float32
        )
        command = state / np.array(self.config["command_to_motor_scale"])
        encoded_state, action = self.codec.encode(state, command)
        np.testing.assert_allclose(action[:, :6], 0, atol=1e-15)
        np.testing.assert_allclose(action[:, 6], [0.3, -1 / 90], atol=1e-7)
        np.testing.assert_array_equal(encoded_state, self.codec.state(state))
        np.testing.assert_allclose(
            self.codec.gripper_command(action[:, 6]), command[:, 6], atol=1e-6
        )

    def test_each_row_uses_its_measured_anchor(self):
        state = np.array(
            [[0, 0, 0, 0, 0, 0, 0], [40, 20, 10, 50, 30, 60, 100]], np.float32
        )
        command = np.array([[10, 15, -30, -20, -10, 30, 12]] * 2, np.float32)
        _, action = self.codec.encode(state, command)
        self.assertGreater(np.abs(action[0, :6] - action[1, :6]).max(), 0.1)
        current = self.codec.fk(np.deg2rad(state[:, :6]))
        target = self.codec.fk(np.deg2rad(command[:, :6] * self.codec.scale[:6]))
        position, rotation = compose_delta(
            current[:, :3, 3], current[:, :3, :3], action
        )
        np.testing.assert_allclose(position, target[:, :3, 3], atol=1e-7)
        np.testing.assert_allclose(rotation, target[:, :3, :3], atol=1e-7)

    def test_analytic_fk_and_rpy_order(self):
        urdf = self.root / "analytic.urdf"
        urdf.write_text(
            '<robot><link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="q" type="continuous"><parent link="a"/><child link="b"/>'
            '<origin xyz="1 0 0" rpy="0.2 0.3 0.4"/><axis xyz="0 0 1"/></joint>'
            '<joint name="tool" type="fixed"><parent link="b"/><child link="c"/>'
            '<origin xyz="1 0 0"/></joint></robot>'
        )
        fk = URDFKinematics(urdf, "a", "c", ["q"])
        expected_r = (
            rotation_exp([0, 0, 0.4])
            @ rotation_exp([0, 0.3, 0])
            @ rotation_exp([0.2, 0, 0])
            @ rotation_exp([0, 0, np.pi / 2])
        )
        pose = fk([np.pi / 2])
        np.testing.assert_allclose(pose[:3, :3], expected_r, atol=1e-15)
        np.testing.assert_allclose(
            pose[:3, 3], [1, 0, 0] + expected_r[:, 0], atol=1e-15
        )
        with self.assertRaises(ValueError):
            URDFKinematics(urdf, "a", "c", ["wrong"])

    def test_nonfinite_or_wrong_shape_fails(self):
        for value in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                self.codec.encode(value, np.zeros(7))


class BundleTests(_CodecFixture):
    def make_source(self):
        source = self.root / "source"
        (source / "meta/episodes/chunk-000").mkdir(parents=True)
        (source / "data/chunk-000").mkdir(parents=True)
        (source / "videos").mkdir()
        (source / "videos/unchanged.mp4").write_bytes(b"video-copy-fixture")
        feature = {
            "dtype": "float32",
            "shape": [7],
            "names": self.config["recorded_feature_names"],
        }
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "total_frames": 24,
            "total_episodes": 6,
            "features": {"action": feature, "observation.state": feature},
        }
        write_json(source / "meta/info.json", info)
        write_json(source / "meta/stats.json", {})
        values = np.arange(24 * 7, dtype=np.float32).reshape(24, 7) * 0.1
        table = pa.table(
            {
                "index": range(24),
                "episode_index": np.repeat(range(6), 4),
                "frame_index": np.tile(range(4), 6),
                "timestamp": np.tile(np.arange(4, dtype=np.float32) / 30, 6),
                "task_index": np.repeat(range(3), 8),
                "action": pa.array(values.tolist(), type=pa.list_(pa.float32(), 7)),
                "observation.state": pa.array(
                    (values * self.codec.scale).tolist(), type=pa.list_(pa.float32(), 7)
                ),
            }
        )
        pq.write_table(table, source / "data/chunk-000/file-000.parquet")
        pq.write_table(
            pa.table({"episode_index": range(6), "length": [4] * 6}),
            source / "meta/episodes/chunk-000/file-000.parquet",
        )
        return source

    def test_build_verify_recompute_and_raw_preservation(self):
        source, output = self.make_source(), self.root / "bundle"
        template = REPO / "configs/data/b601_2cam_ee_delta_lerobot_v30.yaml"
        report = build_bundle(source, output, self.config_path, self.urdf, template)
        verified, artifact = verify_bundle(output, check_source=True)
        self.assertEqual(report, verified)
        self.assertEqual(artifact["action"]["default"]["global_count"], 12)
        self.assertEqual(
            file_hash(output / "videos/unchanged.mp4"),
            file_hash(source / "videos/unchanged.mp4"),
        )
        rows = read_rows(output)
        raw = read_rows(source)
        np.testing.assert_array_equal(
            rows["observation.joint_state"], raw["observation.state"]
        )
        np.testing.assert_array_equal(rows["action.joint_command"], raw["action"])
        split = artifact["ee_delta_contract"]["split"]
        # A large validation-only perturbation must never affect fitted stats.
        val = np.isin(rows["episode_index"], split["validation_episodes"])
        rows["action"][val] += 10000
        rows["observation.state"][val] -= 10000
        self.assertEqual(artifact, frozen_statistics(output, rows, self.config, split))
        with self.assertRaises(ValueError):
            build_bundle(source, output, self.config_path, self.urdf, template)
        (output / "norm_stats.json").write_text("{}")
        with self.assertRaises(ValueError):
            verify_bundle(output)

    def test_bad_time_grid_and_frame_order_rejected(self):
        source = self.make_source()
        rows = read_rows(source)
        info = json.loads((source / "meta/info.json").read_text())
        for key in ("timestamp", "frame_index"):
            bad = copy.deepcopy(rows)
            bad[key][2] += 1
            with self.assertRaises(ValueError):
                validate_rows(bad, info)

    def test_split_is_complete_disjoint_and_stratified(self):
        rows = read_rows(self.make_source())
        split = split_episodes(rows, 10222, 1)
        self.assertEqual(split, split_episodes(rows, 10222, 1))
        self.assertEqual(
            sorted(split["train_episodes"] + split["validation_episodes"]),
            list(range(6)),
        )
        self.assertEqual(len(split["validation_episodes"]), 3)


class NormalizationTests(unittest.TestCase):
    def test_constant_dimension_and_unclipped_round_trip(self):
        values = np.array([[1, 4], [3, 4]], np.float32)
        stats = fit_statistics(values, "min/max")
        lo, span = (
            np.array(stats["global_lower"], np.float32),
            np.array(stats["global_span"], np.float32),
        )
        self.assertEqual(span[1], 1)
        outside = np.array([[5, 6]], np.float32)
        normalized = 2 * (outside - lo) / span - 1
        self.assertTrue(np.all(normalized > 1))
        np.testing.assert_allclose((normalized + 1) * span / 2 + lo, outside)


if __name__ == "__main__":
    unittest.main()
