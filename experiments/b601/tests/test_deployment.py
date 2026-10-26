"""Geometry, observation protocol and failure handling without hardware or weights."""

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from experiments.b601.fastwam_policy.controller import (
    EEDeltaController,
    IKResult,
    IKUnreachable,
)
from experiments.b601.fastwam_policy.observation import (
    ObservationAdapter,
    file_sha256,
    json_sha256,
)
from fastwam.datasets.lerobot3.ee_delta import rotation_log
from torchvision.transforms import Resize


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = json.loads(
            (Path(__file__).parents[1] / "configs/ee_delta.json").read_text()
        )
        links = ["base_link", *[f"link{i}" for i in range(1, 6)], "gripper_end"]
        axes = ["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
        joints = [
            f'<joint name="joint{i + 1}" type="revolute">'
            f'<parent link="{links[i]}"/><child link="{links[i + 1]}"/>'
            f'<origin xyz="0.1 0 0"/><axis xyz="{axes[i]}"/></joint>'
            for i in range(6)
        ]
        self.urdf = self.root / "robot.urdf"
        self.urdf.write_text(
            '<robot name="fixture">'
            + "".join(f'<link name="{name}"/>' for name in links)
            + "".join(joints)
            + "</robot>"
        )
        config["urdf_sha256"] = file_sha256(self.urdf)
        self.config = config
        (self.root / "codec.json").write_text(json.dumps(config))
        stats = {
            key: {"default": {"global_lower": [-1.0] * 7, "global_span": [2.0] * 7}}
            for key in ("state", "action")
        }
        stats["ee_delta_contract"] = {
            "representation": "base_frame_ee_delta_v2",
            "clip": False,
            "normalization": "min/max",
            "anchor": "same_row_measured_pose",
            "statistics_scope": "training_episodes_valid_rows_once",
            "horizon": 32,
            "fps": 30,
            "codec_sha256": json_sha256(config),
            "statistics_sha256": json_sha256(copy.deepcopy(stats)),
        }
        (self.root / "norm_stats.json").write_text(json.dumps(stats))
        self.adapter = ObservationAdapter(
            self.root / "codec.json", self.urdf, self.root / "norm_stats.json"
        )
        self.codec = self.adapter.codec
        self.controller = EEDeltaController(self.codec)

    def test_recorded_command_round_trip(self):
        measured = np.array([20.0, 60.0, 35.0, -25.0, 15.0, 20.0, 90.0])
        goal = measured.copy()
        goal[:6] += [2, -3, 4, -2, 3, 1]
        command = goal / self.codec.scale
        _, action = self.codec.encode(measured, command)
        result = self.controller.solve(measured, action)
        reconstructed = np.array(list(result.command.values()))
        _, round_trip = self.codec.encode(measured, reconstructed)
        np.testing.assert_allclose(round_trip, action, atol=2e-6, rtol=0)

    def test_each_action_uses_new_measured_pose(self):
        first = np.array([20.0, 60.0, 35.0, -25.0, 15.0, 20.0, 90.0])
        second = first.copy()
        second[0] += 10
        action = np.array([0.001, 0.002, -0.001, 0.01, -0.02, 0.01, 0.5])
        for measured in (first, second):
            result = self.controller.solve(measured, action)
            _, encoded = self.codec.encode(
                measured, np.array(list(result.command.values()))
            )
            np.testing.assert_allclose(encoded, action, atol=2e-6, rtol=0)

    def test_no_clipping_of_gripper_or_normalized_values(self):
        measured = np.array([20.0, 60.0, 35.0, -25.0, 15.0, 20.0, 90.0])
        action = np.array([0, 0, 0, 0, 0, 0, 1.2])
        result = self.controller.solve(measured, action)
        self.assertAlmostEqual(result.command["gripper.pos"], 54.0)
        self.assertFalse(result.within_driver_limits)
        decoded = self.adapter.decode_actions(torch.full((32, 7), 3.0))
        np.testing.assert_array_equal(decoded, np.full((32, 7), 3.0))

    def test_unreachable_target_is_rejected(self):
        with self.assertRaisesRegex(IKUnreachable, "IK did not reach target") as raised:
            self.controller.solve(np.zeros(7), [10, 10, 10, 0, 0, 0, 0.5])
        self.assertGreater(raised.exception.details["position_error_m"], 1)

    def test_nonfinite_input_is_not_a_recoverable_ik_failure(self):
        with self.assertRaises(ValueError) as raised:
            self.controller.solve(np.zeros(7), [float("nan"), 0, 0, 0, 0, 0, 0.5])
        self.assertNotIsInstance(raised.exception, IKUnreachable)

    def test_offline_report_distinguishes_raw_and_driver_clipped_targets(self):
        from experiments.b601.deploy import offline

        measured = np.tile([20.0, 60.0, 35.0, -25.0, 15.0, 20.0, 90.0], (32, 1))
        action = np.tile([0, 0, 0, 0, 0, 0, 1.2], (32, 1)).astype(np.float32)
        fixtures, output = self.root / "fixtures", self.root / "output"
        fixtures.mkdir()
        output.mkdir()
        np.savez_compressed(
            fixtures / "sample.npz",
            task="fixture",
            episode=0,
            frame_index=0,
            motor_state=measured,
            state=self.codec.state(measured),
            action=action,
            front=np.zeros((32, 32, 3), np.uint8),
            side=np.zeros((32, 32, 3), np.uint8),
        )
        policy = SimpleNamespace(
            checkpoint_step=1,
            checkpoint=self.root / "unused.pt",
            root=self.root,
            model=SimpleNamespace(device=torch.device("cpu")),
            num_inference_steps=1,
            seed=0,
            adapter=self.adapter,
            predict=lambda *args: action,
        )
        self.assertEqual(offline(policy, self.controller, output), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertTrue(report["passed"])
        self.assertFalse(report["hardware_connected"])
        self.assertFalse(report["all_predicted_targets_within_driver_limits"])
        self.assertTrue(report["all_applied_targets_within_driver_limits"])
        self.assertEqual(report["fixtures"][0]["predicted_outside_driver_limits"], 32)
        diagnostic = json.loads((output / "ik_0.json").read_text())["predicted"][0]
        self.assertEqual(diagnostic["driver_target_physical"][6], 270)
        self.assertAlmostEqual(diagnostic["native_clip_delta_deg"][6], -54, places=3)

    def test_nonfinite_and_wrong_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.solve(np.zeros(7), [0, 0, 0, np.nan, 0, 0, 0.5])
        with self.assertRaises(ValueError):
            self.adapter.decode_actions(torch.zeros(1, 7))
        with self.assertRaises(ValueError):
            self.adapter.encode_images(
                {"front": np.zeros((10, 10, 3)), "side": np.zeros((10, 10, 3))}
            )

    def test_camera_order_and_training_resize(self):
        rng = np.random.default_rng(42)
        images = {
            key: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            for key in ("front", "side")
        }
        actual = self.adapter.encode_images(images)
        expected = torch.cat(
            [
                Resize([224, 224])(
                    torch.from_numpy(images[key]).permute(2, 0, 1).float() / 255
                )
                for key in ("front", "side")
            ],
            -1,
        )
        torch.testing.assert_close(actual[0], (expected - 0.5) / 0.5, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (1, 3, 224, 448))

    def test_state_is_ee_pose_and_continuous_gripper(self):
        measured = np.array([20.0, 60.0, 35.0, -25.0, 15.0, 20.0, 324.0])
        pose = self.codec.fk(np.deg2rad(measured[:6]))
        state = self.codec.state(measured)
        np.testing.assert_allclose(state[:3], pose[:3, 3], atol=1e-7)
        np.testing.assert_allclose(state[3:6], rotation_log(pose[:3, :3]), atol=1e-7)
        self.assertAlmostEqual(float(state[6]), 1.2, places=6)

    def test_urdf_and_statistics_mismatch_rejected(self):
        with self.urdf.open("a") as out:
            out.write("\n")
        with self.assertRaisesRegex(ValueError, "frozen EE contract"):
            ObservationAdapter(
                self.root / "codec.json", self.urdf, self.root / "norm_stats.json"
            )

    def test_legacy_live_is_rejected_before_hardware_access(self):
        from experiments.b601.deploy import live

        with self.assertRaisesRegex(RuntimeError, "retired"):
            live(None, None, None, None)


if __name__ == "__main__":
    unittest.main()
