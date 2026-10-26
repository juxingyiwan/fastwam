"""Decode pre-driver targets, and report native clamping separately from IK."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.b601.fastwam_policy.controller import EEDeltaController
from fastwam.datasets.lerobot3.ee_delta import EEDeltaCodec, rotation_exp


def controller_and_action(target_shoulder, measured_shoulder=0.2):
    # Decoupled translation/rotation gives an exact expected post-clamp error.
    # Nonzero URDF offsets and negative driver signs are deliberate.
    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3] * 0.1
        pose[:3, :3] = rotation_exp(q[3:])
        return pose

    codec = SimpleNamespace(
        fk=fk,
        offset=np.array([0.1, -0.2, 0.15, 0.1, 0.2, -0.1]),
        scale=np.array([1, 1, -1, -1, -1, 1, 6]),
        gripper_command=lambda fraction: fraction * 45,
        config={
            "recorded_feature_names": [f"j{i}.pos" for i in range(7)],
            "driver_motor_limits_deg": [
                [-145, 145],
                [0, 170],
                [0, 200],
                [-80, 90],
                [-90, 90],
                [-130, 130],
                [0, 270],
            ],
        },
    )
    codec._check = lambda value: EEDeltaCodec._check(codec, value)
    measured = np.array([1, measured_shoulder, 0.2, 1, 1, 1, 0])
    action = np.array(
        [0, np.deg2rad(target_shoulder - measured_shoulder) * 0.1, 0, 0, 0, 0, 0.5]
    )
    return EEDeltaController(codec), measured, action


def test_outside_target_preserves_command_and_reports_native_clamp():
    controller, measured, action = controller_and_action(-0.004)
    result = controller.solve(measured, action)
    physical = np.array(list(result.command.values())) * controller.codec.scale
    assert not result.within_driver_limits
    assert physical[1] == pytest.approx(-0.004, abs=1e-7)
    assert result.driver_target_physical[1] == 0
    assert result.native_limit_clipped == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result.position_error_m < 1e-9
    assert result.rotation_error_rad < 1e-9
    assert result.driver_position_error_m == pytest.approx(np.deg2rad(0.004) * 0.1)
    np.testing.assert_allclose(physical[[0, 2, 3, 4, 5]], [1, 0.2, 1, 1, 1], atol=1e-5)


def test_clamp_pose_error_is_not_mistaken_for_ik_failure():
    controller, measured, action = controller_and_action(-1.0)
    result = controller.solve(measured, action)
    assert result.command["j1.pos"] == pytest.approx(-1)
    assert result.position_error_m < 1e-9
    assert result.driver_position_error_m > controller.position_tolerance_m
    assert result.driver_position_error_m == pytest.approx(np.deg2rad(1) * 0.1)


def test_feedback_outside_bounds_is_not_clipped_before_fk():
    controller, measured, action = controller_and_action(
        -0.004, measured_shoulder=-0.02
    )
    result = controller.solve(measured, action)
    assert not result.within_driver_limits
    assert result.command["j1.pos"] == pytest.approx(-0.004, abs=1e-7)
    assert result.position_error_m < 1e-9
    assert result.driver_position_error_m == pytest.approx(np.deg2rad(0.004) * 0.1)


def test_in_range_solution_keeps_existing_unconstrained_path():
    controller, measured, action = controller_and_action(0.3)
    result = controller.solve(measured, action)
    assert result.within_driver_limits and not any(result.native_limit_clipped)
    assert result.command["j1.pos"] == pytest.approx(0.3, abs=1e-7)
    assert result.driver_position_error_m < 1e-9
    assert result.driver_rotation_error_rad < 1e-9


def test_gripper_command_is_preserved_and_native_scale_and_clamp_are_reported():
    controller, measured, action = controller_and_action(0.3)
    action[-1] = 1.2
    result = controller.solve(measured, action)
    assert not result.within_driver_limits
    assert result.command["j6.pos"] == pytest.approx(54)
    assert result.driver_target_physical[6] == 270
    assert result.native_limit_clipped[6]
    assert result.driver_position_error_m < 1e-9
