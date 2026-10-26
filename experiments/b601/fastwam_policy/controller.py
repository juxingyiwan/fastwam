"""Base-frame EE deltas to pre-driver joint targets; never clips commands."""

from dataclasses import dataclass, field

import numpy as np
from fastwam.datasets.lerobot3.ee_delta import compose_delta, rotation_log
from scipy.optimize import least_squares


class IKUnreachable(ValueError):
    """A finite requested pose has no sufficiently accurate local IK solution."""


@dataclass
class IKResult:
    command: dict
    position_error_m: float
    rotation_error_rad: float
    evaluations: int
    within_driver_limits: bool
    driver_target_physical: list = field(default_factory=list)
    native_limit_clipped: list = field(default_factory=list)
    native_clip_delta_deg: list = field(default_factory=list)
    driver_position_error_m: float = 0.0
    driver_rotation_error_rad: float = 0.0


class EEDeltaController:
    def __init__(self, codec, position_tolerance_m=1e-4, rotation_tolerance_rad=1e-3):
        self.codec = codec
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        if min(self.position_tolerance_m, self.rotation_tolerance_rad) <= 0:
            raise ValueError("IK tolerances must be positive")

    def solve(self, motor_state, action):
        """Use fresh feedback for each action, including actions in one chunk.

        Decode the recorded pre-driver command representation. Validate IK
        against the requested EE pose before the native driver's joint clamp.
        Report that clamp and its pose error separately; do not confuse a
        saturated actuator target with a failed geometric conversion.
        """
        measured = self.codec._check(motor_state)
        action = self.codec._check(action)
        if measured.shape != (7,) or action.shape != (7,):
            raise ValueError("The controller accepts one observation and action")
        initial = np.deg2rad(measured[:6]) + self.codec.offset
        pose = self.codec.fk(initial)
        position, rotation = compose_delta(pose[:3, 3], pose[:3, :3], action)

        def residual(joints):
            candidate = self.codec.fk(joints)
            return np.concatenate(
                [
                    (candidate[:3, 3] - position) / 0.1,
                    rotation_log(rotation @ candidate[:3, :3].T),
                ]
            )

        solution = least_squares(
            residual,
            initial,
            method="trf",
            max_nfev=150,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        limits = np.asarray(self.codec.config["driver_motor_limits_deg"])
        motor_goal = np.rad2deg(solution.x - self.codec.offset)
        target = self.codec.fk(solution.x)
        position_error = float(np.linalg.norm(target[:3, 3] - position))
        rotation_error = float(
            np.linalg.norm(rotation_log(rotation @ target[:3, :3].T))
        )
        if (
            not np.isfinite(solution.x).all()
            or not np.isfinite([position_error, rotation_error]).all()
        ):
            raise ValueError("IK produced a nonfinite solution")
        if (
            not solution.success
            or position_error > self.position_tolerance_m
            or rotation_error > self.rotation_tolerance_rad
        ):
            error = IKUnreachable(
                f"IK did not reach target: position={position_error:.6g} m, "
                f"rotation={rotation_error:.6g} rad; {solution.message}"
            )
            error.details = {
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
                "evaluations": int(solution.nfev),
            }
            raise error
        command = np.concatenate(
            [
                motor_goal / self.codec.scale[:6],
                [float(self.codec.gripper_command(action[6]))],
            ]
        )
        motor_goal = command * self.codec.scale
        within_limits = bool(
            ((motor_goal >= limits[:, 0]) & (motor_goal <= limits[:, 1])).all()
        )
        driver_target = np.clip(motor_goal, *limits.T)
        applied_pose = self.codec.fk(np.deg2rad(driver_target[:6]) + self.codec.offset)
        return IKResult(
            dict(zip(self.codec.config["recorded_feature_names"], command.tolist())),
            position_error,
            rotation_error,
            int(solution.nfev),
            within_limits,
            driver_target.tolist(),
            (np.abs(driver_target - motor_goal) > 1e-5).tolist(),
            (driver_target - motor_goal).tolist(),
            float(np.linalg.norm(applied_pose[:3, 3] - position)),
            float(np.linalg.norm(rotation_log(rotation @ applied_pose[:3, :3].T))),
        )
