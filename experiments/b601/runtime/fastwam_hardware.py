"""Keep the audited hardware lifecycle; adapt only the FastWAM action boundary."""

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from .contracts import JOINTS, HardwareError
from .hardware.b601 import B601
from .hardware.front_home import FORWARD_LIMIT_M, CLEARANCE_M, plan_home, execute_home
from .hardware.homing import homing_deadline


class FastWAMHardware(B601):
    def __init__(self, cfg, event, decoder, *, shutdown=None):
        super().__init__(cfg, event, shutdown=shutdown)
        self.decoder = decoder

    def open_observation(self):
        verify_driver_source()
        super().open_observation()
        codec = self.decoder.codec
        if not self.robot.is_calibrated:
            raise HardwareError(
                "Existing B601 calibration is required; deployment never recalibrates"
            )
        if (
            tuple(
                n.removesuffix(".pos") for n in codec.config["recorded_feature_names"]
            )
            != JOINTS
        ):
            raise HardwareError(
                "Training joint order differs from the hardware adapter"
            )
        if not np.array_equal(self.directions, codec.scale) or not np.array_equal(
            self.limits, codec.config["driver_motor_limits_deg"]
        ):
            raise HardwareError(
                "Driver direction/limits differ from the trained EE codec"
            )

    def _native_home(self):
        # Keep the native finish and gripper refinement. Only an outward arm
        # sweep gains a preplanned, continuous coordinated return to near zero.
        measured, _ = self._home_feedback(refresh=True)
        path = plan_home(self.decoder.codec, measured, self.limits)
        self._event("home_forward_plan", phase=(self._home_progress_state or {}).get("phase"),
                    **path.summary())
        if not path.native:
            with homing_deadline(self.shutdown_cfg.home_timeout_s):
                execute_home(self, self.decoder.codec, path)
            if self._home_progress_state is not None:
                self._home_progress_state.update(stage="native_arm", emitted=None)
        return super()._native_home()

    def decode(self, measured, action):
        result = self.decoder.solve(measured, action)
        raw = np.asarray([result.command[j + ".pos"] for j in JOINTS])
        # send() applies the verified native direction/limit mapping and checks
        # the driver's returned target. Keep the pre-driver command intact.
        return raw, asdict(result)

    def manifest(self):
        return {
            **super().manifest(),
            "home_forward_limit_m": FORWARD_LIMIT_M,
            "home_forward_clearance_m": CLEARANCE_M,
            "home_forward_reference": "base_link +X / gripper_end (nominal URDF)",
            "policy_action": "[dx,dy,dz,rx,ry,rz,gripper]",
            "policy_anchor": "fresh measured pose for each action",
            "policy_limit_behavior": "native_driver_clip_with_pose_diagnostics",
        }


def verify_driver_source():
    """Check the audited plugin before any CAN/socket/camera handle is created."""
    spec = importlib.util.find_spec("lerobot_robot_seeed_b601")
    if spec is None or not spec.origin:
        raise HardwareError("Install the pinned B601 hardware dependencies")
    root = Path(spec.origin).parent
    expected = json.loads((Path(__file__).parent / "driver-source.json").read_text())
    for name, digest in expected.items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise HardwareError(
                f"B601 driver source differs from the audited deployment: {name}"
            )
    return expected
