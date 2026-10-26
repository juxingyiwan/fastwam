"""Run native B601 inference on recorded observations or selected hardware."""

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .fastwam_policy.controller import EEDeltaController
from .fastwam_policy.policy import B601Policy


def guard_offline_hardware(event, arguments):
    """Reject Python CAN sockets and camera/serial device opens in offline mode."""
    if event == "socket.__new__" and arguments[1] == getattr(socket, "AF_CAN", 29):
        raise RuntimeError("CAN access is forbidden during offline deployment tests")
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        path = (
            arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
        )
        if path.startswith(("/dev/video", "/dev/tty", "/dev/serial")):
            raise RuntimeError(
                "Camera/serial access is forbidden during offline deployment tests"
            )


def observation_from_fixture(arrays, names):
    return {
        **dict(zip(names, arrays["motor_state"][0])),
        "front": arrays["front"],
        "side": arrays["side"],
    }


def offline(policy, controller, output, fixtures=None):
    """No robot/camera/CAN imports, device discovery, connection or writes."""
    report = {
        "mode": "offline",
        "hardware_connected": False,
        "checkpoint_step": policy.checkpoint_step,
        "checkpoint": str(policy.checkpoint),
        "device": str(policy.model.device),
        "torch": torch.__version__,
        "num_inference_steps": policy.num_inference_steps,
        "seed": policy.seed,
        "clip": False,
        "joint_limit_processing": "native_driver_clip_simulated_without_hardware",
        "fixtures": [],
    }
    if policy.model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        report["gpu"] = torch.cuda.get_device_name()
    failures = []
    limits = np.asarray(controller.codec.config["driver_motor_limits_deg"])

    def applied_target_valid(result):
        physical = np.asarray(result["driver_target_physical"])
        return (
            physical.shape == (7,)
            and np.isfinite(physical).all()
            and ((physical >= limits[:, 0]) & (physical <= limits[:, 1])).all()
        )

    fixture_root = Path(fixtures) if fixtures is not None else policy.root / "fixtures"
    paths = sorted(fixture_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No offline test observations in {fixture_root}")
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as arrays:
            fixture = {
                "path": str(path),
                "task": str(arrays["task"].item()),
                "episode": int(arrays["episode"].item()),
                "frame_index": int(arrays["frame_index"].item()),
            }
            observation = observation_from_fixture(
                arrays, policy.adapter.codec.config["recorded_feature_names"]
            )
            encoded = policy.adapter.codec.state(arrays["motor_state"])
            np.testing.assert_allclose(encoded, arrays["state"], atol=2e-6, rtol=0)
            normalized = policy.adapter.normalizer.transform(
                torch.from_numpy(arrays["action"]), "action"
            )
            np.testing.assert_allclose(
                policy.adapter.decode_actions(normalized),
                arrays["action"],
                atol=2e-7,
                rtol=0,
            )
            baseline, baseline_errors = [], []
            for step, (motor, action) in enumerate(
                zip(arrays["motor_state"], arrays["action"])
            ):
                try:
                    baseline.append(asdict(controller.solve(motor, action)))
                except ValueError as exc:
                    baseline_errors.append({"step": step, "error": str(exc)})
            if policy.model.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            predicted = policy.predict(observation, fixture["task"])
            if policy.model.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            generated, errors = [], []
            # Teacher-forced recorded feedback verifies per-row geometry only;
            # it is not a simulated closed-loop trajectory or a success rate.
            for step, (motor, action) in enumerate(
                zip(arrays["motor_state"], predicted)
            ):
                try:
                    generated.append(asdict(controller.solve(motor, action)))
                except ValueError as exc:
                    errors.append({"step": step, "error": str(exc)})
            item = {
                **fixture,
                "inference_seconds": elapsed,
                "action_shape": list(predicted.shape),
                "all_finite": bool(np.isfinite(predicted).all()),
                "reference_ik": {"solved": len(baseline), "errors": baseline_errors},
                "predicted_ik": {"solved": len(generated), "errors": errors},
                "predicted_outside_driver_limits": sum(
                    not x["within_driver_limits"] for x in generated
                ),
                "predicted_position_error_max_m": max(
                    (x["position_error_m"] for x in generated), default=None
                ),
                "predicted_rotation_error_max_rad": max(
                    (x["rotation_error_rad"] for x in generated), default=None
                ),
                "driver_position_error_max_m": max(
                    (x["driver_position_error_m"] for x in generated), default=None
                ),
                "driver_rotation_error_max_rad": max(
                    (x["driver_rotation_error_rad"] for x in generated), default=None
                ),
                "all_applied_targets_within_driver_limits": bool(generated)
                and all(applied_target_valid(x) for x in generated),
                "physical_action_min": predicted.min(0).tolist(),
                "physical_action_max": predicted.max(0).tolist(),
            }
            if index == 0:
                repeated = policy.predict(observation, fixture["task"])
                item["repeat_max_abs_difference"] = float(
                    np.max(np.abs(repeated - predicted))
                )
                np.testing.assert_allclose(repeated, predicted, atol=1e-6, rtol=0)
            np.savez_compressed(
                output / f"prediction_{index}.npz",
                action=predicted,
                reference=arrays["action"],
                motor_state=arrays["motor_state"],
            )
            (output / f"ik_{index}.json").write_text(
                json.dumps({"reference": baseline, "predicted": generated}, indent=2)
                + "\n"
            )
            if (
                errors
                or baseline_errors
                or not item["all_applied_targets_within_driver_limits"]
            ):
                failures.append(fixture["path"])
            report["fixtures"].append(item)
            print(json.dumps(item), flush=True)
    if policy.model.device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["passed"] = not failures
    report["hardware_modules_loaded"] = [
        name
        for name in sys.modules
        if name.startswith(
            ("lerobot.robots.", "lerobot_robot_seeed_b601", "motorbridge")
        )
    ]
    if report["hardware_modules_loaded"]:
        raise RuntimeError("Offline execution unexpectedly imported a robot driver")
    report["all_predicted_targets_within_driver_limits"] = not failures and not any(
        x["predicted_outside_driver_limits"] for x in report["fixtures"]
    )
    report["all_applied_targets_within_driver_limits"] = not failures and all(
        x["all_applied_targets_within_driver_limits"] for x in report["fixtures"]
    )
    report["limitations"] = (
        "Recorded open-loop inputs only; no robot connection, contact dynamics, "
        "calibration validation, or task success evaluation."
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "passed": report["passed"],
                "all_predicted_targets_within_driver_limits": report[
                    "all_predicted_targets_within_driver_limits"
                ],
                "all_applied_targets_within_driver_limits": report[
                    "all_applied_targets_within_driver_limits"
                ],
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


def live(*args, **kwargs):
    raise RuntimeError(
        "Legacy direct-driver live mode is retired; use experiments.b601.local run"
    )


def main():
    cache_root = Path(
        os.environ.get(
            "FASTWAM_CACHE_ROOT",
            str(
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "fastwam"
            ),
        )
    ).expanduser()
    deployment_root = Path(
        os.environ.get("FASTWAM_DEPLOYMENT_ROOT", str(cache_root / "deployment"))
    ).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixtures", type=Path, help="Directory of recorded offline test observations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=deployment_root / "reports" / time.strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--replan-steps", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "live":
        parser.error(
            "Use bash experiments/b601/local.sh run <cube|pen|drawer>; direct-driver live mode is retired"
        )
    if args.mode == "offline":
        sys.addaudithook(guard_offline_hardware)
    if args.mode == "live" and (args.robot_config is None or args.task is None):
        parser.error("live mode requires --robot-config and --task")
    if (
        not 1 <= args.replan_steps <= 32
        or not math.isfinite(args.duration)
        or args.duration <= 0
    ):
        parser.error("Use 1 <= replan-steps <= 32 and a finite positive duration")
    if args.num_inference_steps < 1:
        parser.error("num-inference-steps must be positive")
    args.checkpoint = args.checkpoint.expanduser().absolute()
    args.output = args.output.expanduser().absolute()
    if args.robot_config is not None:
        args.robot_config = args.robot_config.expanduser().resolve()
        if args.mode == "live" and not args.robot_config.is_file():
            parser.error(f"Robot configuration does not exist: {args.robot_config}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arguments.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )
    start = time.perf_counter()
    policy = B601Policy(
        args.checkpoint,
        args.device,
        args.num_inference_steps,
        args.seed,
    )
    print(
        json.dumps(
            {
                "loaded_step": policy.checkpoint_step,
                "load_seconds": time.perf_counter() - start,
            }
        ),
        flush=True,
    )
    controller = EEDeltaController(policy.adapter.codec)
    return (
        offline(policy, controller, args.output, args.fixtures)
        if args.mode == "offline"
        else live(policy, controller, args, args.output)
    )


if __name__ == "__main__":
    raise SystemExit(main())
