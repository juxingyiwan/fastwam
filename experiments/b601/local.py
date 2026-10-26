"""One local entry point for FastWAM B601 check, preflight and attended runs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .runtime.config import expand, load_config, validate_config
from .runtime.recording import Recorder, write_json

TASKS = {
    "cube": "Grasp the cube to the bowl",
    "pen": "Grasp the pen to the box",
    "drawer": "Grasp the cube into the closed top drawer",
}


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


@contextmanager
def termination_signals():
    def stop(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    root = Path(__file__).resolve().parent
    files = [p for p in root.rglob("*.py") if "tests" not in p.parts]
    files += [
        root / "runtime" / name for name in ("upstream.json", "driver-source.json")
    ]
    result = {str(p.relative_to(root)): file_hash(p) for p in files}
    for name in (
        "fastwam.datasets.lerobot3.ee_delta",
        "fastwam.datasets.lerobot.processors.ee_delta_processor",
        "fastwam.models.wan22.fastwam",
        "fastwam.models.wan22.rtc_adapter",
        "fastwam.models.rtc.condition",
        "fastwam.models.rtc.guidance",
    ):
        import importlib.util

        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            result[name] = file_hash(spec.origin)
    return result


def run_directory(cfg, args, model):
    if args.output:
        return expand(args.output)
    stamp = datetime.now().astimezone()
    task = next((k for k, v in TASKS.items() if v == cfg.task), "custom")
    # Model names are validated before use as path components.
    return (
        expand(cfg.output_root)
        / stamp.strftime("%Y-%m-%d")
        / f"{model}-{task}-{args.operation}-{stamp:%H%M%S}-{uuid.uuid4().hex[:8]}"
    )


def hardware_session(cfg, mode, recorder, connection, decoder, *, locks_held=False):
    from .runtime.controller import control_loop, operator_terminal, preflight
    from .runtime.fastwam_hardware import FastWAMHardware
    from .runtime.lifecycle import protect_shutdown
    from .runtime.hardware.homing import HomingTimeout
    from .runtime.locks import robot_locks

    if cfg.rtc.enabled:
        from .runtime.rtc_controller import control_loop, validate_capability

        write_json(
            recorder.path / "rtc_capability.json", validate_capability(connection, cfg)
        )
    if not cfg.robot.camera_mapping_verified:
        raise ValueError(
            "Confirm the installed front/side mapping before preflight or run"
        )
    import logging

    logging.getLogger("experiments.b601.runtime.hardware.b601").setLevel(logging.INFO)
    hw = FastWAMHardware(cfg.robot, recorder.event, decoder, shutdown=cfg.shutdown)
    report = {"outcome": "running", "policy_commands": 0}
    # Keep the machine's shared robot lock inodes until cleanup ends.
    with (nullcontext() if locks_held else robot_locks(cfg.robot.lock_paths)):
        try:
            hw.open_observation()
            write_json(recorder.path / "hardware.json", hw.manifest())
            calibration = getattr(hw.robot, "calibration_fpath", None)
            if calibration and Path(calibration).is_file():
                value = Path(calibration).read_bytes()
                (recorder.path / "calibration.json").write_bytes(value)
                recorder.event(
                    "calibration_snapshot",
                    source=str(calibration),
                    sha256=hashlib.sha256(value).hexdigest(),
                )
            recorder.start_telemetry()
            prediction = preflight(hw, connection, cfg, recorder)
            if cfg.rtc.enabled:
                from .runtime.rtc_warmup import warmup_rtc

                warmup_rtc(connection, prediction, cfg, recorder)
            if mode == "preflight":
                report.update(outcome="preflight_complete", motors_enabled=False)
            else:
                with operator_terminal() as read_key:
                    hw.enable(cfg.control)
                    report = control_loop(hw, connection, cfg, recorder, read_key)
        except KeyboardInterrupt:
            report.update(outcome="aborted")
        except (Exception, HomingTimeout) as exc:
            report.update(outcome="fault", error=f"{type(exc).__name__}: {exc}")
            if hasattr(exc, "homing_context"):
                report["homing_diagnostic"] = exc.homing_context
            if hasattr(exc, "action_context"):
                report["rejected_action"] = exc.action_context
        finally:
            with protect_shutdown():
                try:
                    hw.close()
                except BaseException as exc:
                    report.update(
                        outcome="fault", shutdown_error=f"{type(exc).__name__}: {exc}"
                    )
                report["shutdown"] = dict(hw.shutdown_report)
    return report


def execute(cfg, args, model, checkpoint, recorder):
    from .runtime.locks import robot_locks

    # Reserve before allocating the model: an offline replay and an attended
    # launch otherwise race for VRAM before the hardware-session lock is checked.
    with robot_locks(cfg.robot.lock_paths):
        return _execute_locked(cfg, args, model, checkpoint, recorder)


def _execute_locked(cfg, args, model, checkpoint, recorder):
    from .deploy import guard_offline_hardware, offline
    from .fastwam_policy.controller import EEDeltaController
    from .fastwam_policy.policy import B601Policy
    from .runtime.connection import FastWAMConnection

    if args.operation == "check":
        sys.addaudithook(guard_offline_hardware)
    print(f"Loading {model}: {checkpoint}", flush=True)
    if cfg.rtc.enabled and args.operation != "check":
        from .fastwam_policy.observation import ObservationAdapter
        from .runtime.process_connection import FastWAMProcessConnection

        root = Path(checkpoint).expanduser().absolute().parent
        adapter = ObservationAdapter(
            root / "codec.json", root / "robot.urdf", root / "norm_stats.json"
        )
        decoder = EEDeltaController(adapter.codec)
        connection = FastWAMProcessConnection(
            checkpoint, cfg.model.device, cfg.model.num_inference_steps, cfg.model.seed,
            task=cfg.task, log_path=recorder.path / "model-worker.log",
            inference_timeout_s=cfg.control.inference_timeout_s,
        )
        try:
            write_json(recorder.path / "model.json", connection.open())
            return hardware_session(cfg, args.operation, recorder, connection, decoder,
                                    locks_held=True)
        finally:
            connection.close()
    policy = B601Policy(
        checkpoint,
        cfg.model.device,
        cfg.model.num_inference_steps,
        cfg.model.seed,
    )
    policy.load_context(cfg.task)
    decoder = EEDeltaController(policy.adapter.codec)
    connection = FastWAMConnection(policy)
    write_json(recorder.path / "model.json", connection.open())
    if args.operation == "check":
        code = offline(policy, decoder, recorder.path)
        report = json.loads((recorder.path / "report.json").read_text())
        if cfg.rtc.enabled:
            from .check_rtc import check_rtc

            report["rtc_check"] = check_rtc(policy, recorder.path, cfg.rtc)
            if not report["rtc_check"]["all_teacher_forced_ik_solved"]:
                code = 2
        return {
            **report,
            "outcome": (
                "check_complete"
                if code == 0 and report["all_predicted_targets_within_driver_limits"]
                else "fault"
            ),
        }
    try:
        return hardware_session(cfg, args.operation, recorder, connection, decoder,
                                locks_held=True)
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["models", "config", "check", "preflight", "run"]
    )
    parser.add_argument("task", nargs="?", choices=list(TASKS))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("FASTWAM_DEPLOY_CONFIG", "~/.config/fastwam/deploy.yml")
        ).expanduser(),
    )
    parser.add_argument("--model", help="Explicit model name from deploy.yml")
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--rtc",
        choices=("true", "false"),
        default=None,
        help="true enables guided RTC; false selects synchronous execution; omitted uses config (default false)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--display-data",
        "--display",
        dest="display_data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show Rerun by default; --no-display-data disables the viewer",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.rtc is not None:
        cfg = replace(cfg, rtc=replace(cfg.rtc, enabled=args.rtc == "true"))
    model = args.model or cfg.model.default
    if model not in cfg.model.checkpoints:
        parser.error(f"Unknown model {model}; choices: {list(cfg.model.checkpoints)}")
    if not model or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in model
    ):
        parser.error(
            "Model names must contain only letters, digits, underscore or hyphen"
        )
    if args.task:
        cfg = replace(cfg, task=TASKS[args.task])
    if args.duration is not None:
        cfg = replace(cfg, control=replace(cfg.control, duration_s=args.duration))
    if args.display_data is not None:
        cfg = replace(cfg, display_data=args.display_data)
    validate_config(cfg)
    if args.operation == "models":
        rows = []
        for name, path in cfg.model.checkpoints.items():
            p = expand(path)
            row = {
                "model": name,
                "checkpoint": str(p),
                "state": "present_requires_check" if p.is_file() else "unavailable",
            }
            rows.append(row)
        print(json.dumps(rows, indent=2))
        return 0
    if args.operation == "config":
        print(json.dumps({"selected_model": model, **cfg.to_dict()}, indent=2))
        return 0
    if args.operation == "run" and not sys.stdin.isatty():
        parser.error("run requires an on-site interactive terminal")
    checkpoint = expand(cfg.model.checkpoints[model])
    # A failed model selection never falls back to another model.
    if not checkpoint.is_file():
        parser.error(
            f"Model {model} is unavailable: {checkpoint}; run models to inspect installed assets"
        )
    path = run_directory(cfg, args, model)
    recorder = Recorder(cfg, args.operation, str(path))
    print(f"Run directory: {path}", flush=True)
    report = {
        "outcome": "fault",
        "mode": args.operation,
        "selected_model": model,
        "task": cfg.task,
        "started_wall": time.time(),
    }
    with (
        (path / "console.log").open("x", encoding="utf-8") as log,
        redirect_stdout(Tee(sys.stdout, log)),
        redirect_stderr(Tee(sys.stderr, log)),
    ):
        try:
            print(f"RTC enabled: {str(cfg.rtc.enabled).lower()}", flush=True)
            versions = {}
            for package in (
                "torch",
                "lerobot",
                "lerobot-robot-seeed-b601",
                "motorbridge",
                "python-can",
                "rerun-sdk",
            ):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            write_json(
                path / "manifest.json",
                {
                    "python": sys.executable,
                    "platform": platform.platform(),
                    "versions": versions,
                    "source_sha256": source_manifest(),
                    "checkpoint_source": str(checkpoint),
                    "operation": args.operation,
                    "selected_model": model,
                    "task": cfg.task,
                },
            )
            with termination_signals():
                report.update(execute(cfg, args, model, checkpoint, recorder))
        except KeyboardInterrupt:
            report.update(outcome="aborted")
        except Exception as exc:
            report.update(outcome="fault", error=f"{type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            report["finished_wall"] = time.time()
            try:
                recorder.close(report)
            except Exception as exc:
                report.update(outcome="fault", recording_error=str(exc))
                write_json(path / "report.json", report)
            print(
                json.dumps(
                    {
                        "outcome": report["outcome"],
                        "error": report.get("error"),
                        "homing": report.get("shutdown", {}).get("homing"),
                        "homing_error": report.get("shutdown", {}).get("homing_error"),
                        "report": str(path / "report.json"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return (
        130
        if report["outcome"] == "aborted"
        else 2 if report["outcome"] in ("fault", "no_actions") else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
