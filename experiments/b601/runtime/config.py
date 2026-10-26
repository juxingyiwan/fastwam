"""Strict FastWAM deployment configuration; machine choices stay outside source."""

import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RobotConfig:
    can_channel: str = "can0"
    robot_id: str = "follower1"
    camera_devices: dict = field(default_factory=dict)
    camera_buffer_size: int = 2
    camera_mapping_verified: bool = False
    width: int = 640
    height: int = 480
    fps: int = 30
    max_camera_age_s: float = 0.15
    max_camera_skew_s: float = 0.05
    camera_recovery_timeout_s: float = 3.0
    camera_max_consecutive_failures: int = 3
    camera_max_reopens: int = 3
    max_feedback_age_s: float = 0.12
    min_temperature_c: float = -25.0
    max_temperature_c: float = 65.0
    temperature_feedback_grace_s: float = 0.5
    lock_paths: tuple = ()


@dataclass(frozen=True)
class ControlConfig:
    hz: int = 30
    execution_horizon: int = 16
    duration_s: float = 300.0
    max_policy_chunks: int = 100000
    max_tick_s: float = 0.15
    inference_timeout_s: float = 15.0
    # Legacy configuration keys, accepted for existing machine configs only.
    # Synchronous execution does not reject chunks by response age/state drift.
    max_response_age_s: float = 2.0
    max_state_drift_arm_deg: float = 2.0
    max_state_drift_gripper_deg: float = 15.0
    settle_gripper_deg: float = 3.0
    startup_home: bool = True
    startup_gripper_speed_deg_s: float = 60.0


@dataclass(frozen=True)
class ShutdownConfig:
    worker_timeout_s: float = 2.0
    home_timeout_s: float = 30.0
    arm_tolerance_deg: float = 2.0
    settle_s: float = 0.25
    feedback_recovery_timeout_s: float = 1.0


@dataclass(frozen=True)
class PolicyConfig:
    horizon: int = 32
    camera_names: tuple = ("front", "side")


@dataclass(frozen=True)
class ModelConfig:
    default: str = "5epoch"
    checkpoints: dict = field(default_factory=dict)
    device: str = "cuda"
    num_inference_steps: int = 10
    seed: int = 42


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    guidance_mode: str = "lerobot"
    guidance_start: float = 0.0
    replan_interval_s: float = 0.0
    max_guidance: float = 10.0
    latency_budget_s: float = 0.20
    guard_s: float = 0.0
    blend_steps: int = 1


@dataclass(frozen=True)
class Config:
    task: str = "Grasp the cube to the bowl"
    output_root: str = "~/.cache/fastwam/runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    shutdown: ShutdownConfig = field(default_factory=ShutdownConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    display_data: bool = True
    save_rollout: bool = True
    display_ip: str | None = None
    display_port: int = 9878
    rtc: RTCConfig = field(default_factory=RTCConfig)

    def to_dict(self):
        return asdict(self)


def section(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path):
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Deployment configuration must be a mapping")
    data = dict(data)
    for key, cls in [
        ("model", ModelConfig),
        ("robot", RobotConfig),
        ("control", ControlConfig),
        ("shutdown", ShutdownConfig),
        ("rtc", RTCConfig),
    ]:
        data[key] = section(cls, data.get(key, {}))
    if "policy" in data:
        raise ValueError(
            "FastWAM camera/action representation is fixed by the trained model"
        )
    cfg = section(Config, data)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    r, c, s, m = cfg.robot, cfg.control, cfg.shutdown, cfg.model
    rtc = cfg.rtc
    if type(rtc.enabled) is not bool or rtc.guidance_mode not in ("lerobot", "pi_vjp"):
        raise ValueError("RTC requires enabled=true/false and guidance_mode=lerobot/pi_vjp")
    if (
        type(rtc.guidance_start) not in (float, int)
        or not math.isfinite(rtc.guidance_start)
        or not 0 <= rtc.guidance_start < 1
    ):
        raise ValueError("rtc.guidance_start must be in [0,1)")
    for name in ("max_guidance", "latency_budget_s"):
        value = getattr(rtc, name)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"rtc.{name} must be finite and positive")
    for name in ("replan_interval_s", "guard_s"):
        value = getattr(rtc, name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"rtc.{name} must be finite and nonnegative")
    if (
        rtc.max_guidance > 20
        or type(rtc.blend_steps) is not int
        or not 1 <= rtc.blend_steps <= 16
    ):
        raise ValueError("RTC requires max_guidance <= 20 and blend_steps in 1..16")
    for value in [
        cfg.display_data,
        cfg.save_rollout,
        r.camera_mapping_verified,
        c.startup_home,
    ]:
        if type(value) is not bool:
            raise ValueError("Use YAML boolean true/false")
    if not isinstance(cfg.task, str) or not cfg.task.strip() or not cfg.output_root:
        raise ValueError("task and output_root are required")
    if (
        set(r.camera_devices) != {"front", "side"}
        or len(set(r.camera_devices.values())) != 2
    ):
        raise ValueError("Configure two distinct front/side camera devices")
    if not r.lock_paths or len(set(r.lock_paths)) != len(r.lock_paths):
        raise ValueError("Configure the shared robot lock paths")
    if c.hz != 30 or r.fps != 30 or (r.width, r.height) != (640, 480):
        raise ValueError("B601 requires 30 Hz and 640x480 camera input")
    for label, value, low, high in [
        ("execution_horizon", c.execution_horizon, 1, 32),
        ("camera_buffer_size", r.camera_buffer_size, 1, 2),
        ("camera_max_reopens", r.camera_max_reopens, 0, 10),
        ("camera_max_consecutive_failures", r.camera_max_consecutive_failures, 1, 100),
        ("max_policy_chunks", c.max_policy_chunks, 1, 10000000),
        ("num_inference_steps", m.num_inference_steps, 1, 1000),
    ]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{label} must be an integer in {low}..{high}")
    for group in [r, c, s]:
        for name, value in asdict(group).items():
            if name.endswith(("_s", "_deg", "_c")) and name not in (
                "min_temperature_c",
                "temperature_feedback_grace_s",
            ):
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{name} must be finite and positive")
    if (
        type(r.min_temperature_c) not in (int, float)
        or not math.isfinite(r.min_temperature_c)
        or r.min_temperature_c >= r.max_temperature_c - 5
    ):
        raise ValueError(
            "min_temperature_c must be finite and below max_temperature_c minus 5"
        )
    if (
        type(r.temperature_feedback_grace_s) not in (int, float)
        or not math.isfinite(r.temperature_feedback_grace_s)
        or not 0 <= r.temperature_feedback_grace_s <= 1
    ):
        raise ValueError("temperature_feedback_grace_s must be in 0..1")
    if not s.settle_s < s.feedback_recovery_timeout_s <= 5:
        raise ValueError("feedback_recovery_timeout_s must exceed settle_s and be <= 5")
    if not isinstance(m.checkpoints, dict) or m.default not in m.checkpoints:
        raise ValueError("Model default must name a configured checkpoint")
    if any(not isinstance(v, str) or not v for v in m.checkpoints.values()):
        raise ValueError("Model checkpoints must contain explicit paths")


def expand(path):
    return Path(os.path.expandvars(str(path))).expanduser().absolute()
