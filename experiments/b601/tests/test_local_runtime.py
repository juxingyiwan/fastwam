"""FastWAM model/action boundaries and episode lifecycle, with no hardware."""

import json
import time
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import pytest
import yaml

from experiments.b601.local import hardware_session, main
from experiments.b601.fastwam_policy.controller import IKResult
from experiments.b601.runtime import controller
from experiments.b601.runtime.config import Config, ModelConfig, RobotConfig
from experiments.b601.runtime.connection import FastWAMConnection
from experiments.b601.runtime.contracts import (
    JOINTS,
    HardwareError,
    Observation,
    Prediction,
)
from experiments.b601.runtime.fastwam_hardware import FastWAMHardware
from experiments.b601.runtime.locks import robot_locks
from experiments.b601.runtime.recording import Recorder


@pytest.fixture
def cfg(tmp_path):
    return Config(
        task="Grasp the cube to the bowl",
        output_root=str(tmp_path / "runs"),
        model=ModelConfig(
            checkpoints={
                "5epoch": str(tmp_path / "5/checkpoint.pt"),
                "10epoch": str(tmp_path / "10/checkpoint.pt"),
            }
        ),
        robot=RobotConfig(
            camera_devices={"front": "front", "side": "side"},
            camera_mapping_verified=True,
            lock_paths=(str(tmp_path / "shared.lock"),),
        ),
        save_rollout=False,
    )


def observed(stamp, state=None):
    return Observation(
        np.zeros(7) if state is None else state,
        {k: np.zeros((32, 32, 3), np.uint8) for k in ("front", "side")},
        stamp,
        time.time(),
        stamp,
    )


def test_model_bridge_keeps_rgb_and_physical_motor_order():
    captured = {}

    def predict(obs, task):
        captured.update(obs)
        return np.ones((32, 7))

    policy = N(
        adapter=N(
            codec=N(config={"recorded_feature_names": [j + ".pos" for j in JOINTS]})
        ),
        predict=predict,
    )
    obs = observed(time.monotonic(), np.arange(7, dtype=float))
    result = FastWAMConnection(policy).predict(obs, "fixture", 3)
    assert result.generation == 3
    assert list(captured)[:7] == [j + ".pos" for j in JOINTS]
    assert captured["front"] is obs.views["front"]  # retain the original RGB array
    assert result.actions.shape == (32, 7)


def test_decode_preserves_outside_command_for_native_driver(cfg):
    target = [0, -0.015, 0, 0, 0, 0, 0]
    result = IKResult(dict(zip((j + ".pos" for j in JOINTS), target)), 0, 0, 4, False)
    codec = N(scale=np.ones(7), config={"driver_motor_limits_deg": [[0, 170]] * 7})
    hw = FastWAMHardware(
        cfg.robot,
        lambda *a, **k: None,
        N(solve=lambda *a: result, codec=codec),
    )
    raw, diagnostic = hw.decode(np.zeros(7), np.zeros(7))
    np.testing.assert_array_equal(raw, target)
    assert diagnostic["within_driver_limits"] is False


@pytest.mark.parametrize("wrong_mapping", [False, True])
def test_native_send_clamps_physical_targets_and_still_rejects_mapping_mismatch(
    cfg, wrong_mapping
):
    hw = FastWAMHardware(cfg.robot, lambda *a, **k: None, None)
    hw.directions = np.array([1, 1, -1, -1, -1, 1, 6])
    hw.limits = np.array(
        [[-145, 145], [0, 170], [0, 200], [-80, 90], [-90, 90], [-130, 130], [0, 270]]
    )
    recorded = np.array([150, -0.2, 0.1, 85, -100, -140, 46])
    expected = np.array([145, 0, 0, -80, 90, -130, 270])
    received, inhibited = [], []

    def send_action(action):
        received.append(action)
        target = expected + 1 if wrong_mapping else expected
        return dict(zip((j + ".pos" for j in JOINTS), target))

    hw.robot = N(send_action=send_action)
    hw.inhibit_home = inhibited.append
    if wrong_mapping:
        with pytest.raises(HardwareError, match="direction/limit mapping"):
            hw.send(recorded)
        assert inhibited == ["Driver target mapping mismatch"]
    else:
        result = hw.send(recorded)
        assert result["driver_target_physical"] == expected.tolist()
        assert result["native_limit_clipped"] == [True] * 7
        assert not inhibited
    assert list(received[0].values()) == recorded.tolist()


def test_rejection_logs_measured_state_and_unsent_action(capsys):
    measured = np.arange(7, dtype=float)
    action = np.arange(7, dtype=float) / 100
    events = []

    def reject(*args):
        raise ValueError("IK did not reach target")

    recorder = N(event=lambda kind, **fields: events.append((kind, fields)))
    with pytest.raises(ValueError) as error:
        controller.decode_action(
            N(decode=reject),
            measured,
            action,
            recorder,
            phase="policy",
            query_id="q1",
            action_index=5,
            state_monotonic=12.5,
        )
    kind, evidence = events[0]
    assert kind == "action_rejected"
    assert evidence == error.value.action_context
    assert evidence["observed_physical"] == measured.tolist()
    assert evidence["ee_delta"] == action.tolist()
    assert evidence["action_index"] == 5 and evidence["dispatched"] is False
    assert "index=5" in capsys.readouterr().err


@pytest.mark.parametrize("persistent", [False, True])
def test_preflight_retries_image_expired_during_feedback_read(
    cfg, tmp_path, monkeypatch, persistent
):
    clock = [10.0]
    reads, predicted, events = [], [], []
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    def observe(**kwargs):
        reads.append(kwargs)
        stale = persistent or len(reads) == 1
        obs = observed(clock[0])
        if stale:
            obs = replace(obs, captured_monotonic=clock[0] - 0.16)
        return obs, {}

    def predict(obs, task):
        predicted.append(obs)
        return N(actions=np.zeros((32, 7)), request_id="preflight", roundtrip_s=0)

    hw = N(
        observe=observe,
        observe_state=lambda **kw: (np.zeros(7), {}),
        decode=lambda *args: (np.zeros(7), {}),
    )
    connection = N(open=lambda: {}, predict=predict)
    recorder = N(
        path=tmp_path,
        observation=lambda *a: None,
        prediction=lambda *a: None,
        event=lambda kind, **kw: events.append(kind),
    )
    if persistent:
        with pytest.raises(HardwareError, match="Camera recovery timed out"):
            controller.preflight(hw, connection, cfg, recorder)
        assert not predicted
        assert clock[0] <= 10 + cfg.robot.camera_recovery_timeout_s + 1 / cfg.control.hz
    else:
        controller.preflight(hw, connection, cfg, recorder)
        assert len(reads) == 2 and len(predicted) == 1
        assert "camera_wait_recovered" in events
        assert clock[0] - predicted[0].captured_monotonic <= cfg.robot.max_camera_age_s


@pytest.mark.parametrize("timestamp", [float("nan"), float("-inf"), 20.0])
def test_preflight_invalid_timestamp_is_not_retried(
    cfg, tmp_path, monkeypatch, timestamp
):
    monkeypatch.setattr(controller.time, "monotonic", lambda: 10.0)
    obs = replace(observed(10.0), captured_monotonic=timestamp)
    hw = N(observe=lambda **kw: (obs, {}))
    recorder = N(path=tmp_path, event=lambda *a, **kw: None)
    with pytest.raises(controller.ProtocolError, match="invalid timestamp"):
        controller.preflight(hw, N(open=lambda: {}), cfg, recorder)


def run_loop(
    cfg,
    tmp_path,
    monkeypatch,
    *,
    solve_seconds=0,
    pending_forever=False,
    stale=False,
    inference_timeout_s=15,
    reject_action=False,
    clip_target=False,
):
    clock = [0.0]
    sent = []
    anchors = []
    records = []
    closed = []
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    def decode(state, action):
        anchors.append(state.copy())
        if reject_action:
            raise ValueError("IK did not reach target")
        clock[0] += solve_seconds
        return state + action, {"within_driver_limits": True}

    def send(raw):
        sent.append((clock[0], raw.copy()))
        applied = np.clip(raw, 0, 0.05) if clip_target else raw
        return {
            "driver_target_physical": applied.tolist(),
            "native_limit_clipped": (np.abs(applied - raw) > 1e-5).tolist(),
        }

    hw = N(
        directions=np.ones(7),
        observe=lambda: (observed(clock[0]), {}),
        observe_state=lambda: (np.full(7, clock[0]), {}),
        hold_command=lambda state: np.zeros(7),
        decode=decode,
        send=send,
        close=lambda: closed.append("hw"),
    )

    def predict(obs, task, generation):
        source = (
            replace(obs, captured_monotonic=obs.captured_monotonic - 5)
            if stale
            else obs
        )
        return Prediction(
            "query", generation, np.full((32, 7), 0.1), source, clock[0], 0, {}
        )

    connection = N(predict=predict, close=lambda: closed.append("model"))

    class Worker:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, *args):
            f = Future()
            if not pending_forever:
                f.set_result(fn(*args))
            return f

    monkeypatch.setattr(controller, "ThreadPoolExecutor", Worker)

    def finish(hw, connection, report, pending, *a, **kw):
        hw.close()
        connection.close()

    monkeypatch.setattr(controller, "finish_episode", finish)
    recorder = N(
        path=tmp_path,
        event=lambda *a, **kw: None,
        check_media=lambda: None,
        prediction_async=lambda *a: None,
        sample=lambda *a, **kw: None,
        append=lambda *a, **kw: records.append(kw),
    )
    cfg = replace(
        cfg,
        control=replace(
            cfg.control,
            duration_s=0.15,
            execution_horizon=2,
            max_tick_s=0.15,
            inference_timeout_s=inference_timeout_s,
        ),
    )
    report = controller.control_loop(hw, connection, cfg, recorder, lambda: "")
    return report, sent, anchors, records, closed


def test_chunk_ik_uses_new_feedback_for_every_action(cfg, tmp_path, monkeypatch):
    report, sent, anchors, records, closed = run_loop(cfg, tmp_path, monkeypatch)
    assert report["policy_commands"] >= 2
    assert anchors[1][0] > anchors[0][0]
    commands = [r for r in records if r["phase"] == "policy"]
    assert commands[0]["ee_delta"] == [0.1] * 7
    np.testing.assert_allclose(
        commands[0]["executed_recorded_command"], anchors[0] + 0.1
    )
    assert closed == ["hw", "model"]


def test_slow_ik_never_dispatches_after_episode_deadline(cfg, tmp_path, monkeypatch):
    report, sent, anchors, records, closed = run_loop(
        cfg, tmp_path, monkeypatch, solve_seconds=1
    )
    assert report["outcome"] == "no_actions"
    assert anchors and all(t < 0.15 for t, _ in sent)
    assert not [r for r in records if r["phase"] == "policy"]


def test_native_limit_clipping_continues_episode_and_is_counted(
    cfg, tmp_path, monkeypatch
):
    report, _, _, records, closed = run_loop(
        cfg, tmp_path, monkeypatch, clip_target=True
    )
    assert report["outcome"] == "timeout"
    assert report["policy_commands"] >= 2
    assert report["policy_clipped_commands"] == report["policy_commands"]
    assert all(
        n == report["policy_commands"] for n in report["policy_clipped_joints"].values()
    )
    commands = [r for r in records if r["phase"] == "policy"]
    assert all(r["driver_target_physical"] == [0.05] * 7 for r in commands)
    assert all(min(r["raw_physical"]) > 0.05 for r in commands)
    assert closed == ["hw", "model"]


def test_episode_report_retains_rejection_and_sends_no_policy_target(
    cfg, tmp_path, monkeypatch
):
    report, sent, anchors, records, closed = run_loop(
        cfg,
        tmp_path,
        monkeypatch,
        reject_action=True,
    )
    assert report["outcome"] == "fault" and report["policy_commands"] == 0
    assert report["rejected_action"]["observed_physical"] == anchors[0].tolist()
    assert report["rejected_action"]["ee_delta"] == [0.1] * 7
    assert report["rejected_action"]["dispatched"] is False
    assert not [r for r in records if r["phase"] == "policy"]
    assert closed == ["hw", "model"]


def test_hung_inference_has_no_actions_outcome_and_cleanup(cfg, tmp_path, monkeypatch):
    report, _, _, _, closed = run_loop(cfg, tmp_path, monkeypatch, pending_forever=True)
    assert report["outcome"] == "no_actions"
    assert closed == ["hw", "model"]


def test_inference_timeout_ends_episode_before_total_budget(cfg, tmp_path, monkeypatch):
    report, sent, _, _, closed = run_loop(
        cfg, tmp_path, monkeypatch, pending_forever=True, inference_timeout_s=0.05
    )
    assert report["outcome"] == "fault" and "inference_timeout_s" in report["error"]
    assert all(t <= 0.05 for t, _ in sent)
    assert closed == ["hw", "model"]


def test_changed_driver_is_rejected_before_devices_open(cfg, tmp_path, monkeypatch):
    from experiments.b601.runtime import fastwam_hardware

    monkeypatch.setattr(
        fastwam_hardware.importlib.util,
        "find_spec",
        lambda name: N(origin=str(tmp_path / "__init__.py")),
    )
    opened = []
    monkeypatch.setattr(
        fastwam_hardware.B601, "open_observation", lambda self: opened.append(True)
    )
    with pytest.raises(HardwareError, match="source differs"):
        FastWAMHardware(cfg.robot, lambda *a, **k: None, None).open_observation()
    assert not opened


def test_sync_response_age_alone_does_not_reject_current_session(
    cfg, tmp_path, monkeypatch
):
    report, _, anchors, _, closed = run_loop(cfg, tmp_path, monkeypatch, stale=True)
    assert report["outcome"] == "timeout" and "error" not in report
    assert report["discarded_responses"] == 0
    assert anchors and closed == ["hw", "model"]


def test_partial_open_is_always_closed(cfg, tmp_path, monkeypatch):
    from experiments.b601.runtime import fastwam_hardware

    events = []

    def fail():
        events.append("bus_open")
        raise RuntimeError("camera failed")

    hw = N(
        open_observation=fail,
        close=lambda: events.append("bus_close"),
        shutdown_report={"homing": "not_required"},
    )
    monkeypatch.setattr(fastwam_hardware, "FastWAMHardware", lambda *a, **kw: hw)
    report = hardware_session(
        cfg, "preflight", N(path=tmp_path, event=lambda *a, **kw: None), None, None
    )
    assert events == ["bus_open", "bus_close"] and report["outcome"] == "fault"


def test_same_robot_lock_blocks_another_client(cfg):
    with robot_locks(cfg.robot.lock_paths):
        with pytest.raises(HardwareError, match="already reserved"):
            with robot_locks(cfg.robot.lock_paths):
                pass


def test_missing_model_fails_instead_of_using_default(cfg, tmp_path):
    config = tmp_path / "deploy.yml"
    value = cfg.to_dict()
    value.pop("policy")
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(SystemExit) as exc:
        main(["check", "--config", str(config), "--model", "10epoch"])
    assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_media_can_record_in_fastwam_environment_without_camera(cfg, tmp_path):
    pytest.importorskip("rerun")
    cfg = replace(cfg, save_rollout=True, display_data=False)
    recorder = Recorder(cfg, "synthetic", str(tmp_path / "recording"))
    recorder.start_telemetry()
    recorder.sample(
        observed(time.monotonic()), action=[0] * 7, physical=[0] * 7, phase="hold"
    )
    recorder.close({"outcome": "synthetic_complete"})
    report = json.loads((recorder.path / "report.json").read_text())
    assert report["log_error"] is None
    assert report["rollout_media"]["video_frames"] == 1
    assert (recorder.path / "front.avi").stat().st_size > 0
    assert (recorder.path / "side.avi").stat().st_size > 0
    assert (recorder.path / "rollout.rrd").stat().st_size > 0
