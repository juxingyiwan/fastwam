"""Real spawned-process lifecycle tests using a lightweight deterministic model."""

import os
import time
from types import SimpleNamespace as N

import numpy as np
import pytest

from experiments.b601.runtime.config import Config
from experiments.b601.runtime.contracts import Observation, Prediction, ProtocolError
from experiments.b601.runtime.process_connection import FastWAMProcessConnection
from experiments.b601.runtime.rtc_contract import RTCRequest, acknowledgement
from experiments.b601.runtime.rtc_warmup import warmup_rtc


class FakeConnection:
    def open(self):
        return {"checkpoint_step": 10, "backend": "fake", "loaded_pid": os.getpid()}

    def rtc_capability(self):
        return {"supported": True, "flow_times": [0.0, 0.8, 0.9]}

    def predict(self, obs, task, generation=0, *, rtc_request=None):
        if task == "error":
            raise ValueError("model fixture failure")
        if task == "hang":
            time.sleep(60)
        if task == "exit":
            os._exit(7)
        if task == "hardware":
            import socket
            socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        actions = np.tile(obs.state, (32, 1)) + generation
        return Prediction("test", generation, actions, obs, time.monotonic(), 0.01,
                          {"pid": os.getpid()},
                          rtc=None if rtc_request is None else acknowledgement(rtc_request))

    def close(self):
        pass


def fake_factory(args, task):
    return FakeConnection()


def observation():
    now = time.monotonic()
    return Observation(np.arange(7, dtype=np.float32),
                       {k: np.zeros((16, 16, 3), np.uint8) for k in ("front", "side")},
                       now, time.time(), now)


def connection(**kwargs):
    return FastWAMProcessConnection("unused", "cpu", task="fixture",
                                   _factory=fake_factory, load_timeout_s=30, **kwargs)


def test_spawn_transports_generation_prefix_and_payload_then_reaps_child():
    conn = connection()
    try:
        assert conn.open()["inference_pid"] == conn.open()["loaded_pid"] != os.getpid()
        assert conn.rtc_capability()["supported"]
        obs = observation()
        request = RTCRequest(3, 4, np.ones((10, 7), np.float32), 8, 10)
        result = conn.predict(obs, "fixture", 3, rtc_request=request)
        np.testing.assert_array_equal(result.actions, np.tile(obs.state, (32, 1)) + 3)
        np.testing.assert_array_equal(result.observation.views["front"], obs.views["front"])
        assert result.rtc == acknowledgement(request)
        assert result.server_timing["pid"] != os.getpid()
        second = conn.predict(obs, "fixture", 4)
        assert second.generation == 4 and second.rtc is None
    finally:
        status = conn.close_with_status(2)
    # Slow shared hosts may need the bounded terminate fallback during teardown.
    assert status["status"] in ("joined", "terminated") and not status["timed_out"]
    assert not conn._process.is_alive()
    assert conn.close() == status
    with pytest.raises(ProtocolError, match="closed"):
        conn.predict(obs, "fixture")


@pytest.mark.parametrize("task,match", [
    ("hang", "deadline"), ("error", "model fixture failure"),
    ("exit", "disconnected|exited"), ("hardware", "CAN access is forbidden"),
])
def test_fault_and_timeout_cannot_leave_a_model_or_reuse_a_late_response(task, match):
    conn = connection(inference_timeout_s=0.2)
    started = time.monotonic()
    try:
        with pytest.raises(ProtocolError, match=match):
            conn.predict(observation(), task)
        assert time.monotonic() - started < 3
        assert not conn._process.is_alive()
        with pytest.raises(ProtocolError, match="closed"):
            conn.predict(observation(), "fixture")
    finally:
        conn.close()


def test_rtc_warmup_exercises_configured_guidance_without_exposing_actions():
    cfg = Config()
    obs = observation()
    actions = np.arange(224, dtype=np.float32).reshape(32, 7)
    prediction = N(generation=7, observation=obs, actions=actions)
    calls, events = [], []
    conn = N(predict=lambda *args, **kw: calls.append((args, kw)))
    recorder = N(event=lambda *args, **kw: events.append((args, kw)))
    assert warmup_rtc(conn, prediction, cfg, recorder) is None
    request = calls[0][1]["rtc_request"]
    assert request.guidance_mode == cfg.rtc.guidance_mode
    assert request.guidance_start == cfg.rtc.guidance_start
    assert request.committed_steps == 6 and request.guidance_horizon == 7
    assert calls[0][0] == (obs, cfg.task, 7)
    np.testing.assert_array_equal(request.prefix, actions[:7])
    assert not np.shares_memory(request.prefix, actions)
    assert events[0][0] == ("rtc_model_warmed",)
    assert events[0][1]["actions_executed"] == 0


def test_shutdown_interrupts_active_inference_without_waiting_for_its_deadline():
    import threading

    conn = connection(inference_timeout_s=15)
    errors = []
    def run():
        try:
            conn.predict(observation(), "hang")
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2
    while not conn._busy and time.monotonic() < deadline:
        time.sleep(.001)
    started = time.monotonic()
    status = conn.close_with_status(1)
    thread.join(2)
    assert not thread.is_alive() and errors
    assert time.monotonic() - started < 3
    assert not status['timed_out'] and not conn._process.is_alive()


def test_model_reservation_precedes_loading_and_covers_both_control_modes(tmp_path, monkeypatch):
    from dataclasses import replace
    from experiments.b601 import local
    from experiments.b601.runtime.locks import robot_locks
    from experiments.b601.runtime.contracts import HardwareError

    base = Config()
    cfg = replace(base, robot=replace(base.robot, lock_paths=(str(tmp_path / "robot.lock"),)))
    calls = []
    def load(owned_cfg, *args):
        with pytest.raises(HardwareError, match="already reserved"):
            with robot_locks(owned_cfg.robot.lock_paths):
                pytest.fail("Reservation was not held during model loading")
        calls.append(owned_cfg.rtc.enabled)
        return {"loaded": True}
    monkeypatch.setattr(local, "_execute_locked", load)
    for enabled in (False, True):
        candidate = replace(cfg, rtc=replace(cfg.rtc, enabled=enabled))
        assert local.execute(candidate, None, None, None, None) == {"loaded": True}
    assert calls == [False, True]
    with robot_locks(cfg.robot.lock_paths):
        with pytest.raises(HardwareError, match="already reserved"):
            local.execute(cfg, None, None, None, None)
    assert calls == [False, True]  # No model allocation on a conflicting launch.


def test_rtc_warmup_uses_only_the_configured_execution_window():
    from dataclasses import replace
    cfg = Config(control=replace(Config().control, execution_horizon=4))
    actions = np.arange(224, dtype=np.float32).reshape(32, 7)
    prediction = N(generation=0, observation=observation(), actions=actions)
    requests = []
    conn = N(predict=lambda *args, **kw: requests.append(kw["rtc_request"]))
    warmup_rtc(conn, prediction, cfg, N(event=lambda *a, **kw: None))
    request = requests[0]
    assert request.committed_steps == request.guidance_horizon == 4
    np.testing.assert_array_equal(request.prefix, actions[:4])
