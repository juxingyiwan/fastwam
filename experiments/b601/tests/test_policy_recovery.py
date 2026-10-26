"""Exercise complete control-loop recovery with fake feedback and delayed inference."""

from dataclasses import replace
from types import SimpleNamespace as N

import numpy as np
import pytest

from experiments.b601.runtime import controller
from experiments.b601.runtime.config import Config
from experiments.b601.runtime.contracts import (
    CameraUnavailable,
    HardwareError,
    Observation,
    Prediction,
    ProtocolError,
)
from experiments.b601.fastwam_policy.controller import IKUnreachable


def simulate(
    tmp_path,
    monkeypatch,
    *,
    first="healthy",
    duration=2.0,
    stress=False,
    interrupt=None,
    fatal=None,
    execution_horizon=3,
    engine=None,
):
    """Model calls finish after six ticks; only the control loop may send targets."""
    controller = engine or globals()["controller"]
    clock = [10.0]
    tick = [0]
    generation = [0]
    last_target = np.zeros(7)
    calls, events, commands, closed = [], [], [], []
    decoded = []
    views = {name: np.zeros((8, 8, 3), np.uint8) for name in ("front", "side")}
    cfg = Config()
    cfg = replace(
        cfg,
        control=replace(
            cfg.control, execution_horizon=execution_horizon, duration_s=duration
        ),
    )
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        controller.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt)
    )

    def observation():
        # These tests hold measured state constant so only injected invalidation
        # changes response admission; no claim about simulated robot dynamics.
        return Observation(
            np.zeros(7), views, clock[0], 0, clock[0], camera_generation=generation[0]
        )

    def observe():
        tick[0] += 1
        if interrupt == "camera" and tick[0] == 3:
            raise CameraUnavailable("fixture USB reconnect")
        if interrupt == "camera" and tick[0] >= 4:
            generation[0] = 1
        return observation(), {}

    def predict(obs, task, epoch, *, rtc_request=None):
        number = len(calls) + 1
        kind = first if number == 1 else "healthy"
        if stress:
            kind = {
                0: "state_drift",
                1: "response_expired",
                2: "ik_unreachable",
                3: "input_expired",
            }.get(number % 11, "healthy")
        calls.append(
            dict(number=number, kind=kind, epoch=epoch, observed=obs.state.copy())
        )
        if kind == "input_expired":
            raise controller.StalePrediction("input_expired")
        if kind == "invalid_payload":
            raise ProtocolError("invalid model output")
        source = obs
        if kind in ("state_drift", "gripper_drift"):
            state = obs.state.copy()
            state[1 if kind == "state_drift" else 6] += (
                2.3731327056884766 if kind == "state_drift" else 15.1
            )
            source = replace(obs, state=state)
        if kind == "response_expired":
            source = replace(obs, captured_monotonic=obs.captured_monotonic - 3)
        if kind == "invalid_timestamp":
            source = replace(obs, captured_monotonic=float("nan"))
        # Distinct actions let the assertions prove obsolete chunks never reach IK.
        actions = np.full((32, 7), number / 100)
        actions[:, 6] = np.arange(32) / 1000
        rtc = None
        if rtc_request is not None:
            from experiments.b601.runtime.rtc_contract import acknowledgement

            rtc = acknowledgement(rtc_request)
        return Prediction(
            str(number), epoch, actions, source, clock[0], 0.2, {}, rtc=rtc
        )

    class Delayed:
        def __init__(self, fn, args):
            self.ready = clock[0] + 0.2
            try:
                self.value, self.error = fn(*args), None
            except Exception as exc:
                self.value, self.error = None, exc

        def done(self):
            return clock[0] >= self.ready

        def result(self):
            assert self.done()
            if self.error:
                raise self.error
            return self.value

    class Worker:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, *args):
            return Delayed(fn, args)

    monkeypatch.setattr(controller, "ThreadPoolExecutor", Worker)
    monkeypatch.setattr(
        controller,
        "finish_episode",
        lambda hw, connection, *a, **k: (hw.close(), connection.close()),
    )

    def decode(state, action):
        number = int(round(action[0] * 100))
        count = sum(x == number for x in decoded)
        decoded.append(number)
        if calls[number - 1]["kind"] == "ik_unreachable" and count == 1:
            raise IKUnreachable("finite pose did not converge")
        if fatal == "decode":
            raise ValueError("nonfinite input or programming error")
        return action.copy(), {}

    def send(raw):
        if fatal == "hardware" and len(calls) >= 2:
            raise HardwareError("lost motor feedback")
        last_target[:] = raw
        return dict(
            driver_target_physical=raw.tolist(), native_limit_clipped=[False] * 7
        )

    hw = N(
        directions=np.ones(7),
        observe=observe,
        observe_state=lambda: (np.zeros(7), {}),
        hold_command=lambda state: last_target.copy(),
        decode=decode,
        send=send,
        close=lambda: closed.append("hardware"),
    )
    connection = N(
        predict=predict,
        close=lambda: closed.append("model"),
        rtc_capability=lambda: dict(
            supported=True,
            guidance_modes=["lerobot", "pi_vjp"],
            representation="base_frame_ee_delta_v2",
            model_dt_s=1 / 30,
            horizon=32,
        ),
    )
    recorder = N(
        path=tmp_path,
        event=lambda kind, **kw: events.append(dict(kind=kind, **kw)),
        check_media=lambda: None,
        prediction_async=lambda result: None,
        sample=lambda *a, **kw: None,
        append=lambda kind, **kw: commands.append(kw),
    )

    def key():
        if interrupt == "pause" and tick[0] in (2, 4):
            return "p"
        if interrupt == "stop" and tick[0] == 4:
            return "q"
        return ""

    report = controller.control_loop(hw, connection, cfg, recorder, key)
    return N(
        report=report,
        calls=calls,
        events=events,
        commands=commands,
        decoded=decoded,
        closed=closed,
    )


@pytest.mark.parametrize("reason", ["state_drift", "gripper_drift", "response_expired"])
def test_sync_accepts_tracking_motion_and_latency_without_drift_or_age_gate(
    tmp_path, monkeypatch, reason
):
    run = simulate(tmp_path, monkeypatch, first=reason)
    assert run.report["outcome"] == "timeout" and "error" not in run.report
    assert run.report["discarded_responses"] == 0
    assert run.decoded[:3] == [1, 1, 1]
    assert run.report["policy_replans"] == 0
    accepted = next(e for e in run.events if e["kind"] == "chunk_accepted")
    if reason == "state_drift":
        assert abs(accepted["state_change_during_inference_deg"][1]) == pytest.approx(
            2.3731327056884766
        )
    if reason == "response_expired":
        assert accepted["observation_to_accept_s"] > 2


def test_default_sync_executes_first_sixteen_actions_then_requeries(
    tmp_path, monkeypatch
):
    run = simulate(
        tmp_path, monkeypatch, execution_horizon=Config().control.execution_horizon
    )
    assert run.report["outcome"] == "timeout" and "error" not in run.report
    assert run.report["execution_horizon"] == 16
    for query in ("1", "2"):
        commands = [
            c for c in run.commands if c["phase"] == "policy" and c["query_id"] == query
        ]
        assert [c["action_index"] for c in commands] == list(range(16))
        assert [c["ee_delta"][6] for c in commands] == [i / 1000 for i in range(16)]
    assert all(c["action_index"] < 16 for c in run.commands if c["phase"] == "policy")
    second_index = next(i for i, c in enumerate(run.commands) if c["query_id"] == "2")
    previous = run.commands[second_index - 1]
    last_first = next(
        c for c in reversed(run.commands[:second_index]) if c["phase"] == "policy"
    )
    assert previous["phase"] == "hold"
    assert previous["driver_target_physical"] == last_first["driver_target_physical"]


@pytest.mark.parametrize("reason", ["input_expired"])
def test_obsolete_response_is_held_and_requeried_then_new_actions_execute(
    tmp_path, monkeypatch, reason
):
    run = simulate(tmp_path, monkeypatch, first=reason)
    assert run.report["outcome"] == "timeout" and "error" not in run.report
    assert run.report["discarded_responses"] == 1
    assert run.report["policy_commands"] >= 3 and 1 not in run.decoded
    assert all(c["query_id"] != "1" for c in run.commands if c["phase"] == "policy")
    assert run.closed == ["hardware", "model"]


def test_mid_chunk_ik_failure_holds_last_applied_target_and_drops_remaining_deltas(
    tmp_path, monkeypatch
):
    run = simulate(tmp_path, monkeypatch, first="ik_unreachable")
    assert run.report["outcome"] == "timeout" and "error" not in run.report
    assert run.report["rejected_policy_actions"] == 1
    assert run.report["discarded_chunk_actions"] == 2
    first_commands = [
        c for c in run.commands if c["phase"] == "policy" and c["query_id"] == "1"
    ]
    assert len(first_commands) == 1
    first_index = next(i for i, c in enumerate(run.commands) if c is first_commands[0])
    following = run.commands[first_index + 1]
    assert following["phase"] == "hold"
    assert (
        following["driver_target_physical"]
        == first_commands[0]["driver_target_physical"]
    )
    assert any(c["query_id"] == "2" for c in run.commands if c["phase"] == "policy")
    rejected = next(e for e in run.events if e["kind"] == "action_rejected")
    assert rejected["dispatched"] is False and rejected["action_index"] == 1


@pytest.mark.parametrize("interrupt", ["pause", "camera"])
def test_inflight_chunk_is_invalidated_across_pause_or_camera_recovery(
    tmp_path, monkeypatch, interrupt
):
    run = simulate(tmp_path, monkeypatch, interrupt=interrupt)
    assert run.report["outcome"] == "timeout" and "error" not in run.report
    assert 1 not in run.decoded and run.report["policy_commands"] >= 3
    assert run.report["discarded_responses"] >= 1


@pytest.mark.parametrize(
    "first,fatal",
    [
        ("invalid_payload", None),
        ("invalid_timestamp", None),
        ("healthy", "decode"),
        ("state_drift", "hardware"),
    ],
)
def test_recovery_does_not_swallow_invalid_payloads_or_hardware_faults(
    tmp_path, monkeypatch, first, fatal
):
    run = simulate(tmp_path, monkeypatch, first=first, fatal=fatal)
    assert run.report["outcome"] == "fault" and "error" in run.report
    assert (
        (run.report["policy_commands"] > 0)
        if fatal == "hardware"
        else (run.report["policy_commands"] == 0)
    )
    assert run.closed == ["hardware", "model"]


def test_operator_can_stop_during_policy_recovery(tmp_path, monkeypatch):
    run = simulate(tmp_path, monkeypatch, first="state_drift", interrupt="stop")
    assert run.report["outcome"] == "aborted"
    assert run.report["policy_commands"] == 0


def test_repeated_recoveries_complete_full_five_minute_control_budget(
    tmp_path, monkeypatch
):
    run = simulate(tmp_path, monkeypatch, duration=300, stress=True)
    assert run.report["outcome"] == "timeout" and "error" not in run.report
    assert run.report["policy_commands"] > 1000
    assert all(
        run.report["replan_reasons"].get(k, 0) > 20
        for k in ("input_expired", "ik_unreachable")
    )
    assert run.report["policy_replans"] > 100
    assert not {"state_drift", "response_expired"} & run.report["replan_reasons"].keys()


@pytest.mark.parametrize("age", [0.151, 0.3])
def test_input_that_expires_before_worker_starts_can_be_requeried(monkeypatch, age):
    monkeypatch.setattr(controller.time, "monotonic", lambda: 10)
    obs = Observation(
        np.zeros(7),
        {k: np.zeros((8, 8, 3), np.uint8) for k in ("front", "side")},
        10 - age,
        0,
        10,
    )
    with pytest.raises(controller.StalePrediction, match="input_expired"):
        controller.check_inference_observation(obs, Config())


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 11.0])
def test_corrupt_inference_timestamps_still_fail(monkeypatch, timestamp):
    monkeypatch.setattr(controller.time, "monotonic", lambda: 10)
    obs = Observation(
        np.zeros(7),
        {k: np.zeros((8, 8, 3), np.uint8) for k in ("front", "side")},
        timestamp,
        0,
        10,
    )
    with pytest.raises(ProtocolError, match="invalid timestamp"):
        controller.check_inference_observation(obs, Config())
