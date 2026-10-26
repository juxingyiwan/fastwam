"""Replay reports must retain camera-wait commands with their sparse schema."""

import json
import numpy as np
import pytest

from experiments.b601 import replay_rtc
from experiments.b601.replay_rtc import ReplayHardware, summarize_stream
from experiments.b601.runtime.contracts import CameraUnavailable


def test_recorded_camera_gap_keeps_feedback_but_does_not_reuse_last_image(monkeypatch):
    hw = ReplayHardware.__new__(ReplayHardware)
    hw.started, hw.source_start = 10, 100
    hw.command_times = np.array([100, 101, 102])
    hw.commands = [
        dict(phase="policy", observed_physical=[0] * 7),
        dict(phase="camera_wait", observed_physical=[1] * 7),
        dict(phase="policy", observed_physical=[2] * 7),
    ]
    monkeypatch.setattr(replay_rtc.time, "monotonic", lambda: 11.5)
    with pytest.raises(CameraUnavailable, match="Recorded camera gap"):
        hw.observe()
    np.testing.assert_array_equal(hw.observe_state()[0], np.ones(7))


@pytest.mark.parametrize("window", [16, 32])
def test_summary_handles_camera_wait_without_tick_duration(tmp_path, window):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    def rows(name, values):
        (tmp_path / name).write_text("\n".join(json.dumps(v) for v in values))

    write(
        "report.json",
        dict(
            source_run="recorded",
            outcome="timeout",
            policy_commands=2,
            rejected_policy_actions=0,
            policy_clipped_commands=0,
            queue_starvations=0,
            camera_recovery={"gaps": 1},
            ik_latency_s={},
            observation_replay_latency_s={},
        ),
    )
    rows(
        "events.jsonl",
        [
            dict(
                kind="rtc_queue_response",
                accepted=True,
                guided=False,
                request_id="a",
                merge_delay_steps=0,
                remaining=window,
                latency_s=0.2,
            ),
            dict(
                kind="rtc_queue_response",
                accepted=True,
                guided=True,
                request_id="b",
                merge_delay_steps=8,
                remaining=window - 8,
                latency_s=0.27,
            ),
        ],
    )
    rows(
        "commands.jsonl",
        [
            dict(phase="policy", query_id="a", sent_monotonic=1, dt_s=0.034),
            dict(phase="camera_wait", query_id=None, sent_monotonic=1.04),
            dict(phase="policy", query_id="b", sent_monotonic=2, dt_s=0.034),
        ],
    )
    for query, mode, start in [("a", "bootstrap", 0.7), ("b", "guided", 1.7)]:
        write(
            "query_" + query + ".json",
            dict(
                request_id=query,
                roundtrip_s=0.24,
                rtc={"mode": mode},
                captured_monotonic=start - 0.1,
                policy_timing={"preparation_started_monotonic": start},
            ),
        )
    result = summarize_stream(tmp_path)
    assert result["camera_wait_ticks"] == result["holds_after_bootstrap"] == 1
    assert result["control_tick_s"]["n"] == 2
    assert result["latency_crop_fraction"]["max"] == 0.25
    assert result["available_actions_after_admission"]["min"] == window - 8
    assert result["completed_plan_unexecuted_tail"]["max"] == window - 1


def test_paired_replay_forwards_explicit_guidance_window_to_warmup_and_query(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from dataclasses import replace
    from types import SimpleNamespace as N
    from experiments.b601.runtime.config import Config

    class CapturedRequests(Exception):
        pass

    requests = []
    actions = np.zeros((32, 7), np.float32)
    class Connection:
        def predict(self, *args, rtc_request=None, **kwargs):
            if rtc_request is not None:
                requests.append(rtc_request)
                if len(requests) == 2:
                    raise CapturedRequests
            return N(actions=actions)

    query = dict(request_id="q", policy_timing={"dispatched_monotonic": 0})
    monkeypatch.setattr(replay_rtc, "dataset", lambda run: ([query], [], {"q": []}))
    monkeypatch.setattr(replay_rtc, "EEDeltaController", lambda codec: None)
    monkeypatch.setattr(replay_rtc, "FastWAMConnection", lambda policy: Connection())
    monkeypatch.setattr(replay_rtc, "source_observation", lambda *args: N())
    monkeypatch.setattr(replay_rtc.np, "load", lambda *args: nullcontext({"actions": actions}))
    monkeypatch.setattr(replay_rtc.torch.cuda, "synchronize", lambda: None)
    cfg = Config(rtc=replace(Config().rtc, guidance_mode="pi_vjp", guidance_start=0.4))
    with pytest.raises(CapturedRequests):
        replay_rtc.paired_replay(N(adapter=N(codec=None)), tmp_path / "run", "task", cfg, tmp_path)
    assert all(r.guidance_start == 0.4 and r.guidance_mode == "pi_vjp" for r in requests)


def test_paired_replay_never_reuses_the_synchronous_discarded_tail(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace as N
    from experiments.b601.runtime.config import Config

    class CapturedSecondQuery(Exception):
        pass

    requests = []
    actions = np.zeros((32, 7), np.float32)
    actions[16:] = 1e6
    class Connection:
        def predict(self, *args, rtc_request=None, **kwargs):
            if rtc_request is not None and rtc_request.plan_id == 1:
                requests.append(rtc_request)
                raise CapturedSecondQuery
            return N(actions=actions)

    queries = [dict(request_id=q, generation=0, camera_generation=0,
                    policy_timing={"dispatched_monotonic": t},
                    roundtrip_s=0.2, captured_monotonic=t)
               for q, t in [("first", 0), ("second", 1)]]
    commands = [dict(query_id="first", sent_monotonic=0.5, action_index=15)]
    monkeypatch.setattr(replay_rtc, "dataset", lambda run: (queries, commands, {"first": [], "second": []}))
    monkeypatch.setattr(replay_rtc, "EEDeltaController", lambda codec: None)
    monkeypatch.setattr(replay_rtc, "FastWAMConnection", lambda policy: Connection())
    monkeypatch.setattr(replay_rtc, "source_observation", lambda *args: N())
    monkeypatch.setattr(replay_rtc.np, "load", lambda *args: nullcontext({"actions": actions}))
    monkeypatch.setattr(replay_rtc.torch.cuda, "synchronize", lambda: None)
    with pytest.raises(CapturedSecondQuery):
        replay_rtc.paired_replay(N(adapter=N(codec=None)), tmp_path / "run", "task", Config(), tmp_path)
    assert requests[0].prefix is None
