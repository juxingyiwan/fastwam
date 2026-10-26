"""Timing/consumption contracts without threads, model imports or hardware."""

from dataclasses import replace
from types import SimpleNamespace as N

import numpy as np
import pytest

from experiments.b601.runtime.config import Config
from experiments.b601.runtime.contracts import ProtocolError
from experiments.b601.runtime.rtc_contract import acknowledgement
from experiments.b601.runtime.rtc_queue import ActionQueue
from experiments.b601.runtime.rtc_session import RTCSession


def result(request, camera=0, ack=True):
    return N(
        request_id="fixture",
        generation=request.generation,
        observation=N(camera_generation=camera),
        actions=np.arange(224, dtype=np.float32).reshape(32, 7),
        rtc=acknowledgement(request) if ack else None,
    )


def test_bootstrap_does_not_skip_even_after_long_wait():
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    outcome = s.receive(result(r), 2, 0)
    assert outcome["merge_delay_steps"] == 0 and s.queue.peek()[1] == 0


def test_replan_preserves_soft_transition_after_previous_chunk_was_cropped():
    cfg = Config(
        control=replace(Config().control, execution_horizon=32),
        rtc=replace(Config().rtc, replan_interval_s=0.35, latency_budget_s=0.25, guard_s=0.05, blend_steps=6),
    )
    s = RTCSession(cfg, 0)
    request = s.prepare_request(0, 0)
    s.receive(result(request), 0.2, 0)
    for _ in range(16):
        s.queue.commit()
    request = s.prepare_request(1, 0)
    for _ in range(8):
        s.queue.commit()
    s.receive(result(request), 1.27, 0)
    # A 24-row admitted chunk must replan before only its committed rows remain.
    assert s.next_query == pytest.approx(1.27 + 8 / 30)
    for _ in range(8):
        s.queue.commit()
    request = s.prepare_request(s.next_query, 0)
    assert request.guidance_horizon - request.committed_steps == 6


def test_merge_is_bounded_by_successful_sends_and_holds_do_not_consume():
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    s.receive(result(r), 0.2, 0)
    for _ in range(12):
        s.queue.commit()
    r = s.prepare_request(1, 0)
    assert r.prefix.shape == (4, 7) and r.committed_steps == 4
    assert not r.prefix.flags.writeable
    for _ in range(4):
        s.queue.commit()
    for _ in range(30):
        assert s.queue.peek() is None
    outcome = s.receive(result(r), 2, 0)
    assert outcome["elapsed_steps"] == 30 and outcome["merge_delay_steps"] == 4
    assert s.queue.peek()[1] == 4


def test_pause_then_resume_invalidates_inflight_without_starting_second_worker():
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    s.toggle_pause(0.1)
    s.toggle_pause(0.2)
    assert s.prepare_request(0.3, 0) is None
    assert not s.receive(result(r), 0.4, 0)["accepted"]
    assert s.prepare_request(0.5, 0).prefix is None


def test_ack_mismatch_is_not_silent_unguided_fallback():
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    with pytest.raises(ProtocolError, match="acknowledgement"):
        s.receive(result(r, ack=False), 0.2, 0)


def test_old_error_discarded_but_current_error_is_fatal():
    s = RTCSession(Config(), 0)
    s.prepare_request(0, 0)
    s.invalidate(0.1)
    assert not s.failed(0.2)
    s.prepare_request(0.3, 0)
    assert s.failed(0.4)


def test_stop_is_terminal_and_late_chunk_does_not_erase_queue():
    q = ActionQueue(32)
    actions = np.zeros((32, 7))
    q.replace(actions, 0)
    assert not q.replace(actions, 32) and len(q) == 32
    s = RTCSession(Config(), 0)
    s.stop()
    with pytest.raises(RuntimeError):
        s.prepare_request(1, 0)


def test_prefix_copies_original_values_without_hold_padding():
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    s.receive(result(r), 0.2, 0)
    for _ in range(14):
        s.queue.commit()
    left = s.queue.leftover()
    r = s.prepare_request(1, 0)
    np.testing.assert_array_equal(r.prefix, left)
    assert len(r.prefix) == 2
    s.queue.actions[-1] = 0
    np.testing.assert_array_equal(r.prefix, left)


def test_rtc_schedule_is_independent_of_sync_execution_horizon():
    schedules = []
    for horizon in [1, 16, 32]:
        cfg = Config(control=replace(Config().control, execution_horizon=horizon))
        session = RTCSession(cfg, 0)
        request = session.prepare_request(0, 0)
        session.receive(result(request), 0.2, 0)
        schedules.append(session.next_query)
    assert schedules == pytest.approx([0.2, 0.2, 0.2])


@pytest.mark.parametrize("delay_ticks", [6, 7, 8])
def test_default_replans_on_admission_without_starvation_or_duplicate_requests(delay_ticks):
    cfg = Config()
    assert cfg.rtc.guidance_mode == "lerobot"
    assert cfg.rtc.guidance_start == cfg.rtc.replan_interval_s == 0
    session = RTCSession(cfg, 0)
    pending = None
    indices = []
    replacements = 0
    for tick in range(120):
        now = tick / 30
        if pending is not None and tick == pending[0]:
            session.receive(result(pending[1]), now, 0)
            pending = None
            replacements += 1
            assert session.next_query == now
        if pending is None:
            request = session.prepare_request(now, 0)
            assert request is not None
            pending = (tick + delay_ticks, request)
        assert session.prepare_request(now, 0) is None
        item = session.queue.peek()
        if tick >= delay_ticks:
            assert item is not None
            indices.append(item[1])
            session.queue.commit()
    assert replacements > 5
    assert max(indices) <= 2 * delay_ticks - 1


@pytest.mark.parametrize("interval", [-0.1, float("nan"), float("inf"), True])
def test_invalid_replan_wait_rejected(interval):
    from experiments.b601.runtime.config import validate_config
    cfg = Config(rtc=replace(Config().rtc, replan_interval_s=interval))
    with pytest.raises(ValueError, match="replan_interval_s"):
        validate_config(cfg)


def test_discarded_prediction_tail_is_never_executed_or_used_as_a_prefix():
    cfg = Config()
    session = RTCSession(cfg, 0)
    request = session.prepare_request(0, 0)
    prediction = result(request)
    prediction.actions[16:] = 1e6  # Distinct bad tail must not leak through guidance.
    session.receive(prediction, 0.2, 0)
    assert len(session.queue) == 16
    for _ in range(7):
        session.queue.commit()
    request = session.prepare_request(0.2 + 7 / 30, 0)
    assert request.guidance_horizon <= 9
    assert np.max(request.prefix) < 1e6
    while session.queue.peek() is not None:
        assert session.queue.peek()[1] < 16
        session.queue.commit()
    assert len(session.queue) == 0


def test_slow_inference_holds_instead_of_executing_the_discarded_tail():
    session = RTCSession(Config(), 0)
    request = session.prepare_request(0, 0)
    session.receive(result(request), 0.2, 0)
    for _ in range(7):
        session.queue.commit()
    request = session.prepare_request(0.5, 0)
    for _ in range(9):
        session.queue.commit()
    assert session.queue.peek() is None
    outcome = session.receive(result(request), 1.0, 0)
    assert outcome["merge_delay_steps"] == 9
    assert len(session.queue) == 7
    assert session.queue.peek()[1] == 9


def test_execution_window_expiry_does_not_reintroduce_tail_actions():
    queue = ActionQueue(32, 16)
    actions = np.zeros((32, 7))
    queue.replace(actions, 7)
    assert len(queue) == 9
    assert not queue.replace(actions, 16)
    assert len(queue) == 9


@pytest.mark.parametrize("phase", [0.001, 0.015, 0.030])
def test_committed_window_uses_successful_sends_not_subtick_wall_phase(phase):
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    s.receive(result(r), 0.2, 0)
    r = s.prepare_request(0.2, 0)
    for _ in range(7):
        s.queue.commit()
    now = 0.2 + 7 / 30 + phase
    effect = s.receive(result(r), now, 0)
    assert effect["elapsed_steps"] == 8
    assert effect["merge_delay_steps"] == 7
    r = s.prepare_request(now, 0)
    assert r.committed_steps == 7 and r.guidance_horizon == 8


def test_queue_exhaustion_does_not_train_a_false_short_delay():
    s = RTCSession(Config(), 0)
    r = s.prepare_request(0, 0)
    s.receive(result(r), 0.2, 0)
    for _ in range(12):
        s.queue.commit()
    r = s.prepare_request(1.0, 0)
    for _ in range(4):
        s.queue.commit()
    s.receive(result(r), 1.4, 0)
    assert s.estimate() == pytest.approx(0.4)
    assert s.prepare_request(1.4, 0).committed_steps == 12


@pytest.mark.parametrize("guard", [-0.1, float("nan"), float("inf"), True])
def test_invalid_delay_guard_rejected(guard):
    from experiments.b601.runtime.config import validate_config
    cfg = Config(rtc=replace(Config().rtc, guard_s=guard))
    with pytest.raises(ValueError, match="guard_s"):
        validate_config(cfg)
