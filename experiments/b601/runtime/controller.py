"""FastWAM EE execution with explicit hardware ownership.

Only the control thread runs IK and sends hardware commands.
"""

import select
import signal
import sys
import time
from contextlib import contextmanager

import numpy as np

from ..fastwam_policy.controller import IKUnreachable
from .camera_recovery import CameraRecovery, record_camera_hold
from .contracts import (
    JOINTS,
    CameraUnavailable,
    HardwareError,
    ProtocolError,
    TransportError,
)
from .lifecycle import EpisodeBudget, EpisodeLimit, finish_episode
from .recording import write_json
from .workers import DaemonWorker as ThreadPoolExecutor


class StalePrediction(ValueError):
    """Discard obsolete policy inputs/results without ending healthy control."""

    def __init__(self, reason, **details):
        message = {
            "camera_generation_changed": "Response belongs to an earlier camera stream generation",
            "control_generation_changed": "Response belongs to an earlier pause/resume generation",
        }.get(reason, reason)
        super().__init__(message)
        self.reason, self.details = reason, details


def check_inference_observation(observation, cfg):
    observation.validate_payload(cfg.policy.camera_names)
    now = time.monotonic()
    ages = np.array(
        [
            now - observation.captured_monotonic,
            now - observation.state_monotonic,
        ]
    )
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise ProtocolError("Inference observation has an invalid timestamp")
    if (ages > cfg.robot.max_camera_age_s).any():
        raise StalePrediction("input_expired", image_age_s=ages[0], state_age_s=ages[1])


def check_prediction(prediction, observation, cfg, generation):
    """Admit a synchronous chunk only from the current control/camera session.

    Holding a target still allows the arm to approach it during inference.
    Feedback movement and response age are diagnostics, not admission gates.
    ``cfg`` is retained for compatibility with the vendored runtime interface.
    """
    if prediction.observation.camera_generation != observation.camera_generation:
        raise StalePrediction("camera_generation_changed")
    if prediction.generation != generation:
        raise StalePrediction("control_generation_changed")
    prediction.observation.validate_payload(("front", "side"))
    observation.validate_payload(("front", "side"))
    now = time.monotonic()
    ages = np.array(
        [
            now - prediction.observation.captured_monotonic,
            now - prediction.observation.state_monotonic,
            now - observation.captured_monotonic,
            now - observation.state_monotonic,
        ]
    )
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise ProtocolError("Prediction observation has an invalid timestamp")
    if (
        not isinstance(prediction.actions, np.ndarray)
        or prediction.actions.shape != (32, 7)
        or not np.isfinite(prediction.actions).all()
    ):
        raise ProtocolError(
            "FastWAM must produce 32 finite seven-dimensional EE actions"
        )


def decode_action(
    hw, measured, action, recorder, *, phase, query_id, action_index, state_monotonic
):
    """Preserve rejected actions before shutdown can change the measured pose."""
    try:
        return hw.decode(measured, action)
    except Exception as exc:
        evidence = {
            "phase": phase,
            "query_id": query_id,
            "action_index": action_index,
            "observed_physical": np.asarray(measured).tolist(),
            "ee_delta": np.asarray(action).tolist(),
            "state_monotonic": state_monotonic,
            "dispatched": False,
            "error": f"{type(exc).__name__}: {exc}",
            **getattr(exc, "details", {}),
        }
        exc.action_context = evidence
        recorder.event("action_rejected", **evidence)
        if phase != "policy" or not isinstance(exc, IKUnreachable):
            print(
                f"Action rejected ({phase}, index={action_index}): {exc}",
                file=sys.stderr,
                flush=True,
            )
        raise


@contextmanager
def operator_terminal():
    import termios
    import tty

    if not sys.stdin.isatty():
        raise HardwareError(
            "run must be launched from the on-site operator's interactive terminal"
        )
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    previous = {}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    try:
        tty.setcbreak(fd)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        yield lambda: (
            sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else ""
        )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def preflight(hw, connection, cfg, recorder):
    write_json(recorder.path / "model.json", connection.open())
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)

    def read_fresh():
        obs, status = hw.observe(disabled=True)
        # Reading disabled-motor feedback can age a previously fresh image pair.
        # Retry within the existing recovery budget instead of failing at the
        # second freshness check. Invalid timestamps/payloads still fail closed.
        obs.validate_payload(cfg.policy.camera_names)
        try:
            obs.validate(cfg.policy.camera_names, cfg.robot.max_camera_age_s)
        except ProtocolError:
            now = time.monotonic()
            image_age = now - obs.captured_monotonic
            state_age = now - obs.state_monotonic
            if (
                np.isfinite([image_age, state_age]).all()
                and image_age > cfg.robot.max_camera_age_s
                and 0 <= state_age <= cfg.robot.max_camera_age_s
            ):
                raise CameraUnavailable(
                    "Preflight image expired while reading feedback"
                ) from None
            raise
        return obs, status

    while (value := recovery.poll(read_fresh)) is None:
        hw.observe_state(disabled=True)
        time.sleep(1 / cfg.control.hz)
    obs, status = value
    recorder.observation(obs, "preflight")
    prediction = connection.predict(obs, cfg.task)
    recorder.prediction(prediction)
    # Check the first target from this observed pose. Future EE deltas need their
    # own measured pose; do not convert an entire chunk against one anchor.
    raw, ik = decode_action(
        hw,
        obs.state,
        prediction.actions[0],
        recorder,
        phase="preflight",
        query_id=prediction.request_id,
        action_index=0,
        state_monotonic=obs.state_monotonic,
    )
    recorder.event(
        "preflight_passed",
        roundtrip_s=prediction.roundtrip_s,
        first_ee_action=prediction.actions[0].tolist(),
        first_recorded_command=raw.tolist(),
        ik=ik,
        **status,
    )
    return prediction


def control_loop(hw, connection, cfg, recorder, read_key):
    control = cfg.control
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fastwam-inference")
    pending = None
    pending_started = None
    pending_generation = None
    generation = 0
    chunk = None
    chunk_commands = 0
    action_index = 0
    paused = False
    hold = None
    last_tick = time.monotonic()
    deadline = last_tick
    budget = EpisodeBudget(control, last_tick)
    report = {
        "executor": "sync",
        "response_admission": "current_control_and_camera_generation",
        "outcome": "running",
        "queries": 0,
        "policy_commands": 0,
        "policy_clipped_commands": 0,
        "policy_clipped_joints": {name: 0 for name in JOINTS},
        "source_actions_completed": 0,
        "discarded_responses": 0,
        "rejected_policy_actions": 0,
        "discarded_chunk_actions": 0,
        "policy_replans": 0,
        "replan_reasons": {},
        "planned_tail_actions": 0,
        "started_wall": time.time(),
        "execution_horizon": control.execution_horizon,
        "target_processing": "ee_delta_latest_feedback_ik_native_driver_limits",
    }
    active_query = None
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)

    def infer(obs, epoch):
        check_inference_observation(obs, cfg)
        result = connection.predict(obs, cfg.task, epoch)
        recorder.prediction_async(result)
        return result

    def replan(reason, **evidence):
        report["policy_replans"] += 1
        counts = report["replan_reasons"]
        counts[reason] = counts.get(reason, 0) + 1
        recorder.event(
            "policy_replan",
            reason=reason,
            generation=generation,
            recovery="hold_last_target_then_fresh_inference",
            **evidence,
        )

    print(
        "Running: p=pause/resume, s=success, f=failure, q/Ctrl+C=stop and native shutdown",
        flush=True,
    )
    try:
        while True:
            time.sleep(max(0, deadline - time.monotonic()))
            now = time.monotonic()
            dt = now - last_tick
            last_tick, deadline = now, now + 1 / control.hz
            if dt > control.max_tick_s:
                raise HardwareError(f"Control tick took {dt:.3f}s")
            key = read_key()
            if key in ("s", "f", "q") or (recorder.path / "STOP").exists():
                report["outcome"] = {"s": "success", "f": "failure"}.get(key, "aborted")
                break
            budget.check_time(now)
            if (
                pending is not None
                and not pending.done()
                and now - pending_started > control.inference_timeout_s
            ):
                raise HardwareError("FastWAM inference exceeded inference_timeout_s")
            value = recovery.poll(hw.observe)
            recorder.check_media()
            if value is None:
                state, status = hw.observe_state()
                if recovery.entered:
                    generation += 1
                    chunk = None
                    hold = hw.hold_command(state)
                if key == "p":
                    paused = not paused
                    recorder.event(
                        "paused" if paused else "resumed", generation=generation
                    )
                budget.check_time(time.monotonic())
                if time.monotonic() - now > control.max_tick_s:
                    raise HardwareError(
                        "Camera-wait feedback exceeded the control tick budget"
                    )
                record_camera_hold(
                    hw, recorder, hold, state, status, generation=generation
                )
                continue
            observation, status = value
            if hold is None:
                hold = hw.hold_command(observation.state)
            if key == "p":
                generation += 1
                paused = not paused
                chunk = None
                hold = hw.hold_command(observation.state)
                recorder.event("paused" if paused else "resumed", generation=generation)
                print(
                    "Paused" if paused else "Resuming from the current observation",
                    flush=True,
                )
            if pending is not None and pending.done():
                future, pending = pending, None
                try:
                    result = future.result()
                except StalePrediction as exc:
                    report["discarded_responses"] += 1
                    replan(exc.reason, **exc.details)
                except (TransportError, ProtocolError):
                    if pending_generation != generation:
                        report["discarded_responses"] += 1
                        replan("superseded_inference")
                    else:
                        chunk = None
                        generation += 1
                        raise  # Invalid payloads/transport failures are not stale results.
                else:
                    report["queries"] += 1
                    try:
                        if paused:
                            raise StalePrediction("paused")
                        check_prediction(result, observation, control, generation)
                    except StalePrediction as exc:
                        report["discarded_responses"] += 1
                        replan(
                            exc.reason,
                            request_id=result.request_id,
                            query_physical=result.observation.state.tolist(),
                            current_physical=observation.state.tolist(),
                            **exc.details,
                        )
                    else:
                        budget.plan_started()
                        chunk = result.actions[: control.execution_horizon]
                        chunk_commands = 0
                        active_query, action_index = result.request_id, 0
                        report["planned_tail_actions"] += cfg.policy.horizon - len(
                            chunk
                        )
                        recorder.event(
                            "chunk_accepted",
                            request_id=active_query,
                            predicted=cfg.policy.horizon,
                            selected=len(chunk),
                            planned_motion_seconds=len(chunk) / control.hz,
                            observation_to_accept_s=time.monotonic()
                            - result.observation.captured_monotonic,
                            state_change_during_inference_deg=(
                                observation.state - result.observation.state
                            ).tolist(),
                        )
            phase = "policy" if chunk is not None and not paused else "hold"
            raw = chunk[action_index] if phase == "policy" else hold
            measured, status = hw.observe_state()
            state_monotonic = time.monotonic()
            ik = None
            if phase == "policy":
                try:
                    executed, ik = decode_action(
                        hw,
                        measured,
                        raw,
                        recorder,
                        phase=phase,
                        query_id=active_query,
                        action_index=action_index,
                        state_monotonic=state_monotonic,
                    )
                except IKUnreachable as exc:
                    # Drop the entire obsolete chunk, never skip directly to its
                    # next delta or send the failed candidate. Retain last target.
                    report["rejected_policy_actions"] += 1
                    report["last_rejected_action"] = exc.action_context
                    remaining = len(chunk) - action_index
                    report["discarded_chunk_actions"] += remaining
                    generation += 1
                    chunk = None
                    replan(
                        "ik_unreachable",
                        discarded_actions=remaining,
                        **exc.action_context,
                    )
                    phase, raw, executed, ik = "hold", hold, hold, None
            else:
                executed = raw
            # IK and observation can take time. Never send a policy/hold command
            # after the episode budget expires.
            budget.check_time(time.monotonic())
            if time.monotonic() - now > control.max_tick_s:
                raise HardwareError("IK/feedback exceeded the control tick budget")
            target = hw.send(executed)
            # Keep the most recently applied target even if a chunk is dropped
            # halfway through; reverting to its starting hold can move the arm.
            hold = np.asarray(target["driver_target_physical"]) / hw.directions
            if phase == "policy" and any(target["native_limit_clipped"]):
                report["policy_clipped_commands"] += 1
                for name, clipped in zip(
                    JOINTS, target["native_limit_clipped"], strict=True
                ):
                    report["policy_clipped_joints"][name] += int(clipped)
                recorder.event(
                    "policy_limit_clipped",
                    query_id=active_query,
                    action_index=action_index,
                    raw_physical=(np.asarray(executed) * hw.directions).tolist(),
                    ik=ik,
                    **target,
                )
            sent_at = time.monotonic()
            recorder.sample(
                observation,
                action=np.asarray(executed).tolist(),
                physical=target["driver_target_physical"],
                phase=phase,
                query_id=active_query if phase == "policy" else None,
            )
            recorder.append(
                "command",
                phase=phase,
                generation=generation,
                dt_s=dt,
                query_id=active_query if phase == "policy" else None,
                action_index=action_index if phase == "policy" else None,
                ee_delta=np.asarray(raw).tolist() if phase == "policy" else None,
                ik=ik,
                executed_recorded_command=np.asarray(executed).tolist(),
                raw_physical=(np.asarray(executed) * hw.directions).tolist(),
                target_processing=report["target_processing"],
                observed_physical=np.asarray(measured).tolist(),
                image_monotonic=observation.captured_monotonic,
                state_monotonic=state_monotonic,
                sent_monotonic=sent_at,
                **status,
                **target,
            )
            if phase == "policy":
                report["policy_commands"] += 1
                chunk_commands += 1
                report["source_actions_completed"] += 1
                action_index += 1
                if action_index == len(chunk):
                    recorder.event(
                        "chunk_completed",
                        request_id=active_query,
                        commands=chunk_commands,
                        source_actions=action_index,
                    )
                    chunk = None
            # Query once per exhausted chunk using this tick's current observation.
            # After a policy tick, acquire again on the next tick; do not require
            # its camera exposure timestamp to postdate the last motor command.
            if not paused and chunk is None and pending is None and phase == "hold":
                budget.require_plan_capacity()
                pending = pool.submit(infer, observation, generation)
                pending_started = time.monotonic()
                pending_generation = generation
    except EpisodeLimit as exc:
        report.update(
            outcome="timeout" if str(exc) == "duration_s" else "budget_exhausted",
            stop_reason=str(exc),
        )
    except KeyboardInterrupt:
        report["outcome"] = "aborted"
    except Exception as exc:
        report.update(outcome="fault", error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "action_context"):
            report["rejected_action"] = exc.action_context
    finally:
        generation += 1
        chunk = None
        if report["policy_commands"] == 0 and report["outcome"] not in (
            "fault",
            "aborted",
        ):
            report.update(
                outcome="no_actions", stop_reason="No policy command was executed"
            )
        report["policy_chunks"] = budget.chunks
        report["camera_recovery"] = recovery.summary()
        finish_episode(
            hw,
            connection,
            report,
            pending,
            pool,
            timeout_s=cfg.shutdown.worker_timeout_s,
        )
    return report
