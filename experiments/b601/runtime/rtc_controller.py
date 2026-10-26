"""RTC actor: sole owner of the action queue, per-step IK and motor commands."""

from dataclasses import replace
import time

import numpy as np

from ..fastwam_policy.controller import IKUnreachable
from .camera_recovery import CameraRecovery, record_camera_hold
from .contracts import JOINTS, HardwareError, ProtocolError
from .controller import (
    StalePrediction,
    check_inference_observation,
    check_prediction,
    decode_action,
)
from .lifecycle import EpisodeLimit, finish_episode
from .rtc_session import RTCSession
from .workers import DaemonWorker as ThreadPoolExecutor


def validate_capability(connection, cfg):
    cap = connection.rtc_capability()
    if (
        not cap.get("supported")
        or cfg.rtc.guidance_mode not in cap.get("guidance_modes", ())
        or cap.get("representation") != "base_frame_ee_delta_v2"
        or cap.get("horizon") != cfg.policy.horizon
        or not np.isclose(cap.get("model_dt_s", 0), 1 / cfg.control.hz)
    ):
        raise ProtocolError(
            "Model backend/checkpoint does not support the selected B601 RTC contract"
        )
    if "flow_times" in cap and not any(
        t + 1e-7 >= cfg.rtc.guidance_start for t in cap["flow_times"]
    ):
        raise ProtocolError("RTC guidance window contains no model sampling steps")
    return cap


def control_loop(hw, connection, cfg, recorder, read_key):
    validate_capability(connection, cfg)
    control = cfg.control
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fastwam-rtc")
    last_tick = deadline = time.monotonic()
    session = RTCSession(cfg, last_tick)
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)
    pending, hold = None, None
    starved = False
    report = dict(
        executor="rtc_fifo",
        execution_horizon=control.execution_horizon,
        prefix_source="remaining_executable_actions",
        outcome="running",
        started_wall=time.time(),
        target_processing="ee_delta_latest_feedback_ik_native_driver_limits",
        policy_commands=0,
        hold_commands=0,
        source_actions_completed=0,
        rejected_policy_actions=0,
        discarded_chunk_actions=0,
        policy_replans=0,
        replan_reasons={},
        policy_clipped_commands=0,
        policy_clipped_joints={name: 0 for name in JOINTS},
        queue_starvations=0,
    )

    def infer(obs, request):
        check_inference_observation(obs, cfg)
        result = connection.predict(
            obs, cfg.task, request.generation, rtc_request=request
        )
        recorder.prediction_async(result)
        return result

    def checked(now):
        session.budget.check_time(time.monotonic())
        if time.monotonic() - now > control.max_tick_s:
            raise HardwareError("RTC observation/IK exceeded the control tick budget")

    print(
        "RTC: p=pause/resume, s=success, f=failure, q/Ctrl+C=stop and native shutdown",
        flush=True,
    )
    try:
        while True:
            time.sleep(max(0, deadline - time.monotonic()))
            now = time.monotonic()
            dt, last_tick, deadline = now - last_tick, now, now + 1 / control.hz
            if dt > control.max_tick_s:
                raise HardwareError(f"RTC control tick took {dt:.3f}s")
            key = read_key()
            if key in ("s", "f", "q") or (recorder.path / "STOP").exists():
                report["outcome"] = {"s": "success", "f": "failure"}.get(key, "aborted")
                break
            session.budget.check_time(now)
            if (
                pending is not None
                and now - session.pending.submitted > control.inference_timeout_s
            ):
                raise ProtocolError("RTC inference exceeded its request deadline")
            value = recovery.poll(hw.observe)
            recorder.check_media()
            checked(now)
            if value is None:
                measured, status = hw.observe_state()
                if hold is None:
                    hold = hw.hold_command(measured)
                if recovery.entered:
                    session.invalidate(now)
                if key == "p":
                    session.toggle_pause(now)
                checked(now)
                record_camera_hold(
                    hw, recorder, hold, measured, status, generation=session.generation
                )
                continue
            obs, status = value
            if hold is None:
                hold = hw.hold_command(obs.state)
            if key == "p":
                session.toggle_pause(now)
                recorder.event(
                    "rtc_paused" if session.paused else "rtc_resumed",
                    generation=session.generation,
                )
            if pending is not None and pending.done():
                future, pending = pending, None
                try:
                    result = future.result()
                except Exception as exc:
                    fatal = session.failed(
                        time.monotonic(), retryable=isinstance(exc, StalePrediction)
                    )
                    recorder.event(
                        "rtc_request_failed", discarded=not fatal, error=str(exc)
                    )
                    if fatal:
                        raise
                else:
                    # Do not impose a motion-drift gate on an asynchronously moving arm.
                    if (
                        result.generation == session.generation
                        and result.observation.camera_generation
                        == obs.camera_generation
                    ):
                        check_prediction(result, obs, control, session.generation)
                    effect = session.receive(
                        result, time.monotonic(), obs.camera_generation
                    )
                    recorder.event("rtc_queue_response", **effect)
                    if effect["accepted"]:
                        starved = False
            checked(now)
            if pending is None:
                request = session.prepare_request(
                    time.monotonic(), obs.camera_generation
                )
                if request is not None:
                    snapshot = replace(
                        obs,
                        state=obs.state.copy(),
                        views={k: v.copy() for k, v in obs.views.items()},
                    )
                    pending = worker.submit(infer, snapshot, request)
                    recorder.event(
                        "rtc_queue_request",
                        generation=request.generation,
                        plan_id=request.plan_id,
                        committed_steps=request.committed_steps,
                        guidance_horizon=request.guidance_horizon,
                        guidance_mode=request.guidance_mode,
                        guidance_start=request.guidance_start,
                        max_guidance=request.max_guidance,
                        prefix=(
                            None if request.prefix is None else request.prefix.tolist()
                        ),
                        submitted_monotonic=session.pending.submitted,
                        observation_age_s=now - obs.captured_monotonic,
                        remaining=len(session.queue),
                    )
            item = None if session.paused else session.queue.peek()
            phase = "policy" if item is not None else "hold"
            if (
                item is None
                and not session.paused
                and session.budget.chunks
                and not starved
            ):
                report["queue_starvations"] += 1
                starved = True
                recorder.event("rtc_queue_empty", pending=pending is not None)
            action, model_index = item if item is not None else (None, None)
            measured, status = hw.observe_state()
            state_time = time.monotonic()
            executed, ik = hold, None
            query = session.active_query
            if item is not None:
                try:
                    executed, ik = decode_action(
                        hw,
                        measured,
                        action,
                        recorder,
                        phase="policy",
                        query_id=query,
                        action_index=model_index,
                        state_monotonic=state_time,
                    )
                except IKUnreachable as exc:
                    remaining = len(session.queue)
                    report["rejected_policy_actions"] += 1
                    report["discarded_chunk_actions"] += remaining
                    report["policy_replans"] += 1
                    report["replan_reasons"]["ik_unreachable"] = (
                        report["replan_reasons"].get("ik_unreachable", 0) + 1
                    )
                    report["last_rejected_action"] = exc.action_context
                    session.invalidate(time.monotonic())
                    recorder.event(
                        "policy_replan",
                        reason="ik_unreachable",
                        generation=session.generation,
                        discarded_actions=remaining,
                        **exc.action_context,
                    )
                    phase, executed, ik, item = "hold", hold, None, None
            checked(now)
            target = hw.send(executed)
            hold = np.asarray(target["driver_target_physical"]) / hw.directions
            if item is not None:
                session.queue.commit()
                report["policy_commands"] += 1
                report["source_actions_completed"] += 1
                if any(target["native_limit_clipped"]):
                    report["policy_clipped_commands"] += 1
                    for name, clipped in zip(
                        JOINTS, target["native_limit_clipped"], strict=True
                    ):
                        report["policy_clipped_joints"][name] += int(clipped)
                    recorder.event(
                        "policy_limit_clipped",
                        query_id=query,
                        action_index=model_index,
                        raw_physical=(np.asarray(executed) * hw.directions).tolist(),
                        ik=ik,
                        **target,
                    )
            else:
                report["hold_commands"] += 1
            recorder.sample(
                obs,
                action=np.asarray(executed).tolist(),
                physical=target["driver_target_physical"],
                phase=phase,
                query_id=query if item is not None else None,
            )
            recorder.append(
                "command",
                phase=phase,
                generation=session.generation,
                plan_id=session.plan_id,
                dt_s=dt,
                query_id=query if item is not None else None,
                action_index=model_index if item is not None else None,
                queue_remaining=len(session.queue),
                ee_delta=action.tolist() if item is not None else None,
                ik=ik,
                executed_recorded_command=np.asarray(executed).tolist(),
                raw_physical=(np.asarray(executed) * hw.directions).tolist(),
                target_processing=report["target_processing"],
                observed_physical=np.asarray(measured).tolist(),
                image_monotonic=obs.captured_monotonic,
                state_monotonic=state_time,
                sent_monotonic=time.monotonic(),
                **status,
                **target,
            )
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
        session.stop()
        report.update(session.metrics)
        report["policy_chunks"] = session.budget.chunks
        report["camera_recovery"] = recovery.summary()
        if not report["policy_commands"] and report["outcome"] not in (
            "fault",
            "aborted",
        ):
            report.update(
                outcome="no_actions", stop_reason="No policy command was executed"
            )
        finish_episode(
            hw,
            connection,
            report,
            pending,
            worker,
            timeout_s=cfg.shutdown.worker_timeout_s,
        )
    return report
