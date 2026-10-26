"""Clock-free RTC scheduling. Caller supplies time and performs every side effect."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .contracts import ProtocolError
from .lifecycle import EpisodeBudget
from .rtc_contract import RTCRequest, acknowledgement
from .rtc_queue import ActionQueue, elapsed_steps


@dataclass(frozen=True)
class Pending:
    request: RTCRequest
    submitted: float
    consumed: int
    camera_generation: int


class RTCSession:
    def __init__(self, cfg, now):
        self.cfg = cfg
        self.dt = 1 / cfg.control.hz
        self.budget = EpisodeBudget(cfg.control, now)
        self.queue = ActionQueue(cfg.policy.horizon, cfg.control.execution_horizon)
        self.delay_steps = deque(maxlen=32)
        self.generation = self.plan_id = 0
        self.pending = None
        self.paused = self.stopped = False
        self.next_query = now
        self.active_query = None
        self.metrics = dict(
            queries=0,
            guided_queries=0,
            bootstrap_queries=0,
            discarded_responses=0,
            expired_chunks=0,
            replacements=0,
            consumption_limited_merges=0,
        )

    def ensure_live(self):
        if self.stopped:
            raise RuntimeError("RTC session has stopped")

    def estimate(self):
        return max(
            self.cfg.rtc.latency_budget_s,
            float(np.percentile(self.delay_steps, 95)) * self.dt
            if self.delay_steps else 0,
        )

    def invalidate(self, now):
        self.ensure_live()
        self.generation += 1
        self.queue.clear()
        self.active_query = None
        self.next_query = now
        # Retain pending ownership until its worker completes; never overlap calls.

    def toggle_pause(self, now):
        self.invalidate(now)
        self.paused = not self.paused

    def prepare_request(self, now, camera_generation):
        self.ensure_live()
        if self.paused or self.pending is not None or now < self.next_query:
            return None
        if self.budget.chunks >= self.budget.max_policy_chunks:
            if not len(self.queue):
                self.budget.require_plan_capacity()
            return None
        estimated = elapsed_steps(self.estimate() + self.cfg.rtc.guard_s, self.dt)
        left = self.queue.leftover()
        # Never invent a hold by repeating a nonzero delta. Guide available rows
        # only; keep at least one free model row. Empty queue uses bootstrap.
        end = min(
            len(left), estimated + self.cfg.rtc.blend_steps, self.cfg.policy.horizon - 1
        )
        prefix = left[:end].copy() if end else None
        if prefix is not None:
            prefix.setflags(write=False)
        request = RTCRequest(
            self.generation,
            self.plan_id,
            prefix,
            min(estimated, end),
            end,
            self.cfg.rtc.max_guidance,
            self.cfg.rtc.guidance_mode,
            guidance_start=self.cfg.rtc.guidance_start,
        )
        self.pending = Pending(request, now, self.queue.consumed, camera_generation)
        return request

    def _take(self):
        self.ensure_live()
        if self.pending is None:
            raise ProtocolError("RTC response has no outstanding request")
        pending, self.pending = self.pending, None
        return pending

    def failed(self, now, *, retryable=False):
        context = self._take()
        stale = context.request.generation != self.generation
        if stale or retryable:
            self.metrics["discarded_responses"] += 1
            self.next_query = now
            return False
        return True

    def receive(self, result, now, camera_generation):
        context = self._take()
        req = context.request
        self.metrics["queries"] += 1
        stale = (
            self.paused
            or req.generation != self.generation
            or result.generation != self.generation
            or req.plan_id != self.plan_id
            or result.observation.camera_generation != camera_generation
            or context.camera_generation != camera_generation
        )
        if stale:
            self.metrics["discarded_responses"] += 1
            self.next_query = now
            return dict(
                accepted=False, reason="superseded", request_id=result.request_id
            )
        if result.rtc != acknowledgement(req):
            raise ProtocolError("RTC acknowledgement mismatch")
        duration = now - context.submitted
        elapsed = elapsed_steps(duration, self.dt)
        consumed = self.queue.consumed - context.consumed
        if consumed < 0:
            raise RuntimeError("RTC consumption counter went backwards")
        skip = min(elapsed, consumed)
        guided = req.prefix is not None
        self.metrics["guided_queries" if guided else "bootstrap_queries"] += 1
        if guided and duration > 0:
            # Wall time includes the phase within the control tick. Ceil(duration/dt)
            # regularly counts a tick that sent no action and unnecessarily freezes
            # the next free row. Use the same successful-send clock as admission.
            # If the queue ran dry, consumption is censored; retain the wall-time
            # upper bound so holds cannot teach the estimator a shorter delay.
            self.delay_steps.append(consumed if len(self.queue) else elapsed)
        accepted = self.queue.replace(result.actions, skip)
        if accepted:
            self.budget.plan_started()
            self.plan_id += 1
            self.active_query = result.request_id
            self.metrics["replacements"] += int(guided)
            self.metrics["consumption_limited_merges"] += int(skip < elapsed)
        else:
            self.metrics["discarded_responses"] += 1
            self.metrics["expired_chunks"] += 1
        # Keep the soft transition available while the worker runs. Reserving
        # only latency+guard leaves an almost fully committed short prefix and
        # silently removes most of the configured blend window.
        reserve_steps = (
            elapsed_steps(self.estimate() + self.cfg.rtc.guard_s, self.dt)
            + self.cfg.rtc.blend_steps
        )
        wait = min(
            self.cfg.rtc.replan_interval_s,
            max(0, len(self.queue) - reserve_steps) * self.dt,
        )
        self.next_query = now + wait
        return dict(
            accepted=accepted,
            reason="active" if accepted else "expired",
            request_id=result.request_id,
            latency_s=duration,
            elapsed_steps=elapsed,
            consumed_steps=consumed,
            merge_delay_steps=skip,
            guided=guided,
            remaining=len(self.queue),
            plan_id=self.plan_id,
        )

    def stop(self):
        self.stopped = True
        self.generation += 1
        self.pending = None
        self.queue.clear()
