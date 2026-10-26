"""Warm the selected model guidance branch while motors are still disabled."""

import time

from .rtc_contract import RTCRequest
from .rtc_queue import elapsed_steps


def warmup_rtc(connection, prediction, cfg, recorder):
    committed = elapsed_steps(cfg.rtc.latency_budget_s + cfg.rtc.guard_s,
                              1 / cfg.control.hz)
    end = min(committed + cfg.rtc.blend_steps,
              cfg.control.execution_horizon, cfg.policy.horizon - 1)
    request = RTCRequest(
        prediction.generation, 0, prediction.actions[:end].copy(),
        min(committed, end), end, cfg.rtc.max_guidance, cfg.rtc.guidance_mode,
        guidance_start=cfg.rtc.guidance_start,
    )
    started = time.monotonic()
    # A synthetic prefix exercises the real RTC path; this result is discarded.
    connection.predict(prediction.observation, cfg.task, prediction.generation,
                       rtc_request=request)
    recorder.event("rtc_model_warmed", elapsed_s=time.monotonic()-started,
                   guidance_mode=request.guidance_mode,
                   guidance_start=request.guidance_start, actions_executed=0)
