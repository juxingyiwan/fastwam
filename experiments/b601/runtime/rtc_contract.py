"""Typed B601 RTC request/acknowledgement; no model, worker or hardware ownership."""

from dataclasses import dataclass
import math

import numpy as np


REPRESENTATION = "base_frame_ee_delta_v2"


@dataclass(frozen=True)
class RTCRequest:
    generation: int
    plan_id: int
    prefix: np.ndarray | None = None
    committed_steps: int = 0
    guidance_horizon: int = 0
    max_guidance: float = 10.0
    guidance_mode: str = "lerobot"
    representation: str = REPRESENTATION
    model_dt_s: float = 1 / 30
    guidance_start: float = 0.0


def validate_request(request, horizon=32):
    if not isinstance(request, RTCRequest):
        raise TypeError("Expected a typed RTCRequest")
    if request.representation != REPRESENTATION or request.guidance_mode not in ("lerobot", "pi_vjp"):
        raise ValueError("RTC action representation or guidance mode mismatch")
    if (
        type(request.guidance_start) not in (float, int)
        or not math.isfinite(request.guidance_start)
        or not 0 <= request.guidance_start < 1
    ):
        raise ValueError("RTC guidance_start must be in [0,1)")
    if not math.isclose(request.model_dt_s, 1 / 30, abs_tol=1e-9):
        raise ValueError("B601 RTC requires the 30 Hz checkpoint timebase")
    for value in (
        request.generation,
        request.plan_id,
        request.committed_steps,
        request.guidance_horizon,
    ):
        if type(value) is not int or value < 0:
            raise ValueError("RTC IDs and step counts must be nonnegative integers")
    if (
        type(request.max_guidance) not in (float, int)
        or not math.isfinite(request.max_guidance)
        or not 0 < request.max_guidance <= 20
    ):
        raise ValueError("RTC max_guidance must be in (0,20]")
    if request.prefix is None:
        if request.committed_steps or request.guidance_horizon:
            raise ValueError("Bootstrap cannot commit a prefix")
    elif (
        not isinstance(request.prefix, np.ndarray)
        or request.prefix.shape != (request.guidance_horizon, 7)
        or request.prefix.dtype.kind not in "fiu"
        or not np.isfinite(request.prefix).all()
        or not 1 <= request.committed_steps <= request.guidance_horizon < horizon
    ):
        raise ValueError(
            "RTC prefix must contain only finite real remaining EE actions"
        )


def acknowledgement(request):
    return dict(
        generation=request.generation,
        plan_id=request.plan_id,
        committed_steps=request.committed_steps,
        guidance_horizon=request.guidance_horizon,
        guidance_mode=request.guidance_mode,
        guidance_start=request.guidance_start,
        representation=request.representation,
        mode="bootstrap" if request.prefix is None else "guided",
    )
