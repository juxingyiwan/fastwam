"""Normalized model-space RTC inputs. No queues, clocks or robot units."""

import math
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class RTCCondition:
    prefix: torch.Tensor  # [B, H, D], normalized with the checkpoint statistics
    weights: torch.Tensor  # [1 or B, H, 1]
    max_guidance: float = 10.0
    guidance_mode: str = "pi_vjp"
    guidance_start: float = 0.0  # Forward flow time: noise=0, action=1.


def prepare_condition(condition, x):
    """Validate once; zero guidance takes the exact unmodified sampling path."""
    if not isinstance(condition, RTCCondition):
        raise TypeError("rtc_condition must be an RTCCondition")
    if condition.guidance_mode not in ("lerobot", "pi_vjp"):
        raise ValueError("RTC guidance_mode must be lerobot or pi_vjp")
    if (
        type(condition.guidance_start) not in (float, int)
        or not math.isfinite(condition.guidance_start)
        or not 0 <= condition.guidance_start < 1
    ):
        raise ValueError("RTC guidance_start must be a forward flow time in [0,1)")
    prefix, weights = condition.prefix, condition.weights
    if not isinstance(prefix, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("RTC prefix and weights must be tensors")
    if (
        x.ndim != 3
        or prefix.shape != x.shape
        or weights.shape not in ((1, x.shape[1], 1), (x.shape[0], x.shape[1], 1))
    ):
        raise ValueError("RTC requires prefix [B,H,D] and weights [1 or B,H,1]")
    if not prefix.is_floating_point() or not weights.is_floating_point():
        raise TypeError("RTC tensors must be floating point")
    if (
        not torch.isfinite(prefix).all()
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or (weights > 1).any()
    ):
        raise ValueError("RTC requires finite inputs and weights in [0,1]")
    if (
        type(condition.max_guidance) not in (float, int)
        or not math.isfinite(condition.max_guidance)
        or condition.max_guidance < 0
    ):
        raise ValueError("RTC max_guidance must be finite and nonnegative")
    if condition.max_guidance == 0 or not weights.any():
        return None
    # Evaluate the small guidance residual in FP32 even for a BF16 denoiser.
    return replace(
        condition,
        prefix=prefix.detach().to(x.device, torch.float32),
        weights=weights.detach().to(x.device, torch.float32),
    )
