"""Convert physical, per-step EE deltas into normalized RTC conditions."""

import torch
from fastwam.models.rtc import RTCCondition
from ..runtime.rtc_contract import REPRESENTATION, validate_request


def build_condition(request, adapter):
    validate_request(request, adapter.contract["horizon"])
    if (
        adapter.contract["fps"] != 30
        or adapter.contract["representation"] != REPRESENTATION
    ):
        raise ValueError("RTC checkpoint timebase/representation mismatch")
    if request.prefix is None:
        return None
    # No reanchoring: each delta is relative to its own future measured pose.
    end, strong = request.guidance_horizon, request.committed_steps
    prefix = torch.zeros(1, adapter.contract["horizon"], 7)
    prefix[0, :end] = adapter.normalizer.transform(
        torch.from_numpy(request.prefix.copy()), "action"
    )
    weights = torch.zeros(1, adapter.contract["horizon"], 1)
    weights[:, :strong] = 1
    weights[0, strong:end, 0] = torch.linspace(1, 0, end - strong + 2)[1:-1]
    return RTCCondition(
        prefix, weights, request.max_guidance, request.guidance_mode,
        request.guidance_start,
    )
