"""Request-scoped endpoint guidance in noise-at-zero/action-at-one flow time.

The `pi_vjp` mode differentiates the endpoint through the denoiser. The legacy
`lerobot` mode substitutes an identity endpoint Jacobian, matching LeRobot b9cb121:
https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/rtc/modeling_rtc.py
"""

import torch


@torch.no_grad()
def guided_velocity(denoise, x, t, condition):
    """Correct a velocity; caller owns the denoiser, cache and integration grid."""
    t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
    if condition.guidance_start and t < condition.guidance_start:
        return denoise(t, x)
    if condition.guidance_mode == "lerobot":
        velocity = denoise(t, x)
        endpoint = x.float() + (1 - t) * velocity.float()
        correction = (condition.prefix - endpoint) * condition.weights
    elif condition.guidance_mode == "pi_vjp":
        if torch.is_inference_mode_enabled():
            raise RuntimeError("pi_vjp RTC requires no_grad, not inference_mode")
        with torch.enable_grad():
            value = x.detach().requires_grad_(True)
            velocity = denoise(t, value)
            endpoint = value.float() + (1 - t) * velocity.float()
            residual = ((condition.prefix - endpoint) * condition.weights).detach()
            # J_endpoint.T @ residual. Do not mask again after the VJP: the
            # denoiser couples prefix and future tokens. Parameter grads stay unset.
            correction = torch.autograd.grad(endpoint, value, grad_outputs=residual)[0]
        velocity, correction = velocity.detach(), correction.detach().float()
    else:
        raise ValueError("Unsupported RTC guidance mode")
    # Same capped gain and operation order as the reference, in forward time.
    tau = 1 - (1 - t)
    squared = (1 - tau) ** 2
    inv_r2 = (squared + tau**2) / squared
    c = torch.nan_to_num((1 - tau) / tau, posinf=condition.max_guidance)
    gain = torch.nan_to_num(c * inv_r2, posinf=condition.max_guidance)
    gain = gain.clamp(max=condition.max_guidance)
    result = velocity.float() + gain * correction
    return result.to(dtype=x.dtype)
