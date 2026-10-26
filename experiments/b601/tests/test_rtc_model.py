"""RTC mathematics, action-space conversion and native-path equivalence."""

from dataclasses import replace
from types import SimpleNamespace as N

import numpy as np
import pytest
import torch

from fastwam.models.rtc import RTCCondition, guided_velocity, prepare_condition
from fastwam.models.wan22.rtc_adapter import guided_wan_velocity
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from experiments.b601.fastwam_policy.rtc import build_condition
from experiments.b601.runtime.rtc_contract import RTCRequest


@pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 0.9, 0.999])
def test_direct_residual_formula_without_autograd(t, monkeypatch):
    x = torch.tensor([[[0.2, 0.3], [0.4, 0.5]]])
    v = torch.ones_like(x) * 0.25
    y = torch.zeros_like(x)
    w = torch.tensor([[[1.0], [0.0]]])
    cond = prepare_condition(RTCCondition(y, w, 5.0, "lerobot"), x)
    monkeypatch.setattr(
        torch.autograd, "grad", lambda *a, **k: pytest.fail("Unexpected backwards")
    )
    with torch.inference_mode():
        actual = guided_velocity(lambda _t, _x: v, x, torch.tensor(t), cond)
    gain = 5.0 if t == 0 else min(5.0, (t * t + (1 - t) ** 2) / (t * (1 - t)))
    expected = v + gain * (y - (x + (1 - t) * v)) * w
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 1], v[:, 1], atol=0, rtol=0)


def test_wan_guidance_sign_on_actual_shifted_grid():
    scheduler = WanContinuousFlowMatchScheduler(1000, 5)
    steps, deltas = scheduler.build_inference_schedule(
        10, torch.device("cpu"), torch.float32
    )
    x = torch.zeros(1, 32, 7)
    prefix = torch.ones_like(x) * 0.1
    cond = prepare_condition(RTCCondition(prefix, torch.ones(1, 32, 1), 1.0), x)
    for step, delta in zip(steps, deltas):
        velocity = guided_wan_velocity(
            lambda **kw: torch.zeros_like(kw["latents_action"]),
            x,
            step[None],
            1000,
            cond,
        )
        after = scheduler.step(velocity, delta, x)
        assert (
            after > x
        ).all()  # Negative sigma step must move toward positive target.


def test_zero_weights_return_native_path_without_consuming_rng():
    x = torch.randn(2, 5, 9)
    rng = torch.get_rng_state().clone()
    assert (
        prepare_condition(RTCCondition(torch.ones_like(x), torch.zeros(1, 5, 1)), x)
        is None
    )
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "change",
    [
        dict(guidance_mode="unknown"),
        dict(guidance_start=float("nan")),
        dict(guidance_start=1.0),
        dict(max_guidance=float("nan")),
        dict(weights=torch.full((1, 4, 1), 2.0)),
        dict(prefix=torch.zeros(1, 3, 2)),
    ],
)
def test_bad_condition_rejected(change):
    x = torch.zeros(1, 4, 2)
    condition = replace(RTCCondition(x, torch.ones(1, 4, 1)), **change)
    with pytest.raises((ValueError, TypeError)):
        prepare_condition(condition, x)


def test_ee_prefix_is_not_reanchored_clipped_or_physically_zero_padded():
    raw = np.arange(21, dtype=np.float32).reshape(3, 7) / 100
    # Nonzero affine offset proves padding is applied in model space afterwards.
    adapter = N(
        contract={"horizon": 32, "fps": 30, "representation": "base_frame_ee_delta_v2"},
        normalizer=N(transform=lambda x, field: 2 * x + 3),
    )
    cond = build_condition(RTCRequest(0, 0, raw, 2, 3), adapter)
    torch.testing.assert_close(cond.prefix[0, :3], torch.from_numpy(raw) * 2 + 3)
    assert cond.prefix[0, 3:].eq(0).all() and cond.weights[0, 3:].eq(0).all()
    assert cond.weights[0, 2, 0] == 0.5
    np.testing.assert_array_equal(
        raw, np.arange(21, dtype=np.float32).reshape(3, 7) / 100
    )


def test_full_vjp_matches_analytic_coupled_endpoint_and_keeps_parameter_grads_empty():
    # Endpoint depends on both action rows, so a prefix residual must also
    # propagate to the future row. Identity-Jacobian guidance cannot do this.
    x = torch.tensor([[[0.2], [0.4]]])
    matrix = torch.nn.Parameter(torch.tensor([[0.4, 0.3], [-0.1, 0.2]]))
    t = torch.tensor(0.8)
    cond = prepare_condition(
        RTCCondition(torch.ones_like(x), torch.tensor([[[1.0], [0.0]]]), 10, "pi_vjp"), x
    )
    def denoise(_t, value):
        assert torch.is_grad_enabled() and value.requires_grad
        return (matrix @ value.reshape(2)).reshape_as(value)
    actual = guided_velocity(denoise, x, t, cond)
    with torch.no_grad():
        v = matrix @ x.reshape(2)
        endpoint = x.reshape(2) + (1 - t) * v
        residual = (torch.ones(2) - endpoint) * torch.tensor([1., 0.])
        jacobian = torch.eye(2) + (1 - t) * matrix
        gain = (t.square() + (1 - t).square()) / (t * (1 - t))
        expected = v + gain * (jacobian.T @ residual)
    torch.testing.assert_close(actual.reshape(2), expected)
    assert actual[0, 1, 0] != v[1]
    assert matrix.grad is None and not actual.requires_grad and x.grad is None


def test_constant_endpoint_does_not_invent_a_guidance_gradient():
    x = torch.tensor([[[0.2], [0.4]]])
    target = torch.tensor([[[0.5], [0.6]]])
    t = torch.tensor(0.75)
    cond = prepare_condition(RTCCondition(torch.zeros_like(x), torch.ones(1, 2, 1)), x)
    velocity = lambda time, value: (target - value) / (1 - time)
    actual = guided_velocity(velocity, x, t, cond)
    torch.testing.assert_close(actual, velocity(t, x), atol=0, rtol=0)


def test_window_skips_backward_without_substituting_identity_guidance(monkeypatch):
    x = torch.randn(1, 2, 1)
    cond = RTCCondition(torch.ones_like(x), torch.ones(1, 2, 1), guidance_start=0.7)
    monkeypatch.setattr(torch.autograd, "grad", lambda *a, **k: pytest.fail("early backward"))
    def denoise(t, value):
        assert not torch.is_grad_enabled()
        return value * 2
    torch.testing.assert_close(guided_velocity(denoise, x, 0.5, cond), x * 2, atol=0, rtol=0)


def test_active_vjp_rejects_inference_mode_instead_of_silently_losing_gradients():
    x = torch.zeros(1, 2, 1)
    cond = RTCCondition(x, torch.ones(1, 2, 1))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="requires no_grad"):
        guided_velocity(lambda t, v: v, x, 0.8, cond)


def test_policy_builds_normal_caches_for_vjp_and_preserves_native_context():
    from experiments.b601.fastwam_policy.policy import B601Policy
    policy = object.__new__(B601Policy)
    def capture(*args, **kwargs):
        value = torch.ones(1)
        return torch.is_inference_mode_enabled(), torch.is_grad_enabled(), value.is_inference()
    policy._predict = capture
    guided = RTCRequest(0, 0, np.zeros((3, 7)), 2, 3, guidance_mode="pi_vjp")
    with torch.inference_mode():
        assert policy.predict({}, "task", rtc_request=guided) == (False, False, False)
        assert torch.is_inference_mode_enabled()
        assert policy.predict({}, "task") == (True, False, True)
        assert policy.predict({}, "task", rtc_request=RTCRequest(0, 0)) == (True, False, True)


def test_native_capability_declares_gradient_requirements():
    from fastwam.models.wan22.fastwam import FastWAM
    model = object.__new__(FastWAM)
    capability = model.model_rtc_capability()
    assert capability["input_gradients"] is True
    assert capability["gradient_modes"] == ("pi_vjp",)
    assert "pi_vjp" in capability["guidance_modes"]
