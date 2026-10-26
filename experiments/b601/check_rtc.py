"""Recorded-input RTC probe; no camera, CAN, motor or live-session access."""

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from .deploy import observation_from_fixture
from .fastwam_policy.controller import EEDeltaController, IKUnreachable
from .fastwam_policy.rtc import build_condition
from .runtime.rtc_contract import RTCRequest
from .runtime.recording import write_json


def check_rtc(policy, output, rtc_config=None):
    from .runtime.config import RTCConfig
    rtc_config = rtc_config or RTCConfig()
    rows = []
    decoder = EEDeltaController(policy.adapter.codec)
    for path in sorted((policy.root / "fixtures").glob("*.npz")):
        with np.load(path, allow_pickle=False) as fixture:
            task = str(fixture["task"].item())
            obs = observation_from_fixture(
                fixture, policy.adapter.codec.config["recorded_feature_names"]
            )
            # Teacher-forced ground-truth prefix checks the model interface, not
            # a simulated robot trajectory or predicted task success.
            request = RTCRequest(
                0, 0, fixture["action"][:16].copy(), 8, 16,
                rtc_config.max_guidance, rtc_config.guidance_mode,
                guidance_start=rtc_config.guidance_start,
            )
            motor_states = fixture["motor_state"].copy()
        context, mask = policy.load_context(task)
        kwargs = dict(
            prompt=None,
            input_image=policy.adapter.encode_images(obs),
            proprio=policy.adapter.encode_state(policy.adapter.motor_state(obs)),
            action_horizon=policy.horizon,
            context=context,
            context_mask=mask,
            num_inference_steps=policy.num_inference_steps,
            seed=policy.seed,
            text_cfg_scale=1.0,
            compile_action_infer=False,
        )
        condition = build_condition(request, policy.adapter)

        def sample(value=None):
            if policy.model.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(value is None), torch.no_grad():
                result = policy.model.infer_action(**kwargs, rtc_condition=value)[
                    "action"
                ]
            if policy.model.device.type == "cuda":
                torch.cuda.synchronize()
            return result, time.perf_counter() - started

        plain, _ = sample()
        zero, _ = sample(
            replace(condition, weights=torch.zeros_like(condition.weights))
        )
        torch.testing.assert_close(plain, zero, atol=0, rtol=0)
        guided, _ = sample(condition)
        assert guided.shape == plain.shape == (32, 7) and torch.isfinite(guided).all()
        repeated, _ = sample(condition)
        # CUDA BF16 VJPs may differ in their final reduction bits. Report that
        # difference; the zero-guidance/native path above remains bit-exact.
        torch.testing.assert_close(guided, repeated, atol=0.004, rtol=0.01)
        times = [sample(condition)[1] for _ in range(10)]
        physical = policy.adapter.decode_actions(guided)
        ik_errors, clipped = [], 0
        for step, (motor, action) in enumerate(
            zip(motor_states, physical, strict=True)
        ):
            try:
                decoded = decoder.solve(motor, action)
                clipped += int(any(decoded.native_limit_clipped))
            except IKUnreachable as exc:
                ik_errors.append(dict(step=step, error=str(exc)))
        policy_times = []
        for _ in range(10):
            if policy.model.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = policy.predict(obs, task, rtc_request=request)
            if policy.model.device.type == "cuda":
                torch.cuda.synchronize()
            policy_times.append(time.perf_counter() - started)
            torch.testing.assert_close(
                policy.adapter.normalizer.transform(torch.from_numpy(result), "action"),
                policy.adapter.normalizer.transform(torch.from_numpy(physical), "action"),
                atol=0.004, rtol=0.01,
            )
        np.savez_compressed(
            Path(output) / (path.stem + "-rtc.npz"),
            baseline_normalized=plain.numpy(),
            guided_normalized=guided.numpy(),
            physical=physical,
            prefix_normalized=condition.prefix.numpy(),
            weights=condition.weights.numpy(),
        )
        target = condition.prefix[0, :8]
        rows.append(
            dict(
                task=task,
                zero_guidance_exact=True,
                repeat_exact=bool(torch.equal(guided, repeated)),
                repeat_max_abs_normalized=float((guided - repeated).abs().max()),
                finite=True,
                baseline_prefix_rmse=float((plain[:8] - target).square().mean().sqrt()),
                guided_prefix_rmse=float((guided[:8] - target).square().mean().sqrt()),
                latency_samples_s=times,
                latency_p95_s=float(np.percentile(times, 95)),
                latency_max_s=max(times),
                policy_latency_samples_s=policy_times,
                policy_latency_p95_s=float(np.percentile(policy_times, 95)),
                policy_latency_max_s=max(policy_times),
                teacher_forced_ik_solved=len(physical) - len(ik_errors),
                teacher_forced_ik_errors=ik_errors,
                teacher_forced_native_clipped=clipped,
            )
        )
    if not rows:
        raise ValueError("No recorded fixtures for RTC validation")
    report = dict(
        mode="offline_rtc",
        hardware_connected=False,
        guidance_mode=request.guidance_mode,
        guidance_start=request.guidance_start,
        all_teacher_forced_ik_solved=all(
            not row["teacher_forced_ik_errors"] for row in rows
        ),
        latency_scope={
            "latency": "Warm model sampling with preprocessed inputs",
            "policy_latency": "Warm policy call including observation preprocessing and action decoding; excludes queue, IK and hardware",
        },
        limitations="Recorded teacher-forced feedback only; does not establish closed-loop stability or task success.",
        fixtures=rows,
    )
    write_json(Path(output) / "rtc-check.json", report)
    return report
