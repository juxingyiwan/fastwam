"""Visual-only adaptation primitives; no simulator or cluster dependencies."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def window_starts(frame_count, horizon, count):
    """Each window has observations [s, ..., s+H], with H executed actions."""
    available = int(frame_count) - int(horizon)
    if horizon <= 0 or count <= 0 or available < count:
        raise ValueError(
            f"Need {count} distinct complete windows with horizon {horizon}; "
            f"only {frame_count} observations available. No padding or success filtering is applied."
        )
    return np.linspace(0, available - 1, count, dtype=np.int64).tolist()


def frame_indices(start, horizon, stride):
    if horizon <= 0 or stride <= 0 or horizon % stride:
        raise ValueError("Video stride must divide the executed-action horizon.")
    result = np.arange(start, start + horizon + 1, stride, dtype=np.int64)
    if len(result) <= 1 or len(result) % 4 != 1:
        raise ValueError("Wan VAE requires a video length of 4k+1, greater than one.")
    return result


class LoRALinear(nn.Module):
    """Train an FP32 residual; merge for inference through the original Linear."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        self.base = base.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        self.enabled = False
        self.merged = False
        self._original_weight_cpu = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight.copy_(self._original_weight_cpu)
            self.merged = False

    @torch.no_grad()
    def merge(self):
        self.unmerge()
        if self._original_weight_cpu is None:
            self._original_weight_cpu = self.base.weight.detach().cpu().clone()
        # Only adapter inference changes weights; the original is retained exactly.
        delta = (self.lora_B @ self.lora_A) * self.scale
        merged_weight = self.base.weight.float() + delta
        self.base.weight.copy_(merged_weight.to(self.base.weight.dtype))
        self.enabled = False
        self.merged = True

    def forward(self, x):
        result = self.base(x)
        if not self.enabled:
            return result
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return result + residual.to(result.dtype)


def install_lora(model, rank, alpha, pattern):
    model.requires_grad_(False)
    matches = [(name, module) for name, module in model.video_expert.named_modules()
               if isinstance(module, nn.Linear) and re.fullmatch(pattern, name)]
    if not matches:
        raise ValueError(f"No video-expert Linear modules match {pattern!r}")
    adapters = {}
    for name, module in matches:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.video_expert.get_submodule(parent_name)
        adapter = LoRALinear(module, rank, alpha)
        setattr(parent, child_name, adapter)
        adapters[name] = adapter
    return adapters


def adapter_parameters(adapters):
    return [p for module in adapters.values() for p in (module.lora_A, module.lora_B)]


def activate(adapters, enabled, trainable=False):
    for module in adapters.values():
        module.unmerge()
        module.enabled = enabled and trainable
        if enabled and not trainable:
            module.merge()
        module.lora_A.requires_grad_(trainable)
        module.lora_B.requires_grad_(trainable)


def reset_adapters(adapters, seed):
    activate(adapters, False, False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for module in adapters.values():
        module.reset_parameters()
    activate(adapters, True, True)


def adapter_state(adapters):
    return {f"{name}.{key}": getattr(module, key).detach().cpu().clone()
            for name, module in adapters.items() for key in ("lora_A", "lora_B")}


def load_adapter_state(adapters, state):
    expected = {f"{name}.{key}" for name in adapters for key in ("lora_A", "lora_B")}
    if set(state) != expected:
        raise ValueError("Adapter module keys differ from the configured video LoRA modules.")
    activate(adapters, False, False)
    with torch.no_grad():
        for name, module in adapters.items():
            for key in ("lora_A", "lora_B"):
                parameter = getattr(module, key)
                value = state[f"{name}.{key}"]
                if parameter.shape != value.shape:
                    raise ValueError(f"Adapter shape mismatch: {name}.{key}")
                parameter.copy_(value.to(parameter.device, parameter.dtype))
    activate(adapters, True, False)


def video_denoise(model, noisy, timestep, context, context_mask, checkpoint_blocks=True):
    """The IDM noisy-video branch is independent of action/teacher-forced video.

    Use the existing video expert's preparation, masks, blocks and head. Skipping
    the unrelated action branch avoids computing an action imitation objective.
    """
    expert = model.video_expert
    if expert.action_conditioned or expert.video_attention_mask_mode != "first_frame_causal":
        raise ValueError("Visual-only IDM adaptation requires unconditional, first-frame-causal video.")
    tokens, t, t_mod, ctx, ctx_mask, freqs, f, h, w, per_frame = expert.prepare(
        x=noisy, timestep=timestep, context=context, context_mask=context_mask,
        action=None, fuse_vae_embedding_in_latents=True,
    )
    mask = expert.build_video_to_video_mask(tokens.shape[1], per_frame, tokens.device)
    for block in expert.blocks:
        args = (tokens, ctx, t_mod, freqs)
        kwargs = {"context_mask": ctx_mask, "self_attn_mask": mask}
        if checkpoint_blocks and torch.is_grad_enabled():
            tokens = checkpoint(block, *args, use_reentrant=False, **kwargs)
        else:
            tokens = block(*args, **kwargs)
    return expert.post(tokens, t, f, h, w)


def future_flow_loss(model, record, context, context_mask, checkpoint_blocks=True):
    clean = record["latents"].to(model.device, model.torch_dtype)
    proprio = record["proprio"].to(model.device, model.torch_dtype)
    if clean.shape[2] < 2:
        raise ValueError("TTT sample has no future latent frames.")
    with torch.no_grad():
        ctx, mask = model._append_proprio_to_context(context, context_mask, proprio)
    scheduler = model.train_video_scheduler
    noise = torch.randn_like(clean)
    timestep = scheduler.sample_training_t(clean.shape[0], clean.device, clean.dtype)
    noisy = scheduler.add_noise(clean, noise, timestep)
    noisy[:, :, :1] = clean[:, :, :1]
    target = scheduler.training_target(clean, noise, timestep)
    prediction = video_denoise(model, noisy, timestep, ctx, mask, checkpoint_blocks)
    # Native training excludes the clean current frame and supervises only futures.
    per_sample = (prediction[:, :, 1:].float() - target[:, :, 1:].float()).square().flatten(1).mean(1)
    return (per_sample * scheduler.training_weight(timestep)).mean()


@contextlib.contextmanager
def task_lock(directory):
    import os
    import socket
    path = Path(directory) / "active.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
    try:
        yield
    finally:
        path.unlink()
