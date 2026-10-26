"""Inference-only wrapper around the native FastWAM checkpoint format."""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.fastwam import FastWAM
from omegaconf import OmegaConf

from .observation import ObservationAdapter

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"


def context_path(root, task, max_len=128):
    """Resolve the same prompt hash and cache filename used during training."""
    key = hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode()).hexdigest()
    matches = list(Path(root).glob(f"{key}.t5_len{max_len}.*.pt"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one training context for {task}")
    return matches[0]


@contextmanager
def model_cache(root):
    """Scope the native loader's cache settings to model construction."""
    values = {"DIFFSYNTH_MODEL_BASE_PATH": str(root), "DIFFSYNTH_SKIP_DOWNLOAD": "true"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class B601Policy:
    def __init__(
        self,
        checkpoint,
        device="cuda",
        num_inference_steps=10,
        seed=42,
    ):
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        # Preserve the adjacent asset directory for linked training checkpoints.
        self.checkpoint = Path(checkpoint).expanduser().absolute()
        self.root = self.checkpoint.parent
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.adapter = ObservationAdapter(
            self.root / "codec.json",
            self.root / "robot.urdf",
            self.root / "norm_stats.json",
        )
        config = OmegaConf.to_container(
            OmegaConf.load(self.root / "model.yaml"), resolve=True
        )
        self.tokenizer_max_len = int(config["tokenizer_max_len"])
        self.horizon = self.adapter.contract["horizon"]
        if config["model_id"] != MODEL_ID:
            raise ValueError(f"Expected the packaged {MODEL_ID} model")
        if config["proprio_dim"] != 7 or config["action_dit_config"]["action_dim"] != 7:
            raise ValueError("Expected the seven-dimensional B601 model")
        kwargs = {
            key: config[key]
            for key in (
                "model_id",
                "tokenizer_model_id",
                "tokenizer_max_len",
                "proprio_dim",
                "video_dit_config",
                "action_dit_config",
            )
        }
        for expert in ("video", "action"):
            scheduler = config[f"{expert}_scheduler"]
            for key in ("train_shift", "infer_shift", "num_train_timesteps"):
                kwargs[f"{expert}_{key}"] = scheduler[key]
        # A full native checkpoint contains both experts. Only the frozen VAE
        # is loaded separately; cached task contexts avoid loading the large T5.
        with model_cache(self.root / "models"):
            self.model = FastWAM.from_wan22_pretrained(
                **kwargs,
                device=device,
                torch_dtype=torch.bfloat16,
                load_text_encoder=False,
                redirect_common_files=False,
                skip_dit_load_from_pretrain=True,
            )
        payload = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        # Native load_checkpoint uses strict=False for MoT. Deployment must fail
        # closed if either expert or the proprio encoder is missing/mismatched.
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(
            payload["proprio_encoder"], strict=True
        )
        self.checkpoint_step = int(payload["step"])
        del payload
        self.model.eval().requires_grad_(False)
        self.contexts = {}

    def load_context(self, task):
        """Load only the requested task; recorded test samples are not required."""
        if task not in self.contexts:
            context = torch.load(
                context_path(self.root / "contexts", task, self.tokenizer_max_len),
                map_location="cpu",
                weights_only=True,
            )
            if context["context"].shape != (self.tokenizer_max_len, 4096) or context[
                "mask"
            ].shape != (self.tokenizer_max_len,):
                raise ValueError(f"Invalid cached task context: {task}")
            value = context["context"].clone()
            mask = context["mask"].bool()
            value[~mask] = 0
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite task context")
            # This is deliberately identical to the training dataset behavior.
            self.contexts[task] = (value, torch.ones_like(mask))
        return self.contexts[task]

    def rtc_flow_times(self):
        """Expose the actual quantized flow grid for validation before device IO."""
        scheduler = self.model.infer_action_scheduler
        times, _ = scheduler.build_inference_schedule(
            self.num_inference_steps, torch.device("cpu"), self.model.torch_dtype
        )
        return (1 - times.float() / scheduler.num_train_timesteps).tolist()

    @torch.inference_mode()
    def predict(self, observation, task, *, rtc_request=None):
        if (
            rtc_request is not None
            and rtc_request.prefix is not None
            and rtc_request.guidance_mode == "pi_vjp"
        ):
            # Build video/context caches as ordinary tensors before the VJP.
            # Only action inputs need gradients; the deployment model is frozen.
            with torch.inference_mode(False), torch.no_grad():
                return self._predict(observation, task, rtc_request=rtc_request)
        return self._predict(observation, task, rtc_request=rtc_request)

    def _predict(self, observation, task, *, rtc_request=None):
        context, mask = self.load_context(task)
        rtc_kwargs = {}
        if rtc_request is not None:
            from .rtc import build_condition

            if not self.model.supports_rtc():
                raise ValueError("Selected FastWAM backend does not support RTC")
            rtc_kwargs["rtc_condition"] = build_condition(rtc_request, self.adapter)
        prediction = self.model.infer_action(
            prompt=None,
            input_image=self.adapter.encode_images(observation),
            proprio=self.adapter.encode_state(self.adapter.motor_state(observation)),
            action_horizon=self.horizon,
            context=context,
            context_mask=mask,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            text_cfg_scale=1.0,
            compile_action_infer=False,
            **rtc_kwargs,
        )
        return self.adapter.decode_actions(prediction["action"])
