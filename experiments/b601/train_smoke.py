"""Instrument the native trainer for a finite, resumable B601 smoke run.

The model, loss, optimizer, training loop and evaluation remain native.
Checkpoints use the native methods; benchmark_only omits checkpoint I/O.
This entry point records evidence and checks finite values.
"""

import hashlib
import json
import logging
import math
import socket
import time
from pathlib import Path

import hydra
import torch
from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

register_default_resolvers()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_parameters(model):
    result = {}
    for label in ("video_expert", "action_expert", "proprio_encoder"):
        module = getattr(model, label)
        name, parameter = next(
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.ndim >= 2
        )
        result[f"{label}.{name}"] = (
            parameter.detach().flatten()[:8192].float().cpu().clone()
        )
    return result


def fingerprints(parameters):
    return {
        name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        for name, value in parameters.items()
    }


class SmokeTrainer(Wan22Trainer):
    def save_checkpoint(self):
        if bool(self.cfg.get("benchmark_only", False)):
            self.accelerator.wait_for_everyone()
            return {"weights_path": None, "state_path": None}
        return super().save_checkpoint()

    def _wandb_log(self, payload):
        super()._wandb_log(payload)
        if not all(math.isfinite(float(value)) for value in payload.values()):
            raise FloatingPointError(
                f"Non-finite training/evaluation metrics: {payload}"
            )
        event = {
            "step": self.global_step,
            "elapsed_seconds": time.perf_counter() - self.smoke_started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            **payload,
        }
        with (Path(self.output_dir) / "smoke-metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.max_steps is None or int(cfg.log_every) != 1:
        raise ValueError("Smoke runs require explicit max_steps and log_every=1")
    if cfg.get("benchmark_only", False) and cfg.resume is not None:
        raise ValueError("Batch benchmarks start from base weights without resuming")
    setup_logging(log_level=logging.INFO)
    misc.register_work_dir(cfg.output_dir)
    root = Path(cfg.output_dir)
    set_global_seed(int(cfg.seed))
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision),
        device=_resolve_train_device(),
    )
    train, val = build_datasets(cfg.data)
    trainer = SmokeTrainer(model=model, train_dataset=train, val_dataset=val, cfg=cfg)
    trainer.smoke_started = time.perf_counter()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    before = sample_parameters(unwrapped)
    start_step = trainer.global_step
    initial_state = {
        "step": start_step,
        "epoch": trainer.epoch,
        "batch_in_epoch": trainer.batch_in_epoch,
        "scheduler_last_epoch": trainer.scheduler.state_dict().get("last_epoch"),
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "parameter_fingerprints": fingerprints(before),
    }
    stop_at = cfg.get("stop_after_steps")
    if stop_at is not None:
        stop_at = int(stop_at)
        if not start_step < stop_at <= trainer.max_steps:
            raise ValueError(
                "stop_after_steps must lie between the restored and total steps"
            )
        # Keep the scheduler's original horizon while ending this process early.
        trainer.max_steps = stop_at
    if trainer.accelerator.is_main_process:
        OmegaConf.save(cfg, root / "config.yaml", resolve=True)
        write_json(root / f"smoke-start-{start_step:06d}.json", initial_state)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    after = sample_parameters(unwrapped)
    changes = {
        name: {
            "changed_values": int((after[name] != value).sum()),
            "max_abs_change": float((after[name] - value).abs().max()),
        }
        for name, value in before.items()
    }
    if not all(torch.isfinite(value).all() for value in after.values()):
        raise FloatingPointError("Non-finite sampled model parameters")
    if not all(change["changed_values"] > 0 for change in changes.values()):
        raise AssertionError(f"A sampled model branch did not update: {changes}")
    result = {
        "status": "passed",
        "host": socket.gethostname(),
        "rank": trainer.accelerator.process_index,
        "world_size": trainer.accelerator.num_processes,
        "start": initial_state,
        "final_step": trainer.global_step,
        "completed_optimizer_steps": trainer.global_step - start_step,
        "checkpointing_enabled": not bool(cfg.get("benchmark_only", False)),
        "batch_size_per_gpu": trainer.batch_size,
        "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
        "batch_in_epoch": trainer.batch_in_epoch,
        "scheduler_last_epoch": trainer.scheduler.state_dict().get("last_epoch"),
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "train_frames": len(train),
        "validation_frames": len(val),
        "parameter_changes": changes,
        "parameter_fingerprints": fingerprints(after),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "elapsed_seconds": time.perf_counter() - trainer.smoke_started,
    }
    write_json(
        root
        / f"smoke-result-{start_step:06d}-rank-{trainer.accelerator.process_index}.json",
        result,
    )
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        print(json.dumps(result, indent=2), flush=True)
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
