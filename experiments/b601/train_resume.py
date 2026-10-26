"""Resume native B601 training without overwriting the original run metadata."""

import json
import logging
from pathlib import Path

import hydra
from accelerate import PartialState
from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

register_default_resolvers()


def validate_resume_paths(cfg):
    run_dir = Path(cfg.output_dir).expanduser().resolve()
    state_dir = Path(cfg.resume).expanduser().resolve()
    record_dir = Path(cfg.continuation_dir).expanduser().resolve()
    if state_dir.parent != run_dir / "checkpoints/state":
        raise ValueError("Resume state must belong to the original output directory")
    for name in ("trainer_state.json", "scheduler.bin"):
        if not (state_dir / name).is_file():
            raise FileNotFoundError(state_dir / name)
    if not record_dir.is_relative_to(run_dir / "continuations"):
        raise ValueError("Continuation metadata must be inside run/continuations")
    if (record_dir / "config.yaml").exists():
        raise FileExistsError(f"Continuation metadata already exists: {record_dir}")
    state = json.loads((state_dir / "trainer_state.json").read_text())
    return record_dir, state


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    distributed = PartialState()
    record_dir, state = validate_resume_paths(cfg)
    distributed.wait_for_everyone()
    main_process = distributed.is_main_process
    setup_logging(log_level=logging.INFO, is_main_process=main_process)
    # Dataset statistics and configuration snapshots belong to this continuation.
    # The native trainer's output_dir remains the original checkpoint directory.
    misc.register_work_dir(record_dir)
    if main_process:
        OmegaConf.save(cfg, record_dir / "config.yaml", resolve=True)
        (record_dir / "resume.json").write_text(
            json.dumps({"source_state": str(cfg.resume), **state}, indent=2) + "\n"
        )
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision),
        device=_resolve_train_device(),
    )
    train, val = build_datasets(cfg.data)
    trainer = Wan22Trainer(model=model, train_dataset=train, val_dataset=val, cfg=cfg)
    if trainer.global_step != state["global_step"]:
        raise RuntimeError("Native trainer did not restore the requested step")
    if trainer.global_step >= trainer.max_steps:
        raise ValueError("The restored checkpoint already reached the training limit")
    trainer.train()
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
