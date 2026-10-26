"""Native checkpoint loading needs no deployment manifest or recorded fixtures."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from experiments.b601.fastwam_policy import policy as module


def test_linked_native_checkpoint_and_lazy_context(tmp_path, monkeypatch):
    training = tmp_path / "training.pt"
    torch.save(
        {"mot": {"x": torch.ones(1)}, "proprio_encoder": {"y": torch.ones(1)},
         "step": 6470},
        training,
    )
    assets = tmp_path / "deployment"
    assets.mkdir()
    checkpoint = assets / "checkpoint.pt"
    checkpoint.symlink_to(training)
    scheduler = {"train_shift": 1, "infer_shift": 1, "num_train_timesteps": 1000}
    config = {
        "model_id": module.MODEL_ID, "tokenizer_model_id": "fixture", "tokenizer_max_len": 128,
        "proprio_dim": 7, "video_dit_config": {}, "action_dit_config": {"action_dim": 7},
        "video_scheduler": scheduler, "action_scheduler": scheduler,
    }
    OmegaConf.save(OmegaConf.create(config), assets / "model.yaml")
    model = MagicMock()
    monkeypatch.setattr(module.FastWAM, "from_wan22_pretrained", lambda **kw: model)
    monkeypatch.setattr(
        module, "ObservationAdapter", lambda *a: SimpleNamespace(contract={"horizon": 32})
    )
    policy = module.B601Policy(checkpoint, device="cpu")
    assert policy.root == assets and policy.checkpoint_step == 6470
    assert policy.contexts == {} and not (assets / "fixtures").exists()
    assert model.mot.load_state_dict.call_args.kwargs == {"strict": True}
    task = "Grasp the cube to the bowl"
    key = hashlib.sha256(module.DEFAULT_PROMPT.format(task=task).encode()).hexdigest()
    (assets / "contexts").mkdir()
    context = torch.ones(128, 4096)
    mask = torch.arange(128) < 5
    torch.save(
        {"context": context, "mask": mask},
        assets / "contexts" / f"{key}.t5_len128.fixture.pt",
    )
    value, inference_mask = policy.load_context(task)
    assert value[:5].eq(1).all() and value[5:].eq(0).all()
    assert inference_mask.all()
    with pytest.raises(ValueError, match="exactly one training context"):
        policy.load_context("unavailable task")


def test_model_cache_restored_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", "/original")
    monkeypatch.delenv("DIFFSYNTH_SKIP_DOWNLOAD", raising=False)
    with pytest.raises(RuntimeError):
        with module.model_cache(tmp_path):
            raise RuntimeError("load failed")
    assert module.os.environ["DIFFSYNTH_MODEL_BASE_PATH"] == "/original"
    assert "DIFFSYNTH_SKIP_DOWNLOAD" not in module.os.environ
