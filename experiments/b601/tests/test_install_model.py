"""Publish native checkpoints independently of training files and offline samples."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.b601.install_model import MODEL_ID, install


@pytest.fixture
def installation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "model.yaml", "codec.json", "norm_stats.json", "robot.urdf",
        f"models/{MODEL_ID}/Wan2.2_VAE.pth", "contexts/task.pt",
    ):
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    torch.save(
        {"mot": {"x": torch.ones(1)}, "proprio_encoder": {"y": torch.ones(1)},
         "step": 12},
        source / "checkpoint.pt",
    )
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/local.example.yml").read_text()
    )
    config["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "deploy.yml"
    path.write_text(yaml.safe_dump(config))
    return SimpleNamespace(root=source), path


def test_install_dereferences_weights_and_keeps_default(installation):
    fixture, config = installation
    checkpoint = fixture.root / "checkpoint.pt"
    original = fixture.root.parent / "training.pt"
    checkpoint.rename(original)
    checkpoint.symlink_to(original)
    result = install(fixture.root, "newmodel", config)
    published = Path(result["checkpoint"])
    assert not published.is_symlink()
    assert result["default"] == "5epoch"
    registered = yaml.safe_load(config.read_text())["model"]["checkpoints"]
    assert registered["newmodel"] == str(published)
    assert install(fixture.root, "newmodel", config) == result
    original.unlink()
    assert torch.load(published, weights_only=True)["step"] == 12
    assert not (published.parent / "fixtures").exists()


def test_install_partial_model_does_not_change_registry(installation):
    fixture, config = installation
    before = config.read_bytes()
    (fixture.root / "checkpoint.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing deployment asset"):
        install(fixture.root, "newmodel", config)
    assert config.read_bytes() == before
    assert not (config.parent / "models").exists()


def test_install_cannot_replace_published_model(installation):
    fixture, config = installation
    result = install(fixture.root, "newmodel", config)
    published = Path(result["checkpoint"])
    before = published.read_bytes()
    torch.save(
        {"mot": {"x": torch.zeros(1)}, "proprio_encoder": {"y": torch.ones(1)},
         "step": 12},
        fixture.root / "checkpoint.pt",
    )
    with pytest.raises(FileExistsError, match="Different model already published"):
        install(fixture.root, "newmodel", config)
    assert published.read_bytes() == before
