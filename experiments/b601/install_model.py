"""Publish a verified local model directory and update the named model registry."""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from .fastwam_policy.observation import file_sha256
from .fastwam_policy.policy import MODEL_ID
from .runtime.config import expand, load_config


def model_files(source):
    """Files needed by inference; offline test samples are optional."""
    names = [
        "checkpoint.pt", "model.yaml", "codec.json", "norm_stats.json",
        "robot.urdf", f"models/{MODEL_ID}/Wan2.2_VAE.pth",
    ]
    contexts = list((source / "contexts").glob("*.pt"))
    if not contexts:
        raise FileNotFoundError(f"Missing task contexts: {source / 'contexts'}")
    names.extend(str(p.relative_to(source)) for p in contexts)
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing deployment asset: {source / name}")
    return {name: file_sha256(source / name) for name in names}


def install(source, name, config):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Use a simple model name, such as 10epoch")
    cfg = load_config(config)
    config = expand(config)
    before = config.read_bytes()
    source = expand(source)
    files = model_files(source)
    payload = torch.load(
        source / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if not payload.get("mot") or not payload.get("proprio_encoder"):
        raise ValueError("Expected a native FastWAM checkpoint")
    step = int(payload["step"])
    del payload
    root = expand(cfg.output_root).parent
    model_id = f'b601_{name}_step{step:06d}'
    destination = root / "models" / model_id
    stage = root / "staging"
    stage.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = model_files(destination)
        if old != files:
            raise FileExistsError(
                f"Different model already published at {destination}; choose a new name"
            )
    else:
        with tempfile.TemporaryDirectory(prefix="publish-", dir=stage) as temporary:
            prepared = Path(temporary) / "model"
            # Follow source links so a published model is independent of training/cache files.
            shutil.copytree(source, prepared)
            copied = model_files(prepared)
            if copied != files:
                raise ValueError("Model changed during publication")
            prepared.rename(destination)
    raw = yaml.safe_load(before)
    raw["model"]["checkpoints"][name] = str(destination / "checkpoint.pt")
    # Never change the selected default just because a new model was installed.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=config.parent, prefix=".deploy-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        yaml.safe_dump(raw, stream, sort_keys=False)
    try:
        load_config(temporary)
        if config.read_bytes() != before:
            raise RuntimeError("Deployment configuration changed while publishing")
        temporary.replace(config)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "model": name,
        "checkpoint": str(destination / "checkpoint.pt"),
        "verified_assets": len(files),
        "default": cfg.model.default,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("~/.config/fastwam/deploy.yml").expanduser()
    )
    args = parser.parse_args()
    print(json.dumps(install(args.source, args.name, args.config), indent=2))


if __name__ == "__main__":
    main()
