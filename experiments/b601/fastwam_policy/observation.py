"""B601 observation encoding shared by recorded and live inference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fastwam.datasets.lerobot.processors.ee_delta_processor import EEAffineNormalizer
from fastwam.datasets.lerobot3.ee_delta import REPRESENTATION, EEDeltaCodec
from torchvision.transforms import Resize

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class ObservationAdapter:
    def __init__(self, codec_path, urdf_path, stats_path):
        config = json.loads(Path(codec_path).read_text())
        stats = json.loads(Path(stats_path).read_text())
        contract = stats["ee_delta_contract"]
        if (
            contract["representation"] != REPRESENTATION
            or contract["clip"] is not False
            or contract["normalization"] != "min/max"
            or contract["anchor"] != "same_row_measured_pose"
            or contract["statistics_scope"] != "training_episodes_valid_rows_once"
            or contract["horizon"] != config["horizon"]
            or contract["fps"] != config["expected_fps"]
            or contract["codec_sha256"] != json_sha256(config)
            or contract["statistics_sha256"]
            != json_sha256({key: stats[key] for key in ("action", "state")})
            or file_sha256(urdf_path) != config["urdf_sha256"]
        ):
            raise ValueError("Deployment assets disagree with the frozen EE contract")
        self.codec = EEDeltaCodec(urdf_path, config)
        self.normalizer = EEAffineNormalizer(stats)
        self.contract = contract
        self.resize = Resize([224, 224])
        self.camera_keys = ("front", "side")

    def motor_state(self, observation):
        state = np.asarray(
            [observation[key] for key in self.codec.config["recorded_feature_names"]],
            dtype=np.float64,
        )
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(
                "Expected seven finite measured motor positions in degrees"
            )
        return state

    def encode_state(self, motor_state):
        state = self.codec.state(motor_state)
        return self.normalizer.transform(torch.from_numpy(state), "state").unsqueeze(0)

    def encode_images(self, observation):
        images = []
        for key in self.camera_keys:
            image = np.asarray(observation[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must be an RGB uint8 HWC image")
            # Match the training ToTensor -> tensor Resize -> Normalize pipeline.
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            images.append(self.resize(tensor))
        return (torch.cat(images, dim=-1) * 2 - 1).unsqueeze(0)

    def decode_actions(self, normalized):
        actions = self.normalizer.transform(normalized.cpu(), "action", inverse=True)
        if (
            actions.shape != (self.contract["horizon"], 7)
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Expected a finite H x 7 physical action chunk")
        return actions.numpy()
