"""Resolve example configurations independently of a particular host layout."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastwam.utils.config_resolvers import register_default_resolvers
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class ConfigurationPathsTests(unittest.TestCase):
    def resolve(self, environment, task="b601_uncond_2cam224"):
        register_default_resolvers()
        config_dir = Path(__file__).resolve().parents[3] / "configs"
        with (
            patch.dict(os.environ, environment, clear=True),
            initialize_config_dir(config_dir=str(config_dir), version_base="1.3"),
        ):
            cfg = compose(config_name="train", overrides=[f"task={task}"])
            return OmegaConf.to_container(cfg, resolve=True)

    def test_defaults_follow_home_without_machine_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self.resolve(
                {"HOME": temporary, "FASTWAM_DATASET_ROOT": "/data/fixture"}
            )
            cache = f"{temporary}/.cache/fastwam"
            self.assertEqual(
                cfg["data"]["train"]["text_embedding_cache_dir"],
                f"{cache}/text_embeds/b601",
            )
            self.assertTrue(
                cfg["output_dir"].startswith(f"{cache}/runs/b601_ee_delta_v2/")
            )
            self.assertTrue(
                cfg["model"]["action_dit_pretrained_path"].startswith(
                    f"{cache}/checkpoints/"
                )
            )

    def test_explicit_paths_override_cache_root(self):
        cfg = self.resolve(
            {
                "HOME": "/unused/home",
                "FASTWAM_CACHE_ROOT": "/cache/fixture",
                "FASTWAM_DATASET_ROOT": "/data/fixture",
                "FASTWAM_TEXT_CACHE": "/data/text cache",
                "FASTWAM_RUN_ROOT": "/results/runs",
                "FASTWAM_ACTION_BACKBONE": "/models/action.pt",
            }
        )
        self.assertEqual(
            cfg["data"]["train"]["text_embedding_cache_dir"], "/data/text cache"
        )
        self.assertTrue(cfg["output_dir"].startswith("/results/runs/b601_ee_delta_v2/"))
        self.assertEqual(
            cfg["model"]["action_dit_pretrained_path"], "/models/action.pt"
        )

    def test_continuation_keeps_checkpoint_root_and_separates_metadata(self):
        cfg = self.resolve(
            {
                "HOME": "/unused/home",
                "FASTWAM_DATASET_ROOT": "/data/fixture",
                "FASTWAM_RESUME_RUN": "/runs/original",
                "FASTWAM_RESUME_STEP": "step_000123",
            },
            task="archive/legacy/b601_resume_10epoch",
        )
        self.assertEqual(cfg["output_dir"], "/runs/original")
        self.assertEqual(cfg["resume"], "/runs/original/checkpoints/state/step_000123")
        self.assertEqual(
            cfg["continuation_dir"],
            "/runs/original/continuations/step_000123_to_10epochs",
        )


if __name__ == "__main__":
    unittest.main()
