"""Protect original run records when selecting a continuation output path."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.b601.train_resume import validate_resume_paths
from omegaconf import OmegaConf


class ResumePathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        state = self.run / "checkpoints/state/step_003235"
        state.mkdir(parents=True)
        (state / "trainer_state.json").write_text(json.dumps({"global_step": 3235}))
        (state / "scheduler.bin").touch()
        (self.run / "config.yaml").write_text("num_epochs: 5\n")
        self.cfg = OmegaConf.create(
            {
                "output_dir": str(self.run),
                "resume": str(state),
                "continuation_dir": str(
                    self.run / "continuations/step_003235_to_10epoch"
                ),
            }
        )

    def test_original_metadata_and_checkpoint_paths_are_preserved(self):
        record_dir, state = validate_resume_paths(self.cfg)
        self.assertEqual(state["global_step"], 3235)
        self.assertFalse(record_dir.exists())
        self.assertEqual((self.run / "config.yaml").read_text(), "num_epochs: 5\n")
        self.assertFalse((self.run / "checkpoints").is_symlink())

    def test_rejects_overwriting_existing_continuation(self):
        record_dir = Path(self.cfg.continuation_dir)
        record_dir.mkdir(parents=True)
        (record_dir / "config.yaml").touch()
        with self.assertRaises(FileExistsError):
            validate_resume_paths(self.cfg)

    def test_rejects_metadata_outside_continuations(self):
        self.cfg.continuation_dir = str(self.run)
        with self.assertRaises(ValueError):
            validate_resume_paths(self.cfg)

    def test_rejects_state_from_another_run(self):
        self.cfg.output_dir = str(self.run / "another_run")
        with self.assertRaises(ValueError):
            validate_resume_paths(self.cfg)


if __name__ == "__main__":
    unittest.main()
