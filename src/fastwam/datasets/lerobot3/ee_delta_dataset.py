"""LeRobot v3 EE-delta adapter enforcing a frozen training/validation split."""

import json
from functools import partial
from pathlib import Path

from accelerate import PartialState

from fastwam.datasets.lerobot.processors.ee_delta_processor import EEDeltaProcessor
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from fastwam.utils import misc

from .base_lerobot_dataset import BaseLerobotDataset
from .ee_delta_preparation import validate_frozen_statistics, read_rows
from .lerobot_dataset import MultiLeRobotDataset
from .robot_video_dataset import RobotVideoDataset


class _EpisodeSubset(MultiLeRobotDataset):
    def __init__(self, *, selected_episodes, dataset_dirs, episodes=None, **kwargs):
        super().__init__(
            dataset_dirs=dataset_dirs,
            episodes={dataset_dirs[0]: selected_episodes},
            **kwargs,
        )


class _EEBaseDataset(BaseLerobotDataset):
    def __init__(self, *, selected_episodes, **kwargs):
        self.multi_dataset_cls = partial(
            _EpisodeSubset, selected_episodes=selected_episodes
        )
        super().__init__(**kwargs)


class EEDeltaDataset(RobotVideoDataset):
    def __init__(
        self,
        *,
        dataset_dirs,
        pretrained_norm_stats,
        is_training_set,
        selected_task_index=None,
        val_set_proportion=0.0,
        **kwargs,
    ):
        if len(dataset_dirs) != 1 or val_set_proportion != 0:
            raise ValueError("Use one EE bundle and its frozen episode split")
        artifact = json.loads(Path(pretrained_norm_stats).read_text())
        contract = validate_frozen_statistics(dataset_dirs[0], artifact)
        if (
            kwargs.get("num_frames", 33) - 1 != contract["horizon"]
            or kwargs.get("global_sample_stride", 1) != 1
            or kwargs.get("skip_padding_as_possible", False)
        ):
            raise ValueError("Use the saved horizon, stride 1 and all padded windows")
        selected = contract["split"][
            "train_episodes" if is_training_set else "validation_episodes"
        ]
        if selected_task_index is not None:
            rows = read_rows(dataset_dirs[0])
            task_episodes = set(rows["episode_index"][rows["task_index"] == int(selected_task_index)].tolist())
            selected = [episode for episode in selected if episode in task_episodes]
            if not selected:
                raise ValueError("Requested task has no episodes in the frozen split")
        self.base_dataset_cls = partial(_EEBaseDataset, selected_episodes=selected)
        self.ee_delta_contract = contract
        super().__init__(
            dataset_dirs=dataset_dirs,
            pretrained_norm_stats=pretrained_norm_stats,
            is_training_set=is_training_set,
            val_set_proportion=0.0,
            **kwargs,
        )
        if not isinstance(self.lerobot_dataset.processor, EEDeltaProcessor):
            raise TypeError("EEDeltaDataset requires EEDeltaProcessor")
        # Native loading casts every numeric list to FP32. Restore the canonical
        # JSON artifact when exporting, including its high-precision summaries.
        if PartialState().is_main_process:
            save_dataset_stats_to_json(
                artifact, str(Path(misc.get_work_dir()) / "dataset_stats.json")
            )

    def __getitem__(self, idx):
        return self._get(idx)
