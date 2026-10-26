"""Task-only view of the native frozen-split EE dataset; preprocessing is unchanged."""
import numpy as np
from torch.utils.data import Dataset
from fastwam.datasets.lerobot3.ee_delta_dataset import EEDeltaDataset
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


class TaskSubsetDataset(Dataset):
    def __init__(self, *, task_index, task_text, **kwargs):
        self.native_dataset = EEDeltaDataset(selected_task_index=int(task_index), **kwargs)
        self.lerobot_dataset = self.native_dataset.lerobot_dataset
        self.ee_delta_contract = self.native_dataset.ee_delta_contract
        self.task_index = int(task_index)
        self.expected_prompt = DEFAULT_PROMPT.format(task=task_text)
        inner = self.lerobot_dataset.multi_dataset._datasets[0]
        columns = inner.hf_dataset.select_columns(
            ['episode_index', 'frame_index', 'task_index', 'index']
        ).with_format(None)[:]
        self.rows = {k: np.asarray(v) for k, v in columns.items()}
        if len(self.rows['task_index']) != len(self.native_dataset):
            raise ValueError('Native stride-1 index mapping changed')
        self.indices = np.flatnonzero(self.rows['task_index'] == self.task_index)
        if len(self.indices) != len(self.native_dataset):
            raise ValueError('Underlying retry population must contain only the requested task')
        if not len(self.indices):
            raise ValueError('Requested task has no samples in the frozen split')
        self.episodes = np.unique(self.rows['episode_index'][self.indices]).tolist()
        print(f'[task-filter] training={kwargs["is_training_set"]} '
              f'task={self.task_index} samples={len(self)} episodes={len(self.episodes)}', flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.native_dataset[int(self.indices[index])]
        if sample['prompt'] != self.expected_prompt:
            raise RuntimeError('Task filter returned an unexpected instruction')
        return sample
