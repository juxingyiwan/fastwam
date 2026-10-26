"""Adapt LeRobot v3 task metadata to the native text-cache entry point."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import precompute_text_embeds as native
from omegaconf import DictConfig, OmegaConf


def export_tasks(source: Path, destination: Path):
    tasks = pd.read_parquet(source / "meta" / "tasks.parquet")
    if not tasks["task_index"].is_unique:
        raise ValueError(f"Duplicate task indices in {source}")
    tasks = tasks.sort_values("task_index")
    records = []
    for task, row in tasks.iterrows():
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Expected task strings in the parquet index: {source}")
        records.append({"task_index": int(row["task_index"]), "task": task})
    if not records:
        raise ValueError(f"No tasks in {source}")
    destination.mkdir(parents=True)
    (destination / "tasks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Resolve before replacing paths so cache/model settings retain their values.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with TemporaryDirectory(prefix="fastwam-v30-tasks-") as temporary:
        staged = {}
        for _, node in native._iter_dataset_nodes(cfg.data):
            dataset_dirs = []
            for source in node.dataset_dirs:
                source = Path(source).expanduser().resolve()
                if source not in staged:
                    destination = Path(temporary) / str(len(staged))
                    export_tasks(source, destination / "meta")
                    staged[source] = str(destination)
                dataset_dirs.append(staged[source])
            node.dataset_dirs = dataset_dirs
        # Keep tokenization, prompt formatting, BF16 encoding and cache naming
        # in the original implementation; the source dataset is never changed.
        native.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
