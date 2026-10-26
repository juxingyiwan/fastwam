"""Launch local GPU workers and summarize LIBERO-10 results."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(root):
    import numpy as np
    from experiments.libero.visual_ttt.core import write_json
    tasks = []
    seeds = set()
    for folder in sorted(root.glob("task_[0-9][0-9]")):
        row = {"task_id": int(folder.name.split("_")[1]), "adaptations": {}}
        baseline = folder / "baseline.json"
        if baseline.exists():
            result = json.loads(baseline.read_text())
            if result["complete"]:
                row["baseline"] = result["success_rate"]
        for path in folder.glob("adapt_*/evaluation.json"):
            result = json.loads(path.read_text())
            if result["complete"]:
                seed = path.parent.name.removeprefix("adapt_")
                row["adaptations"][seed] = result["success_rate"]
                seeds.add(seed)
        tasks.append(row)
    complete_seeds = {seed: float(np.mean([t["adaptations"][seed] for t in tasks]))
                      for seed in seeds if len(tasks) == 10 and all(seed in t["adaptations"] for t in tasks)}
    baseline_complete = len(tasks) == 10 and all("baseline" in t for t in tasks)
    summary = {"suite": "libero_10", "tasks": tasks, "full_suite_baseline_complete": baseline_complete,
               "baseline_mean": float(np.mean([t["baseline"] for t in tasks])) if baseline_complete else None,
               "full_suite_adaptation_means": complete_seeds}
    values = list(complete_seeds.values())
    summary["ttt_mean"] = float(np.mean(values)) if values else None
    summary["ttt_sd_across_adaptations"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    summary["delta_percentage_points"] = (100*(summary["ttt_mean"]-summary["baseline_mean"])
                                          if baseline_complete and values else None)
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0", help="Comma-separated allocated GPU IDs or UUIDs visible on this node.")
    parser.add_argument("--config", default="libero10_idm_visual_ttt", choices=["libero10_idm_visual_ttt", "libero10_idm_visual_ttt_smoke"])
    parser.add_argument("--stage", default="full", choices=["preflight", "baseline", "collect", "adapt", "evaluate", "ttt", "full", "check"])
    parser.add_argument("--tasks", default=None, help="Unique LIBERO-10 task IDs, e.g. 0,1.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args, overrides = parser.parse_known_args()
    os.chdir(ROOT)
    from cluster.runtime.runtime_environment import runtime_environment
    worker_environment = runtime_environment()
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name=args.config, overrides=overrides)
    tasks = list(cfg.TTT.task_ids) if args.tasks is None else [int(x) for x in args.tasks.split(",")]
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in range(10) for t in tasks):
        parser.error("--tasks must be a unique subset of LIBERO-10 task IDs 0..9.")
    gpus = args.gpus.split(",")
    if not all(gpus) or len(set(gpus)) != len(gpus):
        parser.error("GPU IDs must be unique and nonempty.")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited and not set(gpus).issubset(set(inherited.split(","))):
        parser.error("Requested GPUs are outside inherited CUDA_VISIBLE_DEVICES; use the IDs/UUIDs in that mask.")
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("run-id must be a directory name, not a path.")
    root = Path(str(cfg.TTT.output_root)) / run_id
    if args.summarize:
        if not args.run_id:
            parser.error("--summarize requires --run-id.")
        summarize(root)
        return
    workers = min(len(gpus), len(tasks))
    processes = []
    if not args.dry_run:
        (root / "worker_logs").mkdir(parents=True, exist_ok=True)
    for worker in range(workers):
        shard = tasks[worker::workers]
        command = [sys.executable, "-u", "experiments/libero/run_visual_ttt_worker.py", "--config-name", args.config,
                   *overrides, f"TTT.stage={args.stage}", f"TTT.run_id={run_id}",
                   "TTT.task_ids=[" + ",".join(map(str, shard)) + "]"]
        env = dict(worker_environment)
        env["CUDA_VISIBLE_DEVICES"] = gpus[worker]
        print(f"GPU {gpus[worker]} tasks {shard}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            log = (root / "worker_logs" / f"{args.stage}_worker_{worker}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log").open("a")
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failed = False
    for process, log in processes:
        code = process.wait()
        log.close()
        failed |= code != 0
    if not args.dry_run:
        summarize(root)
        print(f"Results: {root.resolve()}", flush=True)
    if failed:
        raise SystemExit("One or more workers failed; inspect worker_logs. Other workers were not terminated.")


if __name__ == "__main__":
    main()
