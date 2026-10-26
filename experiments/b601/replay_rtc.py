"""Evaluate RTC on recorded rollouts; never connect cameras, CAN or motors.

Paired replay uses the lossless query snapshots and actual unexecuted tails.
Stream replay runs the production RTC actor, model, IK and recording workers
against timestamped videos and measured states. Feedback follows the recorded
trajectory independently of new predictions, so this is not a physics simulator.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from .deploy import guard_offline_hardware
from .fastwam_policy.controller import EEDeltaController, IKUnreachable
from .fastwam_policy.policy import B601Policy
from .local import source_manifest
from .runtime.config import load_config
from .runtime.connection import FastWAMConnection
from .runtime.contracts import JOINTS, CameraUnavailable, Observation
from .runtime.locks import robot_locks
from .runtime.recording import Recorder, write_json
from .runtime.rtc_contract import RTCRequest
from .runtime.rtc_controller import control_loop
from .runtime.rtc_queue import elapsed_steps


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0}
    return dict(
        n=len(values),
        min=float(values.min()),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        max=float(values.max()),
        mean=float(values.mean()),
    )


def dataset(run):
    queries = sorted(
        (read_json(p) for p in run.glob("query_*.json")),
        key=lambda q: q["policy_timing"]["dispatched_monotonic"],
    )
    commands = [
        r for r in read_rows(run / "commands.jsonl") if r.get("phase") == "policy"
    ]
    groups = defaultdict(list)
    for row in commands:
        groups[row["query_id"]].append(row)
    return queries, commands, groups


def source_observation(arrays, query):
    now = time.monotonic()
    source_time = query["policy_timing"]["dispatched_monotonic"]
    return Observation(
        arrays["state"].copy(),
        {k: arrays[k].copy() for k in ("front", "side")},
        now - (source_time - query["captured_monotonic"]),
        time.time(),
        now - (source_time - query["state_monotonic"]),
        camera_skew_s=query["camera_skew_s"],
        camera_generation=0,
    )


def paired_replay(policy, run, task, cfg, output):
    queries, commands, groups = dataset(run)
    decoder = EEDeltaController(policy.adapter.codec)
    connection = FastWAMConnection(policy)
    by_id = {q["request_id"]: q for q in queries}
    budget = elapsed_steps(
        cfg.rtc.latency_budget_s + cfg.rtc.guard_s, 1 / cfg.control.hz
    )
    rows = []
    prepared = output / "paired" / run.name
    prepared.mkdir(parents=True)
    # Warm both paths on real saved observations, before paired measurements.
    with np.load(run / ("query_" + queries[0]["request_id"] + ".npz")) as data:
        obs = source_observation(data, queries[0])
        begin = time.perf_counter()
        first = connection.predict(obs, task)
        cold = time.perf_counter() - begin
        warm_end = min(budget + cfg.rtc.blend_steps, cfg.control.execution_horizon, 31)
        warm_request = RTCRequest(
            0,
            0,
            first.actions[:warm_end].copy(),
            min(budget, warm_end),
            warm_end,
            cfg.rtc.max_guidance,
            cfg.rtc.guidance_mode,
            guidance_start=cfg.rtc.guidance_start,
        )
        connection.predict(obs, task, rtc_request=warm_request)
    for number, query in enumerate(queries):
        source_time = query["policy_timing"]["dispatched_monotonic"]
        prior = [c for c in commands if c["sent_monotonic"] <= source_time]
        prefix = None
        previous_id = None
        offset = 0
        # At the recorded query boundary, reconstruct only real unexecuted rows.
        if prior:
            previous_id = prior[-1]["query_id"]
            old_query = by_id.get(previous_id)
            if (
                old_query is not None
                and old_query["generation"] == query["generation"]
                and old_query["camera_generation"] == query["camera_generation"]
            ):
                offset = prior[-1]["action_index"] + 1
                with np.load(run / ("query_" + previous_id + ".npz")) as old:
                    left = old["actions"][offset : cfg.control.execution_horizon].copy()
                end = min(len(left), budget + cfg.rtc.blend_steps, 31)
                if end:
                    prefix = left[:end]
        end = len(prefix) if prefix is not None else 0
        request = RTCRequest(
            0,
            number,
            prefix,
            min(budget, end),
            end,
            cfg.rtc.max_guidance,
            cfg.rtc.guidance_mode,
            guidance_start=cfg.rtc.guidance_start,
        )
        with np.load(run / ("query_" + query["request_id"] + ".npz")) as data:
            obs = source_observation(data, query)
            recorded = data["actions"].copy()
        results, times = {}, {}
        order = ("plain", "rtc") if number % 2 else ("rtc", "plain")
        for mode in order:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = (
                connection.predict(obs, task, rtc_request=request)
                if mode == "rtc"
                else connection.predict(obs, task)
            )
            torch.cuda.synchronize()
            times[mode] = time.perf_counter() - begin
            results[mode] = result.actions
        diagnostics = {}
        for mode, actions in results.items():
            failures, clipped, ik_times = [], 0, []
            for command in groups[query["request_id"]]:
                step = command["action_index"]
                begin = time.perf_counter()
                try:
                    ik = decoder.solve(command["observed_physical"], actions[step])
                    clipped += int(any(ik.native_limit_clipped))
                except IKUnreachable as exc:
                    failures.append(dict(step=step, error=str(exc)))
                ik_times.append(time.perf_counter() - begin)
            diagnostics[mode] = dict(
                attempted=len(ik_times),
                failed=len(failures),
                errors=failures,
                native_clipped=clipped,
                latency_s=stats(ik_times),
            )
        row = dict(
            query_id=query["request_id"],
            previous_query=previous_id,
            old_consumed=offset,
            guided=prefix is not None,
            committed_steps=request.committed_steps,
            guidance_horizon=end,
            latency_s=times,
            historical_latency_s=query["roundtrip_s"],
            historical_image_age_s=source_time - query["captured_monotonic"],
            original_prediction_max_abs_difference=float(
                np.max(np.abs(recorded - results["plain"]))
            ),
            ik=diagnostics,
        )
        if prefix is not None:
            normalized = {
                k: policy.adapter.normalizer.transform(
                    torch.from_numpy(v), "action"
                ).numpy()
                for k, v in {**results, "prefix": prefix}.items()
            }
            row["strong_prefix_rmse"] = {
                k: float(
                    np.sqrt(
                        np.mean(
                            (
                                normalized[k][: request.committed_steps]
                                - normalized["prefix"][: request.committed_steps]
                            )
                            ** 2
                        )
                    )
                )
                for k in results
            }
        np.savez_compressed(
            prepared / ("query_" + query["request_id"] + ".npz"),
            plain=results["plain"],
            rtc=results["rtc"],
            prefix=np.empty((0, 7)) if prefix is None else prefix,
        )
        rows.append(row)
        if (number + 1) % 20 == 0 or number + 1 == len(queries):
            print(
                json.dumps(
                    dict(
                        stage="paired",
                        run=run.name,
                        done=number + 1,
                        total=len(queries),
                    )
                ),
                flush=True,
            )
    summary = dict(
        run=str(run),
        task=task,
        queries=len(rows),
        guided_queries=sum(r["guided"] for r in rows),
        cold_start_s=cold,
        latency_s={
            k: stats([r["latency_s"][k] for r in rows]) for k in ("plain", "rtc")
        },
        historical_latency_s=stats([r["historical_latency_s"] for r in rows]),
        historical_image_age_s=stats([r["historical_image_age_s"] for r in rows]),
        original_prediction_max_abs_difference=max(
            r["original_prediction_max_abs_difference"] for r in rows
        ),
        ik={
            k: {
                field: sum(r["ik"][k][field] for r in rows)
                for field in ("attempted", "failed", "native_clipped")
            }
            for k in ("plain", "rtc")
        },
    )
    write_json(prepared / "report.json", dict(summary=summary, queries=rows))
    return summary


class ReplayHardware:
    """Recorded feedback source plus numerical driver clamp, without device IO."""

    def __init__(self, run, decoder):
        import cv2

        self.cv2, self.decoder = cv2, decoder
        self.rows = [
            r
            for r in read_rows(run / "rollout.jsonl")
            if r.get("video_index") is not None
        ]
        self.times = np.array([r["monotonic"] for r in self.rows])
        self.commands = [
            r
            for r in read_rows(run / "commands.jsonl")
            if "observed_physical" in r and "sent_monotonic" in r
        ]
        self.command_times = np.array([r["sent_monotonic"] for r in self.commands])
        self.source_start = next(
            r["monotonic"] for r in self.rows if r["phase"] == "policy"
        )
        self.duration = self.times[-1] - self.source_start
        self.caps = {}
        for name in ("front", "side"):
            path = run / (name + ".avi")
            if not path.is_file():
                raise ValueError("Missing recorded video: " + str(path))
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError("Cannot read recorded video: " + str(path))
            self.caps[name] = cap
        self.index = -1
        self.views = {}
        self.started = None
        self.directions = decoder.codec.scale
        self.limits = np.asarray(decoder.codec.config["driver_motor_limits_deg"])
        self.ik_times = []
        self.observation_times = []

    def current(self):
        source_time = self.source_start + time.monotonic() - self.started
        index = int(
            np.clip(
                np.searchsorted(self.times, source_time, side="right") - 1,
                0,
                len(self.rows) - 1,
            )
        )
        return self.rows[index]

    def observe(self):
        begin = time.monotonic()
        # Camera gaps have motor-only command rows and no video frame. Do not
        # relabel the last available frame as a fresh observation during a gap.
        if self.current_command()["phase"] == "camera_wait":
            raise CameraUnavailable("Recorded camera gap")
        row = self.current()
        index = row["video_index"]
        if index != self.index:
            for name, cap in self.caps.items():
                if index != self.index + 1:
                    cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Failed to decode recorded {name} frame {index}")
                self.views[name] = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            self.index = index
        now = time.monotonic()
        # Preserve recorded image/state ages, without adding sampling-grid aliasing.
        obs = Observation(
            np.asarray(row["observed_physical"]),
            dict(self.views),
            now - (row["monotonic"] - row["image_monotonic"]),
            time.time(),
            now - (row["monotonic"] - row["state_monotonic"]),
            capture_sequence=row["capture_sequence"],
            camera_skew_s=row["camera_skew_s"],
            camera_generation=row["camera_generation"],
        )
        self.observation_times.append(time.monotonic() - begin)
        return obs, {}

    def current_command(self):
        source_time = self.source_start + time.monotonic() - self.started
        index = int(
            np.clip(
                np.searchsorted(self.command_times, source_time, side="right") - 1,
                0,
                len(self.commands) - 1,
            )
        )
        return self.commands[index]

    def observe_state(self):
        return np.asarray(self.current_command()["observed_physical"]), {}

    def hold_command(self, measured):
        return np.clip(measured, self.limits[:, 0], self.limits[:, 1]) / self.directions

    def decode(self, measured, action):
        begin = time.perf_counter()
        try:
            result = self.decoder.solve(measured, action)
            return np.asarray([result.command[j + ".pos"] for j in JOINTS]), asdict(
                result
            )
        finally:
            self.ik_times.append(time.perf_counter() - begin)

    def send(self, recorded):
        raw = np.asarray(recorded) * self.directions
        target = np.clip(raw, self.limits[:, 0], self.limits[:, 1])
        return dict(
            recorded_command=np.asarray(recorded).tolist(),
            driver_target_physical=target.tolist(),
            native_limit_clipped=(np.abs(raw - target) > 1e-5).tolist(),
        )

    def close(self):
        for cap in self.caps.values():
            cap.release()


def stream_replay(policy, run, task, cfg, output, *, connection=None):
    hw = ReplayHardware(run, EEDeltaController(policy.adapter.codec))
    cfg = replace(
        cfg,
        task=task,
        rtc=replace(cfg.rtc, enabled=True),
        display_data=False,
        save_rollout=True,
        control=replace(cfg.control, duration_s=hw.duration, startup_home=False),
    )
    path = output / "stream" / run.name
    recorder = Recorder(cfg, "recorded_rtc_replay", str(path))
    recorder.start_telemetry()
    connection = FastWAMConnection(policy) if connection is None else connection
    hw.started = time.monotonic()
    print(
        json.dumps(dict(stage="stream_start", run=run.name, duration_s=hw.duration)),
        flush=True,
    )
    report = control_loop(hw, connection, cfg, recorder, lambda: None)
    report.update(
        hardware_connected=False,
        source_run=str(run),
        feedback_source="Historical measured trajectory, independent of predicted commands",
        images="Recorded MJPEG videos; timestamps rebased preserving recorded image/state ages",
        observation_replay_latency_s=stats(hw.observation_times),
        ik_latency_s=stats(hw.ik_times),
    )
    recorder.close(report)
    summary = summarize_stream(path)
    print(
        json.dumps(dict(stage="stream_complete", run=run.name, **summary)), flush=True
    )
    return summary


def process_stream_replay(checkpoint, run, task, cfg, output):
    """Use the deployment backend without loading a second CUDA model in the parent."""
    from .fastwam_policy.observation import ObservationAdapter
    from .runtime.process_connection import FastWAMProcessConnection
    from .runtime.rtc_warmup import warmup_rtc

    root = Path(checkpoint).expanduser().absolute().parent
    adapter = ObservationAdapter(root / "codec.json", root / "robot.urdf", root / "norm_stats.json")
    logs = output / "model-workers"
    logs.mkdir(exist_ok=True)
    connection = FastWAMProcessConnection(
        checkpoint, cfg.model.device, cfg.model.num_inference_steps, cfg.model.seed,
        task=task, log_path=logs / (run.name + ".log"),
        inference_timeout_s=cfg.control.inference_timeout_s,
    )
    try:
        queries, _, _ = dataset(run)
        query = queries[0]
        with np.load(run / ("query_" + query["request_id"] + ".npz")) as arrays:
            obs = source_observation(arrays, query)
        first = connection.predict(obs, task)
        events = []
        recorder = SimpleNamespace(event=lambda kind, **fields: events.append({"kind": kind, **fields}))
        warmup_rtc(connection, first, replace(cfg, task=task), recorder)
        write_json(logs / (run.name + ".json"), {"model": connection.open(), "warmup": events})
        return stream_replay(SimpleNamespace(adapter=adapter), run, task, cfg, output,
                             connection=connection)
    finally:
        connection.close()


def summarize_stream(path):
    report = read_json(path / "report.json")
    events, commands = read_rows(path / "events.jsonl"), read_rows(
        path / "commands.jsonl"
    )
    accepted = [
        r for r in events if r["kind"] == "rtc_queue_response" and r["accepted"]
    ]
    guided = [r for r in accepted if r["guided"]]
    cuts = [r["merge_delay_steps"] for r in guided]
    queries = {
        r["request_id"]: r for r in (read_json(p) for p in path.glob("query_*.json"))
    }
    policy_rows = [r for r in commands if r["phase"] == "policy"]
    counts = defaultdict(int)
    first = {}
    for r in policy_rows:
        counts[r["query_id"]] += 1
        first.setdefault(r["query_id"], r)
    started = policy_rows[0]["sent_monotonic"] if policy_rows else math.inf
    after = [r for r in commands if r["sent_monotonic"] >= started]
    summary = dict(
        source_run=report["source_run"],
        outcome=report["outcome"],
        error=report.get("error"),
        policy_commands=report["policy_commands"],
        rejected_policy_actions=report["rejected_policy_actions"],
        native_clipped_commands=report["policy_clipped_commands"],
        guided_responses=len(guided),
        queue_starvations=report["queue_starvations"],
        holds_after_bootstrap=sum(r["phase"] != "policy" for r in after),
        camera_wait_ticks=sum(r["phase"] == "camera_wait" for r in after),
        camera_recovery=report["camera_recovery"],
        ticks_after_bootstrap=len(after),
        latency_crop_steps=stats(cuts),
        latency_crop_fraction=stats(np.array(cuts) / 32),
        chunks_crop_at_least_half=sum(x >= 16 for x in cuts),
        available_actions_after_admission=stats([r["remaining"] for r in guided]),
        request_to_admission_s=stats([r["latency_s"] for r in guided]),
        worker_roundtrip_s=stats(
            [q["roundtrip_s"] for q in queries.values() if q["rtc"]["mode"] == "guided"]
        ),
        request_to_first_command_s=stats(
            [
                r["sent_monotonic"]
                - queries[q]["policy_timing"]["preparation_started_monotonic"]
                for q, r in first.items()
                if queries[q]["rtc"]["mode"] == "guided"
            ]
        ),
        image_to_first_command_s=stats(
            [
                r["sent_monotonic"] - queries[q]["captured_monotonic"]
                for q, r in first.items()
                if queries[q]["rtc"]["mode"] == "guided"
            ]
        ),
        completed_plan_executed_steps=stats(
            [counts[r["request_id"]] for r in accepted[:-1]]
        ),
        completed_plan_unexecuted_tail=stats(
            [
                max(0, r["remaining"] - counts[r["request_id"]])
                for r in accepted[:-1]
            ]
        ),
        control_tick_s=stats([r["dt_s"] for r in commands if "dt_s" in r]),
        control_ticks_over_50ms=sum(r.get("dt_s", 0) > 0.05 for r in commands),
        ik_latency_s=report["ik_latency_s"],
        observation_replay_latency_s=report["observation_replay_latency_s"],
    )
    write_json(path / "timing-summary.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=Path("~/.config/fastwam/deploy.yml").expanduser()
    )
    p.add_argument("--mode", choices=["paired", "stream", "all"], default="all")
    p.add_argument("--stream-backend", choices=["process", "thread"], default="process",
                   help="Use the deployed model process, or the historical thread for comparison")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed reports; do not rerun completed trials",
    )
    args = p.parse_args(argv)
    sys.addaudithook(guard_offline_hardware)
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    results = dict(
        hardware_connected=False,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=source_manifest(),
        limitations=[
            "Recorded feedback does not react to new commands; no closed-loop success or safety claim.",
            "Video replay includes production model, IK and logger, excludes physical CAN/USB IO.",
            "MJPEG replay is lossy; paired query snapshots are lossless.",
        ],
        runs=[],
    )
    if args.resume and (args.output / "report.json").exists():
        previous = read_json(args.output / "report.json")
        results["prior_evaluation_source_sha256"] = previous.get(
            "prior_evaluation_source_sha256", [previous["source_code_sha256"]]
        )
    policy, checkpoint = None, None
    with robot_locks(cfg.robot.lock_paths):
        try:
            for run in args.runs:
                source = read_json(run / "report.json")
                source_cfg = read_json(run / "config.json")["configuration"]
                task = source_cfg["task"]
                selected = Path(
                    cfg.model.checkpoints[source["selected_model"]]
                ).expanduser()
                row = dict(
                    run=str(run),
                    checkpoint=str(selected),
                    stream_backend=args.stream_backend,
                )
                if args.mode != "stream" or args.stream_backend == "thread":
                    if selected != checkpoint:
                        if policy is not None:
                            del policy
                            gc.collect()
                            torch.cuda.empty_cache()
                        policy = B601Policy(
                            selected, cfg.model.device, cfg.model.num_inference_steps, cfg.model.seed
                        )
                        checkpoint = selected
                    policy.load_context(task)
                    row["checkpoint_step"] = policy.checkpoint_step
                if args.mode in ("paired", "all"):
                    saved = args.output / "paired" / run.name / "report.json"
                    row["paired"] = (
                        read_json(saved)["summary"]
                        if args.resume and saved.exists()
                        else paired_replay(policy, run, task, cfg, args.output)
                    )
                if args.mode in ("stream", "all"):
                    saved = args.output / "stream" / run.name
                    if args.resume and (saved / "report.json").exists():
                        row["stream"] = summarize_stream(saved)
                    elif args.stream_backend == "process":
                        # Release the paired-mode model before starting the only CUDA child.
                        if policy is not None:
                            del policy
                            policy = checkpoint = None
                            gc.collect()
                            torch.cuda.empty_cache()
                        row["stream"] = process_stream_replay(selected, run, task, cfg, args.output)
                    else:
                        row["stream"] = stream_replay(policy, run, task, cfg, args.output)
                results["runs"].append(row)
                write_json(args.output / "report.json", results)
        finally:
            write_json(args.output / "report.json", results)
    print(
        json.dumps(dict(stage="complete", report=str(args.output / "report.json"))),
        flush=True,
    )
    return int(
        any(
            r.get("stream", {}).get("outcome", "timeout") != "timeout"
            for r in results["runs"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
