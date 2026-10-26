"""Run LIBERO-10 Visual-only TTT on the current machine and visible GPU."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.libero.visual_ttt.core import (
    LoRALinear, activate, adapter_parameters, adapter_state, array_sha256, file_sha256,
    fingerprint, frame_indices, future_flow_loss, install_lora, load_adapter_state,
    reset_adapters, task_lock, video_denoise,
    window_starts, write_json,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_evaluator():
    # The upstream evaluator registers resolvers at import and uses a sibling import.
    from experiments.libero import eval_libero_single as evaluator
    return evaluator


def validate_config(cfg):
    if cfg.EVALUATION.task_suite_name != "libero_10" or list(cfg.MULTIRUN.task_suite_names) != ["libero_10"]:
        raise ValueError("This entry point accepts only LIBERO-10.")
    if cfg.EVALUATION.action_infer_mode != "idm":
        raise ValueError("This experiment fixes action_infer_mode=idm.")
    if cfg.model._target_ != "fastwam.runtime.create_fastwam_optional_idm":
        raise ValueError("Use the released Optional IDM model for this experiment.")
    if cfg.model.compile_training_denoise:
        raise ValueError("TTT optimization uses its explicit checkpointed video forward.")
    if not isinstance(cfg.TTT.enabled, bool):
        raise ValueError("TTT.enabled must be a boolean.")
    if list(cfg.TTT.adaptation_seeds) != [int(cfg.seed)]:
        raise ValueError("TTT.adaptation_seeds must contain only cfg.seed; collection, adaptation and evaluation share the same seed.")
    if not cfg.TTT.enabled and cfg.TTT.stage not in {"preflight", "baseline", "full"}:
        raise ValueError("With TTT.enabled=false use preflight, baseline or full (baseline only).")
    if cfg.EVALUATION.visualize_future_video or cfg.EVALUATION.use_action_ensembler:
        raise ValueError("Future visualization and action ensembling must be disabled.")
    if not 1 <= int(cfg.EVALUATION.num_trials) <= 50:
        raise ValueError("Evaluation uses at most the 50 official states, without repetition.")
    tasks = list(cfg.TTT.task_ids)
    if not tasks or len(set(tasks)) != len(tasks) or any(not 0 <= int(t) < 10 for t in tasks):
        raise ValueError("task_ids must be a nonempty unique subset of 0..9.")
    horizon = int(cfg.data.train.num_frames) - 1
    stride = int(cfg.data.train.action_video_freq_ratio)
    frame_indices(0, horizon, stride)
    window_starts(int(cfg.TTT.collection_steps) + 1, horizon, int(cfg.TTT.windows_per_rollout))
    if not 1 <= int(cfg.TTT.collection_rollouts) <= 50:
        raise ValueError("Collection uses the official task initial states from index zero.")
    if int(cfg.TTT.epochs) < 1 or float(cfg.TTT.learning_rate) <= 0:
        raise ValueError("Positive epochs and learning rate are required.")
    if cfg.TTT.stage not in {"preflight", "baseline", "collect", "adapt", "evaluate", "ttt", "full", "check"}:
        raise ValueError("Unknown TTT stage.")
    if int(cfg.EVALUATION.replan_steps) > horizon:
        raise ValueError("replan_steps exceeds the predicted action horizon.")
    if float(cfg.EVALUATION.sigma_shift) != 1.0:
        raise ValueError("The released Optional IDM checkpoint uses action sigma_shift=1.0.")
    if cfg.mixed_precision != "bf16":
        raise ValueError("The validated recipe uses BF16.")
    for value in (cfg.ckpt, cfg.EVALUATION.dataset_stats_path):
        if not Path(str(value)).exists():
            raise FileNotFoundError(value)


def experiment_contract(cfg):
    checkpoint_path = Path(str(cfg.ckpt)).resolve()
    stat = checkpoint_path.stat()
    settings = OmegaConf.to_container(cfg, resolve=True)
    settings.pop("output_dir", None)
    settings.pop("gpu_id", None)
    settings["EVALUATION"].pop("output_dir", None)
    settings["EVALUATION"].pop("task_id", None)
    for key in ("task_ids", "stage", "run_id", "output_root"):
        settings["TTT"].pop(key, None)
    tracked = [
        "experiments/libero/run_visual_ttt_worker.py", "experiments/libero/visual_ttt/core.py",
        "experiments/libero/eval_libero_single.py", "experiments/libero/libero_utils.py",
        "src/fastwam/models/wan22/fastwam.py", "src/fastwam/models/wan22/fastwam_idm.py",
        "src/fastwam/models/wan22/fastwam_optional_idm.py", "src/fastwam/models/wan22/mot.py",
        "src/fastwam/models/wan22/wan_video_dit.py",
    ]
    return {
        "schema": 1, "settings": settings,
        "checkpoint": {"path": str(checkpoint_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "dataset_stats_sha256": file_sha256(cfg.EVALUATION.dataset_stats_path),
        "source_sha256": {p: file_sha256(ROOT / p) for p in tracked},
    }


def task_states(cfg, suite, task_id):
    from libero.libero import get_libero_path
    task = suite.get_task(task_id)
    # Match FastWAM's evaluator, including loading the official NumPy state file.
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    evaluation = np.asarray(torch.load(path, weights_only=False), dtype=np.float64)
    if len(evaluation) < 50:
        raise ValueError("The current LIBERO installation does not expose all 50 evaluation states.")
    return task, evaluation[:int(cfg.EVALUATION.num_trials)], evaluation[:int(cfg.TTT.collection_rollouts)]


def snapshot(obs, cfg, model, processor, ev):
    height, width = map(int, cfg.data.train.video_size)
    image, proprio, _ = ev._obs_to_model_input(obs, cfg, processor, width, height, "cpu", torch.float32)
    rgb = ((image[0].permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
    return rgb, proprio[0].numpy(), ev._extract_sim_state(obs)


def predict(obs, task, cfg, model, processor, ev):
    height, width = map(int, cfg.data.train.video_size)
    action, _, _ = ev._predict_action_chunk(
        obs, task.language, model, processor, cfg,
        action_horizon=int(cfg.data.train.num_frames) - 1,
        input_w=width, input_h=height, model_device="cuda",
    )
    return action


def collect(cfg, task, states, model, processor, ev, folder, contract_hash):
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored["contract_hash"] != contract_hash:
            raise ValueError("Collection cache contract differs; choose a new run_id.")
        for entry in stored["rollouts"]:
            if file_sha256(folder / entry["file"]) != entry["sha256"]:
                raise ValueError("Collection cache failed its checksum.")
        return stored
    entries = []
    seed = int(cfg.seed)
    seed_all(seed)
    env, _ = ev.get_libero_env(task, ev.LIBERO_ENV_RESOLUTION, seed)
    try:
        for index, initial in enumerate(states):
            frames, proprios, raw_states, actions = [], [], [], []
            done = False
            env.reset()
            obs = env.set_init_state(initial)
            for _ in range(int(cfg.EVALUATION.num_steps_wait)):
                obs, _, done, _ = env.step(ev.get_libero_dummy_action())
                if done:
                    raise RuntimeError("Collection episode terminated during warmup.")
            frame, proprio, state = snapshot(obs, cfg, model, processor, ev)
            frames.append(frame); proprios.append(proprio); raw_states.append(state)
            pending = []
            for _ in range(int(cfg.TTT.collection_steps)):
                if not pending:
                    pending = predict(obs, task, cfg, model, processor, ev)[:int(cfg.EVALUATION.replan_steps)].tolist()
                executed = np.asarray(pending.pop(0), dtype=np.float32)
                obs, _, done, _ = env.step(executed.tolist())
                actions.append(executed)
                frame, proprio, state = snapshot(obs, cfg, model, processor, ev)
                frames.append(frame); proprios.append(proprio); raw_states.append(state)
                if done:
                    break
            horizon = int(cfg.data.train.num_frames) - 1
            starts = window_starts(len(frames), horizon, int(cfg.TTT.windows_per_rollout))
            path = folder / f"rollout_{index:03d}.npz"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, rgb=np.stack(frames), proprio=np.stack(proprios),
                                    raw_state=np.stack(raw_states), executed_actions=np.stack(actions),
                                    initial_state=initial, sim_steps=np.arange(len(frames)))
            temporary.replace(path)
            entries.append({"file": path.name, "sha256": file_sha256(path), "seed": seed,
                            "initial_state_sha256": array_sha256(initial), "starts": starts,
                            "executed_steps": len(actions), "terminated": bool(done)})
            print(f"Collected rollout {index + 1}/{len(states)}: {len(actions)} executed steps", flush=True)
    finally:
        env.close()
    result = {"contract_hash": contract_hash, "rollouts": entries,
              "samples": sum(len(e["starts"]) for e in entries),
              "horizon": int(cfg.data.train.num_frames) - 1,
              "video_stride": int(cfg.data.train.action_video_freq_ratio),
              "targets": "realized future RGB only; no success/reward/action targets"}
    write_json(manifest_path, result)
    return result


def encode_records(cfg, model, folder, manifest):
    records = []
    horizon = manifest["horizon"]
    stride = manifest["video_stride"]
    cache_path = folder / "latents.pt"
    identity = fingerprint(manifest)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache["identity"] != identity:
            raise ValueError("Latent cache does not match the collection manifest.")
        return cache["records"]
    with torch.no_grad():
        for entry in manifest["rollouts"]:
            with np.load(folder / entry["file"], allow_pickle=False) as rollout:
                frames, proprios = rollout["rgb"], rollout["proprio"]
            for start in entry["starts"]:
                ids = frame_indices(start, horizon, stride)
                video = torch.from_numpy(frames[ids]).permute(3, 0, 1, 2).unsqueeze(0)
                video = video.to(model.device, model.torch_dtype) * (2 / 255.0) - 1
                # Frozen VAE: no compile/cudagraph lifetime or inference detachment involved.
                latents = model.vae.model.encode(video, model.vae.scale).detach().cpu()
                records.append({"latents": latents, "proprio": torch.from_numpy(proprios[start:start+1]).clone(),
                                "rollout": entry["file"], "start": int(start), "frame_indices": ids.tolist()})
        torch.save({"identity": identity, "records": records}, cache_path.with_suffix(".tmp"))
        cache_path.with_suffix(".tmp").replace(cache_path)
    return records


def evaluate(cfg, task, states, model, processor, ev, path, contract_hash, label):
    adapted = label.startswith("visual_ttt_")
    enabled_modules = [m for m in model.modules() if isinstance(m, LoRALinear) and (m.enabled or m.merged)] if model is not None else []
    if adapted != bool(enabled_modules):
        raise RuntimeError("Evaluation label and actual LoRA activation disagree.")
    expected = {"contract_hash": contract_hash, "label": label, "task_id": int(cfg.EVALUATION.task_id),
                "ttt_enabled": adapted, "active_lora_modules": len(enabled_modules)}
    result = {**expected, "episodes": [], "complete": False}
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Evaluation contract mismatch: {path}")
        if result["complete"]:
            return result
        # The historical evaluator carries RNG state through a whole task.
        # Restart incomplete tasks rather than resume with a different RNG stream.
        result = {**expected, "episodes": [], "complete": False}
    height, width = map(int, cfg.data.train.video_size)
    seed_all(int(cfg.seed))
    env, _ = ev.get_libero_env(task, ev.LIBERO_ENV_RESOLUTION, int(cfg.seed))
    started = time.monotonic()
    try:
        for trial in range(len(result["episodes"]), len(states)):
            paired_seed = int(cfg.seed)
            success, frames, _, _ = ev.run_single_episode(
                env=env, initial_state=states[trial], task_description=task.language,
                model=model, processor=processor, cfg=cfg, episode_idx=trial,
                action_horizon=int(cfg.data.train.num_frames) - 1,
                input_w=width, input_h=height, model_device="cuda",
            )
            if cfg.TTT.save_eval_videos:
                video_dir = path.parent / "videos" / label
                video_dir.mkdir(parents=True, exist_ok=True)
                ev.save_rollout_video(video_dir, frames, str(trial), bool(success), task.language)
            result["episodes"].append({"trial": trial, "seed": paired_seed, "success": bool(success),
                                       "initial_state_sha256": array_sha256(states[trial])})
            result["successes"] = sum(int(e["success"]) for e in result["episodes"])
            result["success_rate"] = result["successes"] / len(result["episodes"])
            result["complete"] = len(result["episodes"]) == len(states)
            result["last_session_seconds"] = time.monotonic() - started
            write_json(path, result)
            print(f"{label} task={cfg.EVALUATION.task_id} trial={trial}: {bool(success)}", flush=True)
    finally:
        env.close()
    return result


def adapt(cfg, task, model, adapters, records, seed, folder, contract_hash):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "adapter.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["contract_hash"] != contract_hash or saved["seed"] != seed:
            raise ValueError("Saved adapter contract mismatch.")
        load_adapter_state(adapters, saved["adapter"])
        return saved["summary"]
    reset_adapters(adapters, seed)
    params = adapter_parameters(adapters)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if trainable != {id(p) for p in params}:
        raise RuntimeError("Trainable parameters are not exactly the configured video LoRA parameters.")
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    context, context_mask = model.encode_prompt(DEFAULT_PROMPT.format(task=task.language))
    context, context_mask = context.detach(), context_mask.detach()
    model.eval()  # No dropout or batch-stat updates; gradients are enabled explicitly.
    optimizer = torch.optim.AdamW(params, lr=float(cfg.TTT.learning_rate), weight_decay=0.0)
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    log = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(cfg.TTT.epochs)):
        for sample_idx in torch.randperm(len(records), generator=generator).tolist():
            optimizer.zero_grad(set_to_none=True)
            loss = future_flow_loss(model, records[sample_idx], context, context_mask,
                                    bool(cfg.TTT.gradient_checkpointing))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite visual flow-matching loss.")
            loss.backward()
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in params):
                raise RuntimeError("No nonzero gradient reached video LoRA.")
            norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.TTT.max_grad_norm), error_if_nonfinite=True)
            optimizer.step()
            log.append({"step": len(log)+1, "epoch": epoch, "sample": sample_idx,
                        "loss": float(loss.detach()), "grad_norm": float(norm)})
            print(f"TTT seed={seed} step={len(log)} loss={log[-1]['loss']:.6f} grad={float(norm):.6f}", flush=True)
    delta_norm = sum(float(m.lora_B.detach().square().sum()) for m in adapters.values()) ** 0.5
    if delta_norm == 0:
        raise RuntimeError("Adapter remained unchanged after training.")
    summary = {"steps": len(log), "unique_samples": len(records), "seed": seed,
               "seconds": time.monotonic()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "trainable_parameters": sum(p.numel() for p in params), "lora_B_norm": delta_norm,
               "objective": "future_video_flow_matching_only", "log": log}
    saved = {"contract_hash": contract_hash, "seed": seed, "summary": summary, "adapter": adapter_state(adapters)}
    torch.save(saved, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    write_json(folder / "training.json", summary)
    activate(adapters, True, False)
    return summary


def gradient_check(cfg, model, adapters, task, processor, ev, initial_state, folder):
    """Integration diagnostic only: repeated images, never reported as task performance."""
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    env, _ = ev.get_libero_env(task, ev.LIBERO_ENV_RESOLUTION, int(cfg.seed))
    try:
        env.reset(); obs = env.set_init_state(initial_state)
        for _ in range(int(cfg.EVALUATION.num_steps_wait)):
            obs, _, _, _ = env.step(ev.get_libero_dummy_action())
        rgb, proprio, _ = snapshot(obs, cfg, model, processor, ev)
    finally:
        env.close()
    nframes = len(frame_indices(0, int(cfg.data.train.num_frames)-1, int(cfg.data.train.action_video_freq_ratio)))
    video = torch.from_numpy(rgb).permute(2, 0, 1)[None, :, None].repeat(1, 1, nframes, 1, 1)
    with torch.no_grad():
        latent = model.vae.model.encode(video.to(model.device, model.torch_dtype)*(2/255)-1, model.vae.scale)
        ctx, mask = model.encode_prompt(DEFAULT_PROMPT.format(task=task.language))
        ctx, mask = model._append_proprio_to_context(ctx, mask, torch.from_numpy(proprio[None]).to(model.device, model.torch_dtype))
        t = torch.full((1,), 500, device=model.device, dtype=model.torch_dtype)
        noisy = model.train_video_scheduler.add_noise(latent, torch.randn_like(latent), t)
        noisy[:, :, :1] = latent[:, :, :1]
        expected, _ = model._teacher_forcing_training_denoise_core(
            latents_noisy=noisy, latents_cond=latent,
            noisy_action=torch.zeros(1, int(cfg.data.train.num_frames)-1, 7, device=model.device, dtype=model.torch_dtype),
            timestep_video=t, timestep_video_cond=torch.zeros_like(t), timestep_action=t,
            context=ctx, context_mask=mask, fuse_vae_embedding_in_latents=True)
        actual = video_denoise(model, noisy, t, ctx, mask, False)
        rms = (actual.float()-expected.float()).square().mean().sqrt().item()
        relative = rms / max(expected.float().square().mean().sqrt().item(), 1e-8)
        if relative > float(cfg.TTT.parity_relative_tolerance):
            raise RuntimeError(f"Video-only/native IDM forward mismatch: relative RMS={relative}")
    activate(adapters, False)
    before = predict(obs, task, cfg, model, processor, ev)
    reset_adapters(adapters, int(cfg.seed))
    activate(adapters, True, False)
    zero_adapter = predict(obs, task, cfg, model, processor, ev)
    np.testing.assert_allclose(before, zero_adapter, rtol=0, atol=0)
    record = {"latents": latent.detach().cpu(), "proprio": torch.from_numpy(proprio[None]).clone()}
    summary = adapt(cfg, task, model, adapters, [record], int(cfg.seed), folder / "diagnostic_adapter", "diagnostic-only")
    after = predict(obs, task, cfg, model, processor, ev)
    delta = float(np.mean(np.abs(after-before)))
    if not np.isfinite(delta) or delta == 0:
        raise RuntimeError("Adapter updated but fixed-input IDM actions did not change.")
    write_json(folder / "integration_check.json", {"diagnostic_only": True, "native_video_relative_rms": relative,
               "zero_adapter_action_max_error": float(np.max(np.abs(before-zero_adapter))),
               "adapted_action_mean_abs_delta": delta, "training": summary})
    activate(adapters, False)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="libero10_idm_visual_ttt")
def main(cfg):
    os.chdir(ROOT)
    from cluster.runtime.runtime_environment import runtime_environment
    os.environ.update(runtime_environment())
    validate_config(cfg)
    ev = load_evaluator()
    suite = ev.benchmark.get_benchmark_dict()["libero_10"]()
    contract = experiment_contract(cfg)
    contract_hash = fingerprint(contract)
    root = Path(str(cfg.TTT.output_root)) / str(cfg.TTT.run_id)
    states = {int(t): task_states(cfg, suite, int(t)) for t in cfg.TTT.task_ids}
    for task_id, (_, evaluation, collection) in states.items():
        folder = root / f"task_{task_id:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        contract_path = folder / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text())["hash"] != contract_hash:
            raise ValueError("Existing run has different settings/source; use a new run_id.")
        write_json(contract_path, {"hash": contract_hash, "contract": contract})
        print(f"Preflight task={task_id}: eval={len(evaluation)}, collection={len(collection)}, "
              f"seed={cfg.seed}, TTT.enabled={cfg.TTT.enabled}, compile={cfg.EVALUATION.compile_action_infer}", flush=True)
    if cfg.TTT.stage == "preflight":
        return
    seed_all(int(cfg.seed))
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    ev._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to("cuda").eval().requires_grad_(False)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(ev.load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path)))
    adapters = (install_lora(model, int(cfg.TTT.lora_rank), float(cfg.TTT.lora_alpha), str(cfg.TTT.lora_pattern))
                if cfg.TTT.enabled and cfg.TTT.stage not in {"baseline", "collect"} else {})
    for task_id, (task, evaluation, collection) in states.items():
        cfg.EVALUATION.task_id = task_id
        folder = root / f"task_{task_id:02d}"
        with task_lock(folder):
            activate(adapters, False, False)
            if cfg.TTT.stage == "check":
                gradient_check(cfg, model, adapters, task, processor, ev, collection[0], folder)
                continue
            if cfg.TTT.stage in {"baseline", "full"}:
                evaluate(cfg, task, evaluation, model, processor, ev, folder / "baseline.json", contract_hash, "frozen")
            if not cfg.TTT.enabled:
                continue
            if cfg.TTT.stage in {"collect", "ttt", "full"}:
                collect(cfg, task, collection, model, processor, ev, folder / "collection", contract_hash)
            records = None
            if cfg.TTT.stage in {"adapt", "ttt", "full"}:
                manifest = json.loads((folder / "collection/manifest.json").read_text())
                if manifest["contract_hash"] != contract_hash:
                    raise ValueError("Collection contract differs from adaptation.")
                for entry in manifest["rollouts"]:
                    if file_sha256(folder / "collection" / entry["file"]) != entry["sha256"]:
                        raise ValueError("Collection data changed after recording.")
                records = encode_records(cfg, model, folder / "collection", manifest)
            if cfg.TTT.stage in {"adapt", "evaluate", "ttt", "full"}:
                for seed in cfg.TTT.adaptation_seeds:
                    seed = int(seed)
                    adaptation_dir = folder / f"adapt_{seed}"
                    if cfg.TTT.stage in {"adapt", "ttt", "full"}:
                        adapt(cfg, task, model, adapters, records, seed, adaptation_dir, contract_hash)
                    else:
                        saved = torch.load(adaptation_dir / "adapter.pt", map_location="cpu", weights_only=True)
                        if saved["contract_hash"] != contract_hash:
                            raise ValueError("Adapter contract differs from evaluation.")
                        load_adapter_state(adapters, saved["adapter"])
                    if cfg.TTT.stage in {"evaluate", "ttt", "full"}:
                        evaluate(cfg, task, evaluation, model, processor, ev, adaptation_dir / "evaluation.json",
                                 contract_hash, f"visual_ttt_{seed}")
                    activate(adapters, False, False)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
