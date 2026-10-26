#!/usr/bin/env python3
"""Check the submission tree without importing models or connecting to devices."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
ARTIFACTS = {".pt", ".pth", ".ckpt", ".safetensors", ".parquet", ".npy", ".npz", ".tar", ".gz", ".zip"}
TEXT = {".py", ".sh", ".md", ".txt", ".json", ".yml", ".yaml", ".toml"}


def check() -> int:
    errors = []
    files = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_relative_to(ROOT):
                errors.append(f"Broken or external symlink: {rel}")
            continue
        if path.is_file():
            files.append(path)
            if path.suffix.lower() in ARTIFACTS or path.stat().st_size > 5_000_000:
                errors.append(f"Unexpected generated or large artifact: {rel}")

    required = [
        ".gitattributes", "LICENSE", "NOTICE.txt", "README.md", "README.zh-CN.md", "SOURCE_PROVENANCE.json",
        "experiments/b601/runtime/LICENSE", "experiments/b601/runtime/upstream.json",
        "experiments/b601/configs/local.example.yml", "third_party/RoboTwin/LICENSE",
        ".github/workflows/cpu.yml", "environments/requirements-cpu.txt",
        "third_party/RoboTwin/task_config/_camera_config.yml",
        "third_party/RoboTwin/task_config/_config_template.yml",
        "third_party/RoboTwin/task_config/_embodiment_config.yml",
        "third_party/RoboTwin/task_config/_eval_step_limit.yml",
        "third_party/RoboTwin/task_config/demo_clean.yml",
        "third_party/RoboTwin/task_config/demo_randomized.yml",
        "third_party/RoboTwin/task_config/create_task_config.sh",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"Missing required file: {name}")

    parsed = 0
    # Vendor code is retained verbatim and may target other Python versions.
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py" and not rel.startswith("third_party/"):
            try:
                ast.parse(path.read_bytes(), filename=rel)
                parsed += 1
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Python syntax: {rel}: {exc}")
        if path.suffix.lower() in TEXT and path != Path(__file__).resolve():
            content = path.read_text(encoding="utf-8", errors="replace")
            # Keep machine addresses and absolute user homes in external configs.
            private = [
                r"\b(?:10\.\d+|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d+\.\d+\b",
                r"/home/[^/\s'\"]+/",
            ]
            if any(re.search(pattern, content) for pattern in private):
                errors.append(f"Machine-specific address or path: {rel}")

    try:
        provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
        hashes = provenance["preserved_source_sha256"]
        for name, expected in hashes.items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"Audited source fingerprint differs: {name}")
        runtime = ROOT / "experiments/b601/runtime"
        attribution = json.loads((runtime / "upstream.json").read_text(encoding="utf-8"))
        for name, value in {**attribution["unchanged_files"], **attribution["adapted_files"]}.items():
            expected = value if isinstance(value, str) else value["local_sha256"]
            path = runtime / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"Runtime attribution fingerprint differs: {name}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"Invalid provenance: {exc}")
        hashes = {}

    authored_docs = [
        *ROOT.glob("README*.md"), *ROOT.glob("CONTRIBUTING*.md"),
        *(ROOT / "docs").glob("*.md"), ROOT / "environments/README.md",
        ROOT / "experiments/b601/README.md",
    ]
    checked_links = 0
    for path in authored_docs:
        if "upstream" in path.name:
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            checked_links += 1
            if not (path.parent / unquote(url.path)).exists():
                errors.append(f"Broken documentation link: {path.relative_to(ROOT)} -> {target}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"Checked {len(files)} source files, {parsed} Python files, "
          f"{len(hashes)} audited fingerprints and {checked_links} documentation links; "
          f"{len(errors)} error(s).")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(check())
