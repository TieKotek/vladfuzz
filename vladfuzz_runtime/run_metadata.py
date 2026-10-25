import json
import os
import hashlib
import platform
import subprocess
from datetime import datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional


@lru_cache(maxsize=1)
def runtime_provenance() -> Dict[str, Any]:
    """Record reproducibility context without credentials or heavyweight imports."""
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for package in ("torch", "transformers", "numpy", "carla", "flash-attn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    git = {}
    try:
        git["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        git["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        git["available"] = False
    manifest = root / "configs/model-assets.json"
    settings = (
        "CUDA_VISIBLE_DEVICES", "CARLA_ROOT", "CARLA_GRAPHICS_ADAPTER",
        "CARLA_PORT", "CARLA_TM_PORT", "VLADFUZZ_CARLA_MODE",
        "VLADFUZZ_MODEL_HOME", "VLADFUZZ_STRICT_LOCAL_ASSETS",
        "LMDRIVE_CHECKPOINT", "LMDRIVE_LLM_MODEL", "LMDRIVE_VISION_ENCODER",
        "SIMLINGO_CHECKPOINT", "SIMLINGO_VISION_MODEL",
        "BEVDRIVER_CHECKPOINT", "BEVDRIVER_LLM_MODEL", "BEVDRIVER_ENCODER",
        "VLADFUZZ_BERT_MODEL", "LMDRIVE_BERT_MODEL", "BEVDRIVER_BERT_MODEL",
    )
    return {
        "python": platform.python_version(),
        "packages": packages,
        "git": git,
        "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.is_file() else None,
        "environment": {key: os.environ[key] for key in settings if key in os.environ},
    }


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_manifest_fragment(manifest_id: Optional[str]) -> str:
    if not manifest_id:
        return "manual"
    return str(manifest_id).replace("/", "__").replace(" ", "_")


def create_run_dir(output_root: str, method: str, model: str, manifest_id: Optional[str], random_seed: Optional[int]) -> str:
    seed_part = f"seed{random_seed}" if random_seed is not None else "seedna"
    run_name = f"{method}_{safe_manifest_fragment(manifest_id)}_{seed_part}_{timestamp_id()}"
    path = Path(output_root) / model / run_name
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def write_metadata(output_dir: str, metadata: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / "metadata.json"
    with path.open("w", encoding="utf-8") as file:
        payload = dict(metadata)
        payload.setdefault("runtime_provenance", runtime_provenance())
        json.dump(payload, file, indent=4, ensure_ascii=False)
