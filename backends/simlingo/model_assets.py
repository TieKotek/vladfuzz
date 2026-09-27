"""Resolve SimLingo assets from the repository-level model directory."""

from pathlib import Path
from typing import Iterable, Optional

from vladfuzz_runtime.model_assets import resolve_model_asset


def _require_files(directory: Path, required_files: Iterable[str]) -> Path:
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            f"SimLingo base model is incomplete at {directory}. "
            f"Missing: {', '.join(missing)}. Run: python scripts/prepare_models.py "
            "--backend simlingo --accept-model-licenses"
        )
    return directory


def resolve_pretrained_dir(
    variant: str,
    required_files: Optional[Iterable[str]] = None,
) -> Path:
    model_name = variant.rsplit("/", 1)[-1]
    directory = resolve_model_asset(
        f"simlingo/{model_name}",
        override_env_var="SIMLINGO_VISION_MODEL",
    )
    return _require_files(directory, required_files or ("config.json",))


def resolve_checkpoint() -> Path:
    checkpoint = resolve_model_asset(
        "simlingo/pytorch_model.pt",
        override_env_var="SIMLINGO_CHECKPOINT",
    )
    if not checkpoint.is_file():
        raise RuntimeError(
            f"SimLingo checkpoint is missing at {checkpoint}. Run: python "
            "scripts/prepare_models.py --backend simlingo --accept-model-licenses"
        )
    return checkpoint
