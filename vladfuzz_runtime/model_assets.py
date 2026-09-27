import os
from pathlib import Path
from typing import Mapping, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_BERT_MODEL_DIR = DEFAULT_MODEL_ROOT / "shared" / "bert-base-uncased"
REMOTE_BERT_MODEL_ID = "google-bert/bert-base-uncased"
_BERT_METADATA_FILES = ("config.json", "vocab.txt")
_BERT_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")


def resolve_model_root(env: Optional[Mapping[str, str]] = None) -> Path:
    values = os.environ if env is None else env
    configured = Path(values.get("VLADFUZZ_MODEL_HOME", DEFAULT_MODEL_ROOT)).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


def resolve_model_asset(
    relative_path: Union[str, Path],
    *,
    override_env_var: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    values = os.environ if env is None else env
    if override_env_var and values.get(override_env_var):
        return Path(values[override_env_var]).expanduser().resolve()
    return resolve_model_root(values) / relative_path


def validate_bert_model_dir(path: Union[str, Path]) -> Path:
    model_dir = Path(path).expanduser().resolve()
    missing = [name for name in _BERT_METADATA_FILES if not (model_dir / name).is_file()]
    if not any((model_dir / name).is_file() for name in _BERT_WEIGHT_FILES):
        missing.append("pytorch_model.bin or model.safetensors")
    if missing:
        raise RuntimeError(
            f"Local BERT model directory is incomplete: {model_dir}. "
            f"Missing: {', '.join(missing)}"
        )
    return model_dir


def resolve_bert_model(
    backend_env_var: str,
    *,
    default_local_dir: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    values = os.environ if env is None else env
    configured = values.get(backend_env_var) or values.get("VLADFUZZ_BERT_MODEL")
    if configured:
        return str(validate_bert_model_dir(configured))

    local_dir = Path(
        default_local_dir
        or resolve_model_asset("shared/bert-base-uncased", env=values)
    )
    if local_dir.exists():
        return str(validate_bert_model_dir(local_dir))

    if values.get("VLADFUZZ_STRICT_LOCAL_ASSETS", "0").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            f"Local BERT model is required but missing at {local_dir.resolve()}. "
            "Run: python scripts/prepare_models.py --backend lmdrive "
            "--accept-model-licenses (or select bevdriver)."
        )
    return REMOTE_BERT_MODEL_ID
