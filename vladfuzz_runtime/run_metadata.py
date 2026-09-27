import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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
        json.dump(metadata, file, indent=4, ensure_ascii=False)
