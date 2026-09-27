"""Load repository-local configuration without requiring shell profile edits."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env(path: Optional[Path] = None) -> Optional[Path]:
    """Load ``.env`` once and preserve variables already set by the caller."""
    env_path = (path or PROJECT_ROOT / ".env").expanduser().resolve()
    if not env_path.is_file():
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    load_dotenv(dotenv_path=env_path, override=False)
    if os.environ.get("CARLA_ROOT") and not os.environ.get("CARLA_PATH"):
        os.environ["CARLA_PATH"] = os.environ["CARLA_ROOT"]
    elif os.environ.get("CARLA_PATH") and not os.environ.get("CARLA_ROOT"):
        os.environ["CARLA_ROOT"] = os.environ["CARLA_PATH"]
    return env_path
