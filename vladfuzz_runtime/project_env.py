"""Load repository-local configuration without requiring shell profile edits."""

from __future__ import annotations

import os
import argparse
import re
import shlex
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
    except ImportError as error:
        raise RuntimeError(
            "Loading .env requires python-dotenv; install the selected backend "
            "environment from environments/*.yml."
        ) from error

    load_dotenv(dotenv_path=env_path, override=False)
    if os.environ.get("CARLA_ROOT") and not os.environ.get("CARLA_PATH"):
        os.environ["CARLA_PATH"] = os.environ["CARLA_ROOT"]
    elif os.environ.get("CARLA_PATH") and not os.environ.get("CARLA_ROOT"):
        os.environ["CARLA_ROOT"] = os.environ["CARLA_PATH"]
    return env_path


def shell_settings(path: Path) -> str:
    """Render quoted Bash assignments without overriding caller variables."""
    from dotenv import dotenv_values

    statements = []
    for key, value in dotenv_values(path).items():
        if value is None:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid configuration variable name: {key!r}")
        statements.append(
            f"if ! [[ -v {key} ]]; then {key}={shlex.quote(value)}; fi; export {key}"
        )
    return "\n".join(statements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", type=Path, required=True)
    args = parser.parse_args()
    print(shell_settings(args.shell))
