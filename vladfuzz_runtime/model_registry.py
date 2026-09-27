from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Optional, Type

from .env_patches import (
    bootstrap_carla_python_paths,
    bootstrap_project_python_paths,
    preload_cuda11_libs,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    agent_module: str
    agent_class: str
    config_path: Path
    hydra_config_path: Optional[Path]
    source_path: Path
    source_note: Optional[str] = None
    needs_cuda11_preload: bool = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_SPECS = {
    "lmdrive": ModelSpec(
        name="lmdrive",
        agent_module="backends.lmdrive.lmdrive_team_code.agent_lmdrive",
        agent_class="LMDriveAgent",
        config_path=PROJECT_ROOT / "backends" / "lmdrive" / "lmdrive_team_code",
        hydra_config_path=None,
        source_path=PROJECT_ROOT / "backends" / "lmdrive" / "lmdrive_team_code" / "agent_lmdrive.py",
        needs_cuda11_preload=True,
    ),
    "simlingo": ModelSpec(
        name="simlingo",
        agent_module="backends.simlingo.team_code.agent_simlingo",
        agent_class="LingoAgent",
        config_path=PROJECT_ROOT / "backends" / "simlingo",
        hydra_config_path=PROJECT_ROOT / "backends" / "simlingo" / "hydra_config.yaml",
        source_path=PROJECT_ROOT / "backends" / "simlingo" / "team_code" / "agent_simlingo.py",
        needs_cuda11_preload=False,
    ),
    "bevdriver": ModelSpec(
        name="bevdriver",
        agent_module="backends.bevdriver.team_code.agent_bevdriver",
        agent_class="BEVDriverAgent",
        config_path=PROJECT_ROOT / "backends" / "bevdriver" / "team_code",
        hydra_config_path=None,
        source_path=PROJECT_ROOT / "backends" / "bevdriver" / "team_code" / "agent_bevdriver.py",
        needs_cuda11_preload=False,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of: {supported}") from exc


def bootstrap_model_environment(model_name: str) -> ModelSpec:
    spec = get_model_spec(model_name)
    if not spec.source_path.is_file():
        detail = spec.source_note or "Install the backend source before running it."
        raise RuntimeError(
            f"Backend source for '{model_name}' is unavailable at {spec.source_path}. {detail}"
        )
    bootstrap_project_python_paths()
    bootstrap_carla_python_paths()
    if spec.needs_cuda11_preload:
        preload_cuda11_libs()
    return spec


def resolve_agent_class(model_name: str) -> Type:
    spec = bootstrap_model_environment(model_name)
    module = import_module(spec.agent_module)
    return getattr(module, spec.agent_class)
