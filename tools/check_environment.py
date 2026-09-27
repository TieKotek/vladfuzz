#!/usr/bin/env python3
"""Check a VLAD-Fuzz checkout without importing heavyweight model code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vladfuzz_runtime.model_assets import resolve_model_root
from vladfuzz_runtime.model_registry import MODEL_SPECS


MANIFEST_PATH = PROJECT_ROOT / "configs" / "model-assets.json"
SUPPORTED_MODELS = tuple(MODEL_SPECS)

REPOSITORY_PATHS = (
    "scenario",
    "scenario_construction",
    "vladfuzz_runtime",
    "vladfuzz_workflows",
    "leaderboard",
    "scenario_runner",
    "baselines/drivefuzz/src/fuzzer.py",
    "artifact_inputs/static_scenarios",
    "artifact_inputs/rq1/test_cases",
    "artifact_inputs/formal/test_cases",
    "configs/experiment_manifest.jsonl",
    "configs/rq3_manifest.jsonl",
    "configs/model-assets.json",
)

CORE_IMPORTS = {
    "cv2": "opencv-python-headless",
    "deap": "deap",
    "filterpy": "filterpy",
    "hydra": "hydra-core",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "PIL": "pillow",
    "pygame": "pygame",
    "py_trees": "py-trees",
    "scipy": "scipy",
    "shapely": "shapely",
    "skfuzzy": "scikit-fuzzy",
    "torch": "torch",
    "transformers": "transformers",
}

MODEL_IMPORTS = {
    "lmdrive": {"accelerate": "accelerate", "timm": "timm"},
    "simlingo": {"pytorch_lightning": "pytorch-lightning", "timm": "timm"},
    "bevdriver": {"accelerate": "accelerate", "timm": "timm"},
}


@dataclass
class CheckResult:
    level: str
    label: str
    detail: str


def _path_result(label: str, path: Path) -> CheckResult:
    if path.exists():
        return CheckResult("OK", label, str(path))
    return CheckResult("ERROR", label, f"missing: {path}")


def check_repository() -> list[CheckResult]:
    results = [_path_result("repository root", PROJECT_ROOT / "README.md")]
    results.extend(
        _path_result(f"repository/{relative}", PROJECT_ROOT / relative)
        for relative in REPOSITORY_PATHS
    )
    if sys.version_info[:2] == (3, 9):
        results.append(CheckResult("OK", "Python", sys.version.split()[0]))
    else:
        results.append(
            CheckResult(
                "WARN",
                "Python",
                f"{sys.version.split()[0]} detected; backend environments use Python 3.9",
            )
        )
    return results


def check_carla() -> list[CheckResult]:
    root_value = os.environ.get("CARLA_ROOT") or os.environ.get("CARLA_PATH")
    if not root_value:
        return [
            CheckResult(
                "ERROR",
                "CARLA",
                "set CARLA_ROOT in the repository .env to a CARLA 0.9.15 installation",
            )
        ]

    root = Path(root_value).expanduser().resolve()
    results = [_path_result("CARLA root", root)]
    executable = Path(os.environ.get("CARLA_EXECUTABLE", root / "CarlaUE4.sh"))
    results.append(_path_result("CARLA executable", executable))
    results.append(
        _path_result(
            "CARLA navigation agents",
            root / "PythonAPI" / "carla" / "agents" / "navigation" / "global_route_planner.py",
        )
    )

    if importlib.util.find_spec("carla") is None:
        results.append(
            CheckResult(
                "ERROR",
                "CARLA Python API",
                "install carla==0.9.15 in the selected backend environment",
            )
        )
    else:
        try:
            import carla
        except Exception as error:
            results.append(CheckResult("ERROR", "CARLA Python API", repr(error)))
        else:
            results.append(CheckResult("OK", "CARLA Python API", str(Path(carla.__file__).resolve())))
    return results


def check_packages(models: Iterable[str]) -> list[CheckResult]:
    imports = dict(CORE_IMPORTS)
    for model in models:
        imports.update(MODEL_IMPORTS[model])

    results = []
    for module, package in sorted(imports.items()):
        if importlib.util.find_spec(module) is not None:
            results.append(CheckResult("OK", f"Python package/{package}", "importable"))
        else:
            results.append(
                CheckResult("ERROR", f"Python package/{package}", f"missing import '{module}'")
            )
    return results


def _selected_assets(models: Iterable[str]) -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    selected_models = set(models)
    return [
        asset
        for asset in manifest["assets"]
        if selected_models.intersection(asset.get("backends", [asset.get("backend")]))
    ]


def check_models(models: Iterable[str]) -> list[CheckResult]:
    models = tuple(models)
    results = []
    for model in models:
        spec = MODEL_SPECS[model]
        if spec.source_path.is_file():
            results.append(CheckResult("OK", f"backend/{model}", str(spec.source_path)))
        else:
            results.append(
                CheckResult(
                    "ERROR",
                    f"backend/{model}",
                    spec.source_note or f"missing: {spec.source_path}",
                )
            )

    model_root = resolve_model_root()
    for asset in _selected_assets(models):
        destination = model_root / asset["destination"]
        if asset["method"] == "huggingface_snapshot":
            missing = [name for name in asset.get("required_files", []) if not (destination / name).is_file()]
            if not destination.is_dir() or missing:
                detail = f"missing or incomplete: {destination}"
                if missing:
                    detail += f" ({', '.join(missing)})"
                results.append(CheckResult("ERROR", f"asset/{asset['id']}", detail))
            else:
                results.append(CheckResult("OK", f"asset/{asset['id']}", str(destination)))
        else:
            results.append(_path_result(f"asset/{asset['id']}", destination))
    return results


def check_gpu() -> list[CheckResult]:
    results = []
    if shutil.which("nvidia-smi") is None:
        results.append(CheckResult("ERROR", "NVIDIA runtime", "nvidia-smi not found"))
    else:
        command = [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        detail = completed.stdout.strip() or completed.stderr.strip()
        results.append(CheckResult("OK" if completed.returncode == 0 else "ERROR", "NVIDIA runtime", detail))
    if shutil.which("vulkaninfo") is None:
        results.append(CheckResult("WARN", "Vulkan", "vulkaninfo not found; install Vulkan tools and verify the NVIDIA ICD"))
    else:
        results.append(CheckResult("OK", "Vulkan", "vulkaninfo available"))
    try:
        import torch
    except ImportError:
        return results
    if torch.cuda.is_available():
        results.append(
            CheckResult(
                "OK",
                "PyTorch CUDA",
                f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}",
            )
        )
    else:
        results.append(
            CheckResult(
                "ERROR",
                "PyTorch CUDA",
                f"torch {torch.__version__} was built for CUDA {torch.version.cuda}, but CUDA initialization failed",
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a VLAD-Fuzz checkout before running CARLA experiments."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=SUPPORTED_MODELS,
        help="Backend to check; repeat for multiple backends. Defaults to all backends.",
    )
    parser.add_argument(
        "--repository-only",
        action="store_true",
        help="Check only files that belong in the source repository.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = tuple(args.model or SUPPORTED_MODELS)
    results = check_repository()
    if not args.repository_only:
        results.extend(check_carla())
        results.extend(check_packages(models))
        results.extend(check_models(models))
        results.extend(check_gpu())

    for result in results:
        print(f"[{result.level:<5}] {result.label}: {result.detail}")

    errors = sum(result.level == "ERROR" for result in results)
    warnings = sum(result.level == "WARN" for result in results)
    print(f"\nSummary: {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
