import ctypes
import os
import sys
from pathlib import Path


def preload_cuda11_libs() -> None:
    """Preload CUDA 11 user-space libs for LMDrive bitsandbytes compatibility."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    site_packages = os.path.join(conda_prefix, "lib", "python3.9", "site-packages")
    nvidia_base = os.path.join(site_packages, "nvidia")
    libs_to_load = [
        ("cuda_runtime", "libcudart.so.11.0"),
        ("cublas", "libcublas.so.11"),
        ("cusparse", "libcusparse.so.11"),
        ("cudnn", "libcudnn.so.8"),
    ]

    for subdir, libname in libs_to_load:
        lib_path = os.path.join(nvidia_base, subdir, "lib", libname)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception as exc:
                print(f"DEBUG: Failed to preload {libname}: {exc}")


def _append_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if path.exists() and resolved not in sys.path:
        sys.path.insert(0, resolved)


def bootstrap_carla_python_paths() -> None:
    """Add local CARLA PythonAPI paths from the configured CARLA release."""
    carla_root = os.environ.get("CARLA_ROOT") or os.environ.get("CARLA_PATH")
    if not carla_root:
        return

    python_api = Path(carla_root) / "PythonAPI"
    carla_python = python_api / "carla"

    _append_sys_path(python_api)
    _append_sys_path(carla_python)


def bootstrap_project_python_paths() -> None:
    """Add vendored dependency roots relative to this checkout."""
    project_root = Path(__file__).resolve().parent.parent
    _append_sys_path(project_root)
    _append_sys_path(project_root / "leaderboard")
    _append_sys_path(project_root / "scenario_runner")
