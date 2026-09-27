#!/usr/bin/env python3
"""Download and validate external model assets without adding them to Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "model-assets.json"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_model_root(value: str | None) -> Path:
    configured = value or os.getenv("VLADFUZZ_MODEL_HOME", "models")
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    return root.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(asset: dict[str, Any], model_root: Path) -> list[str]:
    destination = model_root / asset["destination"]
    errors: list[str] = []
    if asset["method"] == "huggingface_snapshot":
        if not destination.is_dir():
            return [f"missing directory: {destination}"]
        for relative in asset.get("required_files", []):
            required = destination / relative
            if not required.is_file():
                errors.append(f"missing file: {required}")
        return errors

    if not destination.is_file():
        return [f"missing file: {destination}"]
    expected_size = asset.get("size")
    if expected_size is not None and destination.stat().st_size != expected_size:
        errors.append(
            f"size mismatch: {destination} "
            f"(expected {expected_size}, got {destination.stat().st_size})"
        )
        return errors
    expected_hash = asset.get("sha256")
    if expected_hash and sha256(destination) != expected_hash:
        errors.append(f"SHA-256 mismatch: {destination}")
    return errors


def huggingface_api():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for downloads; install the backend "
            "environment before running this command"
        ) from error
    return hf_hub_download, snapshot_download


def download_asset(
    asset: dict[str, Any], model_root: Path, *, offline: bool
) -> None:
    destination = model_root / asset["destination"]
    method = asset["method"]
    if method == "manual":
        print(f"MANUAL  {asset['id']}")
        print(f"        download: {asset['source_url']}")
        print(f"        place as: {destination}")
        return

    hf_hub_download, snapshot_download = huggingface_api()
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None
    if method == "huggingface_snapshot":
        print(f"DOWNLOAD {asset['repo_id']} -> {destination}")
        snapshot_download(
            repo_id=asset["repo_id"],
            revision=asset["revision"],
            local_dir=str(destination),
            allow_patterns=asset.get("allow_patterns"),
            token=token,
            local_files_only=offline,
        )
        return

    if method == "huggingface_file":
        print(
            f"DOWNLOAD {asset['repo_id']}:{asset['source_file']} -> {destination}"
        )
        cached = hf_hub_download(
            repo_id=asset["repo_id"],
            filename=asset["source_file"],
            revision=asset["revision"],
            token=token,
            local_files_only=offline,
        )
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
        return

    raise ValueError(f"unsupported download method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("lmdrive", "simlingo", "bevdriver", "all"),
        default="all",
    )
    parser.add_argument("--model-root")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--accept-model-licenses", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.verify_only and not args.accept_model_licenses:
        print(
            "Review docs/MODEL_ASSETS.md and pass --accept-model-licenses "
            "before downloading third-party model assets.",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest()
    model_root = resolve_model_root(args.model_root)
    selected = [
        asset
        for asset in manifest["assets"]
        if args.backend == "all"
        or args.backend in asset.get("backends", [asset.get("backend")])
    ]

    if not args.verify_only:
        model_root.mkdir(parents=True, exist_ok=True)
        for asset in selected:
            if verify_asset(asset, model_root):
                download_asset(asset, model_root, offline=args.offline)
            else:
                print(f"READY   {asset['id']}")

    failures: list[str] = []
    for asset in selected:
        errors = verify_asset(asset, model_root)
        if errors:
            failures.extend(f"{asset['id']}: {error}" for error in errors)
        else:
            print(f"VERIFIED {asset['id']}")

    if failures:
        print("\nModel asset validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\nAll selected model assets are valid under {model_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
