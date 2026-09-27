#!/usr/bin/env python3
"""Archive one failed DriveFuzz attempt outside the valid run tree."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _safe_manifest_id(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(":", "_")


def _unique_archive_dir(base: Path) -> Path:
    if not base.exists():
        return base
    suffix = 2
    while base.with_name(f"{base.name}-{suffix}").exists():
        suffix += 1
    return base.with_name(f"{base.name}-{suffix}")


def _absolute_matches(parent: Path, pattern: str) -> list[Path]:
    if not parent.exists():
        return []
    return sorted(parent.glob(pattern))


def archive_failure(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root).resolve()
    archive = _unique_archive_dir(
        results_root
        / "infrastructure_failures"
        / args.run_id
        / f"{_safe_manifest_id(args.manifest_id)}-attempt{args.attempt}"
    )
    archive.mkdir(parents=True)

    moved = []
    run_parent = results_root / "runs" / args.run_id
    for source in _absolute_matches(run_parent, f"{args.output_name}-cmd*"):
        target_dir = archive / "runs"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.move(str(source), str(target))
        moved.append(str(target))

    native_parent = results_root / "native_debug" / args.run_id
    for source in _absolute_matches(native_parent, f"{args.output_name}-cmd*_native.log"):
        target_dir = archive / "native_debug"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.move(str(source), str(target))
        moved.append(str(target))

    if args.carla_log:
        carla_log = Path(args.carla_log)
        if carla_log.exists():
            target = archive / "carla.log"
            shutil.move(str(carla_log), str(target))
            moved.append(str(target))

    status = {
        "status": "infrastructure_failed",
        "reason": args.reason,
        "exit_code": args.exit_code,
        "included_in_analysis": False,
        "run_id": args.run_id,
        "manifest_id": args.manifest_id,
        "attempt": args.attempt,
        "archived_at": datetime.now().isoformat(),
        "artifacts": moved,
    }
    (archive / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--carla-log")
    archive_failure(parser.parse_args())


if __name__ == "__main__":
    main()
