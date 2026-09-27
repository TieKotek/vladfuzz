#!/usr/bin/env python3
"""Archive one invalid VLAD-Fuzz attempt outside the valid result tree."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _safe(value):
    return value.replace("/", "_").replace(" ", "_").replace(":", "_")


def _unique(path):
    if not path.exists():
        return path
    suffix = 2
    while path.with_name(f"{path.name}-{suffix}").exists():
        suffix += 1
    return path.with_name(f"{path.name}-{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--run-dir-file")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    run_dirs = [Path(value).resolve() for value in args.run_dir]
    if args.run_dir_file:
        marker = Path(args.run_dir_file)
        if marker.exists():
            run_dirs.extend(
                Path(line.strip()).resolve()
                for line in marker.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    archive = _unique(
        results_root / "infrastructure_failures" / args.run_id
        / f"{_safe(args.manifest_id)}-attempt{args.attempt}"
    )
    archive.mkdir(parents=True)
    artifacts = []
    for index, run_dir in enumerate(dict.fromkeys(run_dirs), start=1):
        if not run_dir.is_dir():
            continue
        target = archive / ("run" if len(run_dirs) == 1 else f"run_{index}")
        shutil.move(str(run_dir), str(target))
        artifacts.append(str(target))

    status = {
        "status": "infrastructure_failed",
        "reason": args.reason,
        "exit_code": args.exit_code,
        "included_in_analysis": False,
        "run_id": args.run_id,
        "manifest_id": args.manifest_id,
        "attempt": args.attempt,
        "archived_at": datetime.now().isoformat(),
        "artifacts": artifacts,
    }
    (archive / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(archive)


if __name__ == "__main__":
    main()
