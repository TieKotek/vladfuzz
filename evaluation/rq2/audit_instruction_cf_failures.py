#!/usr/bin/env python3
"""Initialize or validate manual attribution for misleading Instruction-CF failures."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vladfuzz_runtime.instruction_cf_audit import (
    AUDIT_ANNOTATIONS_FILENAME,
    annotation_template,
    resolve_failure_accounting,
)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or validate false-positive attribution for one Instruction-CF run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Create an annotation template with all labels set to unclear.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    annotation_path = run_dir / AUDIT_ANNOTATIONS_FILENAME

    if args.initialize:
        if annotation_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing annotations: {annotation_path}")
        rows = annotation_template(run_dir)
        _write_jsonl(annotation_path, rows)
        print(f"Wrote {len(rows)} annotations to {annotation_path}")
        return 0

    accounting = resolve_failure_accounting(run_dir, metadata, require_resolved=True)
    print(
        json.dumps(
            {
                "raw_oracle_positives": accounting.raw_oracle_positives,
                "reportable_failures": accounting.reportable_failures,
                "task_mismatch_false_positives": accounting.task_mismatch_false_positives,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
