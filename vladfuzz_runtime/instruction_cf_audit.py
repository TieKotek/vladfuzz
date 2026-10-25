"""Failure-attribution accounting for goal-conflicting Instruction-CF inputs."""

from dataclasses import dataclass
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple


AUDIT_CANDIDATES_FILENAME = "false_positive_audit_candidates.jsonl"
AUDIT_ANNOTATIONS_FILENAME = "false_positive_audit_annotations.jsonl"
VALID_FAILURE = "valid_failure"
TASK_MISMATCH_FALSE_POSITIVE = "task_mismatch_false_positive"
UNCLEAR = "unclear"
FINAL_LABELS = {VALID_FAILURE, TASK_MISMATCH_FALSE_POSITIVE}
ALLOWED_LABELS = FINAL_LABELS | {UNCLEAR}
SCOPED_RULE_VERSION = "departure-only-semantic-change-v1"


@lru_cache(maxsize=4)
def _scoped_records(directory: str):
    root = Path(directory).expanduser().resolve()
    manifest = json.loads((root / "audit_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("rule_version") != SCOPED_RULE_VERSION:
        raise ValueError("Unsupported Instruction-CF attribution rule version")
    records = _load_jsonl(root / "execution_attribution.jsonl")
    indexed = {}
    for row in records:
        run = str(Path(row["run_dir"]).resolve())
        index = int(row["execution_index"])
        key = (run, index)
        if key in indexed:
            raise ValueError(f"Duplicate scoped attribution record: {key}")
        indexed[key] = row
    hashes = {str(Path(row["path"]).resolve()): row["sha256"] for row in manifest["inputs"]}
    return indexed, hashes


def _resolve_scoped_accounting(run_dir, executions, raw, directory):
    indexed, hashes = _scoped_records(str(Path(directory).expanduser().resolve()))
    run_dir = run_dir.resolve()
    for filename in ("metadata.json", "execution_results.jsonl"):
        source = run_dir / filename
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashes.get(str(source)) != digest:
            raise ValueError(f"Scoped audit input hash mismatch: {source}")
    expected = {int(row["execution_index"]) for row in executions}
    available = {index for run, index in indexed if run == str(run_dir)}
    if len(expected) != len(executions) or available != expected:
        raise ValueError(f"Scoped audit does not cover exactly the run executions: {run_dir}")
    retained = []
    fp_count = 0
    for row in executions:
        audit = indexed[str(run_dir), int(row["execution_index"])]
        if audit.get("instruction") != row.get("instruction"):
            raise ValueError(f"Scoped audit instruction mismatch: {run_dir}")
        label = audit["label"]
        if not row.get("failed"):
            if label != "passed":
                raise ValueError("Passing execution has a failure attribution")
        elif label == "retained_failure":
            retained.append(row)
        elif label == TASK_MISMATCH_FALSE_POSITIVE:
            fp_count += int(row.get("failure_count", 1))
        else:
            raise ValueError(f"Unresolved scoped audit label: {label}")
    reportable = sum(int(row.get("failure_count", 1)) for row in retained)
    if reportable + fp_count != raw:
        raise ValueError(f"Scoped audit counts do not match raw failures: {run_dir}")
    return FailureAccounting(
        raw, reportable, fp_count,
        tuple(float(row.get("elapsed_seconds", 0.0)) for row in retained
              for _ in range(int(row.get("failure_count", 1)))),
        tuple(str(artifact) for row in retained for artifact in row.get("failure_artifacts", [])),
    )


def scoped_instruction_cf_runs():
    """Return an explicit audited run selection, or None for default discovery."""
    directory = os.environ.get("VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION")
    if not directory:
        return None
    indexed, _ = _scoped_records(str(Path(directory).expanduser().resolve()))
    return frozenset(run for run, _ in indexed)


class UnresolvedFailureAttributionError(ValueError):
    """Raised when goal-conflicting oracle positives have not been audited."""


@dataclass(frozen=True)
class FailureAccounting:
    raw_oracle_positives: int
    reportable_failures: int
    task_mismatch_false_positives: int
    reportable_elapsed_seconds: Tuple[float, ...]
    reportable_failure_artifacts: Tuple[str, ...]


def make_audit_id(manifest_id: str, scenario_index: int, variant_index: int) -> str:
    task = manifest_id.replace("/", "__")
    return f"{task}__scenario_{scenario_index:05d}__variant_{variant_index:02d}"


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _index_unique(rows: List[Dict], key: str, path: Path) -> Dict[str, Dict]:
    indexed = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"Missing {key} in {path}")
        if value in indexed:
            raise ValueError(f"Duplicate {key}={value} in {path}")
        indexed[value] = row
    return indexed


def resolve_failure_accounting(
    run_dir: Path,
    metadata: Dict,
    *,
    require_resolved: bool = True,
) -> FailureAccounting:
    """Return the failure count valid for matched-task RQ2 reporting.

    Legacy runs without attribution metadata retain their historical count. New
    four-family runs require a final annotation for every misleading oracle
    positive before they can contribute to paper-facing statistics.
    """

    raw = int(metadata.get("total_raw_oracle_positives", metadata.get("total_failures_detected", 0)))
    schema = metadata.get("failure_attribution_schema_version")
    scoped_directory = os.environ.get("VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION")
    is_instruction_cf = (
        schema is not None
        or metadata.get("configuration", {}).get("method") == "instruction_counterfactual"
    )
    if schema is None and (not scoped_directory or not is_instruction_cf):
        return FailureAccounting(raw, raw, 0, tuple(), tuple())

    execution_path = run_dir / "execution_results.jsonl"
    executions = _load_jsonl(execution_path)
    failed = [row for row in executions if bool(row.get("failed"))]
    counted_raw = sum(int(row.get("failure_count", 1)) for row in failed)
    if counted_raw != raw:
        raise ValueError(
            f"Instruction-CF raw failure count mismatch: metadata={raw}, records={counted_raw}, run={run_dir}"
        )

    if scoped_directory:
        return _resolve_scoped_accounting(run_dir, executions, raw, scoped_directory)

    reportable_rows = [
        row for row in failed if row.get("failure_attribution_status") == VALID_FAILURE
    ]
    pending_rows = [
        row
        for row in failed
        if row.get("failure_attribution_status") == "pending_manual_review"
    ]
    accounted_rows = len(reportable_rows) + len(pending_rows)
    if accounted_rows != len(failed):
        raise ValueError(f"Instruction-CF failed executions have invalid attribution states: {run_dir}")

    provisional_count = sum(
        int(row.get("failure_count", 1)) for row in reportable_rows
    )
    pending_count = sum(int(row.get("failure_count", 1)) for row in pending_rows)
    if provisional_count != int(metadata.get("total_provisionally_valid_failures", -1)):
        raise ValueError(f"Instruction-CF provisional failure count mismatch: {run_dir}")
    if pending_count != int(metadata.get("total_pending_failure_attribution", -1)):
        raise ValueError(f"Instruction-CF pending attribution count mismatch: {run_dir}")

    candidate_path = run_dir / AUDIT_CANDIDATES_FILENAME
    candidates = _index_unique(_load_jsonl(candidate_path), "audit_id", candidate_path)
    pending_ids = {str(row.get("audit_id") or "") for row in pending_rows}
    if not pending_ids or "" in pending_ids:
        if pending_rows:
            raise ValueError(f"Pending Instruction-CF records lack audit IDs: {run_dir}")
    if set(candidates) != pending_ids:
        raise ValueError(f"Instruction-CF audit candidates do not match pending executions: {run_dir}")

    annotation_path = run_dir / AUDIT_ANNOTATIONS_FILENAME
    annotations = _index_unique(_load_jsonl(annotation_path), "audit_id", annotation_path)
    invalid = {
        audit_id: row.get("label")
        for audit_id, row in annotations.items()
        if row.get("label") not in ALLOWED_LABELS
    }
    if invalid:
        raise ValueError(f"Invalid Instruction-CF audit labels in {annotation_path}: {invalid}")

    unresolved = sorted(
        audit_id
        for audit_id in pending_ids
        if audit_id not in annotations or annotations[audit_id].get("label") not in FINAL_LABELS
    )
    extra = sorted(set(annotations) - pending_ids)
    if extra:
        raise ValueError(f"Instruction-CF annotations reference unknown audit IDs: {extra}")
    if unresolved and require_resolved:
        raise UnresolvedFailureAttributionError(
            f"{len(unresolved)} misleading Instruction-CF oracle positives require final audit labels in "
            f"{annotation_path}"
        )

    row_by_id = {str(row["audit_id"]): row for row in pending_rows}
    valid_ids = {
        audit_id
        for audit_id, annotation in annotations.items()
        if annotation.get("label") == VALID_FAILURE
    }
    false_positive_ids = {
        audit_id
        for audit_id, annotation in annotations.items()
        if annotation.get("label") == TASK_MISMATCH_FALSE_POSITIVE
    }
    reportable_rows.extend(row_by_id[audit_id] for audit_id in valid_ids)

    reportable_count = sum(int(row.get("failure_count", 1)) for row in reportable_rows)
    false_positive_count = sum(
        int(row_by_id[audit_id].get("failure_count", 1))
        for audit_id in false_positive_ids
    )
    elapsed = tuple(
        float(row.get("elapsed_seconds", 0.0))
        for row in reportable_rows
        for _ in range(int(row.get("failure_count", 1)))
    )
    artifacts = tuple(
        str(artifact)
        for row in reportable_rows
        for artifact in row.get("failure_artifacts", [])
    )
    return FailureAccounting(
        raw,
        reportable_count,
        false_positive_count,
        elapsed,
        artifacts,
    )


def annotation_template(run_dir: Path) -> List[Dict]:
    candidate_path = run_dir / AUDIT_CANDIDATES_FILENAME
    candidates = _load_jsonl(candidate_path)
    return [
        {
            "audit_id": candidate["audit_id"],
            "label": UNCLEAR,
            "reviewer": "",
            "notes": "",
        }
        for candidate in candidates
    ]
