from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from evaluation.rq1.prepare import sha256_file


ERROR_TYPES = {
    "wrong_maneuver",
    "missing_maneuver",
    "hallucinated_maneuver",
    "scene_mismatch",
    "target_marker_leakage",
    "ambiguous_or_non_navigation",
    "format_or_language_issue",
    "other",
}


class AnnotationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Annotation:
    case_id: str
    score: int
    error_type: str
    notes: str

    @property
    def error_types(self) -> Tuple[str, ...]:
        return tuple(item for item in self.error_type.split(",") if item)


@dataclass(frozen=True)
class MergeSummary:
    case_count: int
    disagreement_count: int
    score_agreement: float
    merged_path: Path
    adjudication_package: Path


def _read_mapping(round_root: Path) -> Dict[str, Dict[str, str]]:
    path = Path(round_root) / "private" / "mapping.csv"
    if not path.is_file():
        raise AnnotationValidationError(f"missing private mapping: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row.get("case_id", ""): row for row in rows}
    if "" in mapping or len(mapping) != len(rows):
        raise AnnotationValidationError("mapping contains empty or duplicate case IDs")
    return mapping


def parse_annotation(path: Path, case_id: str) -> Annotation:
    path = Path(path)
    if not path.is_file():
        raise AnnotationValidationError(f"{case_id}: missing annotation file")
    values: Dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise AnnotationValidationError(f"{case_id}: malformed annotation line {line_number}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in {"score", "error_type", "notes"}:
            raise AnnotationValidationError(f"{case_id}: unknown annotation field {key!r}")
        if key in values:
            raise AnnotationValidationError(f"{case_id}: duplicate annotation field {key!r}")
        values[key] = value.strip()
    if not values.get("score"):
        raise AnnotationValidationError(f"{case_id}: score is required")
    if values["score"] not in {"0", "1", "2"}:
        raise AnnotationValidationError(f"{case_id}: score must be one of 0, 1, or 2")
    score = int(values["score"])
    error_type = values.get("error_type", "")
    notes = values.get("notes", "")
    if score == 0:
        if not error_type:
            raise AnnotationValidationError(f"{case_id}: error_type is required for score 0")
        error_types = tuple(item.strip() for item in error_type.split(",") if item.strip())
        if len(error_types) != len(set(error_types)):
            raise AnnotationValidationError(f"{case_id}: duplicate error_type")
        unknown = [item for item in error_types if item not in ERROR_TYPES]
        if unknown:
            raise AnnotationValidationError(f"{case_id}: unknown error_type {unknown[0]!r}")
        if "other" in error_types and not notes:
            raise AnnotationValidationError(f"{case_id}: notes are required for error_type other")
        error_type = ",".join(error_types)
    elif error_type:
        raise AnnotationValidationError(f"{case_id}: error_type must be empty for score {score}")
    return Annotation(case_id, score, error_type, notes)


def validate_submission(round_root: Path, package_dir: Path) -> Dict[str, Annotation]:
    mapping = _read_mapping(Path(round_root))
    package_dir = Path(package_dir)
    cases_dir = package_dir / "cases"
    if not cases_dir.is_dir():
        raise AnnotationValidationError(f"missing cases directory: {cases_dir}")
    actual = {path.name for path in cases_dir.iterdir() if path.is_dir()}
    expected = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AnnotationValidationError(f"missing case directories: {', '.join(missing)}")
    if extra:
        raise AnnotationValidationError(f"unexpected case directories: {', '.join(extra)}")
    records: Dict[str, Annotation] = {}
    errors = []
    for case_id in sorted(expected):
        case_dir = cases_dir / case_id
        expected_hashes = {
            "instruction.txt": mapping[case_id].get("instruction_sha256", ""),
            "reference.txt": mapping[case_id].get("reference_sha256", ""),
        }
        image_files = [
            path for path in case_dir.iterdir()
            if path.is_file() and path.name.startswith("image.")
        ] if case_dir.is_dir() else []
        if len(image_files) != 1:
            errors.append(f"{case_id}: expected exactly one image file")
        elif mapping[case_id].get("image_sha256") and sha256_file(image_files[0]) != mapping[case_id]["image_sha256"]:
            errors.append(f"{case_id}: case evidence was modified (image)")
        for filename, expected_hash in expected_hashes.items():
            evidence = case_dir / filename
            if not evidence.is_file():
                errors.append(f"{case_id}: missing case evidence {filename}")
            elif expected_hash and sha256_file(evidence) != expected_hash:
                errors.append(f"{case_id}: case evidence was modified ({filename})")
        try:
            records[case_id] = parse_annotation(case_dir / "annotation.txt", case_id)
        except AnnotationValidationError as exc:
            errors.append(str(exc))
    if errors:
        raise AnnotationValidationError("invalid submission:\n" + "\n".join(errors))
    return records


def check_correctness_agreement(round_root: Path, annotator_paths: Mapping[str, Path]) -> Dict:
    if len(annotator_paths) != 2:
        raise AnnotationValidationError("exactly two annotator submissions are required")
    submissions = {
        annotator_id: validate_submission(round_root, path)
        for annotator_id, path in annotator_paths.items()
    }
    first_id, second_id = submissions
    conflicts = []
    for case_id in sorted(submissions[first_id]):
        first = submissions[first_id][case_id]
        second = submissions[second_id][case_id]
        if (first.score == 0) == (second.score == 0):
            continue
        conflicts.append({
            "case_id": case_id,
            "annotator_1": first_id,
            "score_1": first.score,
            "error_type_1": first.error_type,
            "notes_1": first.notes,
            "annotation_1": str(Path(annotator_paths[first_id]) / "cases" / case_id / "annotation.txt"),
            "annotator_2": second_id,
            "score_2": second.score,
            "error_type_2": second.error_type,
            "notes_2": second.notes,
            "annotation_2": str(Path(annotator_paths[second_id]) / "cases" / case_id / "annotation.txt"),
        })
    report_path = Path(round_root) / "reports" / "correctness_conflicts.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "annotator_1", "score_1", "error_type_1", "notes_1", "annotation_1",
        "annotator_2", "score_2", "error_type_2", "notes_2", "annotation_2",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(conflicts)
    case_count = len(submissions[first_id])
    return {
        "case_count": case_count,
        "conflict_count": len(conflicts),
        "conflict_case_ids": [row["case_id"] for row in conflicts],
        "correctness_agreement": (case_count - len(conflicts)) / case_count if case_count else None,
        "zero_score_counts": {
            name: sum(record.score == 0 for record in records.values())
            for name, records in submissions.items()
        },
        "report_path": str(report_path),
    }


def merge_submissions(round_root: Path, annotator_paths: Mapping[str, Path]) -> MergeSummary:
    round_root = Path(round_root)
    adjudication_root = round_root / "adjudication"
    if adjudication_root.exists():
        raise FileExistsError(
            f"adjudication output already exists: {adjudication_root}; archive or remove it explicitly before merging again"
        )
    if len(annotator_paths) != 2:
        raise AnnotationValidationError("exactly two annotator submissions are required")
    annotator_ids = list(annotator_paths)
    submissions = {
        annotator_id: validate_submission(round_root, path)
        for annotator_id, path in annotator_paths.items()
    }
    mapping = _read_mapping(round_root)
    first_id, second_id = annotator_ids
    rows = []
    disagreements = []
    score_matches = 0
    for case_id in sorted(mapping):
        first = submissions[first_id][case_id]
        second = submissions[second_id][case_id]
        score_agrees = first.score == second.score
        label_agrees = score_agrees and (first.score != 0 or set(first.error_types) == set(second.error_types))
        score_matches += int(score_agrees)
        status = "agreed" if label_agrees else "disagreement"
        if status == "disagreement":
            disagreements.append(case_id)
        rows.append({
            "case_id": case_id,
            "annotator_1": first_id,
            "score_1": first.score,
            "error_type_1": first.error_type,
            "notes_1": first.notes,
            "annotator_2": second_id,
            "score_2": second.score,
            "error_type_2": second.error_type,
            "notes_2": second.notes,
            "status": status,
        })

    private_dir = round_root / "private"
    merged_path = private_dir / "merged_annotations.csv"
    fields = list(rows[0]) if rows else [
        "case_id", "annotator_1", "score_1", "error_type_1", "notes_1",
        "annotator_2", "score_2", "error_type_2", "notes_2", "status",
    ]
    with merged_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    agreement = score_matches / len(rows) if rows else 0.0
    (private_dir / "merge_summary.json").write_text(
        json.dumps({
            "case_count": len(rows),
            "disagreement_count": len(disagreements),
            "score_agreement": agreement,
            "annotator_ids": annotator_ids,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    package = adjudication_root / "package"
    cases_dir = package / "cases"
    cases_dir.mkdir(parents=True)
    readme = Path(__file__).with_name("evaluator_readme.md").read_text(encoding="utf-8")
    (package / "README.md").write_text(readme, encoding="utf-8")
    source_cases = Path(annotator_paths[first_id]) / "cases"
    for case_id in disagreements:
        destination = cases_dir / case_id
        shutil.copytree(source_cases / case_id, destination)
        (destination / "annotation.txt").write_text("score: \nerror_type: \nnotes: \n", encoding="utf-8")
    with (adjudication_root / "private_disagreements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if row["status"] == "disagreement")

    return MergeSummary(len(rows), len(disagreements), agreement, merged_path, package)
