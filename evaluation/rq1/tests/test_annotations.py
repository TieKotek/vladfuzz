import csv
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq1.annotations import (
    AnnotationValidationError,
    check_correctness_agreement,
    merge_submissions,
    parse_annotation,
    validate_submission,
)
from evaluation.rq1.prepare import prepare_round


class AnnotationTests(unittest.TestCase):
    def write_annotation(self, path: Path, score: str, error_type: str = "", notes: str = "") -> None:
        path.write_text(f"score: {score}\nerror_type: {error_type}\nnotes: {notes}\n", encoding="utf-8")

    def make_round(self, root: Path) -> Path:
        seeds = root / "test_cases"
        seed = seeds / "town_a" / "seed_1"
        seed.mkdir(parents=True)
        (seed / "ego_camera_view.png").write_bytes(b"ego")
        (seed / "dynamic_scenario.json").write_text(
            json.dumps({"route_info": {"route_description": ["Turn left"], "basic_instruction": "Turn left."}}),
            encoding="utf-8",
        )
        for filename, mode, command in (
            ("command_prior.json", "route_prior", "Turn left at the junction."),
            ("command_no_prior.json", "no_prior", "Turn left."),
        ):
            (seed / filename).write_text(json.dumps({"commands": [command], "mode": mode}), encoding="utf-8")
        round_root = root / "round"
        prepare_round(seeds_root=seeds, output_root=round_root)
        submissions = round_root / "submissions"
        shutil.copytree(round_root / "package", submissions / "a")
        shutil.copytree(round_root / "package", submissions / "b")
        return round_root

    def test_correctness_check_flags_only_zero_nonzero_conflicts(self):
        with TemporaryDirectory() as tmp:
            round_root = self.make_round(Path(tmp))
            packages = {name: round_root / "submissions" / name for name in ("a", "b")}
            case_ids = sorted(path.name for path in (packages["a"] / "cases").iterdir())
            for first in range(3):
                for second in range(3):
                    with self.subTest(first=first, second=second):
                        for name, score in (("a", first), ("b", second)):
                            self.write_annotation(
                                packages[name] / "cases" / case_ids[0] / "annotation.txt",
                                str(score), "wrong_maneuver" if score == 0 else "", "Review note",
                            )
                            self.write_annotation(packages[name] / "cases" / case_ids[1] / "annotation.txt", "1" if name == "a" else "2")
                        originals = {path: path.read_bytes() for package in packages.values() for path in package.rglob("annotation.txt")}
                        summary = check_correctness_agreement(round_root, packages)
                        expected = int((first == 0) != (second == 0))
                        self.assertEqual(summary["conflict_count"], expected)
                        self.assertEqual(summary["case_count"], 2)
                        self.assertEqual(summary["correctness_agreement"], (2 - expected) / 2)
                        with Path(summary["report_path"]).open(newline="", encoding="utf-8") as handle:
                            rows = list(csv.DictReader(handle))
                        self.assertEqual(len(rows), expected)
                        if expected:
                            self.assertEqual(rows[0]["case_id"], case_ids[0])
                            self.assertEqual(rows[0]["score_1"], str(first))
                            self.assertEqual(rows[0]["score_2"], str(second))
                            self.assertEqual(rows[0]["notes_1"], "Review note")
                            self.assertTrue(Path(rows[0]["annotation_1"]).is_file())
                        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
                        self.assertFalse((round_root / "adjudication").exists())

    def test_correctness_check_validates_before_writing_report(self):
        with TemporaryDirectory() as tmp:
            round_root = self.make_round(Path(tmp))
            packages = {name: round_root / "submissions" / name for name in ("a", "b")}
            with self.assertRaises(AnnotationValidationError):
                check_correctness_agreement(round_root, packages)
            self.assertFalse((round_root / "reports" / "correctness_conflicts.csv").exists())

    def test_correctness_check_requires_two_submissions(self):
        with self.assertRaisesRegex(AnnotationValidationError, "exactly two"):
            check_correctness_agreement(Path("unused"), {})

    def test_parse_annotation_accepts_valid_scores(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.txt"
            self.write_annotation(path, "0", "wrong_maneuver", "Wrong direction: left expected.")
            parsed = parse_annotation(path, "case_1")
            self.assertEqual(parsed.score, 0)
            self.assertEqual(parsed.error_type, "wrong_maneuver")
            self.assertIn("left expected", parsed.notes)

    def test_parse_annotation_accepts_multiple_error_types(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.txt"
            self.write_annotation(path, "0", "wrong_maneuver, scene_mismatch")

            parsed = parse_annotation(path, "case_1")

            self.assertEqual(parsed.error_type, "wrong_maneuver,scene_mismatch")
            self.assertEqual(parsed.error_types, ("wrong_maneuver", "scene_mismatch"))

    def test_parse_annotation_rejects_invalid_semantics(self):
        invalid = [
            ("", "", "", "score is required"),
            ("3", "", "", "score must be one of"),
            ("0", "", "", "error_type is required"),
            ("1", "wrong_maneuver", "", "error_type must be empty"),
            ("0", "unknown", "", "unknown error_type"),
            ("0", "wrong_maneuver, unknown", "", "unknown error_type"),
            ("0", "wrong_maneuver, wrong_maneuver", "", "duplicate error_type"),
            ("0", "other", "", "notes are required"),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.txt"
            for score, error_type, notes, message in invalid:
                with self.subTest(score=score, error_type=error_type):
                    self.write_annotation(path, score, error_type, notes)
                    with self.assertRaisesRegex(AnnotationValidationError, message):
                        parse_annotation(path, "case_1")

    def test_validate_submission_requires_exact_case_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_root = self.make_round(root)
            package = round_root / "submissions" / "a"
            cases = sorted((package / "cases").iterdir())
            self.write_annotation(cases[0] / "annotation.txt", "1")
            self.write_annotation(cases[1] / "annotation.txt", "2")
            records = validate_submission(round_root, package)
            self.assertEqual(len(records), 2)
            (cases[1] / "annotation.txt").unlink()
            with self.assertRaisesRegex(AnnotationValidationError, "missing annotation"):
                validate_submission(round_root, package)

    def test_validate_submission_rejects_modified_case_evidence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_root = self.make_round(root)
            package = round_root / "submissions" / "a"
            for case_dir in (package / "cases").iterdir():
                self.write_annotation(case_dir / "annotation.txt", "1")
            first_case = sorted((package / "cases").iterdir())[0]
            (first_case / "instruction.txt").write_text("Modified instruction.\n", encoding="utf-8")
            with self.assertRaisesRegex(AnnotationValidationError, "case evidence was modified"):
                validate_submission(round_root, package)

    def test_merge_builds_blind_adjudication_package_for_disagreements(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_root = self.make_round(root)
            packages = {name: round_root / "submissions" / name for name in ("a", "b")}
            case_ids = sorted(path.name for path in (packages["a"] / "cases").iterdir())
            self.write_annotation(packages["a"] / "cases" / case_ids[0] / "annotation.txt", "2")
            self.write_annotation(packages["b"] / "cases" / case_ids[0] / "annotation.txt", "2")
            self.write_annotation(packages["a"] / "cases" / case_ids[1] / "annotation.txt", "0", "wrong_maneuver")
            self.write_annotation(packages["b"] / "cases" / case_ids[1] / "annotation.txt", "1")

            summary = merge_submissions(round_root, packages)

            self.assertEqual(summary.case_count, 2)
            self.assertEqual(summary.disagreement_count, 1)
            self.assertEqual(summary.score_agreement, 0.5)
            adjudication_cases = list((round_root / "adjudication" / "package" / "cases").iterdir())
            self.assertEqual([path.name for path in adjudication_cases], [case_ids[1]])
            self.assertEqual((adjudication_cases[0] / "annotation.txt").read_text(encoding="utf-8"), "score: \nerror_type: \nnotes: \n")
            package_text = "\n".join(path.read_text(encoding="utf-8") for path in (round_root / "adjudication" / "package").rglob("*.md"))
            self.assertNotIn("route_prior", package_text)
            self.assertNotIn("no_prior", package_text)
            with (round_root / "private" / "merged_annotations.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["status"] for row in rows}, {"agreed", "disagreement"})

    def test_merge_refuses_to_overwrite_existing_adjudication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_root = self.make_round(root)
            packages = {name: round_root / "submissions" / name for name in ("a", "b")}
            for package in packages.values():
                for case_dir in (package / "cases").iterdir():
                    self.write_annotation(case_dir / "annotation.txt", "1")
            merge_submissions(round_root, packages)
            with self.assertRaisesRegex(FileExistsError, "adjudication output already exists"):
                merge_submissions(round_root, packages)



if __name__ == "__main__":
    unittest.main()
