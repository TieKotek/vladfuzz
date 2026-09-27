import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq1.reporting import analyze_round, finalize_adjudication, resolve_final_annotations
from evaluation.rq1.statistics import cluster_bootstrap_risk_difference, exact_mcnemar, linear_weighted_kappa


class StatisticsTests(unittest.TestCase):
    def test_weighted_kappa_is_one_for_identical_scores(self):
        self.assertEqual(linear_weighted_kappa([0, 1, 2], [0, 1, 2]), 1.0)

    def test_exact_mcnemar_uses_two_sided_binomial_probability(self):
        result = exact_mcnemar(prior_only=3, no_prior_only=0)
        self.assertAlmostEqual(result, 0.25)

    def test_cluster_bootstrap_reports_paired_risk_difference(self):
        pairs = [
            {"scenario_name": "a", "route_prior": True, "no_prior": False},
            {"scenario_name": "a", "route_prior": True, "no_prior": True},
            {"scenario_name": "b", "route_prior": False, "no_prior": False},
        ]
        estimate, low, high = cluster_bootstrap_risk_difference(pairs, iterations=200, random_seed=3)
        self.assertAlmostEqual(estimate, 1 / 3)
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)


class ReportingTests(unittest.TestCase):
    def write_csv(self, path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def make_round(self, root: Path) -> Path:
        round_root = root / "round"
        mapping = [
            {"case_id": "case_1", "pair_id": "town_a/seed_1/slot_1", "scenario_name": "town_a", "seed_name": "seed_1", "mode": "route_prior"},
            {"case_id": "case_2", "pair_id": "town_a/seed_1/slot_1", "scenario_name": "town_a", "seed_name": "seed_1", "mode": "no_prior"},
            {"case_id": "case_3", "pair_id": "town_b/seed_1/slot_1", "scenario_name": "town_b", "seed_name": "seed_1", "mode": "route_prior"},
            {"case_id": "case_4", "pair_id": "town_b/seed_1/slot_1", "scenario_name": "town_b", "seed_name": "seed_1", "mode": "no_prior"},
        ]
        self.write_csv(round_root / "private" / "mapping.csv", mapping)
        merged = [
            {"case_id": "case_1", "annotator_1": "a", "score_1": "2", "error_type_1": "", "notes_1": "", "annotator_2": "b", "score_2": "2", "error_type_2": "", "notes_2": "", "status": "agreed"},
            {"case_id": "case_2", "annotator_1": "a", "score_1": "0", "error_type_1": "wrong_maneuver", "notes_1": "", "annotator_2": "b", "score_2": "1", "error_type_2": "", "notes_2": "", "status": "disagreement"},
            {"case_id": "case_3", "annotator_1": "a", "score_1": "1", "error_type_1": "", "notes_1": "", "annotator_2": "b", "score_2": "1", "error_type_2": "", "notes_2": "", "status": "agreed"},
            {"case_id": "case_4", "annotator_1": "a", "score_1": "0", "error_type_1": "scene_mismatch,wrong_maneuver", "notes_1": "", "annotator_2": "b", "score_2": "0", "error_type_2": "scene_mismatch,wrong_maneuver", "notes_2": "", "status": "agreed"},
        ]
        self.write_csv(round_root / "private" / "merged_annotations.csv", merged)
        adjudication = round_root / "adjudication" / "package" / "cases" / "case_2"
        adjudication.mkdir(parents=True)
        (adjudication / "annotation.txt").write_text("score: 0\nerror_type: wrong_maneuver\nnotes: final\n", encoding="utf-8")
        return round_root

    def test_resolve_final_annotations_uses_adjudication_only_for_disagreements(self):
        with TemporaryDirectory() as tmp:
            round_root = self.make_round(Path(tmp))
            final = resolve_final_annotations(round_root)
            self.assertEqual(final["case_1"].score, 2)
            self.assertEqual(final["case_2"].score, 0)
            self.assertEqual(final["case_2"].notes, "final")

    def test_finalize_adjudication_persists_final_labels(self):
        with TemporaryDirectory() as tmp:
            round_root = self.make_round(Path(tmp))
            output = finalize_adjudication(round_root)
            self.assertEqual(output, round_root / "private" / "final_annotations.csv")
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["case_id"] for row in rows}, {"case_1", "case_2", "case_3", "case_4"})

    def test_analyze_round_writes_all_publication_report_formats(self):
        with TemporaryDirectory() as tmp:
            round_root = self.make_round(Path(tmp))
            report = analyze_round(round_root, bootstrap_iterations=200, random_seed=4)
            self.assertEqual(report["methods"]["route_prior"]["correct"], 2)
            self.assertEqual(report["methods"]["route_prior"]["high_quality"], 1)
            self.assertEqual(report["methods"]["no_prior"]["correct"], 0)
            self.assertEqual(report["methods"]["no_prior"]["errors"], 2)
            self.assertEqual(report["agreement"]["raw_score_agreement"], 0.75)
            self.assertEqual(report["error_types"]["no_prior"]["scene_mismatch"], 1)
            self.assertEqual(report["error_types"]["no_prior"]["wrong_maneuver"], 2)
            for filename in (
                "summary.json", "method_summary.csv", "error_types.csv",
                "paired_tests.csv", "summary.md", "summary.tex", "final_annotations.csv",
            ):
                self.assertTrue((round_root / "reports" / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
