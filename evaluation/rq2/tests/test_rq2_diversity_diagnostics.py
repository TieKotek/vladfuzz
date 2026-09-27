import json
import unittest
from pathlib import Path

from evaluation.rq2.analyze_diversity import FailureCase, source_failure_counts


class RQ2DiversityDiagnosticsTests(unittest.TestCase):
    def test_source_failure_counts_are_json_serializable(self):
        scene = {}
        cases = [
            FailureCase("simlingo", "VLAD-Fuzz", "task", "a", scene, Path("a")),
            FailureCase("simlingo", "VLAD-Fuzz", "task", "b", scene, Path("b")),
            FailureCase("simlingo", "DriveFuzz", "task", "c", scene, Path("c")),
        ]

        counts = source_failure_counts(cases)

        self.assertEqual(
            counts,
            {"simlingo": {"DriveFuzz": 1, "VLAD-Fuzz": 2}},
        )
        json.dumps(counts)


if __name__ == "__main__":
    unittest.main()
