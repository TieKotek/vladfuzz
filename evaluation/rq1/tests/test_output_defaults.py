import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq1.__main__ import build_parser
from vladfuzz_runtime.experiment_commands import build_experiment_commands
from evaluation.rq2.summarize_results import _collect_local_runs


class OutputDefaultTests(unittest.TestCase):
    def test_rq1_defaults_to_five_seeds_per_static_scenario(self):
        args = build_parser().parse_args(["prepare"])
        self.assertEqual(args.output_root, Path("results/rq1/formal_round"))
        self.assertEqual(args.max_seeds_per_static_scenario, 5)
        self.assertEqual(args.instructions_per_seed, 1)
        self.assertFalse(hasattr(args, "annotators"))

    def test_scripts_and_workflows_use_rq_scoped_default_roots(self):
        expected_fragments = {
            "scripts/run_random.sh": 'OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq2/runs/random}"',
            "scripts/summarize_rq2.sh": 'EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-results/rq2/runs}"',
            "vladfuzz_workflows/local_fuzzer.py": 'default="results/rq2/runs"',
            "vladfuzz_workflows/baseline_testing.py": 'default="results/rq2/runs"',
            "vladfuzz_workflows/nsga_optimization.py": 'default="results/rq2/runs"',
            "vladfuzz_workflows/ads_baseline_testing.py": 'default="results/rq2/runs"',
            "tools/run_experiment_batch.py": 'default="results/rq2/runs"',
            "evaluation/rq2/summarize_results.py": 'default="results/rq2/runs"',
        }
        for filename, fragment in expected_fragments.items():
            with self.subTest(filename=filename):
                self.assertIn(fragment, Path(filename).read_text(encoding="utf-8"))

    def test_recursive_results_scan_ignores_generation_metadata_and_classifies_methods(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                ("vladfuzz", "vlad_fuzz_cmd1", "simlingo"),
                ("random", "random_cmd1", "simlingo"),
                ("instruction_counterfactual", "instruction_counterfactual", "simlingo"),
            ]
            for family, method, model in records:
                path = root / family / model / "run" / "metadata.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"configuration": {"method": method, "vla_model": model}}), encoding="utf-8")
            generation = root / "vladfuzz" / "simlingo" / "run" / "logs" / "gen_0" / "metadata.json"
            generation.parent.mkdir(parents=True)
            generation.write_text(json.dumps({"generation": 0}), encoding="utf-8")

            rows = _collect_local_runs(root)

            self.assertEqual(len(rows), 3)
            self.assertEqual({row["method_family"] for row in rows}, {
                "vladfuzz", "random", "instruction_counterfactual"
            })

    def test_legacy_experiment_runs_remains_ignored(self):
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("/results/", gitignore)
        self.assertIn("/experiment_runs/", gitignore)


if __name__ == "__main__":
    unittest.main()
