import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvaluationLayoutTests(unittest.TestCase):
    def test_rq_implementations_live_in_evaluation_packages(self) -> None:
        expected = (
            "evaluation/rq2/summarize_results.py",
            "evaluation/rq2/build_tables.py",
            "evaluation/rq2/build_paper_assets.py",
            "evaluation/rq2/analyze_diversity.py",
            "evaluation/rq3/summarize_results.py",
            "evaluation/rq4/summarize_results.py",
            "evaluation/rq4/plot_task_profiles.py",
        )
        removed = (
            "tools/summarize_rq2_results.py",
            "tools/build_rq2_tables.py",
            "tools/build_rq2_paper_assets.py",
            "tools/analyze_rq2_diversity.py",
            "tools/summarize_rq3_results.py",
            "tools/summarize_rq4_results.py",
            "tools/plot_rq4_task_profiles.py",
        )

        for relative in expected:
            self.assertTrue((ROOT / relative).is_file(), relative)
        for relative in removed:
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_rq2_tests_live_with_the_rq2_package(self) -> None:
        expected = (
            "test_build_rq2_tables.py",
            "test_rq2_diversity.py",
            "test_rq2_diversity_collection.py",
            "test_rq2_diversity_diagnostics.py",
            "test_rq2_paper_assets.py",
        )
        for name in expected:
            self.assertTrue((ROOT / "evaluation" / "rq2" / "tests" / name).is_file(), name)
            self.assertFalse((ROOT / "tests" / name).exists(), name)

    def test_run_scripts_use_rq_scoped_result_roots(self) -> None:
        markers = {
            "scripts/run_vladfuzz.sh": "results/rq2/runs/vladfuzz",
            "scripts/run_batch_vladfuzz.sh": "results/rq2/runs/vladfuzz",
            "scripts/run_drivefuzz.sh": "results/rq2/runs/drivefuzz",
            "scripts/run_batch_drivefuzz.sh": "results/rq2/runs/drivefuzz",
            "scripts/run_instruction_counterfactual.sh": "results/rq2/runs/instruction_counterfactual",
            "scripts/run_batch_instruction_counterfactual.sh": "results/rq2/runs/instruction_counterfactual",
            "scripts/run_random.sh": "results/rq2/runs/random",
            "scripts/run_batch_random.sh": "results/rq2/runs/random",
            "scripts/run_rq3.sh": "results/rq3/runs/$CONFIG",
            "scripts/run_batch_rq3.sh": "results/rq3/runs/$CONFIG",
        }
        for relative, marker in markers.items():
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(marker, source, relative)

    def test_analysis_defaults_use_rq_scoped_result_roots(self) -> None:
        markers = {
            "evaluation/rq2/summarize_results.py": "results/rq2/summary",
            "evaluation/rq2/build_tables.py": "results/rq2/tables",
            "evaluation/rq2/build_paper_assets.py": "results/rq2/paper",
            "evaluation/rq2/analyze_diversity.py": "results/rq2/diversity",
            "evaluation/rq3/summarize_results.py": "results/rq3/summary",
            "evaluation/rq4/summarize_results.py": "results/rq4/summary",
            "evaluation/rq4/plot_task_profiles.py": "results/rq4/summary",
        }
        for relative, marker in markers.items():
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(marker, source, relative)


if __name__ == "__main__":
    unittest.main()
