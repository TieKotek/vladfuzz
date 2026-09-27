import subprocess
import unittest
from pathlib import Path


class RQ1LayoutTests(unittest.TestCase):
    def test_framework_documents_and_compatibility_entry_exist_outside_paper(self):
        self.assertTrue(Path("evaluation/rq1/README.md").is_file())
        self.assertTrue(Path("evaluation/rq1/evaluator_readme.md").is_file())
        self.assertTrue(Path("tools/prepare_rq1_annotation_batches.py").is_file())
        self.assertFalse(Path("paper/rq1").exists())

    def test_cli_exposes_complete_workflow(self):
        completed = subprocess.run(
            ["python3", "-m", "evaluation.rq1", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for command in ("prepare", "validate", "merge", "adjudicate", "analyze"):
            self.assertIn(command, completed.stdout)

    def test_generated_rounds_are_ignored(self):
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("/results/", gitignore)
        self.assertIn("/experiment_runs/", gitignore)


if __name__ == "__main__":
    unittest.main()
