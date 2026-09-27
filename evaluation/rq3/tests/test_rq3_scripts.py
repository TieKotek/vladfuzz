import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RQ3ScriptTests(unittest.TestCase):
    def _capture_run(self, config):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        capture = root / "argv.json"
        fake_python = root / "fake-python"
        fake_python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env.update({
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHON_BIN": str(fake_python),
            "CAPTURE": str(capture),
            "CONFIG": config,
            "MANIFEST": str(root / "manifest.jsonl"),
            "MANIFEST_ID": "town01/seed_1",
            "TIME_BUDGET_MINUTES": "17",
            "OUTPUT_ROOT": str(root / "results"),
        })
        completed = subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts/run_rq3.sh")],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        argv = json.loads(capture.read_text(encoding="utf-8")) if capture.exists() else []
        return tmp, completed, argv

    def test_global_only_dispatches_nsga_without_language_mutation(self):
        tmp, completed, argv = self._capture_run("global_only")
        try:
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(argv[:2], ["-m", "vladfuzz_workflows.nsga_optimization"])
            self.assertEqual(argv[argv.index("--mutation-depth") + 1], "0")
            self.assertEqual(argv[argv.index("--method") + 1], "global_only")
            self.assertEqual(argv[argv.index("--instruction-source") + 1], "route_prior")
            self.assertEqual(argv[argv.index("--time-budget-minutes") + 1], "17")
        finally:
            tmp.cleanup()

    def test_local_only_dispatches_timed_fixed_scenario_workflow(self):
        tmp, completed, argv = self._capture_run("local_only")
        try:
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(argv[:2], ["-m", "vladfuzz_workflows.local_only_testing"])
            self.assertEqual(argv[argv.index("--mutation-depth") + 1], "2")
            self.assertEqual(argv[argv.index("--time-budget-minutes") + 1], "17")
        finally:
            tmp.cleanup()

    def test_random_scenario_local_dispatches_randomized_local_workflow(self):
        tmp, completed, argv = self._capture_run("random_scenario_local")
        try:
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                argv[:2],
                ["-m", "vladfuzz_workflows.nsga_optimization"],
            )
            self.assertEqual(argv[argv.index("--mutation-depth") + 1], "2")
            self.assertIn("--initial-population-until-budget", argv)
            self.assertEqual(argv[argv.index("--method") + 1], "random_scenario_local")
            self.assertEqual(argv[argv.index("--ngen") + 1], "0")
            self.assertEqual(argv[argv.index("--min-npc-count") + 1], "0")
            self.assertEqual(argv[argv.index("--max-npc-count") + 1], "3")
            self.assertEqual(argv[argv.index("--time-budget-minutes") + 1], "17")
        finally:
            tmp.cleanup()


    def test_batch_has_four_hour_default_and_managed_carla_recovery(self):
        script = (PROJECT_ROOT / "scripts/run_batch_rq3.sh").read_text(encoding="utf-8")
        self.assertIn('TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"', script)
        self.assertIn('CONFIG="${CONFIG:-global_only}"', script)
        self.assertIn("carla_start", script)
        self.assertIn("tools/archive_vladfuzz_failure.py", script)
        self.assertIn("scripts/run_rq3.sh", script)
        self.assertIn('OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq3/runs/$CONFIG}"', script)
        self.assertIn("global_only|local_only|random_scenario_local", script)


if __name__ == "__main__":
    unittest.main()
