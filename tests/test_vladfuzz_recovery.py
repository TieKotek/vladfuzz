import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from vladfuzz_runtime.infrastructure import (
    CARLA_INFRASTRUCTURE_EXIT_CODE,
    CarlaInfrastructureError,
    is_carla_infrastructure_error,
)
from vladfuzz_workflows.local_fuzzer import LocalFuzzer
from vladfuzz_workflows.nsga_optimization import ScenarioOptimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CarlaInfrastructureClassificationTests(unittest.TestCase):
    def test_recognizes_carla_rpc_timeout(self):
        error = RuntimeError(
            "time-out of 20000ms while waiting for the simulator, "
            "make sure the simulator is ready and connected to localhost:2000"
        )
        self.assertTrue(is_carla_infrastructure_error(error))

    def test_does_not_classify_regular_evaluation_error_as_infrastructure(self):
        self.assertFalse(is_carla_infrastructure_error(ValueError("invalid scenario")))

    def test_local_fuzzer_propagates_carla_timeout(self):
        fuzzer = LocalFuzzer.__new__(LocalFuzzer)
        fuzzer.scenario_manager = Mock()
        fuzzer.scenario_manager.start_scenario.side_effect = RuntimeError(
            "time-out of 20000ms while waiting for the simulator"
        )
        fuzzer.success_distance = 5.0
        node = Mock(node_id="Root", instruction="Turn left.")

        with self.assertRaises(CarlaInfrastructureError):
            fuzzer._execute_scenario(node)

    def test_optimizer_propagates_infrastructure_failure_without_assigning_fitness(self):
        optimizer = ScenarioOptimizer.__new__(ScenarioOptimizer)
        optimizer.evaluation_cache = {}
        optimizer.cache_hits = 0
        optimizer.instruction = "Turn left."
        optimizer.failure_dir = None
        optimizer.enable_logging = True
        optimizer.fuzzer = Mock()
        optimizer.fuzzer.evaluate.side_effect = RuntimeError(
            "time-out of 20000ms while waiting for the simulator"
        )

        with self.assertRaises(CarlaInfrastructureError):
            optimizer.evaluate({"route_info": {}, "weather": "ClearNoon"})


class VladFuzzFailureArchiveTests(unittest.TestCase):
    def test_moves_failed_run_outside_normal_model_tree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "simlingo" / "vlad_fuzz_town01__seed_1_seed0_20260708_120000"
            (run_dir / "logs" / "gen_0").mkdir(parents=True)
            (run_dir / "logs" / "gen_0" / "metadata.json").write_text("{}", encoding="utf-8")

            result = subprocess.run([
                "python3", str(PROJECT_ROOT / "tools/archive_vladfuzz_failure.py"),
                "--results-root", str(root), "--run-dir", str(run_dir),
                "--run-id", "simlingo_test", "--manifest-id", "town01/seed_1",
                "--attempt", "1", "--reason", "carla_rpc_timeout",
                "--exit-code", str(CARLA_INFRASTRUCTURE_EXIT_CODE),
            ], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(run_dir.exists())
            archive = root / "infrastructure_failures" / "simlingo_test" / "town01_seed_1-attempt1"
            self.assertTrue((archive / "run" / "logs" / "gen_0" / "metadata.json").exists())
            status = json.loads((archive / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "infrastructure_failed")
            self.assertFalse(status["included_in_analysis"])


class VladFuzzBatchRecoverySourceTests(unittest.TestCase):
    def test_batch_archives_failed_attempt_before_retrying(self):
        script = (PROJECT_ROOT / "scripts/run_batch_vladfuzz.sh").read_text(encoding="utf-8")
        self.assertIn("tools/archive_vladfuzz_failure.py", script)
        self.assertIn(str(CARLA_INFRASTRUCTURE_EXIT_CODE), script)


if __name__ == "__main__":
    unittest.main()
