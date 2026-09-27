import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVEFUZZ_SRC = PROJECT_ROOT / "baselines" / "drivefuzz" / "src"


class DriveFuzzRecoverySourceTests(unittest.TestCase):
    def test_sensor_failure_uses_dedicated_infrastructure_exit_code(self):
        executor = (DRIVEFUZZ_SRC / "executor.py").read_text(encoding="utf-8")
        fuzzer = (DRIVEFUZZ_SRC / "fuzzer.py").read_text(encoding="utf-8")
        fuzz_utils = (DRIVEFUZZ_SRC / "fuzz_utils.py").read_text(encoding="utf-8")

        self.assertIn("SENSOR_FAILURE_EXIT_CODE = 87", executor)
        self.assertIn("_is_sensor_data_failure", executor)
        self.assertIn("executor.SENSOR_FAILURE_EXIT_CODE", fuzzer)
        self.assertIn("executor.SENSOR_FAILURE_EXIT_CODE", fuzz_utils)
        self.assertIn("return ret", fuzz_utils)

    def test_carla_rpc_timeout_uses_infrastructure_exit_code(self):
        executor = (DRIVEFUZZ_SRC / "executor.py").read_text(encoding="utf-8")

        self.assertIn("is_carla_infrastructure_error", executor)
        self.assertIn("CARLA_INFRASTRUCTURE_EXIT_CODE", executor)

    def test_batch_archives_failed_attempts_and_retries_after_cooldown(self):
        script = (PROJECT_ROOT / "scripts/run_batch_drivefuzz.sh").read_text(encoding="utf-8")

        self.assertIn("tools/archive_drivefuzz_failure.py", script)
        self.assertIn("CARLA_INFRA_RESTART_COOLDOWN_SECONDS", script)
        self.assertIn("87", script)

    def test_carla_startup_timeout_allows_slow_restart(self):
        manager = (PROJECT_ROOT / "scripts/carla_manager.sh").read_text(encoding="utf-8")

        self.assertIn('CARLA_STARTUP_TIMEOUT_SECONDS:-300', manager)


    def test_debug_artifacts_are_ephemeral_by_default(self):
        runner = (PROJECT_ROOT / "scripts/run_drivefuzz.sh").read_text(encoding="utf-8")
        batch = (PROJECT_ROOT / "scripts/run_batch_drivefuzz.sh").read_text(encoding="utf-8")
        manager = (PROJECT_ROOT / "scripts/carla_manager.sh").read_text(encoding="utf-8")

        self.assertIn("KEEP_DEBUG_ARTIFACTS=\"${KEEP_DEBUG_ARTIFACTS:-0}\"", runner)
        self.assertIn("KEEP_DEBUG_ARTIFACTS=\"${KEEP_DEBUG_ARTIFACTS:-0}\"", batch)
        self.assertIn("/tmp/vladfuzz-drivefuzz", runner)
        self.assertIn("/tmp/vladfuzz-carla", manager)
        self.assertIn("KEEP_DEBUG_ARTIFACTS=\"$KEEP_DEBUG_ARTIFACTS\"", batch)


class DriveFuzzFailureArchiveTests(unittest.TestCase):
    def test_archives_outputs_logs_and_status_without_changing_valid_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_id = "bevdriver_test"
            output_name = "out-drivefuzz-bevdriver-town03_seed_2"
            output = root / "runs" / run_id / f"{output_name}-cmd1"
            native = root / "native_debug" / run_id / f"{output_name}-cmd1_native.log"
            carla = root / "carla.log"
            output.mkdir(parents=True)
            (output / "queue").mkdir()
            (output / "queue" / "case.json").write_text("{}", encoding="utf-8")
            native.parent.mkdir(parents=True)
            native.write_text("native failure", encoding="utf-8")
            carla.write_text("carla failure", encoding="utf-8")

            result = subprocess.run(
                [
                    "python3",
                    str(PROJECT_ROOT / "tools/archive_drivefuzz_failure.py"),
                    "--results-root",
                    str(root),
                    "--run-id",
                    run_id,
                    "--output-name",
                    output_name,
                    "--manifest-id",
                    "town03/seed_2",
                    "--attempt",
                    "1",
                    "--reason",
                    "sensor_received_no_data",
                    "--exit-code",
                    "87",
                    "--carla-log",
                    str(carla),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(native.exists())
            archive = root / "infrastructure_failures" / run_id / "town03_seed_2-attempt1"
            self.assertTrue((archive / "runs" / f"{output_name}-cmd1" / "queue" / "case.json").exists())
            self.assertTrue((archive / "native_debug" / native.name).exists())
            self.assertTrue((archive / "carla.log").exists())
            status = json.loads((archive / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "infrastructure_failed")
            self.assertEqual(status["reason"], "sensor_received_no_data")
            self.assertEqual(status["exit_code"], 87)
            self.assertFalse(status["included_in_analysis"])


if __name__ == "__main__":
    unittest.main()
