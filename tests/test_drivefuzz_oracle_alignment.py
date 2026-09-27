import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVEFUZZ_SRC = PROJECT_ROOT / "baselines" / "drivefuzz" / "src"


class DriveFuzzOracleAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (PROJECT_ROOT / "scripts/run_drivefuzz.sh").read_text(encoding="utf-8")
        cls.fuzzer = (DRIVEFUZZ_SRC / "fuzzer.py").read_text(encoding="utf-8")
        cls.executor = (DRIVEFUZZ_SRC / "executor.py").read_text(encoding="utf-8")
        cls.config = (DRIVEFUZZ_SRC / "config.py").read_text(encoding="utf-8")

    def test_script_uses_harmonized_oracle_defaults(self):
        self.assertIn('MAX_SIMULATION_FRAMES="${MAX_SIMULATION_FRAMES:-500}"', self.script)
        self.assertIn('ENABLE_RED_CHECK="${ENABLE_RED_CHECK:-0}"', self.script)
        self.assertIn('LOCK_TRAFFIC_LIGHTS_GREEN="${LOCK_TRAFFIC_LIGHTS_GREEN:-1}"', self.script)
        self.assertIn('NO_STUCK_CHECK="${NO_STUCK_CHECK:-1}"', self.script)
        self.assertIn('EXTRA_ARGS+=(--no-stuck-check)', self.script)
        self.assertIn('--max-simulation-frames "$MAX_SIMULATION_FRAMES"', self.script)

    def test_fuzzer_exposes_500_frame_limit(self):
        self.assertIn('self.max_simulation_frames = None', self.config)
        self.assertIn('conf.max_simulation_frames = args.max_simulation_frames', self.fuzzer)
        self.assertRegex(
            self.fuzzer,
            r'add_argument\("--max-simulation-frames", type=int, default=None',
        )

    def test_executor_marks_timeout_at_frame_limit(self):
        self.assertIn('state.num_frames >= conf.max_simulation_frames', self.executor)
        self.assertIn('state.other_error = "timeout"', self.executor)

    def test_lane_invasion_is_recorded_without_early_termination(self):
        lane_block = re.search(
            r'# Check lane violation(?P<body>.*?)# Check traffic light violation',
            self.executor,
            re.DOTALL,
        )
        self.assertIsNotNone(lane_block)
        self.assertIn('conf.agent_type != c.VLA', lane_block.group('body'))

    def test_timeout_is_normalized_as_timeout_not_other(self):
        summary = (PROJECT_ROOT / "evaluation/rq2/summarize_results.py").read_text(encoding="utf-8")
        self.assertIn('events.get("other") == "timeout"', summary)


if __name__ == "__main__":
    unittest.main()
