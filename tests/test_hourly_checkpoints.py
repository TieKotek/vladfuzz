import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vladfuzz_runtime.hourly_checkpoints import HourlyCheckpointRecorder


class HourlyCheckpointRecorderTests(unittest.TestCase):
    def test_writes_cumulative_hourly_checkpoints(self):
        with TemporaryDirectory() as tmp_dir:
            recorder = HourlyCheckpointRecorder(
                output_dir=tmp_dir,
                start_time=100.0,
                interval_seconds=3600.0,
                method="vlad_fuzz",
                model="simlingo",
                manifest_id="town/seed_1",
            )

            recorder.update(
                now=100.0 + 3600.0,
                executions=12,
                failures=3,
                extra={"generation": 1},
            )
            recorder.update(
                now=100.0 + 7200.0,
                executions=20,
                failures=5,
            )

            checkpoint_path = Path(tmp_dir) / "hourly_checkpoints.json"
            jsonl_path = Path(tmp_dir) / "hourly_checkpoints.jsonl"
            checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            jsonl_lines = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual([item["hour"] for item in checkpoints], [1, 2])
            self.assertEqual(checkpoints[0]["executions"], 12)
            self.assertEqual(checkpoints[0]["failures"], 3)
            self.assertEqual(checkpoints[0]["failure_rate"], 0.25)
            self.assertEqual(checkpoints[1]["executions"], 20)
            self.assertEqual(checkpoints[1]["failures"], 5)
            self.assertEqual(len(jsonl_lines), 2)

    def test_backfills_multiple_elapsed_hours_with_current_counts(self):
        with TemporaryDirectory() as tmp_dir:
            recorder = HourlyCheckpointRecorder(
                output_dir=tmp_dir,
                start_time=0.0,
                interval_seconds=3600.0,
            )

            recorder.update(now=3 * 3600.0 + 5.0, executions=7, failures=2)

            checkpoints = json.loads((Path(tmp_dir) / "hourly_checkpoints.json").read_text(encoding="utf-8"))
            self.assertEqual([item["hour"] for item in checkpoints], [1, 2, 3])
            self.assertTrue(all(item["executions"] == 7 for item in checkpoints))


if __name__ == "__main__":
    unittest.main()
