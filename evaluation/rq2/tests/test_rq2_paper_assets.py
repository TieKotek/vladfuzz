import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq2.build_paper_assets import drive_counts, hourly_failures


class RQ2PaperAssetsTests(unittest.TestCase):
    def test_excludes_spawn_attempts_but_preserves_failed_executions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'queue').mkdir()
            (root / 'errors').mkdir()
            for name, frames in (('1_1_1_1000.0.json', 0), ('1_1_2_1010.0.json', 500),
                                 ('1_1_3_1020.0.json', 25)):
                payload = dict(num_frames=frames, events={}, vehicle_states={'speed': []})
                (root / 'queue' / name).write_text(json.dumps(payload))
                if frames == 25:
                    (root / 'errors' / name).write_text(json.dumps(payload))
            executions, failures, excluded, timestamps = drive_counts(root)
            self.assertEqual((executions, failures, excluded), (2, 1, 1))
            self.assertEqual(timestamps, [1020.0])

    def test_rejects_zero_frame_behavior_instead_of_silently_discarding_it(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'queue').mkdir()
            (root / 'errors').mkdir()
            (root / 'queue/1_1_1_1000.0.json').write_text(json.dumps(
                dict(num_frames=0, events={'crash': True}, vehicle_states={'speed': []})))
            with self.assertRaises(ValueError):
                drive_counts(root)

    def test_rejects_error_records_without_valid_queue_execution(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'queue').mkdir()
            (root / 'errors').mkdir()
            (root / 'errors/1_1_1_1000.0.json').write_text('{}')
            with self.assertRaises(ValueError):
                drive_counts(root)

    def test_uses_event_times_not_delayed_checkpoints(self):
        self.assertEqual(hourly_failures([3599, 3601, 7199, 10801, 14403], 5),
                         [0, 1, 3, 3, 5])

    def test_rejects_incomplete_failure_event_set(self):
        with self.assertRaises(ValueError):
            hourly_failures([100], 2)


if __name__ == '__main__':
    unittest.main()
