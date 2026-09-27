import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq4.summarize_results import build_report, classify_failure, render_figure_source


def failure(reason='collision', checks=None, frames=25):
    return {
        'mutated_instruction': 'Continue straight.',
        'failure_reason': reason,
        'node_outcome': 'failed',
        'valid': False,
        'execution_result': {
            'actual_frames_executed': frames,
            'error': None,
            'oracle_events': {
                'speeding': True,
                'enabled_checks': {'collision': True, 'lane_invasion': True,
                                   'speeding': False},
                'triggered_checks': checks or [],
            },
        },
    }


class RQ4ReportingTests(unittest.TestCase):
    def test_preserves_multiple_enabled_oracle_categories(self):
        self.assertEqual(classify_failure(failure(checks=['collision', 'lane_invasion'])),
                         {'collision', 'lane_invasion'})

    def test_constraint_reason_survives_empty_triggered_checks(self):
        self.assertEqual(classify_failure(failure('speed_limit_exceeded')),
                         {'speed_limit_exceeded'})

    def test_disabled_speed_telemetry_is_not_a_failure_category(self):
        self.assertEqual(classify_failure(failure('maintain_distance_failed')),
                         {'maintain_distance_failed'})

    def test_rejects_disabled_triggered_check(self):
        with self.assertRaisesRegex(ValueError, 'disabled'):
            classify_failure(failure(checks=['speeding']))

    def test_rejects_unknown_reason_and_invalid_execution(self):
        for record in (failure('other'), failure(frames=0), failure('collision, strange')):
            with self.subTest(record=record), self.assertRaises(ValueError):
                classify_failure(record)
        record = failure()
        record['execution_result']['error'] = 'RPC timeout'
        with self.assertRaises(ValueError):
            classify_failure(record)

    def prepare_run(self, root):
        run = root / 'run'
        case = run / 'failures' / 'case_1'
        case.mkdir(parents=True)
        (case / 'result.json').write_text(json.dumps(failure(
            checks=['collision', 'lane_invasion'])))
        (case / 'scenario_config.json').write_text('{"duration_frames":500}')
        meta = {
            'total_simulations_executed': 4, 'total_failures_detected': 1,
            'configuration': {'vla_model': 'simlingo', 'manifest_id': 'region/seed_1',
                              'method': 'vlad_fuzz_cmd1', 'time_budget_seconds': 14400},
        }
        path = run / 'metadata.json'
        path.write_text(json.dumps(meta))
        selection = root / 'selected.csv'
        with selection.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=(
                'model', 'baseline', 'scenario_seed', 'run_dir', 'executions', 'failures'))
            writer.writeheader()
            writer.writerow(dict(model='simlingo', baseline='VLAD-Fuzz',
                                 scenario_seed='region/seed_1', run_dir='run',
                                 executions=4, failures=1))
        audit = root / 'rq2_audit.json'
        audit.write_text(json.dumps({'runs': [{
            'model': 'simlingo', 'method': 'VLAD-Fuzz', 'scenario_seed': 'region/seed_1',
            'executions': 4, 'failures': 1,
            'metadata_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }]}))
        return selection, audit, path, case

    def test_report_uses_failure_denominator_and_keeps_zero_categories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, audit, _, _ = self.prepare_run(root)
            build_report(selection, audit, root / 'output', root)
            with (root / 'output/category_summary.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            categories = {row['category']: row for row in rows}
            self.assertEqual(float(categories['collision']['failure_share']), 1.0)
            self.assertEqual(float(categories['lane_invasion']['failure_share']), 1.0)
            self.assertEqual(int(categories['timeout']['count']), 0)
            self.assertEqual(categories['speed_limit_exceeded']['requirement_group'],
                             'Explicit instruction constraints')
            with (root / 'output/failure_index.csv').open() as file:
                index = list(csv.DictReader(file))
            self.assertEqual(len(index), 1)
            self.assertEqual(index[0]['instruction'], 'Continue straight.')
            self.assertTrue(Path(index[0]['scenario_config_path']).exists())

    def test_rejects_changed_frozen_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, audit, metadata, _ = self.prepare_run(root)
            metadata.write_text(metadata.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                build_report(selection, audit, root / 'output', root)
            self.assertFalse((root / 'output').exists())

    def test_rejects_missing_failure_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, audit, _, case = self.prepare_run(root)
            (case / 'result.json').unlink()
            with self.assertRaisesRegex(ValueError, 'count'):
                build_report(selection, audit, root / 'output', root)

    def test_figure_groups_failure_categories_by_model(self):
        rows = []
        for model in ('simlingo', 'lmdrive', 'bevdriver'):
            for key, _, _ in __import__('evaluation.rq4.summarize_results', fromlist=['CATEGORIES']).CATEGORIES:
                rows.append({'model': model, 'category': key,
                             'failure_share': 0.0 if key == 'maintain_distance_failed' else 0.25})
        source = render_figure_source(rows)
        self.assertIn(r'\usepgfplotslibrary{groupplots}', source)
        self.assertIn(r'\nextgroupplot[title={SimLingo}', source)
        self.assertIn(r'\nextgroupplot[title={LMDrive}', source)
        self.assertIn(r'\nextgroupplot[title={BEVDriver}', source)
        self.assertEqual(source.count('Collision'), 1)
        self.assertIn('(25.0000,5)', source)
        self.assertIn('(0.0000,0)', source)
        self.assertNotIn('Driving safety', source)
        self.assertNotIn('Navigation task completion', source)
        self.assertNotIn('Explicit instruction constraints', source)


if __name__ == '__main__':
    unittest.main()
