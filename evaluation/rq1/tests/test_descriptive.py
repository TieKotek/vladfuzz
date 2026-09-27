import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq1.annotations import AnnotationValidationError
from evaluation.rq1.reporting import analyze_descriptive


class DescriptiveTests(unittest.TestCase):
    def make_round(self, root):
        rows = []
        paths = {name: root / 'submissions' / name for name in ('a', 'b')}
        for index, mode in enumerate(('route_prior', 'no_prior', 'route_prior', 'no_prior')):
            case_id = f'case_{index}'
            rows.append(dict(case_id=case_id, pair_id=f'pair_{index // 2}',
                             scenario_name=f'town_{index // 2}', seed_name='seed_1', mode=mode))
            for name, scores in (('a', (2, 0, 1, 0)), ('b', (1, 0, 1, 0))):
                folder = paths[name] / 'cases' / case_id
                folder.mkdir(parents=True)
                for filename in ('image.png', 'instruction.txt', 'reference.txt'):
                    (folder / filename).write_text('evidence')
                error = 'wrong_maneuver,missing_maneuver' if scores[index] == 0 else ''
                (folder / 'annotation.txt').write_text(
                    f'score: {scores[index]}\nerror_type: {error}\nnotes:\n')
        private = root / 'private'
        private.mkdir()
        with (private / 'mapping.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return paths

    def test_preserves_subjective_scores_and_counts_errors_once(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_round(root)
            original = (paths['a'] / 'cases/case_0/annotation.txt').read_bytes()
            report = analyze_descriptive(root, paths)
            self.assertEqual(report['methods']['route_prior']['correct_rate'], 1)
            self.assertEqual(report['methods']['no_prior']['errors'], 2)
            self.assertEqual(report['annotators']['a']['route_prior']['high_quality_rate'], .5)
            self.assertEqual(report['annotators']['b']['route_prior']['high_quality_rate'], 0)
            self.assertEqual(report['error_types']['no_prior']['wrong_maneuver'], 2)
            self.assertEqual(report['error_types']['no_prior']['missing_maneuver'], 2)
            self.assertNotIn('agreement', report)
            self.assertNotIn('paired_tests', report)
            self.assertEqual(original, (paths['a'] / 'cases/case_0/annotation.txt').read_bytes())
            for filename in ('summary.json', 'summary.md', 'summary.tex', 'method_summary.csv',
                             'ratings.csv', 'error_types.csv', 'scenario_summary.csv'):
                self.assertTrue((root / 'reports/descriptive' / filename).is_file())
            self.assertFalse((root / 'reports/descriptive/paired_tests.csv').exists())

    def test_rejects_unresolved_correctness_and_error_labels_before_writing(self):
        for annotation in ('score: 1\nerror_type:\nnotes:\n',
                           'score: 0\nerror_type: wrong_maneuver\nnotes:\n'):
            with self.subTest(annotation=annotation), TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = self.make_round(root)
                (paths['b'] / 'cases/case_1/annotation.txt').write_text(annotation)
                with self.assertRaises(AnnotationValidationError):
                    analyze_descriptive(root, paths)
                self.assertFalse((root / 'reports/descriptive').exists())

    def test_requires_two_submissions(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationValidationError):
                analyze_descriptive(Path(tmp), {})


if __name__ == '__main__':
    unittest.main()
