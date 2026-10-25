import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.package_paper_data import build_package, safe_name
from tools.verify_paper_data import archived_path
from tools.anonymize_paper_data import sanitize_plan


class PaperDataPackagingTests(unittest.TestCase):
    def test_deidentification_updates_dependent_hashes_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'metadata.json'
            source.write_text('{"path": "/private/author/project/results/run", "count": 17}\n')
            old_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            audit = root / 'audit.json'
            audit.write_text(json.dumps(dict(metadata_sha256=old_hash)))
            files = [dict(source=str(path), destination=f'results/rq2/{path.name}',
                          bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                     for path in [source, audit]]
            plan = dict(packages=[dict(name='rq2', files=files)])
            report = sanitize_plan(plan, root / 'stage', ['/private/author/project'], 'Data\n')
            normalized = json.loads(Path(files[0]['source']).read_text())
            normalized_audit = json.loads(Path(files[1]['source']).read_text())
            self.assertEqual(normalized, dict(path='results/run', count=17))
            self.assertEqual(normalized_audit['metadata_sha256'], files[0]['sha256'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), old_hash)
            self.assertEqual(report['changed_unique_files'], 2)

    def test_historical_paths_are_resolved_only_inside_extraction(self):
        root = Path('/tmp/new-extraction')
        self.assertEqual(archived_path('/old/private/project/results/rq2/run/metadata.json', root),
                         root / 'results/rq2/run/metadata.json')
        self.assertEqual(archived_path('static_scenarios/town01.json', root),
                         root / 'results/paper_data/rq2/static_scenarios/town01.json')
        with self.assertRaises(ValueError):
            archived_path('results/../../escape', root)

    def test_rejects_unsafe_paths(self):
        for path in ['', '/absolute', '../escape', 'results/../../escape', 'x\\y']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_name(path)

    def test_deduplicated_roundtrip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'identical evidence\n')
            row = dict(source=str(source), bytes=source.stat().st_size,
                       sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            package = dict(name='rq1', dependencies=[], readme='Test data\n', files=[
                dict(row, destination='results/rq1/a.txt'),
                dict(row, destination='results/rq1/b.txt')])
            hashes = []
            for name in ['one', 'two']:
                output = root / name
                output.mkdir()
                record = build_package(package, output)
                hashes.append(record['sha256'])
                self.assertEqual(record['unique_content_bytes'], row['bytes'])
                with tarfile.open(output / record['archive']) as archive:
                    self.assertTrue(archive.getmember('results/rq1/b.txt').islnk())
                    archive.extractall(output / 'extract', filter='data')
                extracted = output / 'extract'
                self.assertEqual((extracted / 'results/rq1/b.txt').read_bytes(), source.read_bytes())
                manifest = json.loads((extracted / 'results/paper_data/rq1/manifest.json').read_text())
                self.assertNotIn('source', manifest['files'][0])
            self.assertEqual(*hashes)

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'new')
            package = dict(name='rq1', readme='', files=[dict(
                source=str(source), destination='results/a', bytes=3, sha256='old')])
            with self.assertRaises(ValueError):
                build_package(package, root)

    def test_duplicate_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = dict(name='rq1', readme='', files=[
                dict(destination='results/a'), dict(destination='results/a')])
            with self.assertRaises(ValueError):
                build_package(package, Path(directory))


if __name__ == '__main__':
    unittest.main()
