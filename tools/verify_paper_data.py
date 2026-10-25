"""Verify extracted paper data without driving models, simulator servers, or APIs."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def archived_path(recorded, root):
    """Resolve historical provenance against the extraction, never the old host."""
    text = str(recorded)
    for marker in ['results/', 'static_scenarios/']:
        if text.startswith(marker):
            relative = text
            break
        if '/' + marker in text:
            relative = marker + text.split('/' + marker, 1)[1]
            break
    else:
        raise ValueError(f'Unsupported historical data path: {text}')
    path = Path(relative)
    if '..' in path.parts:
        raise ValueError(f'Unsafe data path: {text}')
    if relative.startswith('static_scenarios/'):
        path = Path('results/paper_data/rq2') / path
    return Path(root) / path


def verify_rq1(manifest, root):
    base = root / 'results/rq1/formal_round'
    mapping = read_csv(base / 'private/mapping.csv')
    ratings = []
    for annotator in ['annotator_a', 'annotator_b', 'annotator_c']:
        counts = defaultdict(Counter)
        for row in mapping:
            text = (base / 'submissions' / annotator / 'cases' /
                    row['case_id'] / 'annotation.txt').read_text()
            score = int(re.search(r'^score:\s*([012])\s*$', text, re.M)[1])
            counts[row['mode']][score] += 1
        for mode, count in counts.items():
            ratings.append(dict(annotator=annotator, mode=mode, total=sum(count.values()),
                                incorrect=count[0], acceptable=count[1], high_quality=count[2]))
    key = lambda row: (row['annotator'], row['mode'])
    if sorted(ratings, key=key) != sorted(manifest['expected_ratings'], key=key):
        raise ValueError('RQ1 ratings differ from frozen results')
    return dict(cases=len(mapping), ratings=ratings)


def verify_rq2(manifest, root, output):
    from evaluation.rq2.auto_attribute_instruction_cf import analyze
    from evaluation.rq2.build_paper_assets import read_run

    attribution = output / 'instruction_cf_attribution'
    analyze(root / manifest['instruction_cf_root'], attribution)
    expected = manifest['expected_attribution']
    actual = read_csv(attribution / 'summary_by_model.csv')
    if actual != expected:
        raise ValueError('Recomputed Instruction-CF attribution differs from paper data')
    previous = os.environ.get('VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION')
    os.environ['VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION'] = str(attribution)
    try:
        runs = [read_run(dict(model=row['model'], scenario_seed=row['task'],
                             baseline=row['method'], run_dir=row['destination'],
                             executions=row['raw_attempts'], failures=row['failures']),
                         root, ZoneInfo('Asia/Shanghai')) for row in manifest['runs']]
    finally:
        if previous is None:
            os.environ.pop('VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION', None)
        else:
            os.environ['VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION'] = previous
    totals = defaultdict(Counter)
    for row in runs:
        totals[row['model'], row['method']].update(
            executions=row['executions'], failures=row['failures'])
    for row in manifest['expected_effectiveness']:
        actual = totals[row['model'], row['method']]
        if (actual['executions'], actual['failures']) != (row['executions'], row['failures']):
            raise ValueError(f'RQ2 count mismatch: {row}')
    (output / 'rq2_audit.json').write_text(json.dumps(dict(runs=runs), indent=2) + '\n')
    return dict(run_count=len(runs), effectiveness=manifest['expected_effectiveness'])


def verify_rq3(manifest, root):
    totals = defaultdict(Counter)
    for row in manifest['runs']:
        meta = read_json(root / row['destination'] / 'metadata.json')
        if (meta['total_simulations_executed'], meta['total_failures_detected']) != (
                row['executions'], row['failures']):
            raise ValueError(f'RQ3 source count mismatch: {row["destination"]}')
        totals[row['model'], row['method']].update(
            executions=row['executions'], failures=row['failures'])
    for row in manifest['expected_aggregates']:
        if row['configuration'] == 'Full':
            rq2 = read_json(root / 'results/paper_data/rq2/manifest.json')
            selected = [r for r in rq2['runs']
                        if r['model'] == row['model'] and r['method'] == 'VLAD-Fuzz']
            for run in selected:
                meta = read_json(root / run['destination'] / 'metadata.json')
                if (meta['total_simulations_executed'], meta['total_failures_detected']) != (
                        run['executions'], run['failures']):
                    raise ValueError('RQ3 Full metadata mismatch')
            n = sum(r['executions'] for r in selected)
            f = sum(r['failures'] for r in selected)
        else:
            group = totals[row['model'], row['configuration']]
            n, f = group['executions'], group['failures']
        if (n, f) != (int(row['executions']), int(row['failures'])):
            raise ValueError('RQ3 aggregate mismatch')
        if abs(f / n - float(row['pooled_failure_rate'])) > 0.00000051:
            raise ValueError('RQ3 failure rate mismatch')
    return dict(ablation_runs=len(manifest['runs']), aggregates=manifest['expected_aggregates'])


def verify_rq4(manifest, root, output):
    from evaluation.rq4.summarize_results import build_report

    rq2 = read_json(root / 'results/paper_data/rq2/manifest.json')
    runs = [row for row in rq2['runs'] if row['method'] == 'VLAD-Fuzz']
    selection = output / 'rq4_selection.csv'
    with selection.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'model', 'scenario_seed', 'baseline', 'run_dir', 'executions', 'failures'])
        writer.writeheader()
        for row in runs:
            writer.writerow(dict(model=row['model'], scenario_seed=row['task'],
                                 baseline=row['method'], run_dir=row['destination'],
                                 executions=row['executions'], failures=row['failures']))
    build_report(selection, root / 'results/paper_data/rq4/source_rq2_audit.json',
                 output / 'rq4', root, paper_root=None)
    for filename in ['category_summary.csv', 'category_per_seed.csv']:
        if read_csv(output / 'rq4' / filename) != read_csv(root / 'results/rq4/summary' / filename):
            raise ValueError(f'RQ4 recomputation mismatch: {filename}')
    return dict(run_count=len(runs), failures=sum(row['failures'] for row in runs))


def verify_diversity(root, output):
    import numpy as np
    from evaluation.rq2.analyze_diversity import (
        FailureCase, analyze_task_stratified_diversity, carla_weather_resolver,
        drivefuzz_scene_vector, vladfuzz_scene_vector)

    manifest = read_json(root / 'results/paper_data/rq2/manifest.json')
    directory = root / manifest['diversity_directory']
    retained_path = root / manifest['attribution_directory'] / 'retained_failures.jsonl'
    retained = defaultdict(set)
    with retained_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            retained[str(archived_path(row['run_dir'], root))].update(row['failure_artifacts'])
    cases = []
    # Preserve the frozen task/method and per-run artifact order for bootstrap draws.
    for row in read_csv(directory / 'selected_runs.csv'):
        run = archived_path(row['run_dir'], root)
        if row['baseline'] == 'DriveFuzz':
            meta = read_json(run / 'drivefuzz_metadata.json')
            seed = archived_path(meta['seed_dir'], root)
            mapping = [json.loads(line) for line in (seed / 'mapping.jsonl').read_text().splitlines() if line]
            if len(mapping) != 1:
                raise ValueError('Expected one DriveFuzz instruction mapping')
            for path in sorted((run / 'errors').glob('*.json')):
                cases.append(FailureCase(row['model'], row['baseline'], row['scenario_seed'],
                                         mapping[0]['instruction'], drivefuzz_scene_vector(read_json(path)), path))
        else:
            meta = read_json(run / 'metadata.json')
            static = read_json(archived_path(meta['configuration']['static_scenario'], root))
            for path in sorted((run / 'failures').glob('*/result.json')):
                if row['baseline'] == 'Instruction-CF' and path.parent.name not in retained[str(run)]:
                    continue
                record = read_json(path)
                instruction = record.get('mutated_instruction') or record.get('instruction')
                scene = vladfuzz_scene_vector(read_json(path.with_name('scenario_config.json')),
                                             static, weather_resolver=carla_weather_resolver)
                cases.append(FailureCase(row['model'], row['baseline'], row['scenario_seed'],
                                         instruction, scene, path))
    lookup = {}
    for path in [root / 'results/paper_data/rq2/reference_embeddings.npz',
                 *sorted(directory.glob('instruction_embeddings_*.npz'))]:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache['model'].item()) != 'sentence-transformers/all-mpnet-base-v2':
                raise ValueError('Unexpected frozen instruction encoder')
            for text, embedding in zip(cache['texts'].tolist(), cache['embeddings'].copy()):
                if text in lookup and not np.array_equal(lookup[text], embedding):
                    raise ValueError('Conflicting frozen instruction embeddings')
                lookup[text] = embedding
    embeddings = np.asarray([lookup[case.instruction] for case in cases])
    config = read_json(directory / 'diagnostics.json')
    settings = {key: config[key] for key in [
        'scene_weight', 'max_per_method_task', 'bootstrap_iterations', 'random_seed']}
    summary, comparisons, diagnostics = analyze_task_stratified_diversity(cases, embeddings, **settings)
    expected = {(r['model'], r['method']): r for r in read_csv(directory / 'summary.csv')}
    error = max(abs(row[key] - float(expected[row['model'], row['method']][key]))
                for row in summary for key in ['scene_diversity', 'instruction_diversity',
                                               'joint_diversity', 'joint_ci_low', 'joint_ci_high'])
    if error > 1e-10:
        raise ValueError(f'Diversity differs from frozen results: {error}')
    expected_pairs = {(r['model'], r['comparison']): r for r in read_csv(directory / 'comparisons.csv')}
    for row in comparisons:
        for key in ['mean_difference', 'ci_low', 'ci_high']:
            if abs(row[key] - float(expected_pairs[row['model'], row['comparison']][key])) > 1e-10:
                raise ValueError('Diversity comparison mismatch')
    report = dict(summary=summary, comparisons=comparisons, diagnostics=diagnostics,
                  max_error=error, failures=len(cases))
    (output / 'diversity.json').write_text(json.dumps(report, indent=2) + '\n')
    return dict(max_error=error, failures=len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path.cwd())
    parser.add_argument('--rq', nargs='+', choices=['rq1', 'rq2', 'rq3', 'rq4'],
                        default=['rq1', 'rq2', 'rq3', 'rq4'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--diversity', action='store_true',
                        help='Also recompute bootstrap diversity using frozen embeddings; needs NumPy and CARLA Python API.')
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifests = {rq: read_json(root / f'results/paper_data/{rq}/manifest.json') for rq in args.rq}
    required = set(args.rq)
    for manifest in manifests.values():
        required.update(manifest.get('dependencies', []))
    if args.diversity:
        required.add('rq2')
    for rq in sorted(required):
        manifest = manifests.setdefault(rq, read_json(root / f'results/paper_data/{rq}/manifest.json'))
        for row in manifest['files']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe manifest path')
            path = root / relative
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError(f'Archive content mismatch: {path}')
        print(f'{rq}: {len(manifest["files"])} file hashes verified', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir.resolve()
    report = {}
    for rq in args.rq:
        function = globals()[f'verify_{rq}']
        report[rq] = function(manifests[rq], root, output) if rq in {'rq2', 'rq4'} else function(manifests[rq], root)
        print(f'{rq}: paper statistics verified', flush=True)
    if args.diversity:
        report['diversity'] = verify_diversity(root, output)
        print('rq2: cached-embedding diversity verified', flush=True)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
