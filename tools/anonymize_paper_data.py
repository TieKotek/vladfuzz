"""Normalize local checkout paths in release copies, preserving original data."""

import argparse
import hashlib
import json
from pathlib import Path
import re


HASH = re.compile(rb'(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])')
TEXT_SUFFIXES = {'', '.json', '.jsonl', '.csv', '.txt', '.md', '.tex'}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def sanitize_plan(plan, output_dir, prefixes, readme):
    output_dir = Path(output_dir)
    prefixes = sorted({value.rstrip('/').encode() + b'/' for value in prefixes},
                      key=len, reverse=True)
    if not prefixes or b'/' in prefixes:
        raise ValueError('Supply nonempty local checkout prefixes, never the filesystem root')
    output_dir.mkdir(parents=True, exist_ok=False)
    records = [row for package in plan['packages'] for row in package['files']]
    original = {}
    candidates = {}
    by_path = {}
    for row in records:
        by_path[row['destination']] = row
        key = row['sha256']
        if key in original:
            continue
        source = Path(row['source'])
        content = source.read_bytes()
        if len(content) != row['bytes'] or digest(content) != key:
            raise ValueError(f'Source changed since inventory: {source}')
        original[key] = key
        if source.suffix.lower() not in TEXT_SUFFIXES:
            continue
        normalized = content
        for prefix in prefixes:
            normalized = normalized.replace(prefix, b'')
            normalized = re.sub(re.escape(prefix[:-1]) + rb'(?=["\x27\s]|$)', b'.', normalized)
        if normalized != content or HASH.search(content):
            candidates[key] = normalized

    # Recompute the RQ4 digest over result/scenario hashes, not just file hashes.
    special = {}
    for destination, row in by_path.items():
        if destination == 'results/rq4/summary/audit.json':
            special[row['sha256']] = destination

    def aggregate_failure_hash(run, current):
        text = run['run_dir']
        prefix = 'results/' + text.split('results/', 1)[1].rstrip('/') + '/'
        result_paths = sorted(path for path in by_path
                              if path.startswith(prefix + 'failures/') and path.endswith('/result.json'))
        if len(result_paths) != run['failures']:
            raise ValueError('Incomplete RQ4 failure selection')
        result = hashlib.sha256()
        for path in result_paths:
            scenario = str(Path(path).with_name('scenario_config.json'))
            hashes = [current[by_path[p]['sha256']] for p in [path, scenario]]
            result.update(json.dumps([path[len(prefix):], *hashes]).encode())
        return result.hexdigest()

    current = dict(original)
    updated = {}
    for iteration in range(16):
        next_hashes = dict(original)
        for key, content in candidates.items():
            value = HASH.sub(lambda match: current.get(match[0].decode(), match[0].decode()).encode(), content)
            if key in special:
                data = json.loads(value)
                for run in data['runs']:
                    run['failure_records_sha256'] = aggregate_failure_hash(run, current)
                value = (json.dumps(data, indent=2) + '\n').encode()
            updated[key] = value
            next_hashes[key] = digest(value)
        if next_hashes == current:
            break
        current = next_hashes
    else:
        raise ValueError('Hash references did not converge; inspect cyclic or missing dependencies')

    changed = {key: value for key, value in updated.items() if digest(value) != key}
    stage = output_dir / 'files'
    stage.mkdir()
    for key, value in changed.items():
        (stage / key).write_bytes(value)
    for row in records:
        key = row['sha256']
        if key in changed:
            row.update(source=str((stage / key).resolve()), bytes=len(changed[key]),
                       original_sha256=key, sha256=digest(changed[key]))
    for package in plan['packages']:
        package['readme'] = readme
        package['raw_file_policy'] = (
            'Release copies normalize local checkout paths and dependent hashes. '
            'Original research files are unchanged; experimental values are preserved.')
        package['deidentification'] = dict(
            policy='Local checkout prefixes removed; dependent file and RQ4 aggregate hashes updated.',
            scope='Known local path identifiers; not a guarantee of universal anonymity.')
    plan.update(readme=readme, publication_status='Local de-identified archives; not uploaded.')
    path = output_dir / 'archive_plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    report = dict(unique_sources=len(original), changed_unique_files=len(changed),
                  changed_file_locations=sum(bool(row.get('original_sha256')) for row in records),
                  hash_reference_iterations=iteration + 1,
                  source_experiment_files_modified=False)
    (output_dir / 'normalization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--strip-prefix', action='append', required=True,
                        help='Local checkout root to remove from release text; may be repeated.')
    parser.add_argument('--readme', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize_plan(json.loads(args.plan.read_text()), args.output_dir,
                                   args.strip_prefix, args.readme.read_text()), indent=2))


if __name__ == '__main__':
    main()
