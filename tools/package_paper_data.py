"""Create deterministic, content-deduplicated data archives from a local plan."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def safe_name(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe archive path: {value}')
    return str(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def member(name, size=0):
    info = tarfile.TarInfo(safe_name(name))
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    return info


def build_package(package, output_dir):
    name = package['name']
    if name not in {'rq1', 'rq2', 'rq3', 'rq4'}:
        raise ValueError(f'Unknown package: {name}')
    files = sorted(package['files'], key=lambda row: row['destination'])
    destinations = [safe_name(row['destination']) for row in files]
    prefix = f'results/paper_data/{name}'
    if len(set(destinations)) != len(files) or any(
            path in {f'{prefix}/manifest.json', f'{prefix}/README.md'} for path in destinations):
        raise ValueError('Duplicate or reserved archive destination')
    archive = Path(output_dir) / f'vladfuzz-{name}-data.tar.gz'
    partial = archive.with_suffix(archive.suffix + '.partial')
    if archive.exists() or partial.exists():
        raise FileExistsError(archive)
    lookup = {}
    entries = []
    with partial.open('xb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for row in files:
                    source = Path(row['source'])
                    if source.is_symlink() or not source.is_file():
                        raise ValueError(f'Expected a regular source file: {source}')
                    if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
                        raise ValueError(f'Source changed since inventory: {source}')
                    path = safe_name(row['destination'])
                    entry = dict(path=path, bytes=row['bytes'], sha256=row['sha256'])
                    if row.get('original_sha256'):
                        entry['original_sha256'] = row['original_sha256']
                    key = row['sha256'], row['bytes']
                    info = member(path, row['bytes'])
                    if key in lookup:
                        info.type = tarfile.LNKTYPE
                        info.linkname = lookup[key]
                        info.size = 0
                        entry['same_content_as'] = lookup[key]
                        tar.addfile(info)
                    else:
                        lookup[key] = path
                        with source.open('rb') as stream:
                            tar.addfile(info, stream)
                    entries.append(entry)
                manifest = {key: value for key, value in package.items()
                            if key not in {'files', 'readme'}}
                manifest.update(schema_version=1, files=entries,
                                raw_file_policy=package.get('raw_file_policy',
                                    'Source bytes unchanged; historical paths are provenance.'))
                for filename, data in [
                    ('manifest.json', json.dumps(manifest, indent=2).encode() + b'\n'),
                    ('README.md', package['readme'].encode()),
                ]:
                    tar.addfile(member(f'{prefix}/{filename}', len(data)), io.BytesIO(data))
    partial.rename(archive)
    return dict(name=name, archive=archive.name, bytes=archive.stat().st_size,
                sha256=digest(archive), file_count=len(files),
                logical_bytes=sum(row['bytes'] for row in files),
                unique_content_bytes=sum(size for _, size in lookup),
                dependencies=package.get('dependencies', []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True,
                        help='Local inventory plan; may contain private source paths. Do not publish it.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for package in plan['packages']:
        record = build_package(package, args.output_dir)
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / 'SHA256SUMS').write_text(''.join(
        f"{row['sha256']}  {row['archive']}\n" for row in records))
    (args.output_dir / 'packages.json').write_text(json.dumps(dict(
        schema_version=1, publication_status=plan.get('publication_status',
            'Local archives; anonymity review pending.'),
        packages=records), indent=2) + '\n')
    (args.output_dir / 'README.md').write_text(plan['readme'])


if __name__ == '__main__':
    main()
