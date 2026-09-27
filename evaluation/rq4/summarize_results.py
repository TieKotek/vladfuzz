"""Analyze failure requirements in explicitly frozen VLAD-Fuzz runs."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = (
    ('collision', 'Collision', 'Driving safety'),
    ('lane_invasion', 'Prohibited lane crossing', 'Driving safety'),
    ('out_of_bounds', 'Test-region departure', 'Navigation task completion'),
    ('timeout', 'Destination not reached', 'Navigation task completion'),
    ('speed_limit_exceeded', 'Instruction-speed violation', 'Explicit instruction constraints'),
    ('maintain_distance_failed', 'Maintain-distance violation', 'Explicit instruction constraints'),
)
CATEGORY_KEYS = {key for key, _, _ in CATEGORIES}
MODEL_LABELS = {'simlingo': 'SimLingo', 'lmdrive': 'LMDrive', 'bevdriver': 'BEVDriver'}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_failure(record):
    execution = record.get('execution_result') or {}
    if execution.get('error') or execution.get('actual_frames_executed', 0) <= 0:
        raise ValueError('invalid behavioral execution')
    if record.get('node_outcome', 'failed') != 'failed':
        raise ValueError('record is not a failed execution')
    reason = record.get('failure_reason') or record.get('semantic_failure_reason')
    if not reason:
        raise ValueError('missing final failure reason')
    categories = {part.strip() for part in reason.split(',')}
    oracle = execution.get('oracle_events') or {}
    enabled = oracle.get('enabled_checks') or {}
    # Raw telemetry may flag disabled checks; only active triggered checks count.
    for check in oracle.get('triggered_checks') or []:
        if enabled.get(check) is False:
            raise ValueError(f'disabled check in triggered_checks: {check}')
        categories.add(check)
    unknown = categories - CATEGORY_KEYS
    if unknown:
        raise ValueError(f'unclassified failure categories: {sorted(unknown)}')
    return categories


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def category_rows(model, seed, counts, failures, executions):
    return [dict(model=model, scenario_seed=seed, category=key, label=label,
                 requirement_group=group, count=counts[key], failures=failures,
                 executions=executions, failure_share=counts[key] / failures if failures else 0)
            for key, label, group in CATEGORIES]


def render_figure_source(summary):
    values = {(row['model'], row['category']): 100 * float(row['failure_share'])
              for row in summary}
    model_style = {
        'simlingo': 'SimGreen',
        'lmdrive': 'LMOrange',
        'bevdriver': 'BEVBlue',
    }
    category_labels = (
        r'Maintain-distance\\constraint',
        r'Speed\\constraint',
        r'Destination not\\reached',
        r'Test-region\\departure',
        r'Prohibited lane\\crossing',
        r'Collision',
    )
    lines = [
        r'\documentclass[tikz,border=2pt]{standalone}',
        r'\usepackage[scaled=0.95]{helvet}',
        r'\renewcommand{\familydefault}{\sfdefault}',
        r'\usepackage{pgfplots}',
        r'\usepackage{sfmath}',
        r'\usepgfplotslibrary{groupplots}',
        r'\pgfplotsset{compat=1.17}',
        r'\definecolor{Ink}{HTML}{222222}',
        r'\definecolor{GridGray}{HTML}{D5D9DD}',
        r'\definecolor{SimGreen}{HTML}{008B6A}',
        r'\definecolor{LMOrange}{HTML}{D55E00}',
        r'\definecolor{BEVBlue}{HTML}{0072B2}',
        r'\begin{document}', r'\begin{tikzpicture}',
        r'\begin{groupplot}[',
        r'group style={group size=3 by 1,horizontal sep=0.62cm},',
        r'width=4.85cm,height=6.25cm,xmin=0,xmax=70,ymin=-0.55,ymax=5.55,',
        r'ytick={0,1,2,3,4,5},',
        r'xtick={0,20,40,60},',
        r'tick label style={font=\fontsize{7.4}{8.5}\selectfont},',
        r'title style={font=\bfseries\fontsize{9}{10.5}\selectfont,text=Ink,yshift=-1pt},',
        r'axis lines=left,xmajorgrids=true,grid style={GridGray,line width=0.3pt},',
        r'clip=false,enlarge x limits=false,',
        r'nodes near coords,point meta=x,',
        r'every node near coord/.append style={font=\fontsize{6.7}{7.6}\selectfont,text=Ink,anchor=west,xshift=1pt},',
        r'/pgf/number format/fixed,/pgf/number format/precision=1,/pgf/number format/fixed zerofill]',
    ]
    for index, model in enumerate(('simlingo', 'lmdrive', 'bevdriver')):
        options = [f'title={{{MODEL_LABELS[model]}}}']
        if index == 0:
            options.append('yticklabels={{' + '},{'.join(category_labels) + '}}')
            options.append(r'yticklabel style={font=\fontsize{7.2}{8.2}\selectfont,align=right}')
        else:
            options.append('yticklabels={}')
        lines.append(r'\nextgroupplot[' + ','.join(options) + ']')
        coordinates = ' '.join(
            f'({values[(model, key)]:.4f},{5 - category_index})'
            for category_index, (key, _, _) in enumerate(CATEGORIES))
        color = model_style[model]
        lines.append(r'\addplot[xbar,bar width=7.6pt,fill=' + color + r',draw=' + color +
                     r',fill opacity=0.88,line width=0.45pt] coordinates {' + coordinates + '};')
    lines.extend([
        r'\end{groupplot}',
        r'\node[font=\fontsize{8}{9.2}\selectfont,text=Ink] at '
        r'([yshift=-0.67cm]group c2r1.south) {Share of failed executions (\%)};',
        r'\end{tikzpicture}', r'\end{document}', ''
    ])
    return '\n'.join(lines)


def build_report(selection, rq2_audit, output_root, repo_root, paper_root=None):
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    with Path(selection).open(newline='', encoding='utf-8') as file:
        selected = [row for row in csv.DictReader(file) if row['baseline'] == 'VLAD-Fuzz']
    if not selected:
        raise ValueError('no VLAD-Fuzz runs selected')
    keys = [(row['model'], row['scenario_seed']) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate model/task selection')
    frozen = [row for row in load_json(Path(rq2_audit))['runs'] if row['method'] == 'VLAD-Fuzz']
    frozen_keys = [(row['model'], row['scenario_seed']) for row in frozen]
    if len(frozen_keys) != len(set(frozen_keys)) or set(keys) != set(frozen_keys):
        raise ValueError('selection differs from the frozen RQ2 run set')
    frozen_by_key = dict(zip(frozen_keys, frozen))
    aggregates = defaultdict(Counter)
    totals = defaultdict(Counter)
    per_seed, failure_index, source_audit = [], [], []
    for row in sorted(selected, key=lambda item: (item['model'], item['scenario_seed'])):
        model, seed = row['model'], row['scenario_seed']
        root = Path(row['run_dir'])
        root = (root if root.is_absolute() else repo_root / root).resolve()
        if output_root == root or root in output_root.parents:
            raise ValueError('output must not be inside a raw run directory')
        metadata_path = root / 'metadata.json'
        source = frozen_by_key[(model, seed)]
        metadata_hash = sha256(metadata_path)
        if metadata_hash != source['metadata_sha256']:
            raise ValueError(f'frozen metadata hash mismatch: {root}')
        meta = load_json(metadata_path)
        config = meta['configuration']
        if (config.get('vla_model'), config.get('manifest_id'), config.get('method')) != (
                model, seed, 'vlad_fuzz_cmd1'):
            raise ValueError(f'run identity mismatch: {root}')
        if meta.get('status') in ('failed', 'infrastructure_failed'):
            raise ValueError(f'invalid campaign: {root}')
        n, f = int(row['executions']), int(row['failures'])
        if (n, f) != (meta['total_simulations_executed'], meta['total_failures_detected']) or (
                n, f) != (source['executions'], source['failures']) or not 0 <= f <= n:
            raise ValueError(f'execution/failure count mismatch: {root}')
        if float(config.get('time_budget_seconds', 0)) != 14400:
            raise ValueError(f'different time budget: {root}')
        paths = sorted((root / 'failures').glob('*/result.json'))
        if len(paths) != f:
            raise ValueError(f'failure record count mismatch: {root}')
        counts = Counter()
        digest = hashlib.sha256()
        for path in paths:
            record = load_json(path)
            try:
                categories = classify_failure(record)
            except ValueError as error:
                raise ValueError(f'{path}: {error}') from error
            instruction = record.get('mutated_instruction')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f'missing instruction: {path}')
            scenario = path.parent / 'scenario_config.json'
            result_hash, scenario_hash = sha256(path), sha256(scenario)
            digest.update(json.dumps([str(path.relative_to(root)), result_hash, scenario_hash]).encode())
            counts.update(categories)
            execution = record['execution_result']
            speed_stats = execution.get('ego_speed_stats') or {}
            failure_index.append(dict(
                model=model, scenario_seed=seed, failure_id=path.parent.name,
                categories=';'.join(key for key, _, _ in CATEGORIES if key in categories),
                primary_reason=record.get('failure_reason') or record.get('semantic_failure_reason'),
                instruction=instruction, applied_operator=record.get('applied_operator'),
                node_depth=record.get('node_depth'),
                frames=execution['actual_frames_executed'],
                max_speed_kmh=speed_stats.get('max_speed'),
                min_distance=execution.get('min_distance'),
                result_path=str(path), scenario_config_path=str(scenario),
                result_sha256=result_hash, scenario_sha256=scenario_hash,
            ))
        per_seed.extend(category_rows(model, seed, counts, f, n))
        aggregates[model].update(counts)
        totals[model].update(failures=f, executions=n)
        source_audit.append(dict(model=model, scenario_seed=seed, run_dir=str(root),
                                 executions=n, failures=f, metadata_sha256=metadata_hash,
                                 failure_records_sha256=digest.hexdigest()))
    summary = []
    for model in sorted(totals):
        summary.extend(category_rows(model, 'ALL', aggregates[model],
                                     totals[model]['failures'], totals[model]['executions']))
    # Validate all inputs before creating reports; never change raw experiment files.
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / 'category_summary.csv', summary)
    write_csv(output_root / 'category_per_seed.csv', per_seed)
    if failure_index:
        write_csv(output_root / 'failure_index.csv', failure_index)
    else:
        (output_root / 'failure_index.csv').write_text('model,scenario_seed,failure_id\n')
    audit = dict(
        selection_sha256=sha256(Path(selection)), rq2_audit_sha256=sha256(Path(rq2_audit)),
        runs=source_audit,
        classification_policy='Union of final failure reason and enabled triggered checks; raw telemetry ignored.',
        denominator='All valid failed executions for the model/task; multilabel proportions may sum above one.',
        category_order='From conventional driving outcomes to explicit instruction constraints; not a causal or capability ranking.',
        constraint_comparison='Failure shares do not measure constraint compliance; that requires applicable-execution denominators.',
    )
    (output_root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    if paper_root is not None:
        source_dir = Path(paper_root) / 'figures' / 'src'
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / 'rq4-failure-spectrum.tex').write_text(
            render_figure_source(summary), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument('--selection', type=Path, default=repo_root / 'results/rq2/paper/selected_runs.csv')
    parser.add_argument('--rq2-audit', type=Path, default=repo_root / 'results/rq2/paper/audit.json')
    parser.add_argument('--output-dir', type=Path, default=repo_root / 'results/rq4/summary')
    parser.add_argument('--paper-root', type=Path, default=repo_root / 'paper/fse-2027')
    args = parser.parse_args()
    rows = build_report(args.selection, args.rq2_audit, args.output_dir, repo_root, args.paper_root)
    print(f'Wrote {len(rows)} model/category rows to {args.output_dir}')


if __name__ == '__main__':
    main()
