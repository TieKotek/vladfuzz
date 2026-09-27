"""Freeze selected RQ2 runs and build validated publication tables and curves."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODELS = {'simlingo': 'SimLingo', 'lmdrive': 'LMDrive', 'bevdriver': 'BEVDriver'}
METHODS = ('DriveFuzz', 'Instruction-CF', 'VLAD-Fuzz')
EVENTS = ('crash', 'stuck', 'lane_invasion', 'red', 'out_of_bounds', 'speeding', 'other')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def drive_counts(root):
    valid = set()
    excluded = 0
    for path in sorted((root / 'queue').glob('*.json')):
        record = load_json(path)
        frames = record['num_frames']
        if frames == 0:
            if any(record.get('events', {}).get(key) for key in EVENTS) or record.get('vehicle_states', {}).get('speed'):
                raise ValueError(f'zero-frame behavioral record requires inspection: {path}')
            excluded += 1
        elif frames > 0:
            valid.add(path.name)
        else:
            raise ValueError(f'negative frame count: {path}')
    errors = sorted((root / 'errors').glob('*.json'))
    if any(path.name not in valid for path in errors):
        raise ValueError(f'error without a valid queue execution: {root}')
    timestamps = [float(path.stem.rsplit('_', 1)[1]) for path in errors]
    return len(valid), len(errors), excluded, timestamps


def hourly_failures(elapsed, final_count):
    if len(elapsed) != final_count or any(value < -1 for value in elapsed):
        raise ValueError('failure-event records do not match completed run')
    # Intermediate points use event times, not tree-level checkpoint callbacks.
    # The last point is the completed four-hour campaign, including its final test.
    return [0] + [sum(value <= hour * 3600 for value in elapsed) for hour in (1, 2, 3)] + [final_count]


def read_run(row, repo_root, record_timezone):
    root = Path(row['run_dir'])
    if not root.is_absolute():
        root = repo_root / root
    method = row['baseline']
    meta_path = root / ('drivefuzz_metadata.json' if method == 'DriveFuzz' else 'metadata.json')
    meta = load_json(meta_path)
    config = meta.get('configuration', {})
    if meta.get('status') in ('infrastructure_failed', 'failed'):
        raise ValueError(f'failed campaign: {root}')
    start = datetime.fromisoformat(meta['start_time'])
    end = datetime.fromisoformat(meta['end_time'])
    budget = config.get('time_budget_seconds', meta.get('time_budget_seconds', meta.get('target_duration_seconds')))
    if budget != 14400 or (end - start).total_seconds() < budget - 1:
        raise ValueError(f'incomplete or different budget: {root}')
    raw_n, expected_f = int(row['executions']), int(row['failures'])
    if method == 'DriveFuzz':
        n, f, excluded, epochs = drive_counts(root)
        if meta['executed_scenarios'] != raw_n or n + excluded != raw_n or meta['failures'] != f:
            raise ValueError(f'DriveFuzz source count mismatch: {root}')
        aware_start = start if start.tzinfo else start.replace(tzinfo=record_timezone)
        elapsed = [epoch - aware_start.timestamp() for epoch in epochs]
        command = (root / 'meta').read_text(encoding='utf-8')
        for required in ('--max-simulation-frames 500', '--goal-distance-threshold 5',
                         '--no-red-check', '--lock-traffic-lights-green'):
            if required not in command:
                raise ValueError(f'missing matched execution setting {required}: {root}')
    else:
        n, f, excluded = meta['total_simulations_executed'], meta['total_failures_detected'], 0
        if n != raw_n or config['manifest_id'] != row['scenario_seed'] or config['vla_model'] != row['model']:
            raise ValueError(f'local source identity/count mismatch: {root}')
        elapsed = []
        for path in sorted((root / 'failures').glob('*/result.json')):
            record = load_json(path)
            result = record['execution_result']
            if result.get('error') or result.get('actual_frames_executed', 0) <= 0:
                raise ValueError(f'invalid behavioral failure: {path}')
            timestamp = datetime.strptime(record['timestamp'], '%Y%m%d_%H%M%S')
            elapsed.append((timestamp - start).total_seconds())
        if method == 'Instruction-CF':
            records = [json.loads(line) for line in (root / 'execution_results.jsonl').read_text().splitlines() if line.strip()]
            if len(records) != n or sum(bool(item['failed']) for item in records) != f:
                raise ValueError(f'counterfactual execution count mismatch: {root}')
    if f != expected_f or f > n or any(value > (end - start).total_seconds() + 1 for value in elapsed):
        raise ValueError(f'failure count or event-time mismatch: {root}')
    checkpoints = [json.loads(line) for line in (root / 'hourly_checkpoints.jsonl').read_text().splitlines() if line.strip()]
    checkpoint_hours = [item['hour'] for item in checkpoints]
    if len(set(checkpoint_hours)) != len(checkpoint_hours):
        raise ValueError(f'duplicate checkpoint hours: {root}')
    return dict(model=row['model'], method=method, scenario_seed=row['scenario_seed'],
                run_dir=str(root), executions=n, failures=f, excluded_zero_frame_attempts=excluded,
                failure_rate=f / n, duration_seconds=(end - start).total_seconds(),
                hourly_failures=hourly_failures(elapsed, f),
                missing_checkpoint_hours=sorted(set((1, 2, 3, 4)) - set(checkpoint_hours)),
                max_checkpoint_delay_seconds=max(item['recorded_elapsed_seconds'] - item['hour'] * 3600 for item in checkpoints),
                metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest())


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def build_assets(input_csv, output_root, paper_root, repo_root, record_timezone):
    with input_csv.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['baseline'] in METHODS and row['scenario_seed'] != 'ALL']
    keys = [(row['model'], row['baseline'], row['scenario_seed']) for row in rows]
    if len(keys) != len(set(keys)) or any(row['model'] not in MODELS for row in rows):
        raise ValueError('duplicate selection or unexpected model')
    task_sets = [{row['scenario_seed'] for row in rows if row['model'] == model and row['baseline'] == method}
                 for model in MODELS for method in METHODS]
    if any(len(tasks) != 10 or tasks != task_sets[0] for tasks in task_sets):
        raise ValueError('expected the same ten tasks for all nine model/method groups')
    runs = [read_run(row, repo_root, record_timezone) for row in rows]
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run['model'], run['method'])].append(run)
    summary, curves = [], []
    for model in MODELS:
        for method in METHODS:
            selected = grouped[(model, method)]
            n, f = sum(run['executions'] for run in selected), sum(run['failures'] for run in selected)
            summary.append(dict(model=model, method=method, executions=n, failures=f, failure_rate=f / n))
            for hour in range(5):
                curves.append(dict(model=model, method=method, hour=hour,
                                   failures=sum(run['hourly_failures'][hour] for run in selected)))
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / 'summary.csv', summary)
    write_csv(output_root / 'curves.csv', curves)
    write_csv(output_root / 'selected_runs.csv', rows)
    write_csv(output_root / 'per_run.csv', [{k: v for k, v in run.items() if k not in ('hourly_failures', 'missing_checkpoint_hours')}
                                          for run in runs])
    audit = dict(input_csv=str(input_csv), input_sha256=hashlib.sha256(input_csv.read_bytes()).hexdigest(),
                 record_timezone=str(record_timezone), runs=runs, summary=summary,
                 curve_policy='Hours 1-3: saved failure-event times. Hour 4: final completed campaign; last execution may finish after the budget.',
                 execution_policy='DriveFuzz zero-frame attempts with no behavior excluded; source files unchanged.')
    (output_root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')

    tables = paper_root / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    table = [r'\begin{table}[t]', r'\centering',
             r'\caption{Failure-finding effectiveness under a four-hour budget per task. Counts are pooled over ten tasks for each model and method.}',
             r'\label{tab:rq2-effectiveness}', r'\begin{tabular}{llrrr}', r'\toprule',
             r'Model & Method & Executions & Failures & Failure rate (\%) \\', r'\midrule']
    for model, label in MODELS.items():
        for i, method in enumerate(METHODS):
            row = next(item for item in summary if item['model'] == model and item['method'] == method)
            model_cell = r'\multirow{3}{*}{' + label + '}' if i == 0 else ''
            failures, rate = f"{row['failures']:,}", f"{100 * row['failure_rate']:.2f}"
            if method == 'VLAD-Fuzz':
                failures, rate = r'\textbf{' + failures + '}', r'\textbf{' + rate + '}'
            table.append(f"{model_cell} & {method} & {row['executions']:,} & {failures} & {rate} " + r'\\')
        if model != 'bevdriver':
            table.append(r'\midrule')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (tables / 'rq2-effectiveness.tex').write_text('\n'.join(table), encoding='utf-8')

    ymax = max(500, math.ceil(max(item['failures'] for item in summary) / 500) * 500)
    plot = [r'\documentclass[tikz,border=2pt]{standalone}', r'\usepackage[scaled=0.95]{helvet}',
            r'\renewcommand{\familydefault}{\sfdefault}', r'\usepackage{pgfplots}',
            r'\usepgfplotslibrary{groupplots}', r'\pgfplotsset{compat=1.17}',
            r'\definecolor{VladGreen}{HTML}{008B6A}', r'\definecolor{DriveBlue}{HTML}{0072B2}',
            r'\definecolor{CFOrange}{HTML}{D55E00}', r'\begin{document}', r'\begin{tikzpicture}',
            r'\begin{groupplot}[group style={group size=3 by 1,horizontal sep=0.85cm},',
            r'width=5.35cm,height=4.8cm,xmin=0,xmax=4,xtick={0,1,2,3,4},',
            f'ymin=0,ymax={ymax},ytick distance=500,',
            r'xlabel={Time budget per task (h)},',
            r'tick label style={font=\fontsize{8.3}{9.5}\selectfont},',
            r'label style={font=\fontsize{8.3}{9.5}\selectfont},',
            r'title style={font=\bfseries\fontsize{9.2}{10.5}\selectfont},',
            r'axis lines=left,ymajorgrids=true,grid style={gray!25,line width=0.3pt},',
            r'legend style={draw=none,font=\fontsize{8.3}{9.5}\selectfont,legend columns=3,at={(1.75,1.27)},anchor=south},',
            r'legend cell align=left]']
    styles = {'VLAD-Fuzz': 'VladGreen,mark=*,solid', 'DriveFuzz': 'DriveBlue,mark=square*,dashed',
              'Instruction-CF': 'CFOrange,mark=triangle*,densely dotted'}
    for i, (model, label) in enumerate(MODELS.items()):
        ylabel = ',ylabel={Cumulative failed executions}' if i == 0 else ''
        plot.append(r'\nextgroupplot[title={' + f'({chr(97+i)}) {label}' + '}' + ylabel + ']')
        for method in ('VLAD-Fuzz', 'DriveFuzz', 'Instruction-CF'):
            coords = ' '.join(f"({item['hour']},{item['failures']})" for item in curves
                              if item['model'] == model and item['method'] == method)
            plot.append(r'\addplot[' + styles[method] + ',line width=1pt,mark size=2pt] coordinates {' + coords + '};')
            if i == 0:
                plot.append(r'\addlegendentry{' + method + '}')
    plot.extend([r'\end{groupplot}', r'\end{tikzpicture}', r'\end{document}', ''])
    source_dir = paper_root / 'figures' / 'src'
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / 'rq2-failure-discovery.tex').write_text('\n'.join(plot), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=Path('results/rq2/tables/rq2_all_models.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/rq2/paper'))
    parser.add_argument('--paper-root', type=Path, default=Path('paper/fse-2027'))
    parser.add_argument('--record-timezone', default='Asia/Shanghai', help='Timezone of legacy naive run timestamps.')
    args = parser.parse_args()
    summary = build_assets(args.input_csv, args.output_dir, args.paper_root, Path.cwd(), ZoneInfo(args.record_timezone))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
