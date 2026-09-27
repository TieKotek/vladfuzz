# RQ3 Ablation Experiments

RQ3 compares the complete VLAD-Fuzz framework with component ablations under the same four-hour wall-clock budget:

- `global_only`: retain evolutionary scenario search and execute only the original route-prior instruction for each scenario candidate.
- `local_only`: keep the selected dynamic seed scenario unchanged and repeatedly run VLAD-Fuzz language mutation trees.
- `random_scenario_local`: repeatedly sample independent scenarios with 0--3 NPCs and execute a complete language mutation tree on each one, without evolutionary selection, crossover, or offspring generation.

The first two configurations are the original ablations. `random_scenario_local` is an additional candidate Local-only design retained separately so its results can be compared with the fixed-scenario `local_only` before selecting the paper configuration.

All modes use the same selected seeds, route-prior instructions, Oracle configuration, 500-frame execution limit, and success distance as the Full configuration. The backend model is loaded once per campaign.

## Batch execution

The batch runner rebuilds `configs/rq3_manifest.jsonl` from `test_cases_selected`, manages CARLA, retries infrastructure failures, and archives invalid attempts outside the valid model result tree.

```bash
CONFIG=global_only           MODEL=simlingo ./scripts/run_batch_rq3.sh
CONFIG=local_only            MODEL=simlingo ./scripts/run_batch_rq3.sh
CONFIG=random_scenario_local MODEL=simlingo ./scripts/run_batch_rq3.sh
```

Run the same commands with `MODEL=lmdrive` and `MODEL=bevdriver` in their corresponding Conda environments. `GPU_ID`, `SELECTED_ROOT`, `CARLA_ROOT`, `TIME_BUDGET_MINUTES`, and other settings can be overridden as environment variables. The formal default is 240 minutes per seed.

Valid outputs are stored under:

```text
results/rq3/runs/global_only/<model>/
results/rq3/runs/local_only/<model>/
results/rq3/runs/random_scenario_local/<model>/
```

Infrastructure-failed attempts are moved below the corresponding `infrastructure_failures/` directory and are excluded from analysis.

## Pilot execution

A short pilot over a temporary selected-seed directory can be run by overriding the input root and budget:

```bash
CONFIG=global_only           MODEL=simlingo SELECTED_ROOT=tmp_selected TIME_BUDGET_MINUTES=5 ./scripts/run_batch_rq3.sh
CONFIG=local_only            MODEL=simlingo SELECTED_ROOT=tmp_selected TIME_BUDGET_MINUTES=5 ./scripts/run_batch_rq3.sh
CONFIG=random_scenario_local MODEL=simlingo SELECTED_ROOT=tmp_selected TIME_BUDGET_MINUTES=5 ./scripts/run_batch_rq3.sh
```

The single-seed `scripts/run_rq3.sh` entry does not start CARLA itself; use the batch entry for managed execution.

## Result analysis

Build the RQ3 tables after the campaigns finish:

```bash
./scripts/summarize_rq3.sh
```

The command selects the latest valid run for each model, seed, and available configuration listed in `configs/rq3_manifest.jsonl`. It writes per-seed, aggregate, failure-category, hourly, paired-statistics, and Markdown tables to `results/rq3/summary/`. The optional `random_scenario_local` configuration is included automatically once its result directory exists. Failure categories use multi-label counting, so one failed execution may contribute to multiple categories without changing the total failure count.

Archived experiment paths can be supplied explicitly:

```bash
RESULTS_ROOT=/path/to/results \
MANIFEST=/path/to/rq3_manifest.jsonl \
OUTPUT_DIR=/path/to/summary \
./scripts/summarize_rq3.sh
```

Review `warnings.csv` for missing configurations, malformed metadata, and older duplicate runs excluded from the analysis before using the tables in the paper.
