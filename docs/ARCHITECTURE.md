# Architecture

VLAD-Fuzz separates model-independent testing logic from model-specific
inference code. Selecting a backend must not fork the scenario, oracle, or
search implementation.

## Core Modules

### `scenario/`

Owns closed-loop CARLA execution. `carla_scenario.py` loads static and dynamic
scenario JSON, creates actors, runs the selected VLA agent, and records runtime
state. `metrics.py` and `camera.py` provide shared measurements and sensors.

### `scenario_construction/`

Builds test inputs in three stages:

1. `scenario_constructor.py` turns a rectangular CARLA map region into a
   static scenario containing verified ego spawn and destination candidates.
2. `seed_generator.py` chooses route-diverse start/destination pairs and emits
   executable dynamic seed scenarios plus ego-view images.
3. `generate_instructions.py` uses the image and route description to produce
   natural-language instructions. `build_instruction_manifest.py` links these
   artifacts into stable experiment rows.

### `vladfuzz_runtime/`

Contains cross-cutting runtime services:

- `model_registry.py`: backend names, entry points, and configuration paths
- `model_assets.py`: repository-local model paths
- `env_patches.py`: deterministic Python paths for CARLA and vendored modules
- `oracle.py`: failure checks
- `mutation_operators.py`: language mutation operators
- `experiment_manifest.py`: manifest loading and input resolution
- `run_metadata.py` and `hourly_checkpoints.py`: reproducibility records
- `infrastructure.py` and `carla_connection.py`: process and connection recovery

### `vladfuzz_workflows/`

Contains executable workflows. `run_scenario.py` evaluates one test case,
`local_fuzzer.py` performs language-only fuzzing, and `nsga_optimization.py`
runs the full coordinated scenario and language search. The remaining modules
implement ablations and comparison methods used by the evaluation.

## Backends

`backends/lmdrive`, `backends/simlingo`, and `backends/bevdriver` contain the
model-specific agents and upstream-derived source. The shared runtime resolves
them through `vladfuzz_runtime/model_registry.py`. Weights are never stored in
these packages; every backend resolves them from `VLADFUZZ_MODEL_HOME`.

Each backend has a dedicated Conda environment because its CUDA, PyTorch, and
model-library requirements differ. The CARLA scenario format and workflow
commands remain the same across environments.

## Baselines

`baselines/drivefuzz/` contains the lightweight DriveFuzz runtime adapted for
VLA agents and the matched evaluation protocol. It does not duplicate CARLA or
Autoware. The pinned upstream revision and local changes are recorded in
`baselines/drivefuzz/UPSTREAM.md`.

## Frozen Inputs

`artifact_inputs/` contains the immutable static scenarios and dynamic test
cases used in the paper. The RQ1 corpus and the ten-task formal corpus are kept
separate because they serve different evaluations and are not interchangeable.
Newly generated inputs belong in the ignored `static_scenarios/` and
`test_cases/` working directories.

## Evaluation And Results

Research-question-specific analysis belongs under `evaluation/rq1/` through
`evaluation/rq4/`; reusable repository tooling remains under `tools/`.
Generated data follows the same ownership boundary under `results/rqN/`.
RQ2 stores each method's raw executions under `results/rq2/runs/` and its
derived artifacts under `summary/`, `tables/`, `paper/`, and `diversity/`.
RQ3 stores ablation runs under `results/rq3/runs/`. RQ4 analyzes the frozen
RQ2 VLAD-Fuzz selection and stores only its derived summary, avoiding duplicate
raw executions.

## Vendored CARLA Code

`leaderboard/`, `scenario_runner/`, and `baselines/drivefuzz/` are pinned
third-party dependencies. Changes to these directories should be limited to
integration or compatibility fixes and documented in their provenance files.

## Data Flow

```text
region JSONL
  -> static scenario JSON + BEV diagnostics
  -> dynamic seed JSON + ego-view image
  -> route-prior instruction JSON
  -> experiment manifest row
  -> selected backend + VLAD-Fuzz search
  -> execution metadata, failures, and checkpoints
```

Frozen paper inputs are versioned under `artifact_inputs/`. Newly generated
data and experiment results remain outside Git so exploratory runs cannot
modify the reference inputs.
