# Workflow

This guide describes both regeneration from CARLA and execution from the
frozen paper inputs. It does not require precomputed paper results.

## 1. Validate The Host And Backend

Activate exactly one backend environment and run the doctor:

```bash
conda activate vlad-simlingo
python tools/check_environment.py --model simlingo
```

Resolve every reported error before running CARLA experiments. Warnings about
optional utilities do not block execution.

## 2. Start CARLA

The manager reads `.env` and chooses off-screen or windowed rendering from
`VLADFUZZ_CARLA_MODE`:

```bash
scripts/carla_manager.sh start
scripts/carla_manager.sh wait-ready
```

Stop the managed process when finished:

```bash
scripts/carla_manager.sh stop
```

Batch wrappers use the same manager and can restart CARLA after infrastructure
failures. Do not use CARLA's no-rendering mode because camera observations are
required.

## 3. Generate Static Scenarios

```bash
SCENARIO_ID=town01_t_junction_1 scripts/generate_static_scenarios.sh
```

See `docs/STATIC_SCENARIOS.md` for the schema and custom regions.

## 4. Generate Dynamic Seeds

```bash
MODEL=simlingo \
STATIC_SCENARIO=static_scenarios/town01_t_junction_1.json \
SEEDS_PER_SCENARIO=5 \
scripts/generate_seeds.sh
```

Without `DYNAMIC_FILTER=1`, this stage uses CARLA route and scenario utilities
but does not load the VLA model. Each selected seed is written under
`test_cases/<region-id>/seed_<n>/` with its dynamic scenario JSON and image.
For isolated CARLA runs, this stage uses the same `CARLA_PORT` and
`CARLA_TM_PORT` as the managed server. Set `CARLA_ISOLATED=1` and the connection
settings in `.env` so all stages use the same server.

## 5. Generate Instructions

Set `DASHSCOPE_API_KEY` in `.env`, then run:

```bash
SEEDS_ROOT=test_cases MODE=route_prior NUM_COMMANDS=1 \
  scripts/generate_instructions.sh
```

Use `MODE=both` to compare evidence-guided and direct VLM generation.
The existing CLI value `route_prior` writes `command_prior.json` inside each
seed directory; these stable filenames do not name separate methods.
Use `GENERATOR_MODEL` to override the instruction-generation VLM independently
of the language-mutation provider/model used during fuzzing.

## 6. Build The Manifest

```bash
scripts/build_manifest.sh
```

The resulting `test_cases/manifest.jsonl` links each static scenario, dynamic
seed, image, route metadata, and generated instruction. It is ignored by Git
because its paths correspond to local outputs.

## 7. Run VLAD-Fuzz

Configure the mutation provider, model, and corresponding key as described in
[`docs/INSTALL.md`](INSTALL.md#language-services), then choose the input corpus.

### Newly Generated Inputs

To fuzz the inputs generated in steps 3--6, explicitly select their root and
manifest. Select a `MANIFEST_ID` from `test_cases/manifest.jsonl`:

```bash
MODEL=simlingo \
SELECTED_ROOT=test_cases STATIC_ROOT=static_scenarios \
MANIFEST=test_cases/manifest.jsonl \
MANIFEST_ID=town01_t_junction_1/seed_1 \
TIME_BUDGET_MINUTES=10 OUTPUT_ROOT=results/custom/vladfuzz \
scripts/run_vladfuzz.sh
```

Both `SELECTED_ROOT` and `MANIFEST` are needed: the former locates the generated
`command_prior.json`, while the latter selects its scenario and route. The
separate output root prevents custom runs from mixing with formal-task results.

### Bundled Formal Inputs

The repository includes the ten formal tasks used for RQ2--RQ4 under
`artifact_inputs/formal/test_cases/`. Their canonical manifest is
`configs/experiment_manifest.jsonl`; `configs/rq3_manifest.jsonl` fixes the
same task set for the ablation study. The batch and single-run wrappers use
these frozen inputs by default.

The 70-task generation corpus used for RQ1 is kept separately under
`artifact_inputs/rq1/test_cases/`. Do not substitute one corpus for the other:
some formal task instances differ from the corresponding RQ1 generation
inputs.

To run one of those bundled tasks using the default input paths:

```bash
MODEL=simlingo \
MANIFEST_ID=town01_t_junction_1/seed_1 \
TIME_BUDGET_MINUTES=240 \
scripts/run_vladfuzz.sh
```

Bundled-task results default to `results/rq2/runs/vladfuzz/`. Run metadata records model,
inputs, random seed, method, and parameters. New metadata also records the Git
revision and dirty state, package versions, model-manifest checksum, and an
allowlist of runtime path/GPU settings, never API keys. Hourly checkpoint files
record campaign progress. Batch wrappers retry infrastructure failures; they
do not restore an optimizer's exact in-memory search state from these files.

## One-Command Small Run

After installation, `.env` setup, and model preparation, this command performs
steps 2 through 7 for one region and one seed. It uses
`test_cases/quickstart/` so existing test cases are not modified, and writes
results to `results/quickstart/vladfuzz/` rather than the formal-task directory:

```bash
MODEL=simlingo SCENARIO_ID=town01_t_junction_1 \
  TIME_BUDGET_MINUTES=10 scripts/run_pipeline.sh
```

Override `TEST_CASES_ROOT` and `OUTPUT_ROOT` to use different working directories.

The default manager assumes exclusive use of CARLA on the machine and may
stop other CARLA processes. Use `CARLA_ISOLATED=1` with distinct ports for
concurrent jobs, or set `MANAGE_CARLA=0` to use an already running server.

## Batch And Evaluation Workflows

`scripts/run_batch_vladfuzz.sh` runs manifest rows in sequence with CARLA
recovery. `run_random.sh`, `run_instruction_counterfactual.sh`, and
`run_rq3.sh` implement comparison or ablation configurations. The adapted
DriveFuzz runtime is bundled under `baselines/drivefuzz/`; use
`scripts/run_drivefuzz.sh` or `scripts/run_batch_drivefuzz.sh` to run it with
the same VLA backends and execution criteria.

Each complete Instruction-CF scenario batch contains 32 variants: eight from
each of paraphrase, ambiguity, noise, and misleading. Some variants invalidate
the task. Before reporting failures, follow `evaluation/rq2/README.md` to run
the scoped semantic-change attribution rules and configure all downstream
summaries to consume the same audit. The manual annotation path is documented
separately; it is not a substitute for the scoped paper rules.

RQ-specific programs under `evaluation/` consume generated `results/rqN/`
trees. RQ2 keeps raw method executions under `results/rq2/runs/` and derived
summaries, tables, paper assets, and diversity outputs in sibling directories.
RQ4 reads the frozen VLAD-Fuzz selection from `results/rq2/paper/` rather than
running or copying a separate campaign. Paper figures and result archives are
intentionally deferred until the data portion of the artifact is assembled.
