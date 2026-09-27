# Comparison Baselines

The core VLAD-Fuzz workflow and all three VLA backends are contained in this
repository. Evaluation wrappers also exist for random testing,
instruction-counterfactual testing, ablations, and DriveFuzz.

## Built-In Workflows

These use the shared VLAD-Fuzz runtime and need no additional source checkout:

- `scripts/run_random.sh`
- `scripts/run_instruction_counterfactual.sh`
- `scripts/run_rq3.sh`
- `vladfuzz_workflows.ads_baseline_testing`

## Adapted DriveFuzz

The modified DriveFuzz runtime used in the evaluation is bundled at:

```text
baselines/drivefuzz/src/
```

Only the lightweight Python runtime and town metadata are included. DriveFuzz's
embedded CARLA and Autoware trees are unnecessary because all methods use the
same repository-level CARLA installation and VLA backends.

Run one manifest task with `scripts/run_drivefuzz.sh`, or use
`scripts/run_batch_drivefuzz.sh` for a batch. The wrappers convert VLAD-Fuzz
seeds, pass the selected VLA backend and instruction, align the execution
oracles, and write outputs under `results/rq2/runs/drivefuzz/`.

The exact upstream revision and our changes are listed in
`baselines/drivefuzz/UPSTREAM.md`. Upstream does not declare a top-level license
for its own `src/`; maintainers must confirm redistribution permission before
making the artifact public. This is a release issue, not a missing runtime
component: DriveFuzz source tests are mandatory in this repository.
