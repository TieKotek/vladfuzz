# Frozen Experiment Inputs

This directory contains the versioned inputs used by the paper. Treat these
files as immutable reference data. Regeneration scripts write to the ignored
top-level `static_scenarios/` and `test_cases/` directories instead.

## Contents

- `static_scenarios/`: 14 static CARLA regions. Each region has one scenario
  JSON file and two BEV diagnostics.
- `rq1/test_cases/`: 70 route-defined tasks, five for each static region, used
  for the RQ1 instruction-generation evaluation.
- `formal/test_cases/`: ten selected tasks used for RQ2--RQ4, with two tasks
  for each evaluated road region.
- `SHA256SUMS`: checksums for detecting accidental changes or incomplete
  transfers.

The formal corpus is not simply a byte-for-byte subset of the RQ1 corpus;
several task artifacts were finalized separately. Use
`configs/experiment_manifest.jsonl` for RQ2 and RQ4 and
`configs/rq3_manifest.jsonl` for RQ3. Both manifests deliberately select the
same ten task identifiers.

Model weights and experiment outputs are not included. Follow
`docs/MODEL_ASSETS.md` for weights, and write new runs under `results/`.
