# Paper Result Data

The frozen evaluation data are available in the separate
[VLAD-Fuzz Evaluation Data repository](https://anonymous.4open.science/r/vladfuzz-data-5F73/).
They are not needed to install VLAD-Fuzz or run new experiments.

## Get the Data

Open the data repository and use its ZIP download option to obtain a local
copy. Keep that directory separate from the source checkout and from your own
experiment outputs. The directory passed to `--data-root` must contain
`DATASET.json`, `verify_data.py`, and `results/`.

The data repository README and its RQ-specific guides describe the frozen
records, their accounting conventions, and their dependencies.

| Directory | Contents | Dependency |
| --- | --- | --- |
| `results/rq1/` | 140 instructions, scene images, method mapping, and three evaluators' ratings | None |
| `results/rq2/` | 90 selected campaigns, Instruction-CF attribution, and diversity results | None |
| `results/rq3/` | 60 selected ablation campaigns and verified reanalysis summaries | RQ2 supplies the Full reference |
| `results/rq4/` | Failure-category, task-level, and execution-level summaries | RQ2 supplies the Full records |
| `results/paper_data/` | Run selections, expected statistics, file checksums, and supplementary analysis inputs | Keep with the corresponding RQ |

RQ3 and RQ4 share the Full VLAD-Fuzz campaigns in RQ2 rather than duplicating
them. The manifests select specific campaigns, not whichever run is latest.
Older Instruction-CF campaigns, the random baseline, unrelated test runs, and
model weights are not included.

## Check Integrity

From the data repository root, Python 3.9 or newer is sufficient:

```bash
python3 verify_data.py

# Check one RQ and its dependencies:
python3 verify_data.py --rq rq3
```

`DATASET.json` checks the RQ manifests and support files; each manifest checks
its experimental files by size and SHA256. The data repository's `.gitattributes`
preserves exact bytes across platforms by disabling line-ending conversion.

## Recompute Paper Statistics

Activate a project environment and run from the VLAD-Fuzz source checkout:

```bash
python tools/verify_paper_data.py \
  --data-root /path/to/vladfuzz-data \
  --output-dir results/paper-data-check

# Also recompute diversity and its scenario/instruction components:
python tools/verify_paper_data.py \
  --data-root /path/to/vladfuzz-data --rq rq2 --diversity \
  --output-dir results/paper-diversity-check
```

Output directories must not already exist. To analyze one RQ, supply
`--rq rq1`, `--rq rq2`, `--rq rq3`, or `--rq rq4`. Keep RQ2 available for
RQ3/RQ4 analysis.

These commands do not load driving models, use a GPU, call language APIs,
or start CARLA. Diversity needs NumPy and the CARLA Python API for weather
presets; it uses the included instruction embeddings without downloading an
encoder. Paths are resolved against `--data-root`, not a historical checkout.

The local data repository passed file-integrity checks and RQ1--RQ4
recomputation. Diversity components, joint scores, bootstrap intervals, and
paired comparisons matched the frozen results with zero numerical difference.
The data repository includes `reports/verification.json`.

## Accounting and Provenance

- RQ1 retains all three original rating sets. The historical descriptive report
  covers two evaluators; the release manifest and verifier include all three.
  `private/` names the method mapping used during the blinded experiment.
- RQ2 uses the four-family Instruction-CF rerun. Its scoped attribution separates
  raw positives from retained failures. DriveFuzz zero-frame attempts remain
  in the records but are excluded from the paper's execution denominator.
- RQ3 uses `summary_reanalysis`; its Full campaigns are selected from RQ2.
- RQ4 reuses those 30 Full campaigns and their 4,780 failed executions.
- Historical source audits may mention older Instruction-CF campaigns. They
  provide provenance; the per-RQ manifests define the current release selection.

Release copies normalize local checkout paths and update dependent hashes.
Experimental values, instructions, ratings, images, and embedding vectors are
unchanged. Original research records were not rewritten. Treat the frozen
data as read-only.
