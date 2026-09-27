# DriveFuzz Provenance

## Upstream

- Project: DriveFuzz
- Repository: <https://gitlab.com/s3lab-code/public/drivefuzz>
- Revision: `15bc794dba45a096bf2e8504193211a7dc30ef64`
- Upstream date: 2022-09-05
- Bundled scope: `src/` only

`README.upstream.md` preserves the upstream project description. CARLA,
Autoware, logs, and generated artifacts from the upstream repository are not
vendored because VLAD-Fuzz uses the repository-level CARLA installation and
its own VLA backends.

## VLAD-Fuzz Changes

The bundled runtime is the version used by the VLAD-Fuzz DriveFuzz baseline.
Relative to the pinned upstream source, it:

- adds a VLA target backed by the shared model registry;
- consumes VLAD-Fuzz manifests, instructions, and converted seed scenarios;
- aligns frame limits, destination thresholds, traffic-light handling, and
  failure oracles with the other evaluated methods;
- supports wall-clock budgets, repeated seed execution, and hourly metadata;
- records VLA outputs and normalized failure information;
- adds deterministic cleanup and dedicated CARLA, sensor, and CUDA failure
  handling so infrastructure failures are retried instead of scored;
- supports repository-local CARLA Python packages without `.bashrc` changes;
- optionally disables walker generation and supports headless execution.

The local integration evolved in VLAD-Fuzz commits `98b5ece`, `c3b6a68`,
`daaa02a`, and `a3d0dfe` before this clean artifact was assembled.

## License Notice

The official DriveFuzz repository does not currently declare a top-level
license for its own `src/` directory. Licenses present in its nested CARLA and
Autoware directories apply to those projects, not automatically to DriveFuzz.
This source is required for exact baseline reproduction, but maintainers must
obtain or confirm redistribution permission before publishing the artifact.
