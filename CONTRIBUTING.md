# Contributing

## Scope

Shared CARLA execution and search behavior belongs in `scenario/`,
`scenario_construction/`, `vladfuzz_runtime/`, or `vladfuzz_workflows/`.
Model-specific inference and compatibility code belongs under its corresponding
`backends/<name>/` package and must be registered in
`vladfuzz_runtime/model_registry.py`.

Comparison-method source belongs under `baselines/<name>/`. Preserve its
upstream attribution and document local changes in an adjacent `UPSTREAM.md`.

Do not duplicate the framework for each backend or add model conditionals
throughout shared modules. Keep third-party compatibility changes narrow and
record them in the relevant `UPSTREAM.md` or `THIRD_PARTY.md`.

## Repository Hygiene

- Never commit model weights, API keys, CARLA binaries, generated test cases,
  result directories, caches, or machine-specific absolute paths.
- Add new local settings to `.env.example` without real values.
- Do not require edits to `~/.bashrc`.
- Use standard proxy environment variables only; do not add project-specific
  proxy addresses or defaults.
- New backends require a dedicated Conda YAML and model-asset manifest entries.

## Validation

Activate any complete backend environment and run:

```bash
scripts/validate_repository.sh
```

For a narrow Python change, at minimum run `py_compile` on touched modules and
the relevant unit test. CARLA or model execution changes also require an
end-to-end smoke run with the affected backend.

## Public Release Checklist

Before publishing a release, the maintainers must:

- select the project-level source license;
- finalize `CITATION.cff` with author and paper metadata;
- replace every pending backend provenance note with an exact upstream revision;
- confirm redistribution permission for the adapted DriveFuzz source;
- verify model download URLs, revisions, sizes, and hashes;
- create each Conda environment from scratch on a clean machine;
- run one desktop or headless CARLA smoke test for every backend;
- audit the repository for weights, secrets, private paths, and generated data.
