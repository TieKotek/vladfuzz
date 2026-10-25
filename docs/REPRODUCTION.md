# Reproduction Settings and Verification

This page separates software checks from end-to-end experimental validation.
Passing unit tests does not demonstrate that a fresh machine can render CARLA
cameras or load every driving model. No new long-running experiment is needed
to verify repository structure or reporting code.

## Configuration Contract

Use the selected backend's Conda environment and a repository-local `.env`.
Configuration precedence is explicit command-line arguments, existing process
environment variables, `.env`, then launcher defaults. The shell entry points
load `.env` before assigning defaults, using the same dotenv parser as Python.
Do not rely on shell commands or `.bashrc` edits to configure this project.

The reference model asset paths, pinned download revisions, required files,
checkpoint sizes, and hashes are defined in `configs/model-assets.json`.
Download preparation and environment checks share one validation contract.
The environment checker checks files and sizes; `prepare_models.py
--verify-only` additionally hashes task-specific checkpoints. Directory-based
base models are downloaded from pinned revisions, not exhaustively hashed.

## Paper Configuration

The supported launchers retain the experiment defaults; these are not changed
by repository cleanup. Inspect the actual launcher before running a campaign.

| Setting | Reference value / location |
|---|---|
| Simulator | CARLA 0.9.15, 20 frames per simulated second |
| Formal navigation tasks | Ten entries in `configs/experiment_manifest.jsonl` |
| Ablation tasks | Same identifiers in `configs/rq3_manifest.jsonl` |
| Budget | 240 minutes per method, model, and task |
| Execution limit | 500 frames |
| Destination threshold | 5 m |
| Full search population / mutation depth | 8 / 2 |
| Crossover / mutation probability | 0.7 / 1.0 |
| VLAD-Fuzz / Instruction-CF NPC range | 0 to 3 |
| Driving campaign random seed default | 0 |
| Instruction-CF batch | Eight variants per family, four families |
| Language mutation provider / model | DeepSeek / `deepseek-v4-flash` |
| RQ2 diversity | Equal weights, cap 50, 1,000 resamples, seed 2026 |

API availability and stochastic outputs can change. Record any provider/model
override and preserve generated instructions rather than assuming exact text
regeneration. Match actual recorded settings when comparing campaigns.
DriveFuzz retains its own mutation mechanisms and traffic/environment sampling;
do not impose the VLAD-Fuzz NPC limits on it.

The code-facing ablation names `global_only` and `local_only` correspond to the
paper's Scenario-only and Instruction-only configurations. No search operators
or failure checks are renamed or replaced by documentation changes.

New runs written through the shared metadata writer include Git commit/dirty
state, Python and selected package versions, the model-manifest hash, and an
allowlist of model-path and GPU settings. Credentials are not recorded.
Existing run metadata is not retroactively rewritten. For an immutable result,
preserve the exact source release, resolved run configuration, and derived
audit manifests alongside its raw logs.

## Verification Status

The repository checks performed during this cleanup are recorded here; they
are not claims about an independent clean-machine installation.

| Check | Status |
|---|---|
| Frozen inputs, manifests, and checksum inventory | Covered by repository tests |
| Configuration precedence and safe dotenv parsing | Regression tests added |
| Asset requirements, alternate BERT weights, metadata alias | Regression tests added |
| Vulkan initialization and software-renderer rejection | Regression tests added |
| Scoped audit consumption and stale-input rejection | Regression tests added |
| Configuration and generated-input handoff regressions | 227 tests passed (148 core, 51 evaluation, 28 RQ1) after the fixes described below |
| Isolated dynamic-seed generation | Actual generation passed on RPC 2280 / TM 8280, including route metadata and a nonblank 1920x1080 camera image |
| Repository and evaluation suites in existing SimLingo environment | 220 tests passed (141 core, 51 evaluation, 28 RQ1) |
| Local installed model assets | All selected assets passed verification |
| Local graphics runtime | Default container libraries are incompatible; local matching NVIDIA 580 graphics wrapper works; Vulkan detection corrected as described below |
| Fresh BEVDriver environment on the existing host | YAML installation, dependency check, CUDA, 220 tests, and 100-frame closed-loop run passed |
| Fresh LMDrive environment on the existing host | Missing dependencies including tokenizer protobuf added; CUDA, decoding, 220 tests, and 100-frame closed-loop run passed; known decord metadata warning remains |
| Fresh SimLingo environment on the existing host | Offline installation, `pip check`, CUDA/FlashAttention GPU computation, 220 tests, and 96-frame closed-loop run passed |
| Three environments created from YAML on a clean machine | **Pending** |
| Root-only CARLA/inference smoke test in the three fresh environments | Passed on this host with the local graphics override; not a full campaign or language-mutation test |

The existing three-model Instruction-CF records were independently reprocessed
into temporary analysis outputs without editing raw runs. The scoped audit
reproduced 848 / 231 / 99 false positives and 636 / 1,094 / 1,050 retained
failures for SimLingo / LMDrive / BEVDriver, respectively. Both the shared
accounting reader and the table builder consumed the same attribution.
These records are local verification evidence, not bundled result archives.

The initial inspection found that LMDrive and BEVDriver lacked `python-dotenv`,
although all three YAMLs declare it. On 2026-10-25, version 1.1.1 was installed
in both existing environments without changing other dependencies; each passed
all 13 configuration/asset-check regression tests. This is not a clean-install
validation of the complete specifications. Local `pip check` also reported
a SimLingo extra-package conflict (nuscenes-devkit versus Shapely) and an
LMDrive decord platform warning; BEVDriver's dependency check passed. Resolve
and retest in separate environments rather than changing working experiment
environments in place. The SimLingo extra package is not declared in its YAML.
Reinstalling the official Linux decord 0.6.0 wheel reproduced the same platform
warning: its filename and embedded compatibility tag disagree. Import and a
five-frame CPU video-decoding check passed; this does not replace a full
LMDrive inference test. See `environments/README.md` for the known warning.

On 2026-10-25, isolated `vlad-bevdriver-yamlcheck` and
`vlad-lmdrive-yamlcheck` environments were created without cloning the working
environments. LMDrive's initial agent import exposed missing `ipython` and
CUDA 11 runtime/cuBLAS/cuSPARSE packages required by the bundled LAVIS and
bitsandbytes imports. These were declared in the YAML and installed via
`conda env update`; the subsequent import passed. This is fresh-environment
validation on the existing host, not validation on a clean machine. The later
closed-loop checks are described below.
The existing experiment environments were not modified during these checks.

The separate `vlad-simlingo-yamlcheck` environment subsequently completed its
installation using complete cached wheel archives after online attempts failed
on proxy connectivity. No installed environment was cloned. Its requirements
were read from the YAML, with the FlashAttention URL replaced by the matching
local wheel. The three PyTorch-family distribution pins were corrected to
`2.8.0`, `0.23.0`, and `2.8.0`, matching the original environment's package
metadata; those official PyPI wheels still report `+cu128` in their imported
module versions. Their SHA256 hashes matched official PyPI, and CUDA 12.8 and
FlashAttention 2.8.3 passed an actual GPU attention-computation check. All 220
tests passed. The initial Vulkan check was later found to accept an NVIDIA
layer name without a physical NVIDIA device; the checker and regression tests
have been corrected. This does not establish an online YAML installation on
a clean machine. The later real CARLA runs are described below.

### Closed-Loop Validation

All three new environments subsequently loaded real model weights and ran one
root-only test through `vladfuzz_workflows.nsga_optimization`, using the bundled
`town01_t_junction_1/seed_1` task. Each run used a 100-frame cap, no NPCs,
population size one, zero subsequent generations, and mutation depth zero.
There were no language API calls. Independent CARLA ports and output directories
kept these checks separate from formal experiments.

| Model | Executed Frames | Runtime Error | Navigation Outcome |
|---|---:|---|---|
| SimLingo | 96 | false | Destination threshold reached |
| BEVDriver | 100 | false | Normal timeout oracle result at the short frame cap |
| LMDrive | 100 | false | Normal timeout oracle result at the short frame cap |

All runs recorded vehicle motion, fitness, oracle results, and run metadata;
the timeout cases also produced failure-case files. LMDrive's first real weight
load exposed a missing tokenizer dependency, `protobuf==6.33.6`, which was added
to its YAML and installed before the successful rerun. Import-only checks had
not exercised that path.

The default container graphics libraries were NVIDIA 535 while the host driver
was 580.178.04; default Vulkan discovery listed only the llvmpipe software
device and CARLA crashed before model setup. These successful runs used the
existing local `.cache/nvidia-580.178.04/carla-with-vulkan.sh` wrapper via
`CARLA_EXECUTABLE`. That ignored local cache is not part of a clean checkout,
and the host/container graphics configuration was not repaired globally.
Normal installations need matching NVIDIA graphics libraries exposed by the
host driver/container runtime. The doctor now checks physical-device fields,
not arbitrary occurrences of the word NVIDIA in the layer list.

For subsequent runs on this host only, the same graphics overrides are saved
in the Git-ignored `.cache/local-runtime.sh` profile:

```bash
source .cache/local-runtime.sh
```

The profile checks the current host driver before setting the local graphics
paths for the current shell. It does not edit `.bashrc`, replace system
libraries, or change public launcher defaults. It is not included in a clean
checkout and is not a prerequisite for other users.

Local evidence is under `results/validation/env-smoke-20261025-v2/`, with a
`summary.json` and per-model execution results. This validates basic closed-loop
execution only, not language mutations, multi-generation search, all tasks,
long-running stability, or reproduction of the paper's aggregate results.

## Verification Commands

The subsequent configuration fixes connect dynamic-seed generation through
the shared CARLA connection resolver, explicitly document the generated-input
manifest/root pair, and keep quickstart outputs outside formal experiment
results. The local-language smoke launcher now forwards the selected provider
and model, and the instruction-generation Python entry point loads project
settings. Regression tests exercise these contracts without language API calls.

The active LAVIS cache default is now repository-relative (`.cache/lavis`
under the bundled LMDrive root), rather than an upstream absolute path.
Imported LMDrive registry state was checked to resolve inside this checkout.
Unmodified upstream training/example configurations are outside the supported
closed-loop workflows; their sample paths are not configuration prerequisites.
Machine-specific graphics profiles remain ignored local files, not defaults
required by the public workflows.

Activate one complete backend environment, then run sequentially:

```bash
python -m pip check
scripts/validate_repository.sh
python scripts/prepare_models.py --backend simlingo --verify-only
python tools/check_environment.py --model simlingo --gpu-id 0
```

Replace the backend as appropriate. Run repository validations one at a time:
process-management tests use fake CARLA executables and should not compete for
the same host process namespace. They do not run a CARLA simulation.

For an end-to-end smoke test on a prepared host, use the README's short run,
with a separate output directory and available GPU/ports. This invokes CARLA,
loads weights, and may call the configured language API; it is an explicit
user-run acceptance step, not part of unit testing. A fresh installation must
pass that step before claiming full backend reproducibility.

## Release Gates

- Choose the project-level license and confirm DriveFuzz redistribution rights.
- Record exact upstream source revisions for all vendored backends.
- Finalize citation metadata without exposing identities in an anonymous release.
- Commit the complete source, tests, documentation, and frozen inputs; no keys,
  model weights, local caches, temporary files, or generated results.
- Validate the release from a clean checkout/environment and tag that commit.
- Publish original-result archives separately with checksums, explicit run
  selections, and audit records if exact paper-table verification is offered.

These items remain pending until evidence or authorization is available. A
source-only checkout supports new experiments but does not by itself contain
all archived evidence needed to regenerate the paper's exact reported numbers.
