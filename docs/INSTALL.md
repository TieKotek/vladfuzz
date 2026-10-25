# Installation

## Supported Host

VLAD-Fuzz targets x86-64 Linux with:

- an NVIDIA GPU capable of running CUDA;
- an NVIDIA driver that supports the selected PyTorch CUDA runtime;
- Conda or Mamba;
- enough disk space for CARLA and the selected model assets;
- Vulkan support for CARLA camera rendering.

The reference system uses Ubuntu 22.04 and CARLA 0.9.15. Run these checks before
creating a Python environment:

```bash
nvidia-smi
vulkaninfo --summary
```

`vulkaninfo` is provided by the distribution's Vulkan tools package. A working
CUDA compute stack alone is insufficient because CARLA also renders camera
sensors through Vulkan.

The GPU driver must support CUDA 11.8 for LMDrive/BEVDriver or CUDA 12.8 for
SimLingo; see [NVIDIA's compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
For containers, expose both compute and graphics support using
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

## CARLA Prerequisite

Prepare a working CARLA 0.9.15 packaged release following the official
installation instructions below. Import the additional
maps when the package does not already contain the towns used by your tasks.
The installation directory must contain `CarlaUE4.sh` and a `VERSION` file.

Official references:

- <https://carla.readthedocs.io/en/0.9.15/start_quickstart/>
- <https://github.com/carla-simulator/carla/releases>

Configure the local path in `.env`:

```bash
cp .env.example .env
# Edit CARLA_ROOT and any API keys needed by your workflow.
```

The repository scripts determine their root from their own file location.
Do not export `PROJECT_ROOT` and do not add repository-specific paths to
`~/.bashrc`. `.env` is the only project configuration file and is ignored by
Git. Both Python and shell entry points load the same dotenv format without
executing it as shell code. Use literal `KEY=value` settings, not shell commands.
Explicit process environment settings override `.env`; unset values use project
defaults. Command-line flags take precedence over loaded settings.
For example, `CUDA_VISIBLE_DEVICES=2 scripts/run_vladfuzz.sh` overrides a
different value in `.env` without editing the file.

The matching Python client (`carla==0.9.15`) is installed by every backend
environment. The framework uses `PythonAPI/carla/agents` from the CARLA release
for route planning, but does not load the release's often Python-3.7-specific
wheel. Do not combine a different CARLA server version with this client.

## Desktop And Headless Modes

Set `VLADFUZZ_CARLA_MODE=offscreen` in `.env` for a server without a display,
or `windowed` for desktop debugging. Off-screen rendering produces RGB camera
data; CARLA's no-rendering mode does not and cannot be used for VLA evaluation.
See the [official rendering documentation](https://carla.readthedocs.io/en/0.9.15/adv_rendering_options/)
for CARLA deployment details. Use the managed launcher in either mode:

```bash
scripts/carla_manager.sh start
scripts/carla_manager.sh stop
```

The manager supplies the rendering arguments and waits for the RPC server.

## GPU And Concurrent Runs

On a multi-GPU host, `CUDA_VISIBLE_DEVICES` selects the
model GPU; `CARLA_GRAPHICS_ADAPTER` can select CARLA's Vulkan adapter. Verify
the mapping locally because CUDA and Vulkan device numbering can differ.
`GPU_ID` is a logical CUDA index after filtering: with
`CUDA_VISIBLE_DEVICES=2`, use `GPU_ID=0`, not `GPU_ID=2`.
The doctor accepts the same logical index with `--gpu-id 0`.

Concurrent jobs also need separate CARLA RPC, streaming, secondary, and traffic
manager ports. Use `CARLA_ISOLATED=1` and RPC ports separated by at least three
(for example 2200 and 2210), distinct `CARLA_TM_PORT` values, and unique
`RUN_ID` values. Do not run the default non-isolated manager alongside other
CARLA jobs: its cleanup assumes exclusive use and can stop them. These are
resource-isolation settings, not changes to the paper's testing parameters.

## Create One Backend Environment

Choose one environment; users do not need a separate shared environment:

```bash
conda env create -f environments/lmdrive.yml
conda activate vlad-lmdrive
```

or:

```bash
conda env create -f environments/simlingo.yml
conda activate vlad-simlingo
```

or:

```bash
conda env create -f environments/bevdriver.yml
conda activate vlad-bevdriver
```

The YAMLs install the required PyTorch and CUDA user-space packages. A system
CUDA toolkit is not required unless a user deliberately compiles an extension
from source.

Conda YAMLs are the authoritative dependency specifications. Do not install a
second top-level `requirements.txt` over them; that can silently replace the
backend-specific PyTorch and CUDA stack.

## Language Services

Set the credentials and selectors in `.env`. Instruction generation and
language mutation are separate services:

| Workflow / provider | Selection | Credential |
|---|---|---|
| Instruction generation | `GENERATOR_MODEL` (see `scripts/generate_instructions.sh`) | `DASHSCOPE_API_KEY` |
| Mutation / DeepSeek (default) | `LLM_PROVIDER=deepseek`, `LLM_MODEL=deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| Mutation / Gemini | `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-2.5-flash-lite` | `GEMINI_API_KEY` |
| Mutation / Qwen | `LLM_PROVIDER=qwen`, `LLM_MODEL=qwen3-max` | `DASHSCOPE_API_KEY` |

When switching mutation providers, change both `LLM_PROVIDER` and `LLM_MODEL`;
adding a key alone does not switch services. The model names above are
configurable examples; choose one available to your account at the endpoint
used by that provider. `GENERATOR_MODEL` does not change the mutation model.
Shell launchers read these settings from `.env`; direct Python fuzzing CLIs
select the service with `--llm-provider` and `--llm-model`.
The root-only installation check below does not require a language-service key.

## Download Model Assets

After activating the selected environment:

```bash
python scripts/prepare_models.py --backend lmdrive --accept-model-licenses
```

Replace `lmdrive` with the selected backend. See `docs/MODEL_ASSETS.md` for the
official sources, BEVDriver's manual checkpoint step, and complete file tree.

## Validate The Installation

Run the doctor inside the selected backend environment:

```bash
python tools/check_environment.py --model lmdrive
```

The command checks the CARLA executable, CARLA navigation agents, Python client,
backend package availability, required model files/sizes, NVIDIA runtime,
actual Vulkan initialization with an NVIDIA device, and PyTorch CUDA
initialization. Replace `lmdrive` with the selected backend. The doctor does
not load the driving model or prove camera rendering works: complete a short
CARLA run before a long campaign. Use `prepare_models.py --verify-only` for
full task-checkpoint hash verification, which reads the complete weight files.

If the doctor reports no physical NVIDIA Vulkan device, resolve the CARLA/GPU
deployment prerequisite using the official references above before proceeding.
CUDA availability alone does not establish that CARLA can render camera data.

### Short Closed-Loop Check

After preparing model assets and passing the doctor, run one short scenario
before a long campaign. This checks weight loading, sensor input, model
inference, vehicle control, failure evaluation, and result output without
calling a language API:

```bash
scripts/carla_manager.sh start

MODEL=simlingo POP_SIZE=1 NGEN=0 MUTATION_DEPTH=0 \
  SCENARIO_DURATION_FRAMES=100 MIN_NPC_COUNT=0 MAX_NPC_COUNT=0 \
  OUTPUT_ROOT=results/validation/backend-smoke \
  scripts/run_vladfuzz.sh

scripts/carla_manager.sh stop
```

Activate the matching environment and replace `simlingo` for each backend.
Use isolated ports as described above if another CARLA job is running. The
100-frame cap can produce a normal timeout failure; successful environment
validation requires actual driving frames and `error: false` in the execution
result, not necessarily successful navigation. This root-only check does not
validate language-API access, mutation operators, or multi-generation search.

## Configuration Reference

| Variable | Required | Purpose |
|---|---|---|
| `CARLA_ROOT` | yes | Extracted CARLA 0.9.15 directory |
| `VLADFUZZ_CARLA_MODE` | yes | `offscreen` or `windowed` |
| `VLADFUZZ_MODEL_HOME` | yes | Local weight root, normally `./models` |
| `DASHSCOPE_API_KEY` | instruction generation | Qwen-compatible DashScope API |
| `DEEPSEEK_API_KEY` | default language fuzzing | DeepSeek API |
| `GEMINI_API_KEY` | optional provider | Gemini API |
| `LLM_PROVIDER` / `LLM_MODEL` | language mutation | Matching mutation provider and model ID |
| `GENERATOR_MODEL` | instruction generation | DashScope instruction-generation VLM ID |
| `HF_TOKEN` | gated downloads only | Hugging Face authentication |
| `CUDA_VISIBLE_DEVICES` | optional | Model GPU selection |
| `CARLA_GRAPHICS_ADAPTER` | optional | CARLA Vulkan adapter selection |
| `CARLA_EXECUTABLE` | optional | Custom CARLA executable or launch wrapper; normally unset |

Standard `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` variables are inherited if
the host already uses them. VLAD-Fuzz does not create or override proxy values.
