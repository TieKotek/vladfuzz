# Installation

## Supported Host

The reproduction package targets x86-64 Linux with:

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

LMDrive and BEVDriver use PyTorch CUDA 11.8 wheels. SimLingo uses a CUDA 12.8
wheel. R570 is the corresponding CUDA 12.8 driver branch; NVIDIA also documents
minor-version compatibility for CUDA 12.x on Linux drivers 525.60.13 or newer,
with feature limitations. We recommend R570 or newer for SimLingo. In all
cases, `nvidia-smi` showing a GPU is not sufficient; the post-install check also
verifies PyTorch CUDA initialization.

## Install CARLA 0.9.15

Download and extract the CARLA 0.9.15 packaged release. Import the additional
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
Git.

The matching Python client (`carla==0.9.15`) is installed by every backend
environment. The framework uses `PythonAPI/carla/agents` from the CARLA release
for route planning, but does not load the release's often Python-3.7-specific
wheel. Do not combine a different CARLA server version with this client.

## Desktop And Headless Modes

For interactive desktop debugging, CARLA may open its spectator window:

```bash
$CARLA_ROOT/CarlaUE4.sh
```

For servers without a display, use off-screen rendering:

```bash
$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -nosound
```

Off-screen rendering still produces RGB camera data. CARLA's no-rendering mode
does not, so it cannot be used for VLA evaluation. CARLA 0.9.15 uses Vulkan and
supports `-RenderOffScreen` without an emulated X display.

Set the mode once in `.env` and use the managed launcher in both cases:

```bash
# VLADFUZZ_CARLA_MODE=offscreen  # headless server
# VLADFUZZ_CARLA_MODE=windowed   # desktop debugging
scripts/carla_manager.sh start
```

The manager adds the corresponding Vulkan and off-screen arguments and waits
for the RPC server. On a multi-GPU host, `CUDA_VISIBLE_DEVICES` selects the
model GPU; `CARLA_GRAPHICS_ADAPTER` can select CARLA's Vulkan adapter. Verify
the mapping locally because CUDA and Vulkan device numbering can differ.

For bare-metal execution, `NVIDIA_DRIVER_CAPABILITIES` is not a required shell
variable. It belongs to NVIDIA Container Toolkit. A future container profile
must request at least `compute,utility,graphics`; `display` is additionally
needed only for a visible X11 or Wayland window.

On bare metal, install the complete proprietary NVIDIA driver, including its
Vulkan ICD. The same driver provides compute and graphics support; exporting
`NVIDIA_DRIVER_CAPABILITIES=all` in `.bashrc` does not enable missing host
driver components. If `nvidia-smi` succeeds but `vulkaninfo --summary` fails,
fix the driver/Vulkan installation before debugging Python packages.

References:

- <https://carla.readthedocs.io/en/latest/adv_rendering_options/>
- <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html>
- <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>

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

The PyTorch wheels include their CUDA runtime. A system CUDA toolkit is not
required unless a user deliberately compiles an extension from source.

Conda YAMLs are the authoritative dependency specifications. Do not install a
second top-level `requirements.txt` over them; that can silently replace the
backend-specific PyTorch and CUDA stack.

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
backend imports, required model paths, NVIDIA runtime, Vulkan tools, and PyTorch
CUDA initialization. Replace `lmdrive` with the selected backend.

## Configuration Reference

| Variable | Required | Purpose |
|---|---|---|
| `CARLA_ROOT` | yes | Extracted CARLA 0.9.15 directory |
| `VLADFUZZ_CARLA_MODE` | yes | `offscreen` or `windowed` |
| `VLADFUZZ_MODEL_HOME` | yes | Local weight root, normally `./models` |
| `DASHSCOPE_API_KEY` | instruction generation | Qwen-compatible DashScope API |
| `DEEPSEEK_API_KEY` | default language fuzzing | DeepSeek API |
| `GEMINI_API_KEY` | optional provider | Gemini API |
| `HF_TOKEN` | gated downloads only | Hugging Face authentication |
| `CUDA_VISIBLE_DEVICES` | optional | Model GPU selection |
| `CARLA_GRAPHICS_ADAPTER` | optional | CARLA Vulkan adapter selection |

Standard `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` variables are inherited if
the host already uses them. VLAD-Fuzz does not create or override proxy values.
