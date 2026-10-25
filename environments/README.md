# Conda Environments

Each backend has an independent environment because their PyTorch, CUDA, and
model-library versions conflict. A user installs only the backend they need:

```bash
conda env create -f environments/lmdrive.yml
conda env create -f environments/simlingo.yml
conda env create -f environments/bevdriver.yml
```

These curated files pin backend runtime and bundled-tool dependencies, but are
not complete environment lock files. They intentionally
exclude local paths, proxy settings, editor tools, and packages left over from
unrelated experiments. Exact Linux lock files will be generated only after each
environment passes a clean-machine CARLA smoke test.
`nodefaults` prevents Conda from adding the user's default channels.

LMDrive and BEVDriver use PyTorch 2.0.1 with CUDA 11.8 wheels. SimLingo uses
PyTorch 2.8.0 with CUDA 12.8 wheels and the matching official FlashAttention
2.8.3 wheel. The SimLingo environment therefore requires an NVIDIA driver that
can run CUDA 12.x applications; R570 or newer is recommended for the fewest
compatibility restrictions. No environment requires a global `CUDA_HOME` for
installation.

For SimLingo, the Linux PyPI distributions are named `torch==2.8.0`,
`torchvision==0.23.0`, and `torchaudio==2.8.0`. Their imported module versions
include `+cu128`; that runtime suffix is not part of these distributions'
package metadata. The YAML pins distribution metadata versions rather than
the version strings printed by the imported modules.

## Dependency Validation

To test installation without replacing an existing working environment, use a
different name, for example:

```bash
conda env create -n vlad-bevdriver-yamlcheck -f environments/bevdriver.yml
conda run -n vlad-bevdriver-yamlcheck python -m pip check
```

Installation and import checks alone do not establish end-to-end reproducibility;
each backend also needs a CARLA camera and model-inference smoke test. See
[`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) for verification status.

The official Linux `decord==0.6.0` wheel has a Python 3 filename but embeds a
CPython 3.6 tag in its `WHEEL` metadata. Consequently, `pip check` can report
`decord 0.6.0 is not supported on this platform` under Python 3.9, even when
import and CPU video decoding succeed. Reinstalling the same official wheel
does not resolve this upstream metadata mismatch. Do not change Torch/CUDA or
edit the installed metadata to suppress the warning; verify decoding and the
LMDrive backend separately. Any other dependency-check failures still require
investigation.

`nuscenes-devkit` and `mmdet3d` are not dependencies of the supported SimLingo
execution path and are not included. The files retain dependencies of bundled
utilities, such as `pexpect` and `transforms3d` for the ROS agent. They omit
unused `rdp`, `easydict`, and `contexttimer`, as well as `python-magic`, which
is only needed by an upstream dataset-download utility outside this workflow.
LMDrive also needs `ipython`: its bundled LAVIS initializer imports
`IPython.display` through BLIP-Diffusion, even during driving-agent imports.
It needs `protobuf` when loading its LLaMA tokenizer; an agent import alone
does not exercise that dependency.
Its bitsandbytes backend additionally needs the pinned CUDA 11 runtime,
cuBLAS, and cuSPARSE wheels. These provide standard shared-library names used
by the existing CUDA preloader; the PyTorch wheel alone is insufficient.
They are environment-local user-space libraries, not NVIDIA driver packages.

## Verification Record

The three backend environments have passed local installation and basic
closed-loop checks. The exact test scope, host-specific conditions, and
remaining validation work are recorded in
[`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md). Local verification is not a
guarantee of identical results on another machine.
