# Conda Environments

Each backend has an independent environment because their PyTorch, CUDA, and
model-library versions conflict. A user installs only the backend they need:

```bash
conda env create -f environments/lmdrive.yml
conda env create -f environments/simlingo.yml
conda env create -f environments/bevdriver.yml
```

These curated files describe direct runtime dependencies. They intentionally
exclude local paths, proxy settings, editor tools, and packages left over from
unrelated experiments. Exact Linux lock files will be generated only after each
environment passes a clean-machine CARLA smoke test.

LMDrive and BEVDriver use PyTorch 2.0.1 with CUDA 11.8 wheels. SimLingo uses
PyTorch 2.8.0 with CUDA 12.8 wheels and the matching official FlashAttention
2.8.3 wheel. The SimLingo environment therefore requires an NVIDIA driver that
can run CUDA 12.x applications; R570 or newer is recommended for the fewest
compatibility restrictions. No environment requires a global `CUDA_HOME` for
installation.
