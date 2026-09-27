# Model Assets

VLAD-Fuzz does not redistribute third-party model weights. The preparation tool
downloads assets from the model authors' official releases at revisions pinned
in `configs/model-assets.json`, then verifies the task-specific checkpoints.

## Preparation

Install `huggingface_hub`, review the licenses on the linked model pages, and
run one command for the backend you intend to evaluate:

```bash
python scripts/prepare_models.py --backend lmdrive --accept-model-licenses
python scripts/prepare_models.py --backend simlingo --accept-model-licenses
python scripts/prepare_models.py --backend bevdriver --accept-model-licenses
```

Set `HF_TOKEN` when a Hugging Face repository requires authentication. The
tool honors standard proxy variables but never configures a proxy itself.

## Official Sources

### LMDrive

- Base VLM: <https://huggingface.co/liuhaotian/llava-v1.5-7b>
- LMDrive checkpoint: <https://huggingface.co/OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0>
- Vision encoder: <https://huggingface.co/OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0>
- Upstream code: <https://github.com/opendilab/LMDrive>

### SimLingo

- SimLingo checkpoint: <https://huggingface.co/RenzKa/simlingo>
- InternVL2-1B: <https://huggingface.co/OpenGVLab/InternVL2-1B>
- Upstream code: <https://github.com/RenzKa/simlingo>

### BEVDriver

- Base LLM: <https://huggingface.co/huggyllama/llama-7b>
- Main model: <https://syncandshare.lrz.de/getlink/fiWRzThZRF4xY6DN2Ets7/Main%20Model%20Llama-7b>
- BEV encoder: <https://syncandshare.lrz.de/getlink/fijcZ1H9GEXafEyQjBKUf/BEV%20Encoder%20with%20Traffic%20Light%20Loss>
- Upstream code and checkpoint table: <https://github.com/Intelligent-Vehicles-Lab-HM/BEVDriver>

The LRZ links require manual download. Rename the main-model download to
`bevdriver-model.pth` and the encoder download to `bev-encoder.pth`. This avoids
the ambiguous historical filenames used in the research workspace.

## Expected Tree

```text
models/
├── lmdrive/
│   ├── llava-v1.5-7b/
│   │   ├── config.json
│   │   ├── pytorch_model-00001-of-00002.bin
│   │   ├── pytorch_model-00002-of-00002.bin
│   │   └── tokenizer.model
│   ├── llava-v1.5-checkpoint.pth
│   └── vision-encoder-r50.pth.tar
├── simlingo/
│   ├── InternVL2-1B/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer_config.json
│   └── pytorch_model.pt
├── bevdriver/
│   ├── llama-7b/
│   │   ├── config.json
│   │   ├── model-00001-of-00002.safetensors
│   │   ├── model-00002-of-00002.safetensors
│   │   └── tokenizer.model
│   ├── bevdriver-model.pth
│   └── bev-encoder.pth
└── shared/
    └── bert-base-uncased/
        ├── config.json
        ├── model.safetensors
        └── vocab.txt
```

## Validation

Validate an existing directory without downloading anything:

```bash
python scripts/prepare_models.py --backend lmdrive --verify-only
python scripts/prepare_models.py --backend simlingo --verify-only
python scripts/prepare_models.py --backend bevdriver --verify-only
```

The known task-specific checkpoint hashes are:

| Asset | Size (bytes) | SHA-256 |
|---|---:|---|
| LMDrive checkpoint | 946473021 | `1d3c934a4cc44a35ee92ab8b26b3578d151e39a3f21804f41b12ff31cbff0091` |
| LMDrive vision encoder | 376584921 | `718c7f1acae1607dc35355b1782685e29c98d65233de83fa8a6309ed6df6076a` |
| SimLingo checkpoint | 2569679322 | `ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28` |
| BEVDriver full model | 3058237670 | `bc7b9fe787857b4e7d90bc115a7c403da52913711feb62c718e761594310ef9f` |
| BEVDriver BEV encoder | 645341213 | `591072c4e267c202d8dd0d95ab438355f66987a2ffe1af43390507d0d5666741` |

Pinned Hugging Face revisions provide provenance for directory-based assets.
The validation tool additionally checks the files needed by each backend.
