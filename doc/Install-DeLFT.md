# DeLFT Installation

Get the GitHub repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```
It is advised to first set up a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install pip 
```

Install the project in editable state:

```sh
# macOS (torch is included automatically)
uv pip install -e .

# Linux with CUDA 12.8 — modern GPUs (Turing sm_75 and above: RTX 20/30/40 series, A100, H100, L40S, …)
uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# Linux with CUDA 12.6 — pre-Turing GPUs (V100 sm_70, P100 sm_60). See "Older NVIDIA GPUs" below.
uv pip install -e ".[gpu-pre-turing]" --extra-index-url https://download.pytorch.org/whl/cu126
```

> **Note:** Pick exactly one of `[gpu]` or `[gpu-pre-turing]` — they pin different PyTorch CUDA wheels (cu128 vs cu126) and cannot coexist.

> **For training a model:** add the `[dev]` extras (which include `wandb`, `pytest`, `ruff`) — e.g. `uv pip install -e ".[gpu,dev]"`. The base install does not include `wandb`, so training with W&B tracking (`--wandb`) requires the dev extras (or `pip install wandb` ad-hoc). Inference-only setups don't need this.

DeLFT __0.4.x__ has been tested successfully with Python 3.10/3.11 and TensorFlow 2.17. It will exploit your available GPU with the condition that CUDA 12.1 is properly installed. The exact patch version on PyPI may be ahead of this page; check the PyPI badge in the [README](https://github.com/kermitt2/delft#readme) for the current release.

To ensure the availability of GPU devices for the right version of TensorFlow, CUDA, cuDNN and Python, you can check the dependencies [here](https://www.tensorflow.org/install/source#gpu).

### Older NVIDIA GPUs (V100 and other pre-Turing cards)

Pre-Turing GPUs hit two distinct issues with modern PyTorch wheels.

**Issue 1 — cuDNN 9 dropped sm < 7.5.** PyTorch 2.4+ bundles cuDNN 9, which dropped support for compute capabilities below sm_75 — V100 (sm_70), P100 (sm_60), and other pre-Turing GPUs. On those devices, the first cuDNN-backed RNN op (e.g. `nn.LSTM` inside the `BidLSTM_*` architectures) raises `RuntimeError: cuDNN version ... is not compatible with devices with SM < 7.5`.

DeLFT detects this at device-pick time and disables cuDNN automatically. RNN training falls back to PyTorch's native (non-fused) kernels — functional but slower (typically 3–5× on LSTM-heavy workloads). Transformer architectures (`BERT_*`, `BERT_CRF`, etc.) are unaffected since they don't go through cuDNN's RNN path. You'll see a one-line message at startup when the fallback kicks in:

```
cuDNN disabled: Tesla V100-SXM2-32GB (sm_70) is below the cuDNN 9 floor (sm_75); ...
```

**Issue 2 — CUDA 12.8+ wheels dropped sm_70 kernels entirely.** The `[gpu]` extra installs `torch==2.11.0+cu128`, whose binary kernels target sm_75 and above. On a V100, the first generic CUDA op (e.g. `nn.Embedding` lookup — *not* a cuDNN op, so the cuDNN fallback above does not help) crashes with:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

DeLFT detects this at device-pick time and aborts immediately with an actionable error rather than letting the job die mid-training. To diagnose your wheel:

```sh
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

If `sm_70` is missing from the arch list, install the dedicated extra which pins a torch wheel that still ships sm_70 kernels:

```sh
uv pip install -e ".[gpu-pre-turing]" --extra-index-url https://download.pytorch.org/whl/cu126
```

If your cluster's GPU pool is heterogeneous (e.g. mixing V100 with RTX3090/A6000/L40S), the cuDNN auto-disable runs per-job — but the wheel-side issue is global to the venv. Either keep a separate `gpu-pre-turing` venv for the V100 nodes, or use the `gpu-pre-turing` extra for all nodes (modern GPUs run fine on the cu126 wheel, just slightly older toolkit).

### Upgrading from 0.3.4

When upgrading from DeLFT 0.3.4, be aware of the following breaking changes:

- **Python 3.10 or 3.11 required** (3.8 and 3.9 are no longer supported)
- **TensorFlow 2.17 / tf_keras 2.17**: Pre-trained model weights from 0.3.4 are not directly compatible. Use the model conversion utility to migrate existing models without retraining:
  ```sh
  python -m delft.utilities.convert_model --input <old-model-dir> --output <new-model-dir> --verify
  ```
  For models saved with an older `transformers` library (tokenizer errors), add `--redownload-tokenizer`. Use `--force-partial` if the old architecture added layers not present in the current code. Run with `--dry-run` to inspect without writing
- **CUDA 12.1 required** for GPU support (previously CUDA 11.x)
- **LMDB embedding caches must be converted** from the old pickle format to the new float32 format:
  ```sh
  python -m delft.utilities.convert_lmdb_embeddings --input <old-lmdb-path> --output <new-lmdb-path>
  ```
- **ELMo support has been removed** — use transformer models or static embeddings instead
- **torch is no longer installed by default on Linux** to avoid CUDA conflicts — use the `[gpu]` extra (see install commands above)

## Loading resources locally

Required resources to train models (static embeddings, pre-trained transformer models) will be downloaded automatically, in particular via Hugging Face Hub using the model name identifier. However, if you wish to load these resources locally, you need to notify their local path in the resource registry file. 

Edit the file `delft/resources-registry.json` and modify the value for `path` according to the path where you have saved the corresponding embeddings. The embedding files must be unzipped. For instance, for loading glove-840B embeddings from a local path:

```json
{
    "embeddings": [
        {
            "name": "glove-840B",
            "path": "/PATH/TO/THE/UNZIPPED/EMBEDDINGS/FILE/glove.840B.300d.txt",
            "type": "glove",
            "format": "vec",
            "lang": "en",
            "item": "word"
        },
        ...
    ],
    ...
}

```

For pre-trained transformer models (for example downloaded from Hugging Face), you can indicate simply the path to the model directory, as follow:


```json
{
    "transformers": [
        {
            "name": "scilons/scilons-bert-v0.1",
            "model_dir": "/media/lopez/T52/models/scilons/scilons-bert-v0.1/",
            "lang": "en"
        },
        ...
    ],
    ...
}
```

For older transformer formats with just config, vocab and checkpoint weights file, you can indicate the resources like this:

```json
{
    "transformers": [
        {
            "name": "dmis-lab/biobert-base-cased-v1.2",
            "path-config": "/media/lopez/T5/embeddings/biobert_v1.2_pubmed/bert_config.json",
            "path-weights": "/media/lopez/T5/embeddings/biobert_v1.2_pubmed/model.ckpt-1000000",
            "path-vocab": "/media/lopez/T5/embeddings/biobert_v1.2_pubmed/vocab.txt",
            "lang": "en"
        },
        ...
    ],
    ...
}
```

## Logging training runs to Weights & Biases

DeLFT can stream training metrics to [Weights & Biases](https://wandb.ai). To enable it:

1. Create a `.env` file at the root of the project containing your credentials:

    ```
    WANDB_API_KEY=your_api_key
    WANDB_PROJECT=your_project_name
    WANDB_ENTITY=your_entity_name
    ```

2. Pass `--wandb` to any application script, e.g.:

    ```sh
    python -m delft.applications.grobidTagger date train --architecture BidLSTM --wandb
    ```

