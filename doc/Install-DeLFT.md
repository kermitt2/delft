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

# Linux with CUDA 12.1 (recommended for GPU)
uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu121

# Linux with CUDA 12.1 (alternative using requirements file)
uv pip install -e . -r requirements-cuda.txt
```

Current DeLFT version is __0.4.6__, which has been tested successfully with Python 3.10/3.11 and TensorFlow 2.17. It will exploit your available GPU with the condition that CUDA 12.1 is properly installed.

To ensure the availability of GPU devices for the right version of TensorFlow, CUDA, cuDNN and Python, you can check the dependencies [here](https://www.tensorflow.org/install/source#gpu).

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

