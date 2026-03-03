# DeLFT Installation

Get the Github repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```
It is advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
uv venv --python 3.10
source .venv/bin/activate
uv pip install pip 
```

Install the dependencies:

```sh
uv pip install -r requirements.txt
```

Finally install the project in editable state

```sh
uv pip install -e .
```

Current DeLFT version is __0.3.4__, which has been tested successfully with Python 3.8 and tensorflow 2.9.3. It will exploit your available GPU with the condition that CUDA (>=12) is properly installed. 

To ensure the availability of GPU devices for the right version of tensorflow, CUDA, CuDNN and python, you can check the dependencies [here](https://www.tensorflow.org/install/source#gpu).

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

