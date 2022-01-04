# DeLFT Installation

Get the Github repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```
It is advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
virtualenv --system-site-packages -p python3.8 env
source env/bin/activate
```

Install the dependencies:

```sh
pip3 install -r requirements.txt
```

Finally install the project in editable state

```sh
pip3 install -e .
```

Current DeLFT version is __0.3.0__. It uses tensorflow 2.7.0 and will exploit your available GPU with the condition that CUDA (>=11.0) is properly installed. 

## Loading resources locally

Required resources to train models (static embeddings, pre-trained transformer models) will be downloaded automatically. However, if you wish to load these resources locally, you need to notify their local path in the resource registry file. 

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
    ]
}

```

