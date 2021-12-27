# DeFLT Installation

Get the github repo:

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

Current DeLFT version is __0.3.0__. It uses tensorflow 2.7.0 and will exploit your available GPU with the condition that CUDA (>=11.0) is properly installed. 

You need then to download some pre-trained word embeddings and notify their path into the embedding registry. We suggest for exploiting the provided models:

* _glove Common Crawl_ (2.2M vocab., cased, 300 dim. vectors): [glove-840B](http://nlp.stanford.edu/data/glove.840B.300d.zip)

* _fasttext Common Crawl_ (2M vocab., cased, 300 dim. vectors): [fasttext-crawl](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip)

* _word2vec GoogleNews_ (3M vocab., cased, 300 dim. vectors): [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

* _fasttext_wiki_fr_ (1.1M, NOT cased, 300 dim. vectors) for French: [wiki.fr](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec)

* _ELMo_ trained on 5.5B word corpus (will produce 1024 dim. vectors) for English: [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json) and [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)

* _BERT_ for English, we are using BERT-Base, Cased, 12-layer, 768-hidden, 12-heads , 110M parameters: available [here](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)

* _SciBERT_ for English and scientific content: [SciBERT-cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz)

Then edit the file `embedding-registry.json` and modify the value for `path` according to the path where you have saved the corresponding embeddings. The embedding files must be unzipped.

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

You're ready to use DeLFT.
