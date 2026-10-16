# Loading transformer models

DeLFT integrates HuggingFace transformers natively as PyTorch modules. The transformer of a `BERT*` sequence labelling architecture, or of the `bert` text classification architecture, is the one given with `--transformer` (or `transformer_name` in the API). It is loaded with `AutoModel.from_pretrained` and its tokenizer with `AutoTokenizer.from_pretrained`, so it can be:

1. **a model of the HuggingFace Hub**, given by its name, e.g. `allenai/scibert_scivocab_cased`. It is downloaded at the first use, and kept in the HuggingFace cache (`~/.cache/huggingface`, or `HF_HOME`). Private repositories need a token in the environment (`HF_TOKEN`),
2. **a local directory** in the HuggingFace format, holding the configuration, the tokenizer files and the weights, as `save_pretrained` writes them.

```sh
python -m delft.applications.grobidTagger header train --architecture BERT_CRF --transformer allenai/scibert_scivocab_cased
python -m delft.applications.grobidTagger header train --architecture BERT_CRF --transformer /path/to/matbert
```

The `transformers` section of `delft/resources-registry.json` is not used to load the transformer of these architectures. Checkpoints in other formats, such as the original TensorFlow releases of BERT or SciBERT, have to be converted to the HuggingFace format first.

## Saved models

A fine-tuned model is saved with the name of its transformer in its `config.json`, and its weights, the transformer's included, in its weight file. When it is loaded, its transformer is built from the configuration of the transformer alone, then given the weights of the model: the pretrained weights are not downloaded again. The configuration and the tokenizer of the transformer are still loaded by its name, from the HuggingFace cache when they are there (set `HF_HUB_OFFLINE=1` to use the cache without asking the Hub), or from the Hub.

Transformers can also give contextual embeddings to the RNN architectures, without being fine-tuned: see [contextual embeddings](embeddings.md#contextual-embeddings-from-a-frozen-transformer).
