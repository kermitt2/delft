# Loading transformer models

DeLFT integrates HuggingFace transformers natively as PyTorch modules. A transformer can be loaded from three different sources, in order of preference (each falls back to the next if the resource is not found):

1. **Local directory** — a self-contained HuggingFace-format model directory (config, tokenizer, weights), declared via `model_dir` in `delft/resources-registry.json`.
2. **Plain (legacy) checkpoint** — separate config, weights and vocabulary files, declared via `path-config`, `path-weights` and `path-vocab` entries. Used to load older BERT-style checkpoints downloaded directly from the original GitHub releases.
3. **HuggingFace Hub** — fetched online by name when no registry entry matches. Requires network access; private repositories require `HF_ACCESS_TOKEN` in the environment.

The selection logic is implemented in `delft/utilities/Transformer.py` (`configure_from_registry` / `init_preprocessor` / `instantiate_layer`).

## Loading sources

| Transformer                       | Description                                          | Registry entry?                                            | Hugging Face Hub? | Loader path (`Transformer.py`)                              |
|-----------------------------------|------------------------------------------------------|------------------------------------------------------------|-------------------|-------------------------------------------------------------|
| `allenai/scibert_scivocab_cased`  | SciBERT pulled from the Hub by name                  | no                                                         | yes               | `LOADING_METHOD_HUGGINGFACE_NAME` — `AutoModel.from_pretrained(name)` |
| `portiz/matbert`                  | RoBERTa-based model saved locally as a HF directory  | yes — only `model_dir` is needed                           | no                | `LOADING_METHOD_LOCAL_MODEL_DIR` — `AutoModel.from_pretrained(model_dir)` |
| `scibert` (legacy GitHub release) | Original SciBERT TF1 checkpoint with separate files  | yes — `path-config`, `path-weights`, `path-vocab` required | no                | `LOADING_METHOD_PLAIN_MODEL` — `BertTokenizer.from_pretrained(path-vocab)` + manual config/weights load |

## Configuration examples

**Local HuggingFace-format directory** (preferred for local models):

```json
{
    "name": "portiz/matbert",
    "model_dir": "/Users/lfoppiano/development/projects/embeddings/pre-trained-embeddings/matbert",
    "lang": "en"
}
```

**Plain (legacy) checkpoint** with separate config / weights / vocabulary files:

```json
{
    "name": "dmis-lab/biobert-base-cased-v1.2",
    "path-config": "/media/lopez/T5/embeddings/biobert_v1.2_pubmed/bert_config.json",
    "path-weights": "/media/lopez/T5/embeddings/biobert_v1.2_pubmed/model.ckpt-1000000",
    "path-vocab": "/media/lopez/T5/embeddings/biobert_v1.2_pubmed/vocab.txt",
    "lang": "en"
}
```

**Hub-only** (no registry entry needed) — just pass the Hub model id on the command line:

```sh
python -m delft.applications.grobidTagger header train --architecture BERT_CRF --transformer allenai/scibert_scivocab_cased
```

## Notes

- For models whose Hub name contains `cased` or `uncased`, DeLFT forces `do_lower_case` accordingly to work around tokenizers shipped without an explicit casing configuration (see `Transformer.py:122-147` and issue [#144](https://github.com/kermitt2/delft/issues/144)).
- When DeLFT saves a fine-tuned model, the transformer layer is serialized inside the model directory using a fourth (internal) loading method, `LOADING_METHOD_DELFT_MODEL`, with the file name `transformer-config.json`. This is automatic and not user-configurable.
