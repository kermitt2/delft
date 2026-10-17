"""
The transformers inside the models of DeLFT.

A model about to be trained starts from the pretrained weights of its transformer. A trained model being
loaded only needs the architecture of its transformer, the weights coming next from the model file: its
configuration is enough to build it, without downloading or reading the pretrained weights, which weigh
hundreds of megabytes for a BERT-base, and which an offline machine may not have.

A trained model saves that configuration, and the tokenizer of its transformer, next to its weights, as
DeLFT 0.4 did, so that it loads without the Hub, or a cache of the transformer.
"""

import os

import torch

TRANSFORMER_CONFIG_FILE_NAME = "transformer-config.json"
TRANSFORMER_TOKENIZER_DIR = "transformer-tokenizer"


def load_transformer(transformer_name, load_pretrained_weights=True, model_dir=None):
    """
    The transformer ``transformer_name`` (a Hugging Face Hub name or a local directory), with its
    pretrained weights, or with random ones when ``load_pretrained_weights`` is false. Without them, the
    configuration the trained model in ``model_dir`` saved is used, when there is one.

    It is in fp32 whatever the checkpoint says: some (e.g. deberta-v3) ship in fp16, which recent versions
    of transformers honor, and which clashes with the fp32 layers on top of it.
    """
    from transformers import AutoConfig, AutoModel

    if load_pretrained_weights:
        return AutoModel.from_pretrained(transformer_name, torch_dtype=torch.float32)
    config_path = os.path.join(model_dir, TRANSFORMER_CONFIG_FILE_NAME) if model_dir else None
    if config_path is None or not os.path.isfile(config_path):
        config_path = transformer_name
    return AutoModel.from_config(AutoConfig.from_pretrained(config_path)).to(torch.float32)


def save_transformer_files(directory, transformer, tokenizer):
    """Save in the model ``directory`` the configuration of its ``transformer`` and its ``tokenizer``."""
    transformer.config.to_json_file(os.path.join(directory, TRANSFORMER_CONFIG_FILE_NAME))
    tokenizer.save_pretrained(os.path.join(directory, TRANSFORMER_TOKENIZER_DIR))
