"""
The transformers inside the models of DeLFT.

A model about to be trained starts from the pretrained weights of its transformer. A trained model being
loaded only needs the architecture of its transformer, the weights coming next from the model file: its
configuration is enough to build it, without downloading or reading the pretrained weights, which weigh
hundreds of megabytes for a BERT-base, and which an offline machine may not have.
"""

import torch


def load_transformer(transformer_name, load_pretrained_weights=True):
    """
    The transformer ``transformer_name`` (a Hugging Face Hub name or a local directory), with its
    pretrained weights, or with random ones when ``load_pretrained_weights`` is false.

    It is in fp32 whatever the checkpoint says: some (e.g. deberta-v3) ship in fp16, which recent versions
    of transformers honor, and which clashes with the fp32 layers on top of it.
    """
    from transformers import AutoConfig, AutoModel

    if load_pretrained_weights:
        return AutoModel.from_pretrained(transformer_name, torch_dtype=torch.float32)
    return AutoModel.from_config(AutoConfig.from_pretrained(transformer_name)).to(torch.float32)
