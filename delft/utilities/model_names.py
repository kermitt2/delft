"""
Names of the sequence labelling models, which are also the names of their directories
under ``data/models/sequenceLabelling``:

    {prefix}{short name}-{architecture}[-{suffix}]

for instance ``grobid-header-BidLSTM_CRF_FEATURES`` or, with a suffix,
``grobid-header-BidLSTM_CRF_FEATURES-potion-base-8M``. The suffix tells apart models
of the same task and architecture, trained with different embeddings, transformers or
hyper-parameters, which would otherwise overwrite each other.
"""

import re
from typing import NamedTuple, Optional

GROBID_PREFIX = "grobid-"

# The architectures of delft.sequenceLabelling.models.MODEL_REGISTRY, repeated here so
# that naming a model does not import the models. They hold no dash, where short names
# and suffixes do, which is what makes a name splittable.
ARCHITECTURES = (
    "BidLSTM",
    "BidLSTM_CRF",
    "BidLSTM_ChainCRF",
    "BidLSTM_CNN",
    "BidLSTM_CNN_CRF",
    "BidGRU_CRF",
    "BidLSTM_CRF_FEATURES",
    "BidLSTM_ChainCRF_FEATURES",
    "BidLSTM_CRF_CASING",
    "BERT",
    "BERT_CRF",
    "BERT_ChainCRF",
    "BERT_FEATURES",
    "BERT_CRF_FEATURES",
    "BERT_ChainCRF_FEATURES",
)

# Not an architecture of DeLFT, which cannot load them: the Wapiti CRF models of GROBID
# are named and published along the DeLFT ones.
WAPITI = "wapiti"

# a suffix becomes part of a directory name
_SUFFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_suffix(suffix):
    """Return ``suffix``, None when it is empty, and raise a ValueError when it cannot be
    part of a directory name."""
    if suffix is None or suffix == "":
        return None
    if not _SUFFIX_PATTERN.fullmatch(suffix):
        raise ValueError(
            f"Invalid model name suffix {suffix!r}: it can hold letters, digits, '.', '_' and '-', "
            "and starts with a letter or a digit"
        )
    return suffix


def build_model_name(short_name, architecture, suffix=None, prefix=GROBID_PREFIX):
    """Name of the model of a task (``short_name``, e.g. "header") for an architecture."""
    model_name = f"{prefix}{short_name}-{architecture}"
    suffix = validate_suffix(suffix)
    if suffix is not None:
        model_name += "-" + suffix
    return model_name


class ModelName(NamedTuple):
    short_name: str
    architecture: str
    suffix: Optional[str]
    prefix: str


def split_model_name(model_name, prefix=GROBID_PREFIX):
    """
    Split a name made by ``build_model_name``, or return None when it holds no known
    architecture. The architecture is the first part between dashes that is one, so a
    suffix can hold anything. ``prefix`` is dropped when the name starts with it.
    """
    found_prefix = prefix if prefix and model_name.startswith(prefix) else ""
    parts = model_name[len(found_prefix) :].split("-")
    for i, part in enumerate(parts):
        if i > 0 and (part in ARCHITECTURES or part == WAPITI):
            suffix = "-".join(parts[i + 1 :]) or None
            return ModelName("-".join(parts[:i]), part, suffix, found_prefix)
    return None
