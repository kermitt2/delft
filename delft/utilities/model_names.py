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

GROBID_PREFIX = "grobid-"

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
