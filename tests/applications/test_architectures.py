"""
The sequence labelling application scripts each hardcode the architecture list they
advertise in ``--help`` and validate against. Those lists had drifted from
``MODEL_REGISTRY``: they offered ``BERT_CRF_CHAR`` / ``BERT_CRF_CHAR_FEATURES``, which
no longer exist, so selecting one failed only at model instantiation time.

Each script deliberately offers a *subset* of the registry (e.g. nerTagger omits the
``*_FEATURES`` architectures, since CoNLL-style input carries no feature matrix), so
these tests assert a subset relation rather than equality.
"""

import importlib

import pytest

from delft.sequenceLabelling.models import MODEL_REGISTRY

SEQUENCE_LABELLING_APPS = [
    "nerTagger",
    "grobidTagger",
    "datasetTagger",
    "insultTagger",
]


def get_architectures(app_name):
    """Read the architecture list a given application script advertises."""
    module = importlib.import_module(f"delft.applications.{app_name}")
    source = module.__file__
    namespace = {}
    with open(source) as f:
        lines = f.readlines()

    # The lists are local to main(), so exec just the assignment blocks.
    collected = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("architectures_word_embeddings = [", "architectures_transformers_based = [")):
            capturing = True
            collected.append(stripped)
            continue
        if capturing:
            collected.append(stripped)
            if stripped == "]":
                capturing = False

    exec("\n".join(collected), namespace)
    return namespace["architectures_word_embeddings"] + namespace["architectures_transformers_based"]


@pytest.mark.parametrize("app_name", SEQUENCE_LABELLING_APPS)
def test_advertised_architectures_all_exist(app_name):
    """Every architecture offered in --help must be instantiable via MODEL_REGISTRY."""
    advertised = get_architectures(app_name)
    unknown = sorted(set(advertised) - set(MODEL_REGISTRY))
    assert not unknown, (
        f"{app_name} advertises architecture(s) absent from MODEL_REGISTRY: {unknown}. "
        f"Selecting one fails at model instantiation."
    )


@pytest.mark.parametrize("app_name", SEQUENCE_LABELLING_APPS)
def test_advertised_architectures_have_no_duplicates(app_name):
    advertised = get_architectures(app_name)
    duplicates = sorted({a for a in advertised if advertised.count(a) > 1})
    assert not duplicates, f"{app_name} lists architecture(s) more than once: {duplicates}"


def test_text_classification_architectures_derive_from_registry():
    """The text classification apps import the list, so it cannot drift -- lock that in."""
    from delft.textClassification.models import MODEL_REGISTRY as CLASSIFIER_REGISTRY
    from delft.textClassification.models import architectures

    assert set(architectures) == set(CLASSIFIER_REGISTRY)
