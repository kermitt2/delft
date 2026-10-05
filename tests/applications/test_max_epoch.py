"""
Every text classification application takes ``--max-epoch``: the number of epochs asked
for replaces the default one of the architecture, which a run without the option keeps.
"""

import importlib

import pytest

APPLICATIONS = [
    "citationClassifier",
    "dataseerClassifier",
    "licenseClassifier",
    "softwareClassifier",
    "softwareContextClassifier",
    "toxicCommentClassifier",
]
# where the number of epochs is in what `configure` returns
MAX_EPOCH = 4


@pytest.mark.parametrize("app_name", APPLICATIONS)
@pytest.mark.parametrize("architecture", ["gru", "bert"])
def test_epochs_asked_for_replace_the_default(app_name, architecture):
    configure = importlib.import_module(f"delft.applications.{app_name}").configure

    default = configure(architecture)[MAX_EPOCH]
    assert default > 0
    assert configure(architecture, max_epoch=default + 2)[MAX_EPOCH] == default + 2
    assert configure(architecture, max_epoch=-1)[MAX_EPOCH] == default
