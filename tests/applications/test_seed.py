"""
Every application that trains takes ``--seed`` and seeds the random number generators
with it before doing anything else, so that the split of the data, the initial weights
and the order of the batches are the same from a run to the next. ``grobidTagger`` is
not listed: it seeds inside ``train`` / ``train_eval``, once the data is loaded.
"""

import importlib
import runpy
import sys

import pytest

import delft.utilities.Utilities as utilities

SEEDED_ON_START = {
    "citationClassifier": "train",
    "dataseerClassifier": "train",
    "datasetTagger": "train",
    "insultTagger": "train",
    "licenseClassifier": "train",
    "nerTagger": "train",
    "softwareClassifier": "train",
    "softwareContextClassifier": "train",
    "toxicCommentClassifier": "train",
}


class Seeded(Exception):
    """Stops the application where it seeds the generators: nothing is loaded nor trained."""


def run_until_seeded(app_name, arguments, monkeypatch):
    def stop(seed):
        raise Seeded(seed)

    path = importlib.import_module(f"delft.applications.{app_name}").__file__
    monkeypatch.setattr(utilities, "set_random_seed", stop)
    monkeypatch.setattr(sys, "argv", [path, *arguments])
    with pytest.raises(Seeded) as stopped:
        runpy.run_path(path, run_name="__main__")
    return stopped.value.args[0]


@pytest.mark.parametrize("app_name, action", SEEDED_ON_START.items())
def test_the_seed_of_the_command_line_is_the_one_set(app_name, action, monkeypatch):
    assert run_until_seeded(app_name, [action, "--seed", "7"], monkeypatch) == 7


@pytest.mark.parametrize("app_name, action", SEEDED_ON_START.items())
def test_without_a_seed_the_generators_are_left_alone(app_name, action, monkeypatch):
    assert run_until_seeded(app_name, [action], monkeypatch) is None


def test_grobid_tagger_takes_a_seed_too():
    source = importlib.import_module("delft.applications.grobidTagger").__file__
    with open(source) as f:
        assert '"--seed"' in f.read()
