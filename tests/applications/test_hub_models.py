import os

import pytest

from delft.applications import hub_models as application
from tests.fake_hub import HEADER_MODELS

REGISTRY = {"models-hub": {"repo": "lfoppiano/grobid-model-{short_name}", "bucket": "lfoppiano/delft-models"}}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(application, "load_resource_registry", lambda path: REGISTRY)


def test_list_a_task(hub, capsys):
    assert application.main(["list", "header"]) == sorted(HEADER_MODELS)
    assert "hf://lfoppiano/grobid-model-header: 5 model(s)" in capsys.readouterr().out


def test_list_a_reference(hub):
    assert application.main(["list", "hf://buckets/lfoppiano/delft-models", "--architecture", "wapiti"]) == [
        "grobid-header-wapiti"
    ]


def test_bucket_holds_every_task_and_lists_the_one_asked(hub):
    assert application.main(["list", "header", "--bucket"]) == sorted(HEADER_MODELS)
    assert application.main(["list", "date", "--bucket"]) == ["grobid-date-BidLSTM_CRF"]


def test_pull_one_model(hub, models_dir):
    application.main(["pull", "header", "--architecture", "BidLSTM_CRF", "--suffix", "", "--output", models_dir])
    assert os.listdir(models_dir) == ["grobid-header-BidLSTM_CRF"]


def test_pull_all(hub, models_dir):
    application.main(["pull", "header", "--all", "--output", models_dir])
    assert sorted(os.listdir(models_dir)) == sorted(HEADER_MODELS)


def test_pull_without_a_selection_is_refused(hub, models_dir):
    with pytest.raises(SystemExit):
        application.main(["pull", "header", "--output", models_dir])
    assert not os.path.exists(models_dir)


def test_push_a_model_to_the_repository_of_its_task(hub, tmp_path):
    from tests.utilities.test_hub_publish import _save

    _save(tmp_path / "models", "grobid-figure-BidLSTM_CRF-v2")
    reference = application.main(["push", "grobid-figure-BidLSTM_CRF-v2", "--input", str(tmp_path / "models")])
    assert str(reference) == "hf://lfoppiano/grobid-model-figure/grobid-figure-BidLSTM_CRF-v2"
    assert application.main(["list", "figure"]) == ["grobid-figure-BidLSTM_CRF-v2"]


def test_push_a_directory_to_a_given_place(hub, tmp_path):
    from tests.utilities.test_hub_publish import _save

    _, model_dir = _save(tmp_path / "models", "grobid-figure-BidLSTM_CRF")
    reference = application.main(["push", model_dir, "--to", "hf://buckets/someone/scratch", "--private"])
    assert str(reference) == "hf://buckets/someone/scratch/grobid-figure-BidLSTM_CRF"
    assert hub.created == [("someone/scratch", True)]
