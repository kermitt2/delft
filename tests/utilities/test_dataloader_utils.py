"""Log volume of the per-call helpers.

``effective_num_workers`` and ``pick_device`` both run on every ``tag()`` /
``predict()`` call. When DeLFT is embedded in a host process such as GROBID
their output lands on the host's stdout, which the host cannot filter, so
neither may write a line per call.
"""

import logging

import torch

from delft.utilities import Utilities
from delft.utilities.dataloader_utils import effective_num_workers


class TestEffectiveNumWorkersLogging:
    def test_writes_nothing_to_stdout(self, capsys):
        effective_num_workers(4, dataset_size=1, batch_size=20, role="tag")
        effective_num_workers(0, dataset_size=1, batch_size=20, role="tag")
        assert capsys.readouterr().out == ""

    def test_reports_the_outcome_at_debug_level(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="delft.utilities.dataloader_utils"):
            effective_num_workers(4, dataset_size=1, batch_size=20, role="tag")
        assert "num_workers=0" in caplog.text
        assert "DataLoader[tag]" in caplog.text


class TestPickDeviceLogging:
    def test_announces_a_device_once_only(self, capsys, monkeypatch):
        monkeypatch.setattr(Utilities, "_ANNOUNCED_DEVICES", set())

        Utilities.pick_device(torch.device("cpu"))
        first = capsys.readouterr().out
        for _ in range(3):
            Utilities.pick_device(torch.device("cpu"))
        repeated = capsys.readouterr().out

        assert "Running on cpu" in first
        assert repeated == ""

    def test_still_resolves_the_device_when_silent(self, monkeypatch):
        monkeypatch.setattr(Utilities, "_ANNOUNCED_DEVICES", {"cpu"})
        assert Utilities.pick_device(torch.device("cpu")) == torch.device("cpu")
