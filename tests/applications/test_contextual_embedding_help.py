import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["grobidTagger", "datasetTagger", "nerTagger", "insultTagger"])
def test_sequence_tagger_embedding_help_mentions_contextual_names_and_syntax(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", f"delft.applications.{module}", "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-500:]
    compact = "".join(result.stdout.split())
    for name in ("glove-840B", "fasttext-crawl", "word2vec", "potion-base-8M"):
        assert name in compact
    assert "scibert-contextual" in compact
    assert "bert-base-cased-contextual" in compact
    assert "contextual:<HuggingFacemodelorlocalpath>" in compact
