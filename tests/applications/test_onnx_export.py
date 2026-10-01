from types import SimpleNamespace

import pytest

import delft.applications.onnx_export as onnx_export


def test_refuses_contextual_embeddings_before_creating_export_files(tmp_path, monkeypatch):
    wrapper = SimpleNamespace(
        embeddings=SimpleNamespace(extension="contextual-transformer"),
        load=lambda **kwargs: None,
    )
    monkeypatch.setattr(onnx_export, "Sequence", lambda name: wrapper)
    output = tmp_path / "onnx"

    with pytest.raises(ValueError, match="ONNX export does not support.*contextual embeddings"):
        onnx_export.export_to_onnx("contextual-model", str(output), model_path=str(tmp_path / "model"))

    assert not output.exists()
