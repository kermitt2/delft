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


def test_refuses_a_stack_holding_contextual_embeddings(tmp_path, monkeypatch):
    stack = SimpleNamespace(
        extension="stacked",
        model=SimpleNamespace(
            components=[SimpleNamespace(extension="vec"), SimpleNamespace(extension="contextual-transformer")]
        ),
    )
    wrapper = SimpleNamespace(embeddings=stack, load=lambda **kwargs: None)
    monkeypatch.setattr(onnx_export, "Sequence", lambda name: wrapper)
    output = tmp_path / "onnx"

    with pytest.raises(ValueError, match="ONNX export does not support.*contextual embeddings"):
        onnx_export.export_to_onnx("stacked-model", str(output), model_path=str(tmp_path / "model"))

    assert not output.exists()


def test_a_stack_of_static_embeddings_is_not_contextual():
    stack = SimpleNamespace(
        extension="stacked",
        model=SimpleNamespace(components=[SimpleNamespace(extension="vec"), SimpleNamespace(extension="vec")]),
    )
    assert not onnx_export.uses_contextual_vectors(stack)
    assert not onnx_export.uses_contextual_vectors(None)
