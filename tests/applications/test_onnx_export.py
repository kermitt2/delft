from types import SimpleNamespace

import pytest

import delft.applications.onnx_export as onnx_export


def test_refuses_contextual_embeddings_before_creating_export_files(tmp_path, monkeypatch):
    wrapper = SimpleNamespace(
        embeddings=SimpleNamespace(extension="contextual-transformer"),
        load=lambda **kwargs: None,
    )
    monkeypatch.setattr(onnx_export, "Sequence", lambda name, **kwargs: wrapper)
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
    monkeypatch.setattr(onnx_export, "Sequence", lambda name, **kwargs: wrapper)
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


WORDS = ["Received", "12", "March", "2019", ";", "accepted", "3", "June", "2020"]
LABELS = ["O", "B-<day>", "B-<month>", "B-<year>", "O", "O", "B-<day>", "B-<month>", "B-<year>"]


@pytest.mark.parametrize("architecture", ["BidLSTM_CRF", "BidLSTM_ChainCRF"])
def test_the_exported_encoder_gives_the_emissions_of_the_model(tmp_path, monkeypatch, architecture):
    """Sequences of different lengths, unpadded, and padded for the architectures that mask the padding (the
    ChainCRF ones run their LSTM over it, as their Keras counterparts did).

    The default device is one no CPU tensor runs on, as MPS or CUDA where there is one: the export loaded the
    model there and traced it with inputs on the CPU, which failed ("Placeholder storage has not been allocated
    on MPS device")."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import numpy as np
    import torch

    monkeypatch.setattr("delft.utilities.Utilities.best_device", lambda: torch.device("meta"))

    from delft.sequenceLabelling.data_loader import create_dataloader
    from delft.sequenceLabelling.models import get_model
    from delft.sequenceLabelling.preprocess import Preprocessor
    from delft.sequenceLabelling.wrapper import Sequence

    sequence = Sequence("onnx-model", architecture=architecture, embeddings_name=None, device="cpu")
    sequence.p = Preprocessor(return_chars=True)
    sequence.p.fit([WORDS], [LABELS])
    sequence.model_config.char_vocab_size = len(sequence.p.vocab_char)
    sequence.model = get_model(sequence.model_config, len(sequence.p.vocab_tag), load_pretrained_weights=False)
    sequence.save(str(tmp_path / "models"))

    onnx_export.export_to_onnx("onnx-model", str(tmp_path / "onnx"), model_path=str(tmp_path / "models"))
    session = ort.InferenceSession(str(tmp_path / "onnx" / "encoder.onnx"))

    texts = [WORDS, WORDS[:4], WORDS[5:6]]
    loader = create_dataloader(
        texts, None, preprocessor=sequence.p, batch_size=1, shuffle=False, model_config=sequence.model_config
    )
    sequence.model.eval()
    with torch.inference_mode():
        for inputs, _ in loader:
            expected = sequence.model(inputs)["logits"].numpy()
            feeds = {i.name: inputs[i.name].numpy() for i in session.get_inputs()}
            np.testing.assert_allclose(session.run(None, feeds)[0], expected, atol=1e-5)
            if "ChainCRF" in architecture:
                continue

            padded = {name: np.zeros((1, 20, *value.shape[2:]), value.dtype) for name, value in feeds.items()}
            for name, value in feeds.items():
                padded[name][:, : value.shape[1]] = value
            got = session.run(None, padded)[0][:, : expected.shape[1]]
            np.testing.assert_allclose(got, expected, atol=1e-5)


def test_refuses_an_architecture_it_does_not_support(tmp_path, monkeypatch):
    wrapper = SimpleNamespace(
        embeddings=None,
        load=lambda **kwargs: None,
        model=None,
        p=None,
        model_config=SimpleNamespace(architecture="BidLSTM_CRF_CASING"),
        training_config=None,
    )
    monkeypatch.setattr(onnx_export, "Sequence", lambda name, **kwargs: wrapper)
    with pytest.raises(ValueError, match="not BidLSTM_CRF_CASING"):
        onnx_export.export_to_onnx("casing-model", str(tmp_path / "onnx"))
