"""Unit tests for :mod:`delft.utilities.convert_model`.

These tests exercise the pure helpers (HDF5 walk, name lookup, config
rewriting, task autodetection) against tiny on-disk fixtures. They do not
build a full DeLFT model — that path is best exercised by running the
converter against a real pre-upgrade model manually and inspecting the
output.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from delft.utilities import convert_model


def _write_fake_hdf5(path: Path, keras_version: str = "2.9.0") -> None:
    with h5py.File(str(path), "w") as h:
        h.attrs["keras_version"] = np.bytes_(keras_version)
        h.attrs["backend"] = np.bytes_("tensorflow")
        h.attrs["layer_names"] = np.bytes_("['crf' 'model']")

        crf = h.create_group("crf")
        crf.create_dataset("chain_kernel:0", data=np.zeros((4, 4), dtype=np.float32))
        crf.create_dataset("left_boundary:0", data=np.zeros((4,), dtype=np.float32))
        crf.create_dataset("right_boundary:0", data=np.zeros((4,), dtype=np.float32))
        inner = crf.create_group("crf").create_group("dense")
        inner.create_dataset("kernel:0", data=np.ones((8, 4), dtype=np.float32))
        inner.create_dataset("bias:0", data=np.ones((4,), dtype=np.float32))

        model = h.create_group("model")
        bilstm = model.create_group("bidirectional_1").create_group("forward_lstm_1").create_group("lstm_cell")
        bilstm.create_dataset("kernel:0", data=np.full((10, 16), 0.5, dtype=np.float32))
        bilstm.create_dataset("bias:0", data=np.full((16,), 0.5, dtype=np.float32))


class TestCheckSourceHDF5:
    def test_accepts_keras_2(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.hdf5"
        _write_fake_hdf5(f, keras_version="2.9.0")
        info = convert_model.check_source_hdf5(f)
        assert info["keras_version"] == "2.9.0"
        assert info["backend"] == "tensorflow"
        assert set(info["top_keys"]) == {"crf", "model"}

    def test_rejects_pre_keras_2(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.hdf5"
        _write_fake_hdf5(f, keras_version="1.2.0")
        with pytest.raises(ValueError, match="Pre-Keras-2.x"):
            convert_model.check_source_hdf5(f)

    def test_rejects_missing_version_attr(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.hdf5"
        with h5py.File(str(f), "w") as h:
            h.create_group("crf")
        with pytest.raises(ValueError, match="no 'keras_version'"):
            convert_model.check_source_hdf5(f)


class TestFlattenH5Weights:
    def test_produces_slash_joined_names_with_shapes(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.hdf5"
        _write_fake_hdf5(f)
        with h5py.File(str(f), "r") as h:
            flat = convert_model.flatten_h5_weights(h)
        assert "crf/chain_kernel:0" in flat
        assert flat["crf/chain_kernel:0"].shape == (4, 4)
        assert "crf/crf/dense/kernel:0" in flat
        assert flat["crf/crf/dense/kernel:0"].shape == (8, 4)
        assert "model/bidirectional_1/forward_lstm_1/lstm_cell/kernel:0" in flat
        assert flat["model/bidirectional_1/forward_lstm_1/lstm_cell/kernel:0"].shape == (10, 16)


class TestFindOldKey:
    @staticmethod
    def _make(index: dict):
        """Build old_index, suffix_index, and top_groups from a {key: value} dict."""
        suffix_idx = convert_model.build_suffix_index(index)
        top_groups = sorted({k.split("/")[0] for k in index if "/" in k})
        return index, suffix_idx, top_groups

    def test_direct_hit(self) -> None:
        idx, sfx, grps = self._make({"crf/chain_kernel:0": None})
        assert convert_model.find_old_key("crf/chain_kernel:0", idx, sfx, grps) == "crf/chain_kernel:0"

    def test_missing_returns_none(self) -> None:
        idx, sfx, grps = self._make({})
        assert convert_model.find_old_key("nothing/here:0", idx, sfx, grps) is None

    def test_strips_numeric_suffix_fallback(self) -> None:
        idx, sfx, grps = self._make({"bidirectional/forward_lstm/lstm_cell/kernel:0": None})
        fresh = "bidirectional_3/forward_lstm_3/lstm_cell/kernel:0"
        assert convert_model.find_old_key(fresh, idx, sfx, grps) == "bidirectional/forward_lstm/lstm_cell/kernel:0"

    def test_applies_rename_rules(self, monkeypatch) -> None:
        idx, sfx, grps = self._make({"model/tf_roberta_model/layer_._0/kernel:0": None})
        monkeypatch.setattr(
            convert_model,
            "RENAME_RULES",
            [(r"^model/tf_bert_model/", "model/tf_roberta_model/")],
        )
        assert (
            convert_model.find_old_key("model/tf_bert_model/layer_._0/kernel:0", idx, sfx, grps)
            == "model/tf_roberta_model/layer_._0/kernel:0"
        )

    def test_suffix_match_strips_hdf5_group_prefix(self) -> None:
        """Fresh model gives 'chain_kernel:0', old HDF5 has 'crf/chain_kernel:0'."""
        idx, sfx, grps = self._make({"crf/chain_kernel:0": None})
        assert convert_model.find_old_key("chain_kernel:0", idx, sfx, grps) == "crf/chain_kernel:0"

    def test_suffix_match_for_deep_transformer_path(self) -> None:
        """Fresh model gives 'tf_bert_model/bert/.../weight:0', old has 'model/tf_bert_model/bert/.../weight:0'."""
        full = "model/tf_bert_model/bert/embeddings/word_embeddings/weight:0"
        idx, sfx, grps = self._make({full: None})
        fresh = "tf_bert_model/bert/embeddings/word_embeddings/weight:0"
        assert convert_model.find_old_key(fresh, idx, sfx, grps) == full

    def test_prepend_top_group_for_doubled_prefix(self) -> None:
        """Non-CRF Keras-2 pattern: HDF5 has 'dense/dense/kernel:0', fresh var is 'dense/kernel:0'."""
        idx, sfx, grps = self._make({"dense/dense/kernel:0": np.zeros((768, 10))})
        grps = ["dense"]
        assert convert_model.find_old_key("dense/kernel:0", idx, sfx, grps) == "dense/dense/kernel:0"

    def test_prepend_top_group_for_transformer_doubled(self) -> None:
        """Non-CRF: HDF5 has 'tf_bert_model/tf_bert_model/bert/.../weight:0'."""
        full = "tf_bert_model/tf_bert_model/bert/embeddings/word_embeddings/weight:0"
        idx, sfx, grps = self._make({full: np.zeros((31116, 768))})
        grps = ["tf_bert_model"]
        fresh = "tf_bert_model/bert/embeddings/word_embeddings/weight:0"
        assert convert_model.find_old_key(fresh, idx, sfx, grps) == full

    def test_numeric_suffix_remap_for_nth_instance(self) -> None:
        """Old model was 2nd instance (tf_bert_model_1), fresh is 1st (tf_bert_model)."""
        full = "tf_bert_model_1/tf_bert_model_1/bert/embeddings/word_embeddings/weight:0"
        idx, sfx, grps = self._make({full: np.zeros((28996, 768))})
        grps = ["dense_1", "dropout_75", "input_token", "tf_bert_model_1", "top_level_model_weights"]
        fresh = "tf_bert_model/bert/embeddings/word_embeddings/weight:0"
        assert convert_model.find_old_key(fresh, idx, sfx, grps) == full

    def test_numeric_suffix_remap_for_dense(self) -> None:
        """Old model has dense_1/dense_1/kernel:0, fresh has dense/kernel:0."""
        full = "dense_1/dense_1/kernel:0"
        idx, sfx, grps = self._make({full: np.zeros((768, 2))})
        grps = ["dense_1"]
        fresh = "dense/kernel:0"
        assert convert_model.find_old_key(fresh, idx, sfx, grps) == full


class TestDetectTask:
    def test_seq(self, tmp_path: Path) -> None:
        d = tmp_path / "sequenceLabelling" / "grobid-foo"
        d.mkdir(parents=True)
        assert convert_model.detect_task(d) == "seq"

    def test_class(self, tmp_path: Path) -> None:
        d = tmp_path / "textClassification" / "some-classifier"
        d.mkdir(parents=True)
        assert convert_model.detect_task(d) == "class"

    def test_fallback_seq_via_preprocessor(self, tmp_path: Path) -> None:
        d = tmp_path / "custom-models" / "my-model"
        d.mkdir(parents=True)
        (d / "preprocessor.json").write_text("{}")
        assert convert_model.detect_task(d) == "seq"

    def test_fallback_class_via_config_only(self, tmp_path: Path) -> None:
        d = tmp_path / "custom-models" / "my-classifier"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        assert convert_model.detect_task(d) == "class"

    def test_ambiguous_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "someOtherDir" / "model"
        d.mkdir(parents=True)
        with pytest.raises(ValueError, match="auto-detect"):
            convert_model.detect_task(d)


class TestCopyAndRewrite:
    def _make_source(self, tmp_path: Path, *, with_transformer: bool) -> Path:
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text(json.dumps({"model_name": "original-name", "architecture": "BidLSTM_CRF"}))
        (src / "preprocessor.json").write_text(json.dumps({"vocab_tag": {}}))
        if with_transformer:
            (src / "transformer-config.json").write_text("{}")
            tokdir = src / "transformer-tokenizer"
            tokdir.mkdir()
            (tokdir / "vocab.txt").write_text("foo\nbar\n")
        return src

    def test_copies_required_files(self, tmp_path: Path) -> None:
        src = self._make_source(tmp_path, with_transformer=False)
        dst = tmp_path / "dst"
        convert_model.copy_artifacts(src, dst, task="seq")
        assert (dst / "config.json").exists()
        assert (dst / "preprocessor.json").exists()

    def test_copies_transformer_artifacts_if_present(self, tmp_path: Path) -> None:
        src = self._make_source(tmp_path, with_transformer=True)
        dst = tmp_path / "dst"
        convert_model.copy_artifacts(src, dst, task="seq")
        assert (dst / "transformer-config.json").exists()
        assert (dst / "transformer-tokenizer" / "vocab.txt").read_text() == "foo\nbar\n"

    def test_rewrite_model_name_updates_config(self, tmp_path: Path) -> None:
        src = self._make_source(tmp_path, with_transformer=False)
        dst = tmp_path / "my-new-name"
        convert_model.copy_artifacts(src, dst, task="seq")
        convert_model.rewrite_model_name(dst)
        cfg = json.loads((dst / "config.json").read_text())
        assert cfg["model_name"] == "my-new-name"

    def test_class_task_does_not_require_preprocessor(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text(json.dumps({"model_name": "x", "architecture": "bert"}))
        dst = tmp_path / "dst"
        convert_model.copy_artifacts(src, dst, task="class")
        assert (dst / "config.json").exists()
        assert not (dst / "preprocessor.json").exists()

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        with pytest.raises(FileNotFoundError, match="config.json"):
            convert_model.copy_artifacts(src, dst, task="seq")

    def test_redownload_tokenizer_skips_copy(self, tmp_path: Path) -> None:
        src = self._make_source(tmp_path, with_transformer=True)
        dst = tmp_path / "dst"
        convert_model.copy_artifacts(src, dst, task="seq", redownload_tokenizer=True)
        assert not (dst / "transformer-tokenizer").exists()
        assert (dst / "transformer-config.json").exists()


class TestPatchTokenizerConfig:
    def test_removes_conflicting_key(self, tmp_path: Path) -> None:
        tok_dir = tmp_path / "transformer-tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text(
            json.dumps({"add_special_tokens": True, "model_max_length": 512})
        )
        convert_model.patch_tokenizer_config(tmp_path)
        cfg = json.loads((tok_dir / "tokenizer_config.json").read_text())
        assert "add_special_tokens" not in cfg
        assert cfg["model_max_length"] == 512

    def test_noop_when_no_conflict(self, tmp_path: Path) -> None:
        tok_dir = tmp_path / "transformer-tokenizer"
        tok_dir.mkdir()
        original = json.dumps({"model_max_length": 512})
        (tok_dir / "tokenizer_config.json").write_text(original)
        convert_model.patch_tokenizer_config(tmp_path)
        assert (tok_dir / "tokenizer_config.json").read_text() == original

    def test_noop_when_no_tokenizer_dir(self, tmp_path: Path) -> None:
        convert_model.patch_tokenizer_config(tmp_path)  # should not raise


class TestUnmatchedPolicy:
    def test_abort_default(self) -> None:
        convert_model.FORCE_PARTIAL = False
        with pytest.raises(ValueError, match="No old-HDF5 match"):
            convert_model._handle_unmatched("some/var:0", (3,), [])

    def test_force_partial_returns_keep_init(self, capsys) -> None:
        convert_model.FORCE_PARTIAL = True
        try:
            policy = convert_model._handle_unmatched("some/var:0", (3,), [])
        finally:
            convert_model.FORCE_PARTIAL = False
        assert policy == "keep_init"
        assert "unmatched" in capsys.readouterr().out
