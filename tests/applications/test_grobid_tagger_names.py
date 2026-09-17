"""The name of a model decides the directory it is saved to and loaded from, so the
train, eval and tag actions of grobidTagger have to agree on it."""

from unittest.mock import MagicMock, patch

from delft.applications import grobidTagger


def _configured_name(**kwargs):
    arguments = {"model": "header", "architecture": "BidLSTM_CRF", "output_path": None}
    arguments.update(kwargs)
    return grobidTagger.configure(**arguments)[2]


class TestConfigure:
    def test_name_without_suffix_is_unchanged(self):
        assert _configured_name() == "grobid-header-BidLSTM_CRF"

    def test_suffix_is_appended(self):
        assert _configured_name(suffix="potion-base-8M") == "grobid-header-BidLSTM_CRF-potion-base-8M"

    def test_model_saved_under_output_keeps_no_prefix(self):
        assert _configured_name(output_path="/tmp/models", suffix="v2") == "header-BidLSTM_CRF-v2"

    def test_transformer_architecture(self):
        name = _configured_name(architecture="BERT_CRF", suffix="scibert_scivocab_cased")
        assert name == "grobid-header-BERT_CRF-scibert_scivocab_cased"


class TestLoadingActions:
    def test_tag_loads_the_model_with_the_suffix(self):
        with patch.object(grobidTagger, "Sequence") as sequence:
            sequence.return_value.tag.return_value = {}
            grobidTagger.annotate_text(["a text"], "date", "json", architecture="BidLSTM_CRF", suffix="v2")
        assert sequence.call_args.args[0] == "grobid-date-BidLSTM_CRF-v2"

    def test_eval_loads_the_model_with_the_suffix(self):
        data = ([["a"]], [["O"]], [[["f"]]])
        with (
            patch.object(grobidTagger, "Sequence") as sequence,
            patch.object(grobidTagger, "load_data_and_labels_crf_file", return_value=data),
        ):
            sequence.return_value = MagicMock()
            grobidTagger.eval_("date", input_path="some.data", architecture="BidLSTM_CRF", suffix="v2")
        assert sequence.call_args.args[0] == "grobid-date-BidLSTM_CRF-v2"

    def test_tag_loads_the_model_from_the_default_directory(self):
        with patch.object(grobidTagger, "Sequence") as sequence:
            sequence.return_value.tag.return_value = {}
            grobidTagger.annotate_text(["a text"], "date", "json")
        sequence.return_value.load.assert_called_once_with()

    def test_tag_loads_the_model_from_where_it_is_told(self):
        with patch.object(grobidTagger, "Sequence") as sequence:
            sequence.return_value.tag.return_value = {}
            grobidTagger.annotate_text(["a text"], "date", "json", input_model_path="hf://buckets/owner/bucket")
        sequence.return_value.load.assert_called_once_with("hf://buckets/owner/bucket")

    def test_eval_loads_the_model_from_where_it_is_told(self):
        data = ([["a"]], [["O"]], [[["f"]]])
        with (
            patch.object(grobidTagger, "Sequence") as sequence,
            patch.object(grobidTagger, "load_data_and_labels_crf_file", return_value=data),
        ):
            grobidTagger.eval_("date", input_path="some.data", input_model_path="hf://owner/repository@v1")
        sequence.return_value.load.assert_called_once_with("hf://owner/repository@v1")
