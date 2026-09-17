import pytest

from delft.utilities.model_names import ARCHITECTURES, ModelName, build_model_name, split_model_name, validate_suffix


class TestBuildModelName:
    def test_name_without_suffix_is_the_one_used_so_far(self):
        assert build_model_name("header", "BidLSTM_CRF_FEATURES") == "grobid-header-BidLSTM_CRF_FEATURES"

    def test_short_name_may_hold_dashes(self):
        assert build_model_name("affiliation-address", "BidLSTM_CRF") == "grobid-affiliation-address-BidLSTM_CRF"

    def test_suffix_comes_last(self):
        name = build_model_name("header", "BERT_CRF", "scibert_scivocab_cased")
        assert name == "grobid-header-BERT_CRF-scibert_scivocab_cased"

    @pytest.mark.parametrize("suffix", [None, ""])
    def test_empty_suffix_is_no_suffix(self, suffix):
        assert build_model_name("date", "BidLSTM_CRF", suffix) == "grobid-date-BidLSTM_CRF"

    def test_prefix_can_be_dropped(self):
        assert build_model_name("header", "BidLSTM_CRF", "v2", prefix="") == "header-BidLSTM_CRF-v2"


class TestValidateSuffix:
    @pytest.mark.parametrize("suffix", ["potion-base-8M", "batch_size_3", "glove.840B", "v2", "8M"])
    def test_accepts_what_a_directory_name_can_hold(self, suffix):
        assert validate_suffix(suffix) == suffix

    @pytest.mark.parametrize("suffix", ["allenai/scibert", "../up", "with space", "-leading-dash", ".hidden", "tab\t"])
    def test_rejects_what_a_directory_name_cannot_hold(self, suffix):
        with pytest.raises(ValueError, match="Invalid model name suffix"):
            validate_suffix(suffix)


class TestSplitModelName:
    @pytest.mark.parametrize(
        "model_name, expected",
        [
            ("grobid-header-BidLSTM_CRF", ModelName("header", "BidLSTM_CRF", None, "grobid-")),
            (
                "grobid-affiliation-address-BidLSTM_CRF_FEATURES-potion-base-8M",
                ModelName("affiliation-address", "BidLSTM_CRF_FEATURES", "potion-base-8M", "grobid-"),
            ),
            ("grobid-header-BERT_CRF-batch_size_3", ModelName("header", "BERT_CRF", "batch_size_3", "grobid-")),
            ("grobid-name-header-wapiti-biorxiv", ModelName("name-header", "wapiti", "biorxiv", "grobid-")),
            ("datasets-BidLSTM_CRF", ModelName("datasets", "BidLSTM_CRF", None, "")),
            # the architecture is the first one met: a suffix can name another
            ("grobid-date-BidLSTM_CRF-vs-BERT", ModelName("date", "BidLSTM_CRF", "vs-BERT", "grobid-")),
        ],
    )
    def test_splits(self, model_name, expected):
        assert split_model_name(model_name) == expected

    @pytest.mark.parametrize("model_name", ["license_gru", "grobid-header", "BidLSTM_CRF", "grobid-BidLSTM_CRF"])
    def test_name_without_a_short_name_and_an_architecture_does_not_split(self, model_name):
        assert split_model_name(model_name) is None

    @pytest.mark.parametrize("suffix", [None, "potion-base-8M"])
    @pytest.mark.parametrize("architecture", ARCHITECTURES)
    def test_split_undoes_build(self, architecture, suffix):
        name = build_model_name("reference-segmenter", architecture, suffix)
        assert split_model_name(name) == ModelName("reference-segmenter", architecture, suffix, "grobid-")

    def test_architectures_are_the_ones_of_the_model_registry(self):
        from delft.sequenceLabelling.models import MODEL_REGISTRY

        assert set(ARCHITECTURES) == set(MODEL_REGISTRY)
