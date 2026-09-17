import pytest

from delft.utilities.model_names import build_model_name, validate_suffix


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
