import argparse

import pytest

from delft.utilities.Utilities import parse_number_ranges


class TestParseNumberRanges:
    @pytest.mark.parametrize(
        "text, numbers",
        [("9", [9]), ("9-12", [9, 10, 11, 12]), ("9-12,15, 20-21", [9, 10, 11, 12, 15, 20, 21]), ("3-3", [3])],
    )
    def test_numbers_and_ranges(self, text, numbers):
        assert parse_number_ranges(text) == numbers

    @pytest.mark.parametrize("text", ["", "a", "12-9", "1-2-3", "-3", "4,"])
    def test_rejects_anything_else(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_number_ranges(text)
