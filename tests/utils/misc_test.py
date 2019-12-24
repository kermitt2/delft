from delft.utilities.misc import parse_number_ranges, parse_dict

# derived from https://github.com/elifesciences/sciencebeam-trainer-delft/tree/develop/tests

class TestParseNumberRanges:
    def test_should_return_empty_array_for_empty_expr(self):
        assert parse_number_ranges('') == []

    def test_should_parse_single_number(self):
        assert parse_number_ranges('1') == [1]

    def test_should_parse_comma_separated_numbers(self):
        assert parse_number_ranges('1,2,3') == [1, 2, 3]

    def test_should_parse_single_number_range(self):
        assert parse_number_ranges('1-3') == [1, 2, 3]

    def test_should_parse_multiple_number_ranges(self):
        assert parse_number_ranges('1-3,5-6') == [1, 2, 3, 5, 6]

    def test_should_ignore_spaces(self):
        assert parse_number_ranges(' 1 - 3 , 5 - 6 ') == [1, 2, 3, 5, 6]


class TestParseDict:
    def test_should_return_empty_dict_for_empty_expr(self):
        assert parse_dict('') == {}

    def test_should_parse_single_key_value_pair(self):
        assert parse_dict('key1=value1') == {'key1': 'value1'}

    def test_should_allow_equals_sign_in_value(self):
        assert parse_dict('key1=value=1') == {'key1': 'value=1'}

    def test_should_parse_multiple_key_value_pair(self):
        assert parse_dict(
            'key1=value1|key2=value2', delimiter='|'
        ) == {'key1': 'value1', 'key2': 'value2'}

    def test_should_ignore_spaces(self):
        assert parse_dict(
            ' key1 = value1 | key2 = value2 ', delimiter='|'
        ) == {'key1': 'value1', 'key2': 'value2'}
