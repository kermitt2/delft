from delft.utilities.Tokenizer import tokenizeAndFilterSimple, tokenizeAndFilter


class TestTokenizer:

    def test_tokenizer_filter_simple(self):
        input = 'this is a test, but a stupid test!!'

        output = tokenizeAndFilterSimple(input)

        assert len(output) == 11
        assert output == ['this', 'is', 'a', 'test', ',', 'but', 'a', 'stupid', 'test', '!', '!']

    def test_tokenizer_filter(self):
        input = 'this is a test, but a stupid test!!'

        output = tokenizeAndFilter(input)

        assert len(output) == 2
        assert output[0] == ['this', 'is', 'a', 'test', ',', 'but', 'a', 'stupid', 'test', '!', '!']
        assert output[1] == [(0, 4), (5, 7), (8, 9), (10, 14), (14, 15), (16, 19), (20, 21), (22, 28), (29, 33),
                             (33, 34), (34, 35)]

    def test_tokenizer_filter_simple_with_breaklines(self):
        input = '\nthis is yet \u2666 another, dummy... test,\na [stupid] test?!'

        output = tokenizeAndFilterSimple(input)

        assert len(output) == 19
        assert output == ['this', 'is', 'yet', '\u2666', 'another', ',', 'dummy', '.', '.', '.', 'test', ',', 'a',
                          '[', 'stupid', ']', 'test', '?', '!']

    def test_tokenizer_filter_with_breaklines(self):
        input = '\nthis is yet \u2666 another, dummy... test,\na [stupid] test?!'

        output = tokenizeAndFilter(input)

        assert len(output) == 2
        assert output[0] == ['this', 'is', 'yet', '\u2666', 'another', ',', 'dummy', '.', '.', '.', 'test', ',', 'a',
                             '[', 'stupid', ']', 'test', '?', '!']
        assert output[1] == [(1, 5), (6, 8), (9, 12), (13, 14), (15, 22), (22, 23), (24, 29), (29, 30), (30, 31),
                             (31, 32), (33, 37), (37, 38), (39, 40), (41, 42), (42, 48), (48, 49), (50, 54), (54, 55),
                             (55, 56)]
