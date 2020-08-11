from delft.sequenceLabelling.preprocess import WordPreprocessor


class TestWordPreprocessor:

    def test_load_example(self):
        p = WordPreprocessor.load('sequenceLabelling/test_data/preprocessor.json')

        assert len(p.vocab_char) == 70

    def test_load_withUmmappedVariable_shouldIgnore(self):
        p = WordPreprocessor.load('sequenceLabelling/test_data/preprocessor2.json')

        assert len(p.vocab_char) == 70
