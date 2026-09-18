import logging

from delft.sequenceLabelling import reader

LOGGER = logging.getLogger(__name__)


def test_load_data_crf_string():
    x, y = (
        reader.load_data_crf_string("""In in I In In In n In In In INITCAP NODIGIT 0 NOPUNCT In Xx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
just just j ju jus just t st ust just NOCAPS NODIGIT 0 NOPUNCT just xxxx x SAMEFONT SAMEFONTSIZE false false BASELINE false

Soon soon S So Soo Soon n on oon Soon INITCAP NODIGIT 0 NOPUNCT Soon Xxxx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
after after a af aft afte r er ter fter NOCAPS NODIGIT 0 NOPUNCT after xxxx x SAMEFONT SAMEFONTSIZE false false BASELINE false

Therefore therefore T Th The Ther e re ore fore INITCAP NODIGIT 0 NOPUNCT Therefore Xxxx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
, , , , , , , , , , ALLCAPS NODIGIT 1 COMMA , , , SAMEFONT SAMEFONTSIZE false false BASELINE false

By by B By By By y By By By INITCAP NODIGIT 0 NOPUNCT By Xx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
replacing replacing r re rep repl g ng ing cing NOCAPS NODIGIT 0 NOPUNCT replacing xxxx x SAMEFONT SAMEFONTSIZE false false BASELINE false

Meanwhile meanwhile M Me Mea Mean e le ile hile INITCAP NODIGIT 0 NOPUNCT Meanwhile Xxxx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
, , , , , , , , , , ALLCAPS NODIGIT 1 COMMA , , , SAMEFONT SAMEFONTSIZE false false BASELINE false

More more M Mo Mor More e re ore More INITCAP NODIGIT 0 NOPUNCT More Xxxx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
excitingly excitingly e ex exc exci y ly gly ngly NOCAPS NODIGIT 0 NOPUNCT excitingly xxxx x SAMEFONT SAMEFONTSIZE false false BASELINE false
, , , , , , , , , , ALLCAPS NODIGIT 1 COMMA , , , SAMEFONT SAMEFONTSIZE false false BASELINE false
""")
    )

    assert len(x) == 6
    assert x[0][0] == "In"
    assert x[0][1] == "just"

    assert x[3][1] == "replacing"

    assert x[4][0] == "Meanwhile"


# the fields of the second line are two spaces apart, as in some GROBID training files,
# and those of the third a tab apart
SPACED = "Table table T BLOCKSTART I-<content>\n1  1  1  BLOCKIN  <content>\n:\t:\t:\tBLOCKIN\t<content>\n\n"


class TestFieldSeparators:
    """Splitting on every single space gave an empty field for each extra one."""

    def test_split_fields(self):
        assert reader.split_fields("a  b \t c\n") == ["a", "b", "c"]
        assert reader.split_fields("a b") == ["a", "b"]

    def test_with_labels_from_a_string(self):
        x, y, f = reader.load_data_and_labels_crf_string(SPACED)
        assert [list(tokens) for tokens in x] == [["Table", "1", ":"]]
        assert [list(row) for row in f[0]] == [
            ["table", "T", "BLOCKSTART"],
            ["1", "1", "BLOCKIN"],
            [":", ":", "BLOCKIN"],
        ]
        assert list(y[0]) == ["B-<content>", "I-<content>", "I-<content>"]

    def test_with_labels_from_a_file(self, tmp_path):
        path = tmp_path / "spaced.train"
        path.write_text(SPACED + SPACED.replace("Table", "Figure"))
        x, y, f = reader.load_data_and_labels_crf_file(str(path))
        assert {len(row) for document in f for row in document} == {3}
        assert [tokens[0] for tokens in x] == ["Table", "Figure"]

    def test_without_labels(self, tmp_path):
        unlabelled = "Table table T BLOCKSTART\n1  1  1  BLOCKIN\n\n"
        x, f = reader.load_data_crf_string(unlabelled)
        assert {len(row) for document in f for row in document} == {len(f[0][0])}
        path = tmp_path / "spaced.data"
        path.write_text(unlabelled)
        x_file, f_file = reader.load_data_crf_file(str(path))
        assert [list(row) for row in f_file[0]] == [list(row) for row in f[0]]
