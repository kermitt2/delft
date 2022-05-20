import logging

from delft.sequenceLabelling import reader

LOGGER = logging.getLogger(__name__)


def test_load_data_crf_string():
    x, y = reader.load_data_crf_string("""In in I In In In n In In In INITCAP NODIGIT 0 NOPUNCT In Xx Xx SAMEFONT SAMEFONTSIZE false false BASELINE false
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

    assert len(x) == 6
    assert x[0][0] == 'In'
    assert x[0][1] == 'just'

    assert x[3][1] == 'replacing'

    assert x[4][0] == 'Meanwhile'