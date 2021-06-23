import logging

from delft.sequenceLabelling.tagger import get_entities_with_offsets
from delft.utilities.Tokenizer import tokenizeAndFilter

LOGGER = logging.getLogger(__name__)


def test_get_entities_with_offsets():
    original_string = '(Mo -x 1 T x ) 3 Sb 7 with \uf084 x 0.1'
    tokens = ['(', 'Mo', '-', 'x', '1', 'T', 'x', ')', '3', 'Sb', '7', 'with', '\uf084', 'x', '0', '.', '1']
    tags = ['B-<formula>', 'I-<formula>', 'I-<formula>', 'I-<formula>', 'I-<formula>', 'I-<formula>', 'I-<formula>',
            'I-<formula>', 'I-<formula>', 'I-<formula>', 'I-<formula>', 'O', 'O', 'B-<variable>', 'B-<value>',
            'I-<value>', 'I-<value>']
    types = [tag.split('-')[-1] for tag in tags]

    offsets = [(0, 1), (1, 3), (4, 5), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 19), (20, 21),
               (22, 26), (27, 28), (29, 30), (31, 32), (32, 33), (33, 34)]

    spaces = [offsets[offsetIndex][1] != offsets[offsetIndex + 1][0] for offsetIndex in range(0, len(offsets) - 1)]

    for index in range(0, len(offsets)):
        chunk = original_string[offsets[index][0]:offsets[index][1]]

        assert chunk == tokens[index]

    entities_with_offsets = get_entities_with_offsets(tags, offsets)
    # (chunk_type, chunk_start, chunk_end, pos_start, pos_end)

    assert len(entities_with_offsets) == 3
    entity0 = entities_with_offsets[0]
    assert entity0[0] == "<formula>"
    entity_text = original_string[entity0[3]: entity0[4] + 1]
    assert entity_text == "(Mo -x 1 T x ) 3 Sb 7"
    assert tokens[entity0[1]:entity0[2]] == tokenizeAndFilter(entity_text)[0]

    entity1 = entities_with_offsets[1]
    assert entity1[0] == "<variable>"
    entity_text = original_string[entity1[3]: entity1[4] + 1]
    assert entity_text == "x"
    assert tokens[entity1[1]:entity1[2]] == tokenizeAndFilter(entity_text)[0]

    entity2 = entities_with_offsets[2]
    assert entity2[0] == "<value>"
    entity_text = original_string[entity2[3]: entity2[4] + 1]
    assert entity_text == "0.1"
    assert tokens[entity2[1]:entity2[2]] == tokenizeAndFilter(entity_text)[0]

    # for item in entities_with_offsets:
    #     type = item[0]
    #     token_start = item[1]
    #     token_end = item[2]
    #     char_start = item[3]
    #     char_end = item[4]
    #
    #     text = ''.join([tokens[idx] + (' ' if spaces[idx] else '') for idx in range(token_start, token_end)])
    #     if text.endswith(' '):
    #         text = text[0:-1]
    #
    #     assert text == original_string[char_start: char_end + 1]
