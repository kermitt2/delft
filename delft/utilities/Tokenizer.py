import regex as re

# Generic simple tokenizer for Indo-European languages
# also python side of GROBID default tokenizer, used for Indo-European languages

delimiters = "\n\r\t\f\u00A0([ •*,:;?.!/)-−–‐\"“”‘’'`$]*\u2666\u2665\u2663\u2660\u00A0"
regex = '|'.join(map(re.escape, delimiters))
pattern = re.compile('('+regex+')') 
# additional parenthesis above are for capturing delimiters and keep then in the token list

blanks = ' \t\n'

def tokenizeAndFilter(text):
    """
    Tokenization following the above pattern with offset information
    """
    offset = 0
    offsets = []
    tokens = []
    for index, match in enumerate(pattern.split(text)):
        tokens.append(match)
        position = (offset, offset+len(match))
        offsets.append(position)
        offset = offset+len(match)

    finalTokens = []
    finalOffsets = []
    i = 0
    for token in tokens:
        if token not in blanks:
            finalTokens.append(token)
            finalOffsets.append(offsets[i])
        i += 1
    return finalTokens, finalOffsets


def tokenizeAndFilterSimple(text):
    """
    Tokenization following the above pattern without offset information
    """
    tokens = []
    for index, match in enumerate(pattern.split(text)):
        tokens.append(match)

    finalTokens = []
    i = 0
    for token in tokens:
        if token not in blanks:
            finalTokens.append(token)
        i += 1

    return finalTokens


def filterSpace(token):
    return (token not in blanks)