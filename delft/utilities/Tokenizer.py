import regex as re

# Generic simple tokenizer for Indo-European languages
# also python side of GROBID default tokenizer, used for Indo-European languages

delimiters = "\n\r\t\f\u00A0([ •*,:;?.!/)-−–‐\"“”‘’'`$]*\u2666\u2665\u2663\u2660\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B"
regex = '|'.join(map(re.escape, delimiters))
pattern = re.compile('('+regex+')') 
# additional parenthesis above are for capturing delimiters and keep then in the token list

blanks = ' \t\n\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B'

def tokenize(text):
    """
    Tokenization following the above pattern with offset information and keep 
    blank characters (space family) as tokens
    """
    return tokenizeAndFilter(text, filterBlank=False)


def tokenizeAndFilter(text, filterBlank=True):
    """
    Tokenization following the above pattern with offset information and with 
    filtering of blank characters (space family)
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
        if not filterBlank or token not in blanks:
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
    