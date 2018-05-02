import regex as re

# python side of GROBID default tokenizer, used for Indo-European languages

delimiters = "\n\r\t\f\u00A0([ •*,:;?.!/)-−–‐\"“”‘’'\`$]*\u2666\u2665\u2663\u2660\u00A0"
regex = '|'.join(map(re.escape, delimiters))
pattern = re.compile('('+regex+')') 
# additional parenthesis above are for capturing delimiters and keep then in the token list

blanks = ' \t\n'

def tokenize(text):
    return pattern.split(text)

def tokenizeAndFilter(text):
    #tokens = tokenize(text)
    offset = 0
    offsets = []
    tokens = []
    for index, match in enumerate(pattern.split(text)):
        print(index, match)
        tokens.append(match)
        position = (offset, offset+len(match))
        offsets.append(position)
        offset = offset+len(match)
    #tokens = pattern.split(text)

    finalTokens = []
    finalOffsets = []
    i = 0
    for token in tokens:
        if token not in blanks:
            finalTokens.append(token)
            finalOffsets.append(offsets[i])
        i += 1
    return finalTokens, finalOffsets

def filterSpace(token):
    return (token not in blanks)

if __name__ == "__main__":
    # some tests
    test = 'this is a test, but a stupid test!!'
    print(test)
    print(tokenize(test))
    print(tokenizeAndFilter(test))

    test = '\nthis is yet \u2666 another, dummy... test,\na [stupid] test?!'
    print(test)
    print(tokenize(test))
    print(tokenizeAndFilter(test))