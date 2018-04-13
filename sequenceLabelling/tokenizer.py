# python side of GROBID default tokenizer, used for Indo-European languages

import regex as re

delimiters = "\n\r\t\f\u00A0([ •*,:;?.!/)-−–‐\"“”‘’'\`$]*\u2666\u2665\u2663\u2660\u00A0"
regex = '|'.join(map(re.escape, delimiters))
pattern = re.compile('('+regex+')') 
# additional parenthesis above are for capturing delimiters and keep then in the token list

blanks = ' \t\n'

def tokenize(text):
    return pattern.split(text)

def tokenizeAndFilter(text):
    tokens = tokenize(text)
    finalTokens = []
    for token in tokens:
        if token not in blanks:
            finalTokens.append(token)
    return finalTokens

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