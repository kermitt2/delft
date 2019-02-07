import numpy as np
import xml
from xml.sax import make_parser, handler
import pandas as pd


def load_texts_and_classes(filepath):
    """
    Load texts and classes from a file in the following simple tab-separated format:

    id_0    text_0  class_00 ...    class_n0
    id_1    text_1  class_01 ...    class_n1
    ...
    id_m    text_m  class_0m  ...   class_nm

    text has no EOF and no tab

    Returns:
        tuple(numpy array, numpy array): texts and classes

    """
    texts = []
    classes = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if (len(line) is 0):
                continue
            pieces = line.split('\t')
            if (len(pieces) < 3):
                print("Warning: number of fields in the data file too low for line:", line)
            texts.append(pieces[1])
            classes.append(pieces[2:])

    return np.asarray(texts), np.asarray(classes)


def load_texts_and_classes_pandas(filepath):
    """
    Load texts and classes from a file in csv format using pandas dataframe:

    id      text    class_0     ... class_n
    id_0    text_0  class_00    ... class_n0
    id_1    text_1  class_01    ... class_n1
    ...
    id_m    text_m  class_0m    ... class_nm

    It should support any CSV file format.

    Returns:
        tuple(numpy array, numpy array): texts and classes

    """

    df = pd.read_csv(filepath)
    df.iloc[:,1].fillna('MISSINGVALUE', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j,1])

    classes = df.iloc[:,2:]
    classes_list = classes.values.tolist()

    return np.asarray(texts_list), np.asarray(classes_list)


def load_texts_pandas(filepath):
    """
    Load texts from a file in csv format using pandas dataframe:

    id      text
    id_0    text_0
    id_1    text_1
    ...
    id_m    text_m

    It should support any CSV file format.

    Returns:
        numpy array: texts

    """

    df = pd.read_csv(filepath)
    df.iloc[:,1].fillna('MISSINGVALUE', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j,1])

    return np.asarray(texts_list)


def load_citation_sentiment_corpus(filepath):
    """
    Load texts from the citation sentiment corpus:

    Source_Paper  Target_Paper    Sentiment   Citation_Text

    sentiment "value" can o (neutral), p (positive), n (negative)

    Returns:
        tuple(numpy array, numpy array): texts and polarity

    """

    texts = []
    polarities = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if (len(line) is 0):
                continue
            if line.startswith('#'):
                continue

            pieces = line.split('\t')
            if (len(pieces) != 4):
                print("Warning: incorrect number of fields in the data file for line:", line)
                continue
            text = pieces[3]
            # remove start/end quotes
            text = text[1:len(text)-1]
            texts.append(text)

            polarity = []
            if pieces[2] is 'n':
                polarity.append(1)
            else:
                polarity.append(0)
            if pieces[2] is 'o':
                polarity.append(1)
            else:
                polarity.append(0)
            if pieces[2] is 'p':
                polarity.append(1)
            else:
                polarity.append(0)
            polarities.append(polarity)

    return np.asarray(texts), np.asarray(polarities)



