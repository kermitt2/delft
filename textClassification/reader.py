import numpy as np
import xml
from xml.sax import make_parser, handler
import pandas as pd

def load_texts_and_classes(filepath):
    """Load texts and classes from a file in the following simple tab-separated format:

    id_0    text_0  class_00 ...    class_n0
    id_1    text_1  class_01 ...    class_n1
    ...
    id_m    text_m  class_0m  ...   class_nm

    text has no EOF and no tab

    Returns:
        tuple(numpy array, numpy array): texts and classes.

    Example:
        >>> filenameCsv = 'toxic.csv'
        >>> texts, classes = load_texts_and_classes(filenameCsv)
    """
    texts = []
    classes = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if (len(line) is 0):
                continue
            pieces = line.split('\t')
            if (len(pieces) < 3):
                printf("Warning: number of fields in the data file too low for line:", line)
            texts.append(pieces[1])
            classes.append(pieces[2:])

    return np.asarray(texts), np.asarray(classes)


def load_texts_and_classes_pandas(filepath):
    """Load texts and classes from a file in csv format using pandas dataframe:

    id      text    class_0     ... class_n
    id_0    text_0  class_00    ... class_n0
    id_1    text_1  class_01    ... class_n1
    ...
    id_m    text_m  class_0m    ... class_nm

    It should support any CSV file format.

    Returns:
        tuple(numpy array, numpy array): texts and classes.

    Example:
        >>> filenameCsv = 'toxic.csv'
        >>> texts, classes = load_texts_and_classes_pandas(filenameCsv)
    """

    df = pd.read_csv(filepath)
    df.iloc[:,1].fillna('MISSINGVALUE', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j,1])

    classes = df.iloc[:,2:]
    classes_list = classes.values.tolist()

    return texts_list, np.asarray(classes_list)


def load_texts_pandas(filepath):
    """Load texts from a file in csv format using pandas dataframe:

    id      text
    id_0    text_0
    id_1    text_1
    ...
    id_m    text_m

    It should support any CSV file format.

    Returns:
        numpy array: texts

    Example:
        >>> filenameCsv = 'toxic.csv'
        >>> texts = load_texts(filenameCsv)
    """

    df = pd.read_csv(filepath)
    df.iloc[:,1].fillna('MISSINGVALUE', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j,1])

    return texts_list
