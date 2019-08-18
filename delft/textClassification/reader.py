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



def load_dataseer_corpus_csv(filepath):
    """
    Load texts from the Dataseer dataset type corpus in csv format:

        doi,text,datatype,dataSubtype,leafDatatype

    Classification of the datatype follows a 3-level hierarchy, so the possible 3 classes are returned.
    dataSubtype and leafDatatype are optional

    Returns:
        tuple(numpy array, numpy array, numpy array, numpy array): 
            texts, datatype, datasubtype, leaf datatype

    """
    df = pd.read_csv(filepath)
    df = df[pd.notnull(df['text'])]
    df = df[pd.notnull(df['datatype'])]
    df.iloc[:,1].fillna('NA', inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j,1])

    datatypes = df.iloc[:,2]
    datatypes_list = datatypes.values.tolist()
    datatypes_list = np.asarray(datatypes_list)
    list_classes_datatypes = np.unique(datatypes_list)
    datatypes_final = normalize_classes(datatypes_list, list_classes_datatypes)

    print(df.shape, df.shape[0], df.shape[1])

    if df.shape[1] > 3:
        datasubtypes = df.iloc[:,3]
        datasubtypes_list = datasubtypes.values.tolist()
        datasubtypes_list = np.asarray(datasubtypes_list)
        list_classes_datasubtypes = np.unique(datasubtypes_list)
        datasubtypes_final = normalize_classes(datasubtypes_list, list_classes_datasubtypes)

    if df.shape[1] > 4:
        leafdatatypes = df.iloc[:,4]
        leafdatatypes_list = leafdatatypes.values.tolist()
        leafdatatypes_list = np.asarray(leafdatatypes_list)
        list_classes_leafdatatypes = np.unique(leafdatatypes_list)
        leafdatatypes_final = normalize_classes(leafdatatypes_list, list_classes_leafdatatypes)

    if df.shape[1] == 3:
        return np.asarray(texts_list), datatypes_final, None, None, list_classes_datatypes.tolist(), None, None
    elif df.shape[1] == 4:
        return np.asarray(texts_list), datatypes_final, datasubtypes_final, None, list_classes_datatypes.tolist(), list_classes_datasubtypes.tolist(), None
    else:
        return np.asarray(texts_list), datatypes_final, datasubtypes_final, leafdatatypes_final, list_classes_datatypes.tolist(), list_classes_datasubtypes.tolist(), list_classes_leafdatatypes.tolist()


def normalize_classes(y, list_classes):
    '''
        Replace string values of classes by their index in the list of classes
    '''
    def f(x):
        return np.where(list_classes == x)

    intermediate = np.array([f(xi)[0] for xi in y])
    return np.array([vectorize(xi, len(list_classes)) for xi in intermediate])

def vectorize(index, size):
    '''
    Create a numpy array of the provided size, where value at indicated index is 1, 0 otherwise 
    '''
    result = np.zeros(size)
    if index < size:
        result[index] = 1
    else:
        print("warning: index larger than vector size: ", index, size)
    return result
