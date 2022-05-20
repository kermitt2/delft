import numpy as np
import xml
import gzip
import json
from xml.sax import make_parser, handler
import pandas as pd
from delft.utilities.numpy import shuffle_triple_with_view


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
            if len(line) == 0:
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
            if len(line) == 0:
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
            if pieces[2] == 'n':
                polarity.append(1)
            else:
                polarity.append(0)
            if pieces[2] == 'o':
                polarity.append(1)
            else:
                polarity.append(0)
            if pieces[2] == 'p':
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
    if 'datatype' in df.columns:
        df = df[pd.notnull(df['datatype'])]
    if 'reuse' in df.columns:    
        df = df[pd.notnull(df['reuse'])]
    df.iloc[:,1].fillna('NA', inplace=True)

    # shuffle, note that this is important for the reuse prediction, the following shuffle in place
    # and reset the index
    df = df.sample(frac=1).reset_index(drop=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j,1])

    if 'reuse' in df.columns:  
        # we simply get the reuse boolean value for the examples
        datareuses = df.iloc[:,2]
        reuse_list = datareuses.values.tolist()
        reuse_list = np.asarray(reuse_list)
        # map boolean values to [0,1]
        def map_boolean(x):
            return [1.0,0.0] if x else [0.0,1.0]
        reuse_list = np.array(list(map(map_boolean, reuse_list)))
        print(reuse_list)
        return np.asarray(texts_list), reuse_list, None, None, ["not_reuse", "reuse"], None, None

    # otherwise we have the list of datatypes, and optionally subtypes and leaf datatypes
    datatypes = df.iloc[:,2]
    datatypes_list = datatypes.values.tolist()
    datatypes_list = np.asarray(datatypes_list)
    datatypes_list_lower = np.char.lower(datatypes_list)
    list_classes_datatypes = np.unique(datatypes_list_lower)    
    datatypes_final = normalize_classes(datatypes_list_lower, list_classes_datatypes)

    print(df.shape, df.shape[0], df.shape[1])

    if df.shape[1] > 3:
        # remove possible row with 'no_dataset'
        df = df[~df.datatype.str.contains("no_dataset")]
        datasubtypes = df.iloc[:,3]
        datasubtypes_list = datasubtypes.values.tolist()
        datasubtypes_list = np.asarray(datasubtypes_list)
        datasubtypes_list_lower = np.char.lower(datasubtypes_list)
        list_classes_datasubtypes = np.unique(datasubtypes_list_lower)
        datasubtypes_final = normalize_classes(datasubtypes_list_lower, list_classes_datasubtypes)

    '''
    if df.shape[1] > 4:
        leafdatatypes = df.iloc[:,4]
        leafdatatypes_list = leafdatatypes.values.tolist()
        leafdatatypes_list = np.asarray(leafdatatypes_list)
        #leafdatatypes_list_lower = np.char.lower(leafdatatypes_list)
        leafdatatypes_list_lower = leafdatatypes_list
        list_classes_leafdatatypes = np.unique(leafdatatypes_list_lower)  
        print(list_classes_leafdatatypes)
        leafdatatypes_final = normalize_classes(leafdatatypes_list_lower, list_classes_leafdatatypes)
    '''

    if df.shape[1] == 3:
        return np.asarray(texts_list), datatypes_final, None, None, list_classes_datatypes.tolist(), None, None
    #elif df.shape[1] == 4:
    else:
        return np.asarray(texts_list), datatypes_final, datasubtypes_final, None, list_classes_datatypes.tolist(), list_classes_datasubtypes.tolist(), None
    '''
    else:
        return np.asarray(texts_list), datatypes_final, datasubtypes_final, leafdatatypes_final, list_classes_datatypes.tolist(), list_classes_datasubtypes.tolist(), list_classes_leafdatatypes.tolist()
    '''

def load_software_use_corpus_json(json_gz_file_path):
    """
    Load texts and classes from the corresponding Softcite corpus export in gzipped json format

    Classification of the software usage is binary

    Returns:
        tuple(numpy array, numpy array): 
            texts, binary class (used/not_used)

    """

    texts_list = []
    classes_list = []

    with gzip.GzipFile(json_gz_file_path, 'r') as fin:
        data = json.loads(fin.read().decode('utf-8'))
        if not "documents" in data:
            print("There is no usable classified text in the corpus file", json_gz_file_path)
            return None, None 
        for document in data["documents"]:
            for segment in document["texts"]:
                if "entity_spans" in segment:
                    if not "text" in segment:
                        continue
                    text = segment["text"]
                    for entity_span in segment["entity_spans"]:
                        if entity_span["type"] == "software":
                            texts_list.append(text)
                            if "used" in entity_span and entity_span["used"]:
                                classes_list.append("used")
                            else:
                                classes_list.append("not_used")
    list_possible_classes = np.unique(classes_list)
    classes_list_final = normalize_classes(classes_list, list_possible_classes)

    texts_list_final = np.asarray(texts_list)

    texts_list_final, classes_list_final, _ = shuffle_triple_with_view(texts_list_final, classes_list_final)

    return texts_list_final, classes_list_final


def load_software_context_corpus_json(json_gz_file_path):
    """
    Load texts and classes from the corresponding Softcite mention corpus export in gzipped json format

    Classification of the software usage is multiclass/multilabel

    Returns:
        tuple(numpy array, numpy array): 
            texts, classes_list

    """

    texts_list = []
    classes_list = []

    with gzip.GzipFile(json_gz_file_path, 'r') as fin:
        data = json.loads(fin.read().decode('utf-8'))
        if not "documents" in data:
            print("There is no usable classified text in the corpus file", json_gz_file_path)
            return None, None 
        for document in data["documents"]:
            for segment in document["texts"]:
                if "entity_spans" in segment:
                    if not "text" in segment:
                        continue
                    text = segment["text"]
                    for entity_span in segment["entity_spans"]:
                        if entity_span["type"] == "software":
                            texts_list.append(text)
                            classes = []
                            if "used" in entity_span and entity_span["used"]:
                                classes.append(1.0)
                            else:
                                classes.append(0.0)

                            if "contribution" in entity_span and entity_span["contribution"]:
                                classes.append(1.0)
                            else:
                                classes.append(0.0)

                            if "shared" in entity_span and entity_span["shared"]:
                                classes.append(1.0)
                            else:
                                classes.append(0.0)    

                            classes_list.append(classes)

    #list_possible_classes = np.unique(classes_list)
    #classes_list_final = normalize_classes(classes_list, list_possible_classes)

    texts_list_final = np.asarray(texts_list)
    classes_list_final = np.asarray(classes_list)
    
    texts_list_final, classes_list_final, _ = shuffle_triple_with_view(texts_list_final, classes_list_final)

    return texts_list_final, classes_list_final


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
