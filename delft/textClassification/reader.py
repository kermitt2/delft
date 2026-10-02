import gzip
import json
import os

import numpy as np
import pandas as pd

from delft.utilities.numpy import shuffle_triple_with_view


class TextsOnDisk:
    """
    The texts of a tab-separated data file, read from the file when they are asked for
    rather than held in memory, for a data set too big for it (see issue #27). Only the
    byte offsets of the lines are kept: 8 bytes per text.

    It is a sequence, as the arrays of texts the readers return are: ``len``, an
    integer index for the text of a line, a slice or a sequence of indices (or a
    boolean mask) for the ``TextsOnDisk`` of those lines, so that shuffling and
    splitting leave the texts on disk, and iteration reads the lines one after the
    other. The dataset of a ``DataLoader`` reads one text at a time from it, in the
    worker processes as well: the file is opened again in every process.
    """

    def __init__(self, filepath, offsets, column=1, encoding="utf-8"):
        self.filepath = filepath
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.column = column
        self.encoding = encoding
        self._handle = None
        self._pid = None

    @classmethod
    def index(cls, filepath, column=1, encoding="utf-8"):
        """
        Read ``filepath`` once, and return the ``TextsOnDisk`` of its non-empty lines and
        the list of the fields of every line after ``column`` (the classes).
        """
        offsets = []
        classes = []
        with open(filepath, "rb") as f:
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode(encoding).strip()
                if len(line) == 0:
                    continue
                pieces = line.split("\t")
                if len(pieces) < column + 2:
                    print("Warning: number of fields in the data file too low for line:", line)
                offsets.append(offset)
                classes.append(pieces[column + 1 :])
        return cls(filepath, offsets, column=column, encoding=encoding), classes

    @property
    def shape(self):
        return (len(self.offsets),)

    def __len__(self):
        return len(self.offsets)

    def _file(self):
        # a handle is not shared with the workers of a DataLoader: each opens its own
        if self._handle is None or self._pid != os.getpid():
            self._handle = open(self.filepath, "rb")
            self._pid = os.getpid()
        return self._handle

    def _read(self, offset):
        f = self._file()
        f.seek(int(offset))
        line = f.readline().decode(self.encoding).strip()
        pieces = line.split("\t")
        return pieces[self.column] if self.column < len(pieces) else ""

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._read(self.offsets[key])
        if isinstance(key, slice):
            return TextsOnDisk(self.filepath, self.offsets[key], column=self.column, encoding=self.encoding)
        indices = np.asarray(key)
        return TextsOnDisk(self.filepath, self.offsets[indices], column=self.column, encoding=self.encoding)

    def __iter__(self):
        for offset in self.offsets:
            yield self._read(offset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def __repr__(self):
        return f"TextsOnDisk({self.filepath!r}, {len(self)} texts)"

    def close(self):
        if self._handle is not None and self._pid == os.getpid():
            self._handle.close()
        self._handle = None
        self._pid = None


def load_texts_and_classes(filepath, in_memory=True):
    """
    Load texts and classes from a file in the following simple tab-separated format:

    id_0    text_0  class_00 ...    class_n0
    id_1    text_1  class_01 ...    class_n1
    ...
    id_m    text_m  class_0m  ...   class_nm

    text has no EOF and no tab

    With ``in_memory=False`` the texts are not loaded: a ``TextsOnDisk`` reads them from
    the file when they are asked for, for a data set too big for memory (issue #27).
    It is used like the array of texts: the classes are still returned as an array.

    Returns:
        tuple(numpy array, numpy array): texts and classes

    """
    if not in_memory:
        texts, classes = TextsOnDisk.index(filepath)
        return texts, np.asarray(classes)

    texts = []
    classes = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            pieces = line.split("\t")
            if len(pieces) < 3:
                print("Warning: number of fields in the data file too low for line:", line)
            texts.append(pieces[1])
            classes.append(pieces[2:])

    return np.asarray(texts), np.asarray(classes)


def load_texts_and_classes_pandas(filepath):
    """
    Load texts and classes from a file in csv format using pandas dataframe, with format as follow:

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
    df.iloc[:, 1].fillna("MISSINGVALUE", inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j, 1])

    classes = df.iloc[:, 2:]
    classes_list = classes.values.tolist()

    return np.asarray(texts_list), np.asarray(classes_list)


def load_texts_and_classes_pandas_no_id(filepath):
    """
    Load texts and classes from a file in csv format using pandas dataframe, with format as follow:

    text    class_0     ... class_n
    text_0  class_00    ... class_n0
    text_1  class_01    ... class_n1
    ...
    text_m  class_0m    ... class_nm

    It should support any CSV file format.

    Returns:
        tuple(numpy array, numpy array): texts and classes

    """

    df = pd.read_csv(filepath)
    df.iloc[:, 1].fillna("MISSINGVALUE", inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j, 0])

    classes = df.iloc[:, 1:]
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
    df.iloc[:, 1].fillna("MISSINGVALUE", inplace=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j, 1])

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
            if line.startswith("#"):
                continue

            pieces = line.split("\t")
            if len(pieces) != 4:
                print(
                    "Warning: incorrect number of fields in the data file for line:",
                    line,
                )
                continue
            text = pieces[3]
            # remove start/end quotes
            text = text[1 : len(text) - 1]
            texts.append(text)

            polarity = []
            if pieces[2] == "n":
                polarity.append(1)
            else:
                polarity.append(0)
            if pieces[2] == "o":
                polarity.append(1)
            else:
                polarity.append(0)
            if pieces[2] == "p":
                polarity.append(1)
            else:
                polarity.append(0)
            polarities.append(polarity)

    return np.asarray(texts), np.asarray(polarities)


def load_citation_intent_corpus(filepath):
    """
    Load texts from the citation intent corpus multicite https://github.com/allenai/multicite (NAACL 2022)

    Source_Paper  Target_Paper    Sentiment   Citation_Text

    sentiment "value" can o (neutral), p (positive), n (negative)

    Returns:
        tuple(numpy array, numpy array): texts and polarity

    """

    texts = []
    classes = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue

            pieces = line.split("\t")
            if len(pieces) != 4:
                print(
                    "Warning: incorrect number of fields in the data file for line:",
                    line,
                )
                continue
            text = pieces[3]
            # remove start/end quotes
            text = text[1 : len(text) - 1]
            texts.append(text)

            the_class = []
            if pieces[2] == "n":
                the_class.append(1)
            else:
                the_class.append(0)
            if pieces[2] == "o":
                the_class.append(1)
            else:
                the_class.append(0)
            if pieces[2] == "p":
                the_class.append(1)
            else:
                the_class.append(0)
            classes.append(the_class)

    return np.asarray(texts), np.asarray(classes)


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
    df = df[pd.notnull(df["text"])]
    if "datatype" in df.columns:
        df = df[pd.notnull(df["datatype"])]
    if "reuse" in df.columns:
        df = df[pd.notnull(df["reuse"])]
    df.iloc[:, 1].fillna("NA", inplace=True)

    # shuffle, note that this is important for the reuse prediction, the following shuffle in place
    # and reset the index
    df = df.sample(frac=1).reset_index(drop=True)

    texts_list = []
    for j in range(0, df.shape[0]):
        texts_list.append(df.iloc[j, 1])

    if "reuse" in df.columns:
        # we simply get the reuse boolean value for the examples
        datareuses = df.iloc[:, 2]
        reuse_list = datareuses.values.tolist()
        reuse_list = np.asarray(reuse_list)

        # map boolean values to [0,1]
        def map_boolean(x):
            return [1.0, 0.0] if x == "no_reuse" else [0.0, 1.0]

        reuse_list = np.array(list(map(map_boolean, reuse_list)))
        print(reuse_list)
        return (
            np.asarray(texts_list),
            reuse_list,
            None,
            None,
            ["no_reuse", "reuse"],
            None,
            None,
        )

    # otherwise we have the list of datatypes, and optionally subtypes and leaf datatypes
    datatypes = df.iloc[:, 2]
    datatypes_list = datatypes.values.tolist()
    datatypes_list = np.asarray(datatypes_list)
    datatypes_list_lower = np.char.lower(datatypes_list)
    list_classes_datatypes = np.unique(datatypes_list_lower)
    datatypes_final = normalize_classes(datatypes_list_lower, list_classes_datatypes)

    print(df.shape, df.shape[0], df.shape[1])

    if df.shape[1] > 3:
        # remove possible row with 'no_dataset'
        df = df[~df.datatype.str.contains("no_dataset")]
        datasubtypes = df.iloc[:, 3]
        datasubtypes_list = datasubtypes.values.tolist()
        datasubtypes_list = np.asarray(datasubtypes_list)
        datasubtypes_list_lower = np.char.lower(datasubtypes_list)
        list_classes_datasubtypes = np.unique(datasubtypes_list_lower)
        datasubtypes_final = normalize_classes(datasubtypes_list_lower, list_classes_datasubtypes)

    """
    if df.shape[1] > 4:
        leafdatatypes = df.iloc[:,4]
        leafdatatypes_list = leafdatatypes.values.tolist()
        leafdatatypes_list = np.asarray(leafdatatypes_list)
        #leafdatatypes_list_lower = np.char.lower(leafdatatypes_list)
        leafdatatypes_list_lower = leafdatatypes_list
        list_classes_leafdatatypes = np.unique(leafdatatypes_list_lower)
        print(list_classes_leafdatatypes)
        leafdatatypes_final = normalize_classes(leafdatatypes_list_lower, list_classes_leafdatatypes)
    """

    if df.shape[1] == 3:
        return (
            np.asarray(texts_list),
            datatypes_final,
            None,
            None,
            list_classes_datatypes.tolist(),
            None,
            None,
        )
    # elif df.shape[1] == 4:
    else:
        return (
            np.asarray(texts_list),
            datatypes_final,
            datasubtypes_final,
            None,
            list_classes_datatypes.tolist(),
            list_classes_datasubtypes.tolist(),
            None,
        )
    """
    else:
        return np.asarray(texts_list), datatypes_final, datasubtypes_final, leafdatatypes_final, list_classes_datatypes.tolist(), list_classes_datasubtypes.tolist(), list_classes_leafdatatypes.tolist()
    """


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

    with gzip.GzipFile(json_gz_file_path, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
        if "documents" not in data:
            print(
                "There is no usable classified text in the corpus file",
                json_gz_file_path,
            )
            return None, None
        for document in data["documents"]:
            for segment in document["texts"]:
                if "entity_spans" in segment:
                    if "text" not in segment:
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

    with gzip.GzipFile(json_gz_file_path, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
        if "documents" not in data:
            print(
                "There is no usable classified text in the corpus file",
                json_gz_file_path,
            )
            return None, None
        for document in data["documents"]:
            for segment in document["texts"]:
                if "entity_spans" in segment:
                    if "text" not in segment:
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

                            if "created" in entity_span and entity_span["created"]:
                                classes.append(1.0)
                            else:
                                classes.append(0.0)

                            if "shared" in entity_span and entity_span["shared"]:
                                classes.append(1.0)
                            else:
                                classes.append(0.0)

                            classes_list.append(classes)

    # list_possible_classes = np.unique(classes_list)
    # classes_list_final = normalize_classes(classes_list, list_possible_classes)

    texts_list_final = np.asarray(texts_list)
    classes_list_final = np.asarray(classes_list)

    texts_list_final, classes_list_final, _ = shuffle_triple_with_view(texts_list_final, classes_list_final)

    return texts_list_final, classes_list_final


def load_software_dataset_context_corpus_json(json_gz_file_path):
    """
    Load texts and classes for software and dataset mention corpus KISH export in gzipped json format

    Classification of the dataset and software usage is multiclass/multilabel

    Returns:
        tuple(numpy array, numpy array):
            texts, classes_list

    """
    texts_list = []
    classes_list = []

    with gzip.GzipFile(json_gz_file_path, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
        if "documents" not in data:
            print(
                "There is no usable classified text in the corpus file",
                json_gz_file_path,
            )
            return None, None
        for document in data["documents"]:
            for segment in document["texts"]:
                if "class_attributes" not in segment:
                    continue
                if "text" not in segment:
                    continue
                text = segment["text"]
                texts_list.append(text)
                classes = []
                classification = segment["class_attributes"]["classification"]
                if "used" in classification and classification["used"]["value"]:
                    classes.append(1.0)
                else:
                    classes.append(0.0)

                if "created" in classification and classification["created"]["value"]:
                    classes.append(1.0)
                else:
                    classes.append(0.0)

                if "shared" in classification and classification["shared"]["value"]:
                    classes.append(1.0)
                else:
                    classes.append(0.0)

                classes_list.append(classes)

    texts_list_final = np.asarray(texts_list)
    classes_list_final = np.asarray(classes_list)

    texts_list_final, classes_list_final, _ = shuffle_triple_with_view(texts_list_final, classes_list_final)

    return texts_list_final, classes_list_final


def normalize_classes(y, list_classes):
    """
    Replace string values of classes by their index in the list of classes
    """

    def f(x):
        return np.where(list_classes == x)

    intermediate = np.array([f(xi)[0] for xi in y])
    return np.array([vectorize(xi, len(list_classes)) for xi in intermediate])


def vectorize(index, size):
    """
    Create a numpy array of the provided size, where value at indicated index is 1, 0 otherwise
    """
    result = np.zeros(size)
    if index < size:
        result[index] = 1
    else:
        print("warning: index larger than vector size: ", index, size)
    return result
