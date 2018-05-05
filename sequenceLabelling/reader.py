import numpy as np
import xml
from xml.sax import make_parser, handler
from utilities.Tokenizer import tokenizeAndFilterSimple


class TEIContentHandler(xml.sax.ContentHandler):
    """ 
    XML SAX handler for reading mixed content within xml text tags  
    """
    
    # local sentence
    tokens = []
    labels = []

    # all sentences of the document
    sents = []
    allLabels = []

    # working variables
    accumulated = ''
    currentLabel = None

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
     
    def startElement(self, name, attrs):
        if self.accumulated != '':
            localTokens = tokenizeAndFilterSimple(self.accumulated)
            for token in localTokens:
                self.tokens.append(token)
                self.labels.append('O')
        if name == 'TEI' or name == 'tei':
            # beginning of a document
            self.tokens = []
            self.labels = []
            self.sents = []
            self.allLabels = []
        if name == "p":
            # beginning of sentence
            self.tokens = []
            self.labels = []
            self.currentLabel = 'O'
        if name == "rs":
            # beginning of entity
            if attrs.getLength() != 0:
                if attrs.getValue("type") != 'insult' and attrs.getValue("type") != 'threat':
                    print("Invalid entity type:", attrs.getValue("type"))
                self.currentLabel = '<'+attrs.getValue("type")+'>'
        self.accumulated = ''
                
    def endElement(self, name):
        # print("endElement '" + name + "'")
        if name == "p":
            # end of sentence 
            if self.accumulated != '':
                localTokens = tokenizeAndFilterSimple(self.accumulated)
                for token in localTokens:
                    self.tokens.append(token)
                    self.labels.append('O')

            self.sents.append(self.tokens)
            self.allLabels.append(self.labels)
            tokens = []
            labels = []
        if name == "rs":
            # end of entity
            localTokens = tokenizeAndFilterSimple(self.accumulated)
            begin = True
            if self.currentLabel is None:
                self.currentLabel = 'O'
            for token in localTokens:
                self.tokens.append(token)
                if begin:
                    self.labels.append('B-'+self.currentLabel)
                    begin = False
                else:     
                    self.labels.append('I-'+self.currentLabel)
            self.currentLabel = None
        self.accumulated = ''
    
    def characters(self, content):
        self.accumulated += content
     
    def getSents(self):
        return np.asarray(self.sents)

    def getAllLabels(self):
        return np.asarray(self.allLabels)

    def clear(self): # clear the accumulator for re-use
        self.accumulated = ""


def load_data_and_labels_xml_string(stringXml):
    """
    Load data and label from a string 
    the format is as follow:
    <p> 
        bla bla you are a <rs type="insult">CENSURED</rs>, 
        and I will <rs type="threat">find and kill</rs> you bla bla
    </p>
    only the insulting expression is labelled, and similarly only the threat 
    "action" is tagged

    Returns:
        tuple(numpy array, numpy array): data and labels

    """
    # as we have XML mixed content, we need a real XML parser...
    parser = make_parser()
    handler = TEIContentHandler()
    parser.setContentHandler(handler)
    parser.parseString(stringXml)
    tokens = handler.getSents()
    labels = handler.getAllLabels()
    return tokens, labels


def load_data_and_labels_xml_file(filepathXml):
    """
    Load data and label from an XML file
    the format is as follow:
    <p> 
        bla bla you are a <rs type="insult">CENSURED</rs>, 
        and I will <rs type="threat">find and kill</rs> you bla bla
    </p>
    only the insulting expression is labelled, and similarly only the threat 
    "action" is tagged

    Returns:
        tuple(numpy array, numpy array): data and labels

    """
    # as we have XML mixed content, we need a real XML parser...
    parser = make_parser()
    handler = TEIContentHandler()
    parser.setContentHandler(handler)
    parser.parse(filepathXml)
    tokens = handler.getSents()
    labels = handler.getAllLabels()
    return tokens, labels


def load_data_and_labels_crf_file(filepath):
    """
    Load data, features and label from a CRF matrix string 
    the format is as follow:

    token_0 f0_0 f0_1 ... f0_n label_0
    token_1 f1_0 f1_1 ... f1_n label_1
    ...
    token_m fm_0 fm_1 ... fm_n label_m

    Returns:
        tuple(numpy array, numpy array): data and labels

    """
    tokens = []
    labels = []
    features = []

    return tokens, labels, features


def load_data_and_labels_crf_string(crfString):
    """
    Load data, features and label from a CRF matrix file 
    the format is as follow:

    token_0 f0_0 f0_1 ... f0_n label_0
    token_1 f1_0 f1_1 ... f1_n label_1
    ...
    token_m fm_0 fm_1 ... fm_n label_m

    Returns:
        tuple(numpy array, numpy array): data and labels

    """
    tokens = []
    labels = []
    features = []

    return tokens, labels, features


def load_data_and_labels_conll(filename):
    """
    Load data and label from a file.

    Args:
        filename (str): path to the file.

        The file format is tab-separated values.
        A blank line is required at the end of a sentence.

        For example:
        ```
        EU  B-ORG
        rejects O
        German  B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O

        Peter	B-PER
        Blackburn	I-PER
        ...
        ```

    Returns:
        tuple(numpy array, numpy array): data and labels

    """

    # TBD: for consistency the tokenization in the CoNLL files should not be considered, 
    # only the standard DeLFT tokenization, in line with the word embeddings
    sents, labels = [], []
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
    return np.asarray(sents), np.asarray(labels)

"""
def batch_iter(data, labels, batch_size, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                if preprocessor:
                    yield preprocessor.transform(X, y)
                else:
                    yield X, y

    return num_batches_per_epoch, data_generator()
"""

if __name__ == "__main__":
    # some tests
    xmlPath = '../../data/sequence/train.xml'
    print(xmlPath)
    sents, allLabels = load_data_and_labels_xml_file(xmlPath)
    print('toxic tokens:', sents)
    print('toxic labels:', allLabels)

    xmlPath = '../../data/sequence/test.xml'
    print(xmlPath)
    sents, allLabels = load_data_and_labels_xml_file(xmlPath)
    print('toxic tokens:', sents)
    print('toxic labels:', allLabels)
