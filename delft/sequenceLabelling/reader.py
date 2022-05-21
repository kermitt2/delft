import numpy as np
import xml
from xml.sax import make_parser, handler
from delft.utilities.Tokenizer import tokenizeAndFilterSimple, tokenizeAndFilter, tokenize
import re
import os
import gzip
from tqdm import tqdm
import json

class TEIContentHandler(xml.sax.ContentHandler):
    """ 
    TEI XML SAX handler for reading mixed content within xml text tags  
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
            self.tokens = []
            self.labels = []
        if name == "rs":
            # end of entity
            localTokens = tokenizeAndFilterSimple(self.accumulated)
            begin = True
            if self.currentLabel == None:
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
        return np.asarray(self.sents, dtype=object)

    def getAllLabels(self):
        return np.asarray(self.allLabels, dtype=object)

    def clear(self): # clear the accumulator for re-use
        self.accumulated = ""


class ENAMEXContentHandler(xml.sax.ContentHandler):
    """ 
    ENAMEX-style XML SAX handler for reading mixed content within xml text tags  
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
    corpus_type = ''

    def __init__(self, corpus_type='lemonde'):
        xml.sax.ContentHandler.__init__(self)
        self.corpus_type = corpus_type

    def translate_fr_labels(self, mainType, subType):
        #default
        labelOutput = "O"
        senseOutput = ""

        if mainType.lower() == "company":
            labelOutput = 'business'
        elif mainType.lower() == "fictioncharacter":
            labelOutput = "person"
        elif mainType.lower() == "organization": 
            if subType.lower() == "institutionalorganization":
                labelOutput = "institution"
            elif subType.lower() == "company":
                labelOutput = "business"
            else: 
                labelOutput = "organisation"
        elif mainType.lower() ==  "person":
            labelOutput = "person"
        elif mainType.lower() ==  "location":
            labelOutput = "location"
        elif mainType.lower() ==  "poi":
            labelOutput = "location"
        elif mainType.lower() ==  "product":
            labelOutput = "artifact"

        return labelOutput

    def startElement(self, name, attrs):
        if self.accumulated != '':
            localTokens = tokenizeAndFilterSimple(self.accumulated)
            for token in localTokens:
                self.tokens.append(token)
                self.labels.append('O')
        if name == 'corpus' or name == 'DOC':
            # beginning of a document
            self.tokens = []
            self.labels = []
            self.sents = []
            self.allLabels = []
        if name == "sentence":
            # beginning of sentence
            self.tokens = []
            self.labels = []
            self.currentLabel = 'O'
        if name == "ENAMEX" or name == "EX_ENAMEX":
            # beginning of entity
            if attrs.getLength() != 0:
                #if attrs.getValue("type") != 'insult' and attrs.getValue("type") != 'threat':
                #    print("Invalid entity type:", attrs.getValue("type"))
                attribute_names = attrs.getNames()
                mainType = None
                if "type" in attrs:
                    mainType = attrs.getValue("type")
                if "TYPE" in attrs:
                    mainType = attrs.getValue("TYPE")
                if mainType == None:
                    print('ENAMEX element without type attribute!')

                if "sub_type" in attrs:
                    subType = attrs.getValue("sub_type")
                else:
                    subType = ''
                if self.corpus_type == 'lemonde':
                    self.currentLabel = '<'+self.translate_fr_labels(mainType, subType)+'>'
                else:
                    self.currentLabel = '<'+mainType+'>'
        self.accumulated = ''

    def endElement(self, name):
        #print("endElement '" + name + "'")
        if name == "sentence":
            # end of sentence 
            if self.accumulated != '':
                localTokens = tokenizeAndFilterSimple(self.accumulated)
                for token in localTokens:
                    self.tokens.append(token)
                    self.labels.append('O')

            self.sents.append(self.tokens)
            self.allLabels.append(self.labels)
            self.tokens = []
            self.labels = []
        if name == "ENAMEX" or name == "EX_ENAMEX":
            # end of entity
            localTokens = tokenizeAndFilterSimple(self.accumulated)
            begin = True
            if self.currentLabel == None:
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

    field separator can be either space or tab
    input file can be gzipped or not

    Returns:
        tuple(numpy array, numpy array, numpy array): tokens, labels, features

    """
    sents = []
    labels = []
    featureSets = []

    if filepath.endswith(".gz"):
        with gzip.open(filepath, 'rt') as f:
            sents, labels, featureSets = load_data_and_labels_crf_content(f)
    else:
        with open(filepath) as f:
            sents, labels, featureSets = load_data_and_labels_crf_content(f)

    return np.asarray(sents, dtype=object), np.asarray(labels, dtype=object), np.asarray(featureSets, dtype=object)

def load_data_and_labels_crf_content(the_file):
    sents = []
    labels = []
    featureSets = []

    tokens, tags, features = [], [], []
    for line in the_file:
        line = line.strip()
        if len(line) == 0:
            if len(tokens) != 0: 
                sents.append(tokens)
                labels.append(tags)
                featureSets.append(features)
                tokens, tags, features = [], [], []
        else:
            pieces = re.split(' |\t', line)
            token = pieces[0]
            tag = pieces[len(pieces)-1]
            localFeatures = pieces[1:len(pieces)-1]
            tokens.append(token)
            tags.append(_translate_tags_grobid_to_IOB(tag))
            features.append(localFeatures)

    if len(tokens) == len(tags) == len(localFeatures) > 0:
        sents.append(tokens)
        labels.append(tags)
        featureSets.append(features)
        print("Adding the final items from the input file. ")

    assert "Tokens, tags and features haven't got the same size", len(tokens) == len(tags) == len(features)

    return sents, labels, featureSets


def load_data_and_labels_crf_string(crfString):
    """
    Load data, features (no label!) from a CRF matrix file 
    the format is as follow:

    token_0 f0_0 f0_1 ... f0_n
    token_1 f1_0 f1_1 ... f1_n
    ...
    token_m fm_0 fm_1 ... fm_n

    field separator can be either space or tab

    Returns:
        tuple(numpy array, numpy array, numpy array): tokens, features

    """
    sents = []
    labels = []
    featureSets = []
    tokens, tags, features = [], [], []
    for line in crfString.splitlines():    
        line = line.strip(' \t')
        if len(line) == 0:
            if len(tokens) != 0:
                sents.append(tokens)
                labels.append(tags)
                featureSets.append(features)
                tokens, tags, features = [], [], []
        else:
            #pieces = line.split('\t')
            pieces = re.split(' |\t', line)
            token = pieces[0]
            tag = pieces[len(pieces)-1]
            localFeatures = pieces[1:len(pieces)-1]
            tokens.append(token)
            tags.append(_translate_tags_grobid_to_IOB(tag))
            features.append(localFeatures)
    # last sequence
    if len(tokens) != 0:
        sents.append(tokens)
        labels.append(tags)
        featureSets.append(features)
    return sents, labels, featureSets


def load_data_crf_string(crfString):
    """
    Load data and features from a CRF matrix file 
    the format is as follow:

    token_0 f0_0 f0_1 ... f0_n
    token_1 f1_0 f1_1 ... f1_n
    ...
    token_m fm_0 fm_1 ... fm_n

    field separator can be either space or tab

    Returns:
        tuple(numpy array, numpy array): tokens, features

    """
    sents = []
    featureSets = []
    tokens, features = [], []
    #print("crfString:", crfString)
    for line in crfString.splitlines():
        line = line.strip(' \t')
        if len(line) == 0:
            if len(tokens) != 0:
                sents.append(tokens)
                featureSets.append(features)
                tokens, features = [], []
        else:
            pieces = re.split(' |\t', line)
            token = pieces[0]
            localFeatures = pieces[1:len(pieces)]
            tokens.append(token)
            features.append(localFeatures)
    # last sequence
    if len(tokens) != 0:
        sents.append(tokens)
        featureSets.append(features)

    #print('sents:', len(sents))
    #print('featureSets:', len(featureSets))
    return sents, featureSets


def _translate_tags_grobid_to_IOB(tag):
    """
    Convert labels as used by GROBID to the more standard IOB2 
    """
    if tag.endswith('other>'):
        # outside
        return 'O'
    elif tag.startswith('I-'):
        # begin
        return 'B-'+tag[2:]
    elif tag.startswith('<'):
        # inside
        return 'I-'+tag
    else:
        return tag


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

    # TBD: ideally, for consistency, the tokenization in the CoNLL files should not be enforced, 
    # only the standard DeLFT tokenization should be used, in line with the word embeddings
    sents, labels = [], []
    with open(filename, encoding="UTF-8") as f:
        words, tags = [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-') or line.startswith('#begin document'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                if len(line.split('\t')) == 2:
                    word, tag = line.split('\t')
                else:
                    word, _, tag = line.split('\t')
                words.append(word)
                tags.append(tag)

    return np.asarray(sents, dtype=object), np.asarray(labels, dtype=object)


def load_data_and_labels_conll_with_document_context(filename, max_context_window=400):
    """
    Load data and label from a file. In this alternative, we do not segment by sentence and 
    we keep a maximum of document context according to a context window size.

    Args:
        filename (str): path to the file.

        The file format is tab-separated values.
        A blank line is required at the end of a sentence.

        For example:
        ```
        EU  B-ORG
        rejects O
        German  B-MISC
        call    O
        to  O
        boycott O
        British B-MISC
        lamb    O
        .   O

        Peter   B-PER
        Blackburn   I-PER
        ...
        ```

    Returns:
        tuple(numpy array, numpy array): data and labels

    """

    # TBD: ideally, for consistency, the tokenization in the CoNLL files should not be enforced, 
    # only the standard DeLFT tokenization should be used, in line with the word embeddings
    documents, sents, labels = [], [], []
    with open(filename, encoding="UTF-8") as f:
        words, tags = [], []
        for line in f:
            line = line.rstrip()
            if line.startswith('-DOCSTART-') or line.startswith('#begin document'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
                if len(sents) != 0:
                    documents.append(sents)
                    sents = []
            elif len(line) == 0:
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                if len(line.split('\t')) == 2:
                    word, tag = line.split('\t')
                else:
                    word, _, tag = line.split('\t')
                words.append(word)
                tags.append(tag)

    # regroup sentences according document contexts following context window

    return np.asarray(sents, dtype=object), np.asarray(labels, dtype=object)


def load_data_and_labels_lemonde(filepathXml):
    """
    Load data and label from Le Monde XML corpus file
    the format is ENAMEX-style, as follow:
    <sentence id="E14">Les ventes de micro-ordinateurs en <ENAMEX type="Location" sub_type="Country" 
        eid="2000000003017382" name="Republic of France">France</ENAMEX> se sont ralenties en 1991. </sentence>

    Returns:
        tuple(numpy array, numpy array): data and labels

    """
    # as we have XML mixed content, we need a real XML parser...
    parser = make_parser()
    handler = ENAMEXContentHandler()
    parser.setContentHandler(handler)
    parser.parse(filepathXml)
    tokens = handler.getSents()
    labels = handler.getAllLabels()

    return tokens, labels


def load_data_and_labels_ontonotes(ontonotesRoot, lang='en'):
    """
    Load data and label from Ontonotes 5.0 pseudo-XML corpus files
    the format is ENAMEX-style, as follow, with one sentence per line:
    <doc>
        <ENAMEX TYPE="DATE">Today</ENAMEX> , one newspaper headline warned of civil war .
        ...
    </doc>

    Returns:
        tuple(numpy array, numpy array): data and labels

    """
    # assuming we have the root of ontonotes corpus, we iterate through the sub-directories
    # and process all .name files
    nb_files = 0
    # map lang and subdir names
    lang_name = 'english'
    if lang == 'zh':
        lang_name = '/chinese/'
    elif lang == 'ar':
        lang_name = '/arabic/'

    tokens = []
    labels =[]

    # first pass to get number of files
    for subdir, dirs, files in os.walk(ontonotesRoot):
        for file in files:
            if '/english/' in subdir and file.endswith('.name'):
                # remove old/new testament
                if '/pt/' in subdir:
                    continue
                #print(os.path.join(subdir, file))
                nb_files += 1
    nb_total_files = nb_files
    #print(nb_total_files, 'total files')

    nb_files = 0
    pbar = tqdm(total=nb_total_files)
    for subdir, dirs, files in os.walk(ontonotesRoot):
        for file in files:
            if '/english/' in subdir and file.endswith('.name'):
                # remove old/new testament
                if '/pt/' in subdir:
                    continue
                handler = ENAMEXContentHandler(corpus_type="ontonotes")
                # massage a bit the pseudo-XML so that it looks XML
                with open(os.path.join(subdir, file), encoding="UTF-8") as f:
                    content = '<?xml version="1.0" encoding="utf-8"?>\n'
                    for line in f:
                        line = line.strip()
                        if len(line) > 2 and line[-2] == '/':
                            line = line[:len(line)-2] + line[-1]

                        if len(line) != 0:
                            if not '<DOC' in line and not '</DOC' in line:
                                content += '<sentence>' + line + '</sentence>\n'
                            else:
                                content += line + "\n"
                    #print(content)
                    xml.sax.parseString(content, handler)
                    tokens.extend(handler.sents)
                    labels.extend(handler.allLabels)
                    nb_files += 1
                    pbar.update(1)
    pbar.close()
    print('nb total sentences:', len(tokens))
    total_tokens = 0
    for sentence in tokens:
        total_tokens += len(sentence)
    print('nb total tokens:', total_tokens)    

    final_tokens = np.asarray(tokens)
    final_label = np.asarray(labels)

    return final_tokens, final_label


def load_data_and_labels_json_offsets(jsonCorpus, tokenizer=None):
    """
    Load data and labels from json corpus where annotations are expressed with offsets. 
    This requires a tokenizer passed as parameter. If tokenizer is None, we use the generic
    Indo-European tokenizer.

    Note: input file can be gzipped or not

    {
    "lang": "en",
    "level": "sentence",
    "documents": [
        {
            "id": "10.1371/journal.pone.0198300",
            "body_text": [
                {
                    "text": "The test was designed so that bacteria were collected at 1 hour and 6 hours after start time on each day of testing.",
                    "annotation_spans": [
                        {
                            "start": 30,
                            "end": 38,
                            "text": "bacteria",
                            "type": "dataset",
                            "datatype": "Tabular Data:Sample Table"
                        }
                    ]
                },
            ]
        }
    }    

    Returns:
        tuple(numpy array, numpy array): data and labels
    """
    if not os.path.exists(jsonCorpus):
        print("Invalid path file: ", jsonCorpus)
        return None, None

    all_tokens = []
    all_labels = []

    if jsonCorpus.endswith(".gz"):
        corpus_file = gzip.open(jsonCorpus, "rt")
    else:
        corpus_file = open(jsonCorpus, "rt")
    
    jsonDocuments = json.load(corpus_file)
    if "documents" in jsonDocuments:
        for jsonDocument in jsonDocuments["documents"]:
            if "body_text" in jsonDocument:
                for text_piece in jsonDocument["body_text"]:
                    if "text" in text_piece:
                        tokens = []
                        labels = []
                        text = text_piece["text"]
                        local_tokens, local_offsets = tokenizeAndFilter(text)
                        spans = []
                        if "annotation_spans" in text_piece:
                            for annotation_span in text_piece["annotation_spans"]:
                                local_type = None
                                if "type" in annotation_span:
                                    local_type = annotation_span["type"]
                                    local_type = local_type.replace(" ", "_")
                                spans.append([annotation_span["start"], annotation_span["end"], local_type])
                        i =0
                        for local_token in local_tokens:
                            tokens.append(local_token)
                            offset = local_offsets[i]
                            found = False
                            for span in spans:
                                if span[0] <= offset[0] and (offset[1] <= span[1] or offset[0] < span[1]):
                                    if span[0] == offset[0]:
                                        labels.append("B-"+span[2])
                                    else:
                                        labels.append("I-"+span[2])
                                    found = True
                                    break
                            if not found:
                                labels.append("O")
                            i += 1

                        all_tokens.append(tokens)
                        all_labels.append(labels)

    corpus_file.close()

    final_tokens = np.asarray(all_tokens)
    final_labels = np.asarray(all_labels)

    return final_tokens, final_labels

if __name__ == "__main__":
    # some tests
    xmlPath = '../../data/sequenceLabelling/toxic/train.xml'
    print(xmlPath)
    sents, allLabels = load_data_and_labels_xml_file(xmlPath)
    print('toxic tokens:', sents)
    print('toxic labels:', allLabels)

    xmlPath = '../../data/sequenceLabelling/toxic/test.xml'
    print(xmlPath)
    sents, allLabels = load_data_and_labels_xml_file(xmlPath)
    print('toxic tokens:', sents)
    print('toxic labels:', allLabels)

    crfPath = '../../data/sequenceLabelling/grobid/date/date-060518.train'
    print(crfPath)
    x_all, y_all, f_all = load_data_and_labels_crf_file(input)
    print('grobid date tokens:', x_all)
    print('grobid date labels:', y_all)
    print('grobid date features:', f_all)

    with open(crfPath, 'r') as theFile:
        string = theFile.read()

    x_all, f_all = load_data_and_labels_crf_string(string)
    print('grobid date tokens:', x_all)
    print('grobid date features:', f_all)
