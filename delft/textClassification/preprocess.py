import itertools
import regex as re
import numpy as np
# seed is fixed for reproducibility
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)

from unidecode import unidecode
from delft.utilities.Tokenizer import tokenizeAndFilterSimple
from delft.utilities.bert.run_classifier_delft import DataProcessor
import delft.utilities.bert.tokenization as tokenization
from delft.utilities.bert.run_classifier_delft import InputExample

special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)


def to_vector_single(text, embeddings, maxlen=300):
    """
    Given a string, tokenize it, then convert it to a sequence of word embedding 
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
    """
    tokens = tokenizeAndFilterSimple(clean_text(text))
    window = tokens[-maxlen:]

    # TBD: use better initializers (uniform, etc.) 
    x = np.zeros((maxlen, embeddings.embed_size), )

    # TBD: padding should be left and which vector do we use for padding? 
    # and what about masking padding later for RNN?
    for i, word in enumerate(window):
        x[i,:] = embeddings.get_word_vector(word).astype('float32')

    return x

def to_vector_elmo(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on ELMo contextualized embeddings
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_only_ELMo(subtokens)
    """
    if use_token_dump:
        return embeddings.get_sentence_vector_ELMo_with_token_dump(tokens)
    """

def to_vector_bert(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the BERT contextualized embeddings, introducing
    padding token when appropriate
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        subtokens.append(local_tokens)
    vector = embeddings.get_sentence_vector_only_BERT(subtokens)
    return vector

def to_vector_simple_with_elmo(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the concatenation of the provided static embeddings and 
    the ELMo contextualized embeddings, introducing <PAD> and <UNK> 
    padding token vector when appropriate
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        if len(tokens[i]) < maxlen:
            for i in range(0, maxlen-len(tokens[i])):
                local_tokens.append(" ")
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_with_ELMo(subtokens)

def to_vector_simple_with_bert(tokens, embeddings, maxlen=300, lowercase=False, num_norm=False):
    """
    Given a list of tokens convert it to a sequence of word embedding 
    vectors based on the concatenation of the provided static embeddings and 
    the BERT contextualized embeddings, introducing padding token vector 
    when appropriate
    """
    subtokens = []
    for i in range(0, len(tokens)):
        local_tokens = []
        for j in range(0, min(len(tokens[i]), maxlen)):
            if lowercase:
                local_tokens.append(lower(tokens[i][j]))
            else:
                local_tokens.append(tokens[i][j])
        if len(tokens[i]) < maxlen:
            for i in range(0, maxlen-len(tokens[i])):
                local_tokens.append(" ")
        subtokens.append(local_tokens)
    return embeddings.get_sentence_vector_with_BERT(subtokens)

def clean_text(text):
    x_ascii = unidecode(text)
    x_clean = special_character_removal.sub('',x_ascii)
    return x_clean


def lower(word):
    return word.lower() 


def normalize_num(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)


class BERT_classifier_processor(DataProcessor):
    """
    BERT data processor for classification
    """
    def __init__(self, labels=None, x_train=None, y_train=None, x_test=None, y_test=None):
        self.list_classes = labels
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_train_examples(self, x_train=None, y_train=None):
        """See base class."""
        if x_train is not None:
            self.x_train = x_train
        if y_train is not None:
            self.y_train = y_train
        examples, _ = self.create_examples(self.x_train, self.y_train)
        return examples

    def get_labels(self):
        """See base class."""
        return self.list_classes

    def get_test_examples(self, x_test=None, y_test=None):
        """See base class."""
        if x_test is not None:
            self.x_test = x_test
        if y_test is not None:
            self.y_test = y_test
        examples, results = self.create_examples(self.x_test, self.y_test)
        return examples, results

    def create_examples(self, x_s, y_s=None):
        examples = []
        valid_classes = np.zeros((y_s.shape[0],len(self.list_classes)))
        accumul = 0
        for (i, x) in enumerate(x_s):
            y = y_s[i]
            guid = i
            text_a = tokenization.convert_to_unicode(x)
            #the_class = self._rewrite_classes(y, i)
            ind, = np.where(y == 1)
            the_class = self.list_classes[ind[0]]
            if the_class is None:
                #print(text_a)
                continue
            if the_class not in self.list_classes:
                #the_class = 'other'
                continue
            #if the_class not in self.list_classes:
            #    continue
            label = tokenization.convert_to_unicode(the_class)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            valid_classes[accumul] = y
            accumul += 1

        return examples, valid_classes 

    def create_inputs(self, x_s, dummy_label='dummy'):
        examples = []
        # dummy label to avoid breaking the bert base code
        label = tokenization.convert_to_unicode(dummy_label)
        for (i, x) in enumerate(x_s):
            guid = i
            text_a = tokenization.convert_to_unicode(x) 
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
