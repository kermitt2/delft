import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_software_context_corpus_json
from delft.textClassification.reader import vectorize as vectorizer
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import time
from delft.textClassification.models import architectures
import numpy as np

"""
    A multiclass classifier to be used in combination with a software mention recognition model, for characterizing
    the nature of the mention of software in scientific and technical literature. 
    This classifier predicts if the software introduced by a software mention in a sentence is likely:
    - used or not by the described work (class used)
    - a creation of the described work (class creation)
    - shared (class shared)

    For the software mention recognizer, see https://github.com/ourresearch/software-mentions
    and grobidTagger.py in the present project DeLFT.

    Best architecture/model is fine-tuned SciBERT. 
"""

list_classes = ["used", "creation", "shared"]
class_weights = {0: 1.,
                 1: 1.,
                 2: 1.}

def configure(architecture):
    batch_size = 256
    maxlen = 300
    patience = 5
    early_stop = True
    max_epoch = 60

    # default bert model parameters
    if architecture == "bert":
        batch_size = 32
        early_stop = False
        max_epoch = 6
        maxlen = 100

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(embeddings_name, fold_count, architecture="gru", transformer=None):
    print('loading multiclass software context dataset...')
    xtr, y = load_software_context_corpus_json("data/textClassification/software/software-contexts.json.gz")

    report_training_contexts(y)

    model_name = 'software_context_'+architecture
    class_weights = None

    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, patience=patience,
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def train_and_eval(embeddings_name, fold_count, architecture="gru", transformer=None): 
    print('loading multiclass software context dataset...')
    xtr, y = load_software_context_corpus_json("data/textClassification/software/software-contexts.json.gz")

    report_training_contexts(y)

    model_name = 'software_context_'+architecture
    class_weights = None

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    print(list_classes)

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, patience=patience,
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


def train_binary(embeddings_name, fold_count, architecture="gru", transformer=None):
    print('loading multiclass software context dataset...')
    x_train, y_train = load_software_context_corpus_json("data/textClassification/software/software-contexts.json.gz")

    report_training_contexts(y_train)

    for class_rank in range(len(list_classes)):
        model_name = 'software_context_' + list_classes[class_rank] + '_'+architecture
        class_weights = None

        batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

        y_train_class_rank = [ [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_train ]
        y_train_class_rank = np.array(y_train_class_rank)

        list_classes_rank = [list_classes[class_rank], "not_"+list_classes[class_rank]]

        model = Classifier(model_name, architecture=architecture, list_classes=list_classes_rank, max_epoch=max_epoch, fold_number=fold_count, patience=patience,
            use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, early_stop=early_stop,
            class_weights=class_weights, transformer_name=transformer)

        if fold_count == 1:
            model.train(x_train, y_train_class_rank)
        else:
            model.train_nfold(x_train, y_train_class_rank)
        # saving the model
        model.save()


def train_and_eval_binary(embeddings_name, fold_count, architecture="gru", transformer=None): 
    print('loading multiclass software context dataset...')
    xtr, y = load_software_context_corpus_json("data/textClassification/software/software-contexts.json.gz")

    report_training_contexts(y)
    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    for class_rank in range(len(list_classes)):
        model_name = 'software_context_' + list_classes[class_rank] + '_'+architecture
        class_weights = None

        batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

        y_train_class_rank = [ [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_train ]
        y_test_class_rank = [ [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_test ]

        y_train_class_rank = np.array(y_train_class_rank)
        y_test_class_rank = np.array(y_test_class_rank)

        list_classes_rank = [list_classes[class_rank], "not_"+list_classes[class_rank]]

        model = Classifier(model_name, architecture=architecture, list_classes=list_classes_rank, max_epoch=max_epoch, fold_number=fold_count, patience=patience,
            use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, early_stop=early_stop,
            class_weights=class_weights, transformer_name=transformer)

        if fold_count == 1:
            model.train(x_train, y_train_class_rank)
        else:
            model.train_nfold(x_train, y_train_class_rank)
        model.eval(x_test, y_test_class_rank)

    # saving the model
    #model.save()


# classify a list of texts
def classify(texts, output_format, embeddings_name=None, architecture="gru", transformer=None):
    # load model
    model = Classifier('software_context_'+architecture)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)
    if output_format == 'json':
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return result

def report_training_contexts(y):
    nb_used = 0
    nb_creation = 0
    nb_shared = 0
    for the_class in y:
        if the_class[0] == 1.0:
            nb_used += 1
        if the_class[1] == 1.0:
            nb_creation += 1
        if the_class[2] == 1.0:
            nb_shared += 1

    print("\ntotal context training cases:", len(y))

    print("\t- used contexts:", nb_used)
    print("\t  not used contexts:", str(len(y) - nb_used))

    print("\t- creation contexts:", nb_creation)
    print("\t  not creation contexts:", str(len(y) - nb_creation))

    print("\t- shared contexts:", nb_shared)
    print("\t  not shared contexts:", str(len(y) - nb_shared))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classify the context of a mentioned software using the DeLFT library")

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']
    pretrained_transformers_examples = [ 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased' ]

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture",default='gru', help="type of model architecture to be used, one of "+str(architectures))
    parser.add_argument(
        "--embedding", 
        default=None,
        help="The desired pre-trained word embeddings using their descriptions in the file. " + \
            "For local loading, use delft/resources-registry.json. " + \
            "Be sure to use here the same name as in the registry, e.g. " + str(word_embeddings_examples) + \
            " and that the path in the registry to the embedding file is correct on your system."
    )
    parser.add_argument(
        "--transformer", 
        default=None,
        help="The desired pre-trained transformer to be used in the selected architecture. " + \
            "For local loading use, delft/resources-registry.json, and be sure to use here the same name as in the registry, e.g. " + \
            str(pretrained_transformers_examples) + \
            " and that the path in the registry to the model path is correct on your system. " + \
            "HuggingFace transformers hub will be used otherwise to fetch the model, see https://huggingface.co/models " + \
            "for model names"
    )

    args = parser.parse_args()

    if args.action not in ('train', 'train_eval', 'classify', 'train_binary', 'train_eval_binary'):
        print('action not specified, must be one of [train,train_binary,train_eval,train_eval_binary,classify]')

    embeddings_name = args.embedding
    transformer = args.transformer

    architecture = args.architecture
    if architecture not in architectures:
        print('unknown model architecture, must be one of '+str(architectures))

    if transformer == None and embeddings_name == None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if args.action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)

    if args.action == 'train_binary':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train_binary(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)    

    if args.action == 'train_eval_binary':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval_binary(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)    

    if args.action == 'classify':
        someTexts = ['Radiographic errors were recorded on individual tick sheets and the information was captured in an Excel spreadsheet (Microsoft, Redmond, WA).', 
            'This ground-up approach sits in stark contrast to the top-down one used to introduce patient access to the NHS Summary Care Record via HealthSpace, which has so far met with limited success.', 
            'The authors of the GeneWiki project have developed the WikiTrust resource (3), which works via a Firefox plug-in, to mark up Wikipedia articles according to the Wikipedian\'s reputation.']
        result = classify(someTexts, "json", architecture=architecture, embeddings_name=embeddings_name, transformer=transformer)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

