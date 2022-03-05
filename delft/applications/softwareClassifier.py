import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_software_use_corpus_json
from delft.textClassification.reader import vectorize as vectorizer
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import time
from delft.textClassification.models import architectures
import numpy as np

"""
    This binary classifier is used in combination with a software mention recognition model, for characterizing
    the nature of the citation of software in scientific and technical literature. 
    This classifier predicts if the software introduced by a software mention in a sentence is used
    or not by the described work. 

    For the software mention recognizer, see https://github.com/ourresearch/software-mentions
    and grobidTagger.py in the present project DeLFT.

    Best architecture/model is fine-tuned SciBERT. 
"""

list_classes =  [
                    "not_used", 
                    "used"
                ]

class_weights = {
                    0: 1.,
                    1: 4.
                }


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
        max_epoch = 3
        maxlen = 100

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(embeddings_name, fold_count, architecture="gru", transformer=None):
    print('loading binary software use dataset...')
    xtr, y = load_software_use_corpus_json("data/textClassification/software/software-use.json.gz")

    model_name = 'software_use_'+architecture
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
    print('loading binary software use dataset...')
    xtr, y = load_software_use_corpus_json("data/textClassification/software/software-use.json.gz")

    nb_used = 0
    for the_class in y:
        if the_class[1] == 1.0:
            nb_used += 1
    nb_unused = len(y) - nb_used
    print("\ttotal:", len(y))
    print("\tused:", nb_used)
    print("\tnot used:", nb_unused)

    model_name = 'software_use_'+architecture
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


# classify a list of texts
def classify(texts, output_format, embeddings_name=None, architecture="gru", transformer=None):
    # load model
    model = Classifier('software_use_'+architecture, architecture=architecture, list_classes=list_classes, embeddings_name=embeddings_name, transformer_name=transformer)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)
    if output_format == 'json':
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classify whether a mentioned software is used or not, using the DeLFT library")

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

    if args.action not in ('train', 'train_eval', 'classify'):
        print('action not specified, must be one of [train,train_eval,classify]')

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

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)    

    if args.action == 'classify':
        someTexts = ['Radiographic errors were recorded on individual tick sheets and the information was captured in an Excel spreadsheet (Microsoft, Redmond, WA).', 
            'This ground-up approach sits in stark contrast to the top-down one used to introduce patient access to the NHS Summary Care Record via HealthSpace, which has so far met with limited success.', 
            'The authors of the GeneWiki project have developed the WikiTrust resource (3), which works via a Firefox plug-in, to mark up Wikipedia articles according to the Wikipedian\'s reputation.']
        result = classify(someTexts, "json", architecture=architecture, embeddings_name=embeddings_name, transformer=transformer)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

