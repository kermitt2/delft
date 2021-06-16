import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_software_use_corpus_json
from delft.textClassification.reader import vectorize as vectorizer
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import keras.backend as K
import time
from delft.textClassification.models import modelTypes
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

list_classes = ["not_used", "used"]
class_weights = {0: 1.,
                 1: 4.}

def configure(architecture, use_BERT=False, use_ELMo=False):
    batch_size = 256
    if use_ELMo:
        batch_size = 20
    elif use_BERT:
        batch_size = 50
    maxlen = 300
    # default bert model parameters
    if architecture.find("bert") != -1:
        batch_size = 32
        maxlen = 100
    return batch_size, maxlen

def train(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru"):
    print('loading binary software use dataset...')
    xtr, y = load_software_use_corpus_json("data/textClassification/software/software-use.json.gz")

    model_name = 'software_use'
    class_weights = None
    if use_ELMo:
        model_name += '-with_ELMo'
    elif use_BERT:
        model_name += '-with_BERT'

    batch_size, maxlen = configure(architecture, use_BERT, use_ELMo)

    model = Classifier(model_name, model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size, maxlen=maxlen,
        class_weights=class_weights)

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def train_and_eval(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru"): 
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

    model_name = 'software_use'
    class_weights = None
    if use_ELMo:
        model_name += '-with_ELMo'
    elif use_BERT:
        model_name += '-with_BERT'

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    batch_size, maxlen = configure(architecture, use_BERT, use_ELMo)

    print(list_classes)

    model = Classifier(model_name, model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size, maxlen=maxlen,
        class_weights=class_weights)

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


# classify a list of texts
def classify(texts, output_format, architecture="gru"):
    # load model
    model = Classifier('software_use', model_type=architecture, list_classes=list_classes)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)
    if output_format is 'json':
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classify whether a mentioned software is used or not")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture",default='gru', help="type of model architecture to be used, one of "+str(modelTypes))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--use-BERT", action="store_true", help="Use BERT contextual embeddings") 
    parser.add_argument(
        "--embedding", default='glove-840B',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )

    args = parser.parse_args()

    if args.action not in ('train', 'train_eval', 'classify'):
        print('action not specified, must be one of [train,train_eval,classify]')

    embeddings_name = args.embedding
    use_ELMo = args.use_ELMo
    use_BERT = args.use_BERT

    architecture = args.architecture
    if architecture not in modelTypes:
        print('unknown model architecture, must be one of '+str(modelTypes))

    if args.action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(embeddings_name, args.fold_count, use_ELMo=use_ELMo, use_BERT=use_BERT, architecture=architecture)

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(embeddings_name, args.fold_count, use_ELMo=use_ELMo, use_BERT=use_BERT, architecture=architecture)    

    if args.action == 'classify':
        someTexts = ['Radiographic errors were recorded on individual tick sheets and the information was captured in an Excel spreadsheet (Microsoft, Redmond, WA).', 
            'This ground-up approach sits in stark contrast to the top-down one used to introduce patient access to the NHS Summary Care Record via HealthSpace, which has so far met with limited success.', 
            'The authors of the GeneWiki project have developed the WikiTrust resource (3), which works via a Firefox plug-in, to mark up Wikipedia articles according to the Wikipedian\'s reputation.']
        result = classify(someTexts, "json", architecture=architecture)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # See https://github.com/tensorflow/tensorflow/issues/3388
    #K.clear_session()
