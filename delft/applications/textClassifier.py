import argparse
import csv
import json
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from delft.textClassification import Classifier
from delft.textClassification.models import architectures
from delft.textClassification.reader import load_texts_and_classes_generic

pretrained_transformers_examples = ['bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased']

actions = ['train', 'train_eval', 'eval', 'classify']


def get_one_hot(y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y2 = onehot_encoder.fit_transform(integer_encoded)
    return y2


def configure(architecture):
    batch_size = 256
    maxlen = 150
    patience = 5
    early_stop = True
    max_epoch = 60

    # default bert model parameters
    if architecture == "bert":
        batch_size = 32
        # early_stop = False
        # max_epoch = 3

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(model_name, input_file, embeddings_name, fold_count, architecture=None, transformer=None,
          x_index=0, y_indexes=[1]):
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    print('loading ' + model_name + ' training corpus...')
    xtr, y = load_texts_and_classes_generic(input_file, x_index, y_indexes)

    list_classes = list(set([y_[0] for y_ in y]))

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch,
                       fold_number=fold_count, patience=patience, transformer_name=transformer,
                       use_roc_auc=True, embeddings_name=embeddings_name, early_stop=early_stop,
                       batch_size=batch_size, maxlen=maxlen, class_weights=None)

    y_ = get_one_hot(y)

    if fold_count == 1:
        model.train(xtr, y_)
    else:
        model.train_nfold(xtr, y_)
    # saving the model
    model.save()


def eval(model_name, input_file, architecture=None, x_index=0, y_indexes=[1]):
    # model_name += model_name + '-' + architecture

    print('loading ' + model_name + ' evaluation corpus...')

    xtr, y = load_texts_and_classes_generic(input_file, x_index, y_indexes)
    print(len(xtr), 'evaluation sequences')

    model = Classifier(model_name)
    model.load()

    y_ = get_one_hot(y)

    model.eval(xtr, y_)


def train_and_eval(model_name, input_file, embeddings_name, fold_count, transformer=None,
                   architecture="gru", x_index=0, y_indexes=[1]):
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    print('loading ' + model_name + ' corpus...')
    xtr, y = load_texts_and_classes_generic(input_file, x_index, y_indexes)

    list_classes = list(set([y_[0] for y_ in y]))

    y_one_hot = get_one_hot(y)

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch,
                       fold_number=fold_count, patience=patience, transformer_name=transformer,
                       use_roc_auc=True, embeddings_name=embeddings_name, early_stop=early_stop,
                       batch_size=batch_size, maxlen=maxlen, class_weights=None)

    # segment train and eval sets
    x_train, x_test, y_train, y_test = train_test_split(xtr, y_one_hot, test_size=0.1)

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)

    model.eval(x_test, y_test)

    # saving the model
    model.save()


# classify a list of texts
def classify(texts, output_format, architecture="gru", transformer=None):
    # load model
    model = Classifier(model_name)
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
        description="General classification of text ")

    parser.add_argument("action", help="the action", choices=actions)
    parser.add_argument("model", help="The name of the model")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--input", type=str, required=True, help="The file to be used for training/evaluation")
    parser.add_argument("--x-index", type=int, required=True, help="Index of the columns for the X value "
                                                                   "(assuming a TSV file)")
    parser.add_argument("--y-indexes", type=str, required=False, help="Index(es) of the columns for the Y (classes) "
                                                                     "separated by comma, without spaces (assuming "
                                                                     "a TSV file)")
    parser.add_argument("--architecture", default='gru', choices=architectures,
                        help="type of model architecture to be used, one of " + str(architectures))
    parser.add_argument(
        "--embedding", default='word2vec',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."))

    parser.add_argument(
        "--transformer",
        default=None,
        help="The desired pre-trained transformer to be used in the selected architecture. " + \
             "For local loading use, delft/resources-registry.json, and be sure to use here the "
             "same name as in the registry, e.g. " + \
             str(pretrained_transformers_examples) + \
             " and that the path in the registry to the model path is correct on your system. " + \
             "HuggingFace transformers hub will be used otherwise to fetch the model, "
             "see https://huggingface.co/models " + \
             "for model names"
    )

    args = parser.parse_args()

    embeddings_name = args.embedding
    input_file = args.input
    model_name = args.model
    transformer = args.transformer
    architecture = args.architecture
    x_index = args.x_index

    if args.action != "classify":
        if args.y_indexes is None:
            print("--y-indexes is mandatory")
            sys.exit(-1)
        y_indexes = [int(index) for index in args.y_indexes.split(",")]

        if len(y_indexes) > 1:
            print("At the moment we support just one value per class. Taking the first value only. ")
            y_indexes = y_indexes[0]

    if transformer is None and embeddings_name is None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if args.action == 'train':
        train(model_name, input_file, embeddings_name, args.fold_count, architecture=architecture,
              transformer=transformer, x_index=x_index, y_indexes=y_indexes)

    elif args.action == 'eval':
        eval(model_name, input_file, architecture=architecture, x_index=x_index, y_indexes=y_indexes)

    elif args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(model_name, input_file, embeddings_name, args.fold_count, architecture=architecture,
                                transformer=transformer, x_index=x_index, y_indexes=y_indexes)

    elif args.action == 'classify':
        lines = []
        with open(input_file, 'r') as f:
            tsvreader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_ALL)
            for line in tsvreader:
                if len(line) == 0:
                    continue
                lines.append(line[x_index])

        result = classify(lines, "csv")

        result_binary = [np.argmax(line) for line in result]

        for x in result_binary:
            print(x)
    # See https://github.com/tensorflow/tensorflow/issues/3388
    # K.clear_session()
