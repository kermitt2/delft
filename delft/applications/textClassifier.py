import argparse
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from delft.textClassification import Classifier
from delft.textClassification.models import architectures
from delft.textClassification.reader import load_texts_and_classes_generic
from delft.utilities.Utilities import t_or_f

pretrained_transformers_examples = ['bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased']

actions = ['train', 'train_eval', 'eval', 'classify']


def get_one_hot(y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y2 = onehot_encoder.fit_transform(integer_encoded)
    return y2



def configure(architecture, max_sequence_length_=-1, batch_size_=-1, max_epoch_=-1, patience_=-1, early_stop=True):
    batch_size = 256
    maxlen = 150 if max_sequence_length_ == -1 else max_sequence_length_
    patience = 5 if patience_ == -1 else patience_
    max_epoch = 60 if max_epoch_ == -1 else max_epoch_

    # default bert model parameters
    if architecture == "bert":
        batch_size = 32
        # early_stop = False
        # max_epoch = 3

    batch_size = batch_size_ if batch_size_ != -1 else batch_size

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(model_name,
          architecture,
          input_file,
          embeddings_name,
          fold_count,
          transformer=None,
          x_index=0,
          y_indexes=[1],
          batch_size=-1,
          max_sequence_length=-1,
          patience=-1,
          incremental=False,
          learning_rate=None,
          multi_gpu=False,
          max_epoch=50,
          early_stop=True
          ):

    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture,
                                                                    max_sequence_length,
                                                                    batch_size,
                                                                    max_epoch,
                                                                    patience,
                                                                    early_stop=early_stop)

    print('loading ' + model_name + ' training corpus...')
    xtr, y = load_texts_and_classes_generic(input_file, x_index, y_indexes)

    list_classes = list(set([y_[0] for y_ in y]))

    model = Classifier(model_name,
                       architecture=architecture,
                       list_classes=list_classes,
                       max_epoch=max_epoch,
                       fold_number=fold_count,
                       patience=patience,
                       transformer_name=transformer,
                       use_roc_auc=True,
                       embeddings_name=embeddings_name,
                       early_stop=early_stop,
                       batch_size=batch_size,
                       maxlen=maxlen,
                       class_weights=None,
                       learning_rate=learning_rate)

    y_ = get_one_hot(y)

    if fold_count == 1:
        model.train(xtr, y_, incremental=incremental, multi_gpu=multi_gpu)
    else:
        model.train_nfold(xtr, y_)
    # saving the model
    model.save()


def eval(model_name, architecture, input_file, x_index=0, y_indexes=[1]):
    # model_name += model_name + '-' + architecture

    print('loading ' + model_name + ' evaluation corpus...')

    xtr, y = load_texts_and_classes_generic(input_file, x_index, y_indexes)
    print(len(xtr), 'evaluation sequences')

    model = Classifier(model_name, architecture=architecture)
    model.load()

    y_ = get_one_hot(y)

    model.eval(xtr, y_)


def train_and_eval(model_name, architecture, input_file, embeddings_name, fold_count, transformer=None,
                x_index=0, y_indexes=[1], batch_size=-1,
                max_sequence_length=-1, patience=-1, multi_gpu=False):
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture, batch_size, max_sequence_length,
                                                                    patience)

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
        model.train(x_train, y_train, multi_gpu=multi_gpu)
    else:
        model.train_nfold(x_train, y_train, multi_gpu=multi_gpu)

    model.eval(x_test, y_test)

    # saving the model
    model.save()


# classify a list of texts
def classify(model_name, architecture, texts, output_format='json'):
    model = Classifier(model_name, architecture=architecture)
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
    parser.add_argument("--input", type=str, required=True,
                        help="The file to be used for train, train_eval, eval and, classify")
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

    parser.add_argument("--max-sequence-length", type=int, default=-1, help="max-sequence-length parameter to be used.")
    parser.add_argument("--batch-size", type=int, default=-1, help="batch-size parameter to be used.")
    parser.add_argument("--patience", type=int, default=-1, help="patience, number of extra epochs to perform after "
                                                                "the best epoch before stopping a training.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--incremental", action="store_true", help="training is incremental, starting from existing model if present")
    parser.add_argument("--max-epoch", type=int, default=-1,
                        help="Maximum number of epochs for training.")
    parser.add_argument("--early-stop", type=t_or_f, default=None,
                        help="Force early training termination when metrics scores are not improving " +
                             "after a number of epochs equals to the patience parameter.")

    parser.add_argument("--multi-gpu", default=False,
                        help="Enable the support for distributed computing (the batch size needs to be set accordingly using --batch-size)",
                        action="store_true")

    args = parser.parse_args()

    embeddings_name = args.embedding
    input_file = args.input
    model_name = args.model
    transformer = args.transformer
    architecture = args.architecture
    x_index = args.x_index
    patience = args.patience
    batch_size = args.batch_size
    incremental = args.incremental
    max_sequence_length = args.max_sequence_length
    learning_rate = args.learning_rate
    max_epoch = args.max_epoch
    early_stop = args.early_stop
    multi_gpu = args.multi_gpu

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
        train(model_name, architecture, input_file, embeddings_name, args.fold_count,
              transformer=transformer,
              x_index=x_index,
              y_indexes=y_indexes,
              batch_size=batch_size,
              incremental=incremental,
              max_sequence_length=max_sequence_length,
              patience=patience,
              learning_rate=learning_rate,
              max_epoch=max_epoch,
              early_stop=early_stop,
              multi_gpu=multi_gpu)

    elif args.action == 'eval':
        eval(model_name, architecture, input_file, x_index=x_index, y_indexes=y_indexes)

    elif args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train_and_eval(model_name,
                       architecture,
                       input_file,
                       embeddings_name,
                       args.fold_count,
                       transformer=transformer,
                       x_index=x_index,
                       y_indexes=y_indexes,
                       batch_size=batch_size,
                       max_sequence_length=max_sequence_length,
                       patience=patience,
                       multi_gpu=multi_gpu)

    elif args.action == 'classify':
        lines, _ = load_texts_and_classes_generic(input_file, x_index, None)

        result = classify(model_name, lines, "csv")

        result_binary = [np.argmax(line) for line in result]

        for x in result_binary:
            print(x)
    # See https://github.com/tensorflow/tensorflow/issues/3388
    # K.clear_session()
