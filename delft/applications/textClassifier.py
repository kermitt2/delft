import argparse
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from delft.textClassification import Classifier
from delft.textClassification.models import architectures
from delft.textClassification.reader import load_texts_and_classes_generic

pretrained_transformers_examples = ['bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased']


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
        model.model_config.fold_number=1

    model.eval(x_test, y_test)

    # saving the model
    model.save()


# classify a list of texts
def classify(texts, output_format, architecture="gru", transformer=None):
    # load model
    model = Classifier(model_name, architecture=architecture, embeddings_name=embeddings_name,
                       transformer_name=transformer)
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

    parser.add_argument("action")
    parser.add_argument("model", help="The name of the model")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--input", type=str, required=True, help="The file to be used for training/evaluation")
    parser.add_argument("--x-index", type=int, required=True, help="Index of the columns for the X value "
                                                                   "(assuming a TSV file)")
    parser.add_argument("--y-indexes", type=str, required=True, help="Index(es) of the columns for the Y (classes) "
                                                                     "separated by comma, without spaces (assuming a TSV file)")
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

    if args.action not in ('train', 'train_eval', 'eval', 'classify'):
        print('action not specified, must be one of [train, train_eval, eval, classify]')

    embeddings_name = args.embedding
    input_file = args.input
    model_name = args.model
    transformer = args.transformer
    architecture = args.architecture
    x_index = args.x_index
    y_indexes = [int(index) for index in args.y_indexes.split(",")]

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
        someTexts = [
            'One successful strategy [15] computes the set-similarity involving (multi-word) keyphrases about the mentions and the entities, collected from the KG.',
            'Unfortunately, fewer than half of the OCs in the DAML02 OC catalog (Dias et al. 2002) are suitable for use with the isochrone-fitting method because of the lack of a prominent main sequence, in addition to an absence of radial velocity and proper-motion data.',
            'However, we found that the pairwise approach LambdaMART [41] achieved the best performance on our datasets among most learning to rank algorithms.']
        result = classify(model_name, someTexts, "json")
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # See https://github.com/tensorflow/tensorflow/issues/3388
    # K.clear_session()
