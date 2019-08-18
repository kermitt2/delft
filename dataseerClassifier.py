import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_dataseer_corpus_csv
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import keras.backend as K
import time
from delft.textClassification.models import modelTypes
import numpy as np

"""
    Classifier for deciding if a sentence introduce a dataset or not, and prediction of the 
    dataset type. 
"""

def train(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru"): 
    print('loading dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-1.csv")

    class_weights = None
    batch_size = 256
    maxlen = 300
    if use_ELMo:
        batch_size = 20
    elif use_BERT:
        batch_size = 50

    # default bert model parameters
    if architecture.find("bert") != -1:
        batch_size = 32
        maxlen = 100

    model = Classifier('dataseer', model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size,
        class_weights=class_weights)

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def train_and_eval(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru"): 
    print('loading dataset type corpus...')
    #xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-1.csv")
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    # distinct values of classes
    print(list_classes)
    print(len(list_classes), "classes")

    print(len(xtr), "texts")
    print(len(y), "classes")

    class_weights = None
    batch_size = 256
    maxlen = 300
    if use_ELMo:
        batch_size = 20
    elif use_BERT:
        batch_size = 50

    # default bert model parameters
    if architecture.find("bert") != -1:
        batch_size = 32
        maxlen = 100

    model = Classifier('dataseer', model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size, maxlen=maxlen,
        class_weights=class_weights)

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    print(len(x_train), "train texts")
    print(len(y_train), "train classes")

    print(len(x_test), "eval texts")
    print(len(y_test), "eval classes")

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


def classify(texts, output_format, architecture="gru"):
    '''
        Classify a list of texts with an existing model
    '''
    # load model
    model = Classifier('dataseer', model_type=architecture)
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
        description = "Dataset identification and classification for scientific literature")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture",default='gru', help="type of model architecture to be used, one of "+str(modelTypes))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--use-BERT", action="store_true", help="Use BERT contextual embeddings") 
    parser.add_argument(
        "--embedding", default='word2vec',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )

    args = parser.parse_args()

    if args.action not in ('train', 'train_eval', 'classify'):
        print('action not specifed, must be one of [train,train_eval,classify]')

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
        someTexts = ['Labeling yield and radiochemical purity was analyzed by instant thin layered chromatography (ITLC).', 
            'NOESY and ROESY spectra64,65 were collected with typically 128 scans per t1 increment, with the residual water signal removed by the WATERGATE sequence and 1 s relaxation time.', 
            'The concentrations of Cd and Pb in feathers were measured by furnace atomic absorption spectrometry (VARIAN 240Z).']
        result = classify(someTexts, "json", architecture=architecture)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # See https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()