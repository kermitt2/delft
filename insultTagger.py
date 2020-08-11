import os
import json
from delft.utilities.Embeddings import Embeddings
import delft.sequenceLabelling
from delft.sequenceLabelling import Sequence
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll
import argparse
import keras.backend as K
import time


def train(embeddings_name, architecture='BidLSTM_CRF'): 
    root = os.path.join(os.path.dirname(__file__), 'data/sequenceLabelling/toxic/')

    train_path = os.path.join(root, 'corrected.xml')
    valid_path = os.path.join(root, 'valid.xml')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_xml_file(train_path)
    x_valid, y_valid = load_data_and_labels_xml_file(valid_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    model = Sequence('insult', max_epoch=50, embeddings_name=embeddings_name)
    model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
    print('training done')

    # saving the model
    model.save()


# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts, output_format, architecture='BidLSTM_CRF'):
    annotations = []

    # load model
    model = Sequence('insult', architecture=architecture)
    model.load()

    start_time = time.time()

    annotations = model.tag(texts, output_format)
    runtime = round(time.time() - start_time, 3)

    if output_format is 'json':
        annotations["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return annotations


if __name__ == "__main__":

    architectures = ['BidLSTM_CRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 
                     'bert-base-en', 'bert-base-en', 'scibert', 'biobert']

    parser = argparse.ArgumentParser(
        description = "Experimental insult recognizer for the Wikipedia toxic comments dataset")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture", default='BidLSTM_CRF', choices=architectures,
                        help="Type of model architecture to be used, one of "+str(architectures))
    parser.add_argument(
        "--embedding", default='fasttext-crawl',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )

    args = parser.parse_args()

    if args.action not in ('train', 'tag'):
        print('action not specifed, must be one of [train,tag]')

    embeddings_name = args.embedding
    architecture = args.architecture

    if args.action == 'train':
        train(embeddings_name, architecture=architecture)

    if args.action == 'tag':
        someTexts = ['This is a gentle test.', 
                     'you\'re a moronic wimp who is too lazy to do research! die in hell !!', 
                     'This is a fucking test.']
        result = annotate(someTexts, "json", architecture=architecture)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
