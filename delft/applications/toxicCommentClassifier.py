import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_texts_and_classes_pandas
from delft.textClassification.reader import load_texts_pandas
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import pandas as pd
import time
import sys
from delft.textClassification.models import architectures

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
class_weights = {0: 1.,
                 1: 1.,
                 2: 1.,
                 3: 1.,
                 4: 1.,
                 5: 1.}

def configure(architecture):
    batch_size = 256
    maxlen = 300
    patience = 5
    early_stop = True
    max_epoch = 30

    # default bert model parameters
    if architecture == "bert":
        batch_size = 32
        early_stop = False
        max_epoch = 3
        maxlen = 200

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(embeddings_name=None, fold_count=1, architecture="gru", transformer=None): 
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier('toxic_'+architecture, architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, class_weights=class_weights,
        embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop, 
        transformer_name=transformer)

    print('loading train dataset...')
    xtr, y = load_texts_and_classes_pandas("data/textClassification/toxic/train.csv")
    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def test(architecture="gru"):
    # load model
    model = Classifier('toxic_'+architecture)
    model.load()

    print('loading test dataset...')
    xte = load_texts_pandas("data/textClassification/toxic/test.csv")
    print('number of texts to classify:', len(xte))
    start_time = time.time()
    result = model.predict(xte, output_format="csv")
    print("runtime: %s seconds " % (round(time.time() - start_time, 3)))
    return result


# classify a list of texts
def classify(texts, output_format, architecture="gru", transformer=None):
    # load model
    model = Classifier('toxic_'+architecture)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    print("runtime: %s seconds " % (round(time.time() - start_time, 3)))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classification of comments/short texts in toxicity types (toxic, severe_toxic, obscene, threat, insult, identity_hate) based on DeLFT")

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']
    pretrained_transformers_examples = [ 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased' ]

    parser.add_argument("action", help="one of [train, test, classify]")
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

    action = args.action    
    if action not in ('train', 'classify', 'test'):
        print('action not specifed, must be one of [train,test,classify]')

    embeddings_name = args.embedding
    transformer = args.transformer

    architecture = args.architecture
    if architecture not in architectures:
        print('unknown model architecture, must be one of '+str(architectures))

    if transformer == None and embeddings_name == None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if architecture.find("bert") != -1:
        print('BERT models are not supported for multi-label labelling, at least for the moment. Please choose a RNN architecture.')
        sys.exit(0)

    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train(embeddings_name=embeddings_name, fold_count=args.fold_count, architecture=architecture, transformer=transformer)

    if action == 'test':
        y_test = test()    

        # write test predictions as a submission file 
        sample_submission = pd.read_csv("data/textClassification/toxic/sample_submission.csv")
        sample_submission[list_classes] = y_test
        sample_submission.to_csv("data/textClassification/toxic/result.csv", index=False)

    if action == 'classify':
        someTexts = ['This is a gentle test.', 'This is a fucking test!', 'With all due respects, I think you\'re a moron.']
        result = classify(someTexts, "json", architecture=architecture)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
