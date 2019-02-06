import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_texts_and_classes_pandas
from delft.textClassification.reader import load_texts_pandas
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import keras.backend as K
import pandas as pd
import time

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def train(embeddings_name, fold_count): 
    model = Classifier('toxic', "gru", list_classes=list_classes, max_epoch=30, 
        fold_number=fold_count, embeddings_name=embeddings_name)

    print('loading train dataset...')
    xtr, y = load_texts_and_classes_pandas("data/textClassification/toxic/train.csv")
    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def test():
    # load model
    model = Classifier('toxic', "gru", list_classes=list_classes)
    model.load()

    print('loading test dataset...')
    xte = load_texts_pandas("data/textClassification/toxic/test.csv")
    print('number of texts to classify:', len(xte))
    start_time = time.time()
    result = model.predict(xte, output_format="csv")
    print("runtime: %s seconds " % (round(time.time() - start_time, 3)))
    return result


# classify a list of texts
def classify(texts, output_format):
    # load model
    model = Classifier('toxic', "gru", list_classes=list_classes)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    print("runtime: %s seconds " % (round(time.time() - start_time, 3)))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classification of comments/short texts in toxicity types (toxic, severe_toxic, obscene, threat, insult, identity_hate)")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()

    action = args.action    
    if action not in ('train', 'classify', 'test'):
        print('action not specifed, must be one of [train,test,classify]')

    # change bellow for the desired pre-trained word embeddings using their descriptions in the file 
    # embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    embeddings_name = "fasttext-crawl"

    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(embeddings_name, args.fold_count)

    if action == 'test':
        y_test = test()    

        # write test predictions as a submission file 
        sample_submission = pd.read_csv("data/textClassification/toxic/sample_submission.csv")
        sample_submission[list_classes] = y_test
        sample_submission.to_csv("data/textClassification/toxic/result.csv", index=False)

    if action == 'classify':
        someTexts = ['This is a gentle test.', 'This is a fucking test!', 'With all due respects, I think you\'re a moron.']
        result = classify(someTexts, "json")
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # See https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
