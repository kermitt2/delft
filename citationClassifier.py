import os
import json
from utilities.Embeddings import make_embeddings_simple
from utilities.Utilities import split_data_and_labels
from textClassification.reader import load_citation_sentiment_corpus
import textClassification
import argparse
import keras.backend as K
import time

list_classes = ["negative", "neutral", "positive"]

def train(embedding_vector, fold_count): 
    model = textClassification.Classifier('citations', "gru", list_classes=list_classes, max_epoch=50, fold_number=fold_count, 
        use_roc_auc=True, embeddings=embedding_vector)

    print('loading citation sentiment corpus...')
    xtr, y = load_citation_sentiment_corpus("data/textClassification/citations/citation_sentiment_corpus.txt")
    
    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def train_and_eval(embedding_vector, fold_count): 
    model = textClassification.Classifier('citations', "gru", list_classes=list_classes, max_epoch=50, fold_number=fold_count, 
        use_roc_auc=True, embeddings=embedding_vector)

    print('loading citation sentiment corpus...')
    xtr, y = load_citation_sentiment_corpus("data/textClassification/citations/citation_sentiment_corpus.txt")

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


# classify a list of texts
def classify(texts, embedding_vector, output_format):
    # load model
    model = textClassification.Classifier('citations', "gru", list_classes=list_classes, embeddings=embedding_vector)
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
        description = "Sentiment classification of citation passages")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()

    action = args.action    
    if (action != 'train') and (action != 'train_eval') and (action != 'classify'):
        print('action not specifed, must be one of [train,train_eval,classify]')

    print('importing embeddings...')
    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec", True)

    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        else:
            train(embedding_vector, args.fold_count)

    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        else:
            y_test = train_and_eval(embedding_vector, args.fold_count)    

    if action == 'classify':
        someTexts = ['One successful strategy [15] computes the set-similarity involving (multi-word) keyphrases about the mentions and the entities, collected from the KG.', 
            'Unfortunately, fewer than half of the OCs in the DAML02 OC catalog (Dias et al. 2002) are suitable for use with the isochrone-fitting method because of the lack of a prominent main sequence, in addition to an absence of radial velocity and proper-motion data.', 
            'However, we found that the pairwise approach LambdaMART [41] achieved the best performance on our datasets among most learning to rank algorithms.']
        result = classify(someTexts, embedding_vector, "json")
        print(json.dumps(result, sort_keys=False, indent=4))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
