import os
import json
from utilities.Embeddings import make_embeddings_simple
from utilities.Utilities import split_data_and_labels
from textClassification.reader import load_texts_and_classes_pandas
import textClassification
import argparse

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def train(embedding_vector): 
    model = textClassification.Classifier('toxic', "gru", max_epoch=25, embeddings=embedding_vector)

    xtr, y = load_texts_and_classes_pandas("data/textClassification/toxic/train.csv")
    model.train(xtr, y)

    # saving the model
    model.save()


def train_and_eval(embedding_vector): 
    model = textClassification.Classifier('toxic', "gru", max_epoch=25, embeddings=embedding_vector)

    xtr, y = load_texts_and_classes_pandas("data/textClassification/toxic/train.csv")

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y)

    model.train(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


# classify a list of texts
def classify(texts, embedding_vector):
    # load model
    model = textClassification.Classifier('toxic', "gru", embeddings=embedding_vector)
    model.load()
    results = []

    for text in texts:
        results.append(model.predict(text))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classification of comments/short texts in toxicity types (toxic, severe_toxic, obscene, threat, insult, identity_hate)")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()

    action = args.action    
    if (action != 'train') and (action != 'classify'):
        print('action not specifed, must be one of [train,classify]')

    print('importing embeddings...')
    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec", True)

    print('building dataframe...')
    #xtr, xte, y, word_index = make_df("../data/textClassification/toxic/train.csv",
    #                                "../data/textClassification/toxic/test.csv",
    #                                max_features, maxlen, list_classes) 
    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train(embedding_vector)

    if action == 'test':
        y_test = test(embedding_vector)    

        # write test predictions as a submission file 
        sample_submission = pd.read_csv("data/textClassification/toxic/sample_submission.csv")
        sample_submission[list_classes] = y_test
        sample_submission.to_csv("data/textClassification/toxic/"+resultFile, index=False)

    if action == 'classify':
        someTexts = ['This is a gentle test.', 'This is a fucking test!', 'This is a test and I know where you leave.']
        classify(someTexts, embedding_vector)

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
