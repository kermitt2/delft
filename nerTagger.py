import os
import json
import numpy as np
import sequenceLabelling
from utilities.Tokenizer import tokenizeAndFilter
from utilities.Embeddings import Embeddings
from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll
import keras.backend as K
import argparse
import time

# train a model with all available CoNLL 2003 data 
def train(embedding_vector): 
    print('Loading data...')
    x_train1, y_train1 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
    x_train2, y_train2 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')

    # we concatenate train and valid sets
    x_train = np.concatenate((x_train1, x_train2), axis=0)
    y_train = np.concatenate((y_train1, y_train2), axis=0)

    x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    model = sequenceLabelling.Sequence('ner', max_epoch=50, embeddings=embedding_vector)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    model.save()

# train and usual eval on CoNLL 2003 eng.testb 
def train_eval(embedding_vector): 
    root = os.path.join(os.path.dirname(__file__), '../data/sequence/')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
    x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
    x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_test), 'evaluation sequences')

    model = sequenceLabelling.Sequence('ner', max_epoch=50, embeddings=embedding_vector)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    print("\nEvaluation on test set:")
    model.eval(x_test, y_test)

    # saving the model
    model.save()

# usual eval on CoNLL 2003 eng.testb 
def eval(embedding_vector): 
    root = os.path.join(os.path.dirname(__file__), '../data/sequence/')

    print('Loading data...')
    x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
    print(len(x_test), 'evaluation sequences')

    # load model
    model = sequenceLabelling.Sequence('ner', embeddings=embedding_vector)
    model.load()

    start_time = time.time()

    print("\nEvaluation on test set:")
    model.eval(x_test, y_test)
    runtime = round(time.time() - start_time, 3)
    
    print("runtime: %s seconds " % (runtime))

# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts, embedding_vector, output_format):
    annotations = []

    # load model
    model = sequenceLabelling.Sequence('ner', embeddings=embedding_vector)
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
    parser = argparse.ArgumentParser(
        description = "Named Entity Recognizer")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()
    
    action = args.action    
    if (action != 'train') and (action != 'tag') and (action != 'eval') and (action != 'train_eval'):
        print('action not specifed, must be one of [train,train_eval,eval,tag]')

    #embedding_vector = Embeddings("fasttext-crawl").model
    embedding_vector = Embeddings("glove-840B").model

    if action == 'train':
        train(embedding_vector)
    
    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(embedding_vector)

    if action == 'eval':
        eval(embedding_vector)

    if action == 'tag':
        someTexts = ['The University of California has found that 40 percent of its students suffer food insecurity. At four state universities in Illinois, that number is 35 percent.',
                     'President Obama is not speaking anymore from the White House.']
        result = annotate(someTexts, embedding_vector, "json")
        print(json.dumps(result, sort_keys=False, indent=4))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
