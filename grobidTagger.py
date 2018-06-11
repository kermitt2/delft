import os
import json
import numpy as np
from utilities.Embeddings import Embeddings
import sequenceLabelling
from utilities.Tokenizer import tokenizeAndFilter
from sklearn.model_selection import train_test_split
from sequenceLabelling.reader import load_data_and_labels_crf_file
import keras.backend as K
import argparse
import time

models = ['affiliation-address', 'citation', 'date', 'header', 'name-citation', 'name-header']


# train a GROBID model with all available data 
def train(model, embeddings_name): 
    print('Loading data...')
    x_all, y_all, f_all = load_data_and_labels_crf_file('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')

    x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    model = sequenceLabelling.Sequence('grobid-'+model, max_epoch=50, embeddings_name=embeddings_name)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    model.save()

# split data, train a GROBID model and evaluate it 
def train_eval(model, embeddings_name): 
    print('Loading data...')
    x_all, y_all, f_all = load_data_and_labels_crf_file('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')

    x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_eval), 'evaluation sequences')

    model = sequenceLabelling.Sequence('grobid-'+model, max_epoch=50, embeddings_name=embeddings_name)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # evaluation
    print("\nEvaluation:")
    model.eval(x_eval, y_eval)

    # saving the model
    model.save()

# annotate a list of texts, this is relevant only of models taking only text as input 
# (so not text with layout information) 
def annotate_text(texts, model, output_format):
    annotations = []

    # load model
    model = sequenceLabelling.Sequence('grobid-'+model)
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
        description = "Trainer for GROBID models")

    parser.add_argument("model")
    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()
    
    model = args.model    
    if not model in models:
        print('invalid model, should be one of', models)

    action = args.action    
    if (action != 'train') and (action != 'tag') and (action != 'train_eval'):
        print('action not specifed, must be one of [train,train_eval,tag]')

    # change bellow for the desired pre-trained word embeddings using their descriptions in the file 
    # embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    embeddings_name = "glove-840B"

    if action == 'train':
        train(model, embeddings_name)
    
    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(model, embeddings_name)

    if action == 'tag':
        someTexts = []

        if model == 'date':
            someTexts.append("January 2006")
            someTexts.append("March the 27th, 2001")
        elif model == 'citation':
            someTexts.append("N. Al-Dhahir and J. Cioffi, \“On the uniform ADC bit precision and clip level computation for a Gaussian signal,\” IEEE Trans. Signal Processing, pp. 434–438, Feb. 1996.")
            someTexts.append("T. Steinherz, E. Rivlin, N. Intrator, Off-line cursive script word recognition—a survey, Int. J. Doc. Anal. Recognition 2(3) (1999) 1–33.")
        elif model == 'name-citation':
            someTexts.append("L. Romary and L. Foppiano")
            someTexts.append("Maniscalco, S., Francica, F., Zaffino, R.L.")
        elif model == 'name-header':
            someTexts.append("He-Jin Wu 1 · Zhao Jin 2 · Ai-Dong Zhu 1")
            someTexts.append("Irène Charon ⋆ and Olivier Hudry")

        result = annotate_text(someTexts, model, "json")
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()