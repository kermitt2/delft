import os
import json
import numpy as np
from utilities.Embeddings import make_embeddings_simple
import sequenceLabelling
from utilities.Tokenizer import tokenizeAndFilter
from sequenceLabelling.reader import load_data_and_labels_crf_file
import keras.backend as K
import argparse
import time

models = ['affiliation-address', 'citation', 'date', 'header', 'name-citztion', 'name-header']


# train a GROBID model with all available data 
def train(model, embedding_vector): 
    print('Loading data...')
    x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')

    print(len(x_train), 'train sequences')

    model = sequenceLabelling.Sequence('grobid-'+model, max_epoch=50, embeddings=embedding_vector)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    model.save()



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

    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec", True)

    if action == 'train':
        train(model, embedding_vector)
    
    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(model, embedding_vector, args.fold_count)

    if action == 'tag':
        someTexts = []

        if model is 'date':
            someTexts.append("January 2006")
            someTexts.append("March the 27th, 2001")
        elif model is 'citation':
            someTexts.append("N. Al-Dhahir and J. Cioffi, \“On the uniform ADC bit precision and clip level computation for a Gaussian signal,\” IEEE Trans. Signal Processing, pp. 434–438, Feb. 1996.")
            someTexts.append("T. Steinherz, E. Rivlin, N. Intrator, Off-line cursive script word recognition—a survey, Int. J. Doc. Anal. Recognition 2(2) (1999) 1–33.")

        result = annotate(someTexts, model, embedding_vector, "json")
        print(json.dumps(result, sort_keys=False, indent=4))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()