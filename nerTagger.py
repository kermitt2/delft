import os
import json
from utilities.Embeddings import make_embeddings_simple
import sequenceLabelling
from sequenceLabelling.tokenizer import tokenizeAndFilter
from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll
import keras.backend as K
import argparse
import time

# train a model with all available CoNLL 2003 data 
def train(embedding_vector, fold_count): 
    root = os.path.join(os.path.dirname(__file__), '../data/sequence/')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
    x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
    x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_test), 'evaluation sequences')

    model = sequenceLabelling.Sequence('ner', max_epoch=1, embeddings=embedding_vector)
    model.train(x_train, y_train, x_valid, y_valid)

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
    model.train(x_train, y_train, x_valid, y_valid)

    model.eval(x_test, y_test)

    # saving the model
    model.save()

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
    if (action != 'train') and (action != 'tag'):
        print('action not specifed, must be one of [train,tag]')

    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec", True)

    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train(embedding_vector, args.fold_count)
    
    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(embedding_vector, args.fold_count)

    if action == 'tag':
        someTexts = ['The University of California has found that 40 percent of its students suffer food insecurity. At four state universities in Illinois, that number is 35 percent.',
                     'President Obama is not speaking anymore from the White House.']
        result = annotate(someTexts, embedding_vector, "json")
        print(json.dumps(result, sort_keys=False, indent=4))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
