import os
import json
from utilities.Embeddings import Embeddings
import sequenceLabelling
from utilities.Tokenizer import tokenizeAndFilter
from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll
import argparse
import keras.backend as K
import time

def train(embedding_vector): 
    train_path = os.path.join(root, 'corrected.xml')
    valid_path = os.path.join(root, 'valid.xml')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_xml_file(train_path)
    x_valid, y_valid = load_data_and_labels_xml_file(valid_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    model = sequenceLabelling.Sequence('insult', max_epoch=50, embeddings=embedding_vector)
    model.train(x_train, y_train, x_valid, y_valid)
    print('training done')

    # saving the model
    model.save()

# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts, embedding_vector, output_format):
    annotations = []

    # load model
    model = sequenceLabelling.Sequence('insult', embeddings=embedding_vector)
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
        description = "Experimental insult recognizer for the Wikipedia toxic comments dataset")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()
    
    action = args.action    
    if (action != 'train') and (action != 'tag'):
        print('action not specifed, must be one of [train,tag]')

    embedding_vector = Embeddings("fasttext-crawl").model

    if action == 'train':
        train(embedding_vector)

    if action == 'tag':
        someTexts = ['This is a gentle test.', 
                     'you\'re a moronic wimp who is too lazy to do research! die in hell !!', 
                     'This is a fucking test.']
        result = annotate(someTexts, embedding_vector, "json")
        print(json.dumps(result, sort_keys=False, indent=4))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
