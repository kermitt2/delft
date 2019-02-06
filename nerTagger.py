import os
#import numpy as np
#import sequenceLabelling
#from utilities.Tokenizer import tokenizeAndFilter
#from utilities.Embeddings import Embeddings
#from utilities.Utilities import stats
#from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll, load_data_and_labels_lemonde, load_data_and_labels_ontonotes
#from sklearn.model_selection import train_test_split
import keras.backend as K
import argparse
#import time

from delft.ner import train
from delft.ner import train_eval
from delft.ner import eval
from delft.ner import annotate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Neural Named Entity Recognizers")

    parser.add_argument("action", help="one of [train, train_eval, eval, tag]")
    parser.add_argument("--fold-count", type=int, default=1, help="number of folds or re-runs to be used when training")
    parser.add_argument("--lang", default='en', help="language of the model as ISO 639-1 code")
    parser.add_argument("--dataset-type",default='conll2003', help="dataset to be used for training the model")
    parser.add_argument("--train-with-validation-set", action="store_true", help="Use the validation set for training together with the training set")
    parser.add_argument("--architecture",default='BidLSTM_CRF', help="type of model architecture to be used, one of [BidLSTM_CRF, BidLSTM_CNN, BidLSTM_CNN_CRF, BidGRU-CRF]")
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--data-path", default=None, help="path to the corpus of documents for training (only use currently with Ontonotes corpus in orginal XML format)") 
    parser.add_argument("--file-in", default=None, help="path to a text file to annotate") 
    parser.add_argument("--file-out", default=None, help="path for outputting the resulting JSON NER anotations") 

    args = parser.parse_args()

    action = args.action    
    if action not in ('train', 'tag', 'eval', 'train_eval'):
        print('action not specifed, must be one of [train, train_eval, eval, tag]')
    lang = args.lang
    dataset_type = args.dataset_type
    train_with_validation_set = args.train_with_validation_set
    use_ELMo = args.use_ELMo
    architecture = args.architecture
    if architecture not in ('BidLSTM_CRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF'):
        print('unknown model architecture, must be one of [BidLSTM_CRF, BidLSTM_CNN_CRF, BidLSTM_CNN_CRF, BidGRU_CRF]')
    data_path = args.data_path
    file_in = args.file_in
    file_out = args.file_out

    # change bellow for the desired pre-trained word embeddings using their descriptions in the file 
    # embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    if lang == 'en':
        if dataset_type == 'conll2012':
            embeddings_name = 'fasttext-crawl'
        else:
            embeddings_name = "glove-840B"
    elif lang == 'fr':
        embeddings_name = 'wiki.fr'

    if action == 'train':
        train(embeddings_name, 
            dataset_type, 
            lang, 
            architecture=architecture, 
            use_ELMo=use_ELMo,
            data_path=data_path)

    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(embeddings_name, 
            dataset_type, 
            lang, 
            architecture=architecture, 
            fold_count=args.fold_count, 
            train_with_validation_set=train_with_validation_set, 
            use_ELMo=use_ELMo,
            data_path=data_path)

    if action == 'eval':
        eval(dataset_type, lang, architecture=architecture, use_ELMo=use_ELMo)

    if action == 'tag':
        if lang is not 'en' and lang is not 'fr':
            print("Language not supported:", lang)
        else: 
            print(file_in)
            result = annotate("json", 
                            dataset_type, 
                            lang, 
                            architecture=architecture, 
                            use_ELMo=use_ELMo, 
                            file_in=file_in, 
                            file_out=file_out)
            """if result is not None:
                if file_out is None:
                    print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
            """
    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
