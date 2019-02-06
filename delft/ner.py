import os
import numpy as np
from delft.sequenceLabelling import Sequence
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import stats
from delft.sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll, load_data_and_labels_lemonde, load_data_and_labels_ontonotes
from sklearn.model_selection import train_test_split
import keras.backend as K
import argparse
import time


# train a model with all available CoNLL 2003 data 
def train(embedding_name, dataset_type='conll2003', lang='en', architecture='BidLSTM_CRF', use_ELMo=False, data_path=None): 

    if (architecture == "BidLSTM_CNN_CRF"):
        word_lstm_units = 200
        recurrent_dropout=0.5
    else:
        word_lstm_units = 100
        recurrent_dropout=0.5

    if use_ELMo:
        batch_size = 120
    else:
        batch_size = 20

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading data...')
        x_train1, y_train1 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
        x_train2, y_train2 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
        x_train3, y_train3 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')

        # we concatenate all sets
        x_all = np.concatenate((x_train1, x_train2, x_train3), axis=0)
        y_all = np.concatenate((y_train1, y_train2, y_train3), axis=0)

        # split train and valid sets in a random way
        x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=60, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embedding_name,
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        use_ELMo=use_ELMo)
    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_train1, y_train1 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.train')
        x_train2, y_train2 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.dev')
        x_train3, y_train3 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')

        # we concatenate train and valid sets
        x_all = np.concatenate((x_train1, x_train2, x_train3), axis=0)
        y_all = np.concatenate((y_train1, y_train2, y_train3), axis=0)

        # split train and valid sets in a random way
        x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=80, 
                        recurrent_dropout=0.20,
                        embeddings_name=embedding_name, 
                        early_stop=True, 
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        use_ELMo=use_ELMo)
    elif (lang == 'fr'):
        print('Loading data...')
        dataset_type = 'lemonde'
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=60, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embedding_name, 
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        use_ELMo=use_ELMo)
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return

    #elif (dataset_type == 'ontonotes') and (lang == 'en'):
    #    model = sequenceLabelling.Sequence('ner-en-ontonotes', max_epoch=60, embeddings_name=embedding_name)
    #elif (lang == 'fr'):
    #    model = sequenceLabelling.Sequence('ner-fr-lemonde', max_epoch=60, embeddings_name=embedding_name)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    model.save()


# train and usual eval on CoNLL 2003 eng.testb 
def train_eval(embedding_name, 
                dataset_type='conll2003', 
                lang='en', 
                architecture='BidLSTM_CRF', 
                fold_count=1, 
                train_with_validation_set=False, 
                use_ELMo=False, 
                data_path=None): 

    if (architecture == "BidLSTM_CNN_CRF"):
        word_lstm_units = 200
        max_epoch = 30
        recurrent_dropout=0.5
    else:        
        word_lstm_units = 100
        max_epoch = 25
        recurrent_dropout=0.5

    if use_ELMo:
        batch_size = 120
    else:
        batch_size = 20

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL 2003 data...')
        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        if not train_with_validation_set: 
            # restrict training on train set, use validation set for early stop, as in most papers
            model = Sequence(model_name, 
                            max_epoch=60, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embedding_name, 
                            early_stop=True, 
                            fold_number=fold_count,
                            model_type=architecture,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            use_ELMo=use_ELMo)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results (~ +0.5 f1 score with CoNLL-2003)
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embedding_name, 
                            early_stop=False, 
                            fold_number=fold_count,
                            model_type=architecture,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            use_ELMo=use_ELMo)

    elif (dataset_type == 'ontonotes-all') and (lang == 'en'):
        print('Loading Ontonotes 5.0 XML data...')
        x_all, y_all = load_data_and_labels_ontonotes(data_path)
        x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-ontonotes'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=60, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embedding_name, 
                        early_stop=True, 
                        fold_number=fold_count,
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        use_ELMo=use_ELMo)

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.train')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.dev')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        if not train_with_validation_set: 
            model = Sequence(model_name, 
                            max_epoch=80, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embedding_name, 
                            early_stop=True, 
                            fold_number=fold_count,
                            model_type=architecture,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            use_ELMo=use_ELMo)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results 
            model = Sequence(model_name, 
                            max_epoch=40, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embedding_name, 
                            early_stop=False, 
                            fold_number=fold_count,
                            model_type=architecture,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            use_ELMo=use_ELMo)

    elif (lang == 'fr'):
        print('Loading data...')
        dataset_type = 'lemonde'
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=60, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embedding_name, 
                        early_stop=True, 
                        fold_number=fold_count,
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        use_ELMo=use_ELMo)
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return        

    start_time = time.time()
    if fold_count == 1:
        model.train(x_train, y_train, x_valid, y_valid)
    else:
        model.train_nfold(x_train, y_train, x_valid, y_valid, fold_number=fold_count)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    print("\nEvaluation on test set:")
    model.eval(x_eval, y_eval)

    # saving the model
    model.save()


# usual eval on CoNLL 2003 eng.testb 
def eval(dataset_type='conll2003', 
         lang='en', 
         architecture='BidLSTM_CRF', 
         use_ELMo=False, 
         data_path=None): 

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL-2003 NER data...')
        x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture
        model = Sequence(model_name)
        model.load()

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture
        model = Sequence(model_name)
        model.load()

    else:
        print("dataset/language combination is not supported for fixed eval:", dataset_type, lang)
        return

    start_time = time.time()

    print("\nEvaluation on test set:")
    model.eval(x_test, y_test)
    runtime = round(time.time() - start_time, 3)

    print("runtime: %s seconds " % (runtime))


# annotate a list of sentences in a file, provides results in a list of offset mentions 
def annotate(output_format, 
             dataset_type='conll2003', 
             lang='en', 
             architecture='BidLSTM_CRF', 
             use_ELMo=False, 
             file_in=None, 
             file_out=None):
    if file_in is None or not os.path.isfile(file_in):
        raise ValueError("the provided input file is not valid")
    annotations = []

    if (dataset_type == 'conll2003') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture
        model = sequenceLabelling.Sequence(model_name)
        model.load()

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture
        model = sequenceLabelling.Sequence(model_name)
        model.load()

    elif (lang == 'fr'):
        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
        model_name += '-' + architecture
        model = sequenceLabelling.Sequence(model_name)
        model.load()
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return 

    start_time = time.time()

    model.tag_file(file_in=file_in, output_format=output_format, file_out=file_out)
    runtime = round(time.time() - start_time, 3)

    print("runtime: %s seconds " % (runtime))

