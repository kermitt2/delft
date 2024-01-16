import os
import numpy as np
from delft.sequenceLabelling import Sequence
from delft.utilities.Utilities import stats, t_or_f
from delft.utilities.numpy import shuffle_arrays
from delft.sequenceLabelling.reader import load_data_and_labels_conll, load_data_and_labels_lemonde, load_data_and_labels_ontonotes
from sklearn.model_selection import train_test_split
import argparse
import time

def configure(architecture, dataset_type, lang, embeddings_name,
              use_ELMo, max_sequence_length=-1, batch_size=-1,
              patience=-1, max_epoch=-1, early_stop=None):

    o_max_epoch = 60
    o_early_stop = True
    o_multiprocessing = True
    o_max_sequence_length = 300
    o_patience = 5
    o_batch_size = 32

    # general RNN word embeddings input
    if embeddings_name is None:
        o_embeddings_name = 'glove-840B'
        if lang == 'en':
            if dataset_type == 'conll2012':
                o_embeddings_name = 'fasttext-crawl'
        elif lang == 'fr':
            o_embeddings_name = 'wiki.fr'
    else:
        o_embeddings_name = embeddings_name

    if lang == 'fr':
        o_multiprocessing = False

    if architecture == "BidLSTM_CNN_CRF":
        o_word_lstm_units = 200
        o_max_epoch = 30
        o_recurrent_dropout = 0.5
    else:
        o_word_lstm_units = 100
        o_max_epoch = 50
        o_recurrent_dropout = 0.5

    if use_ELMo:
        # following should be done for predicting if max sequence length permits, it also boosts the runtime with ELMo embeddings signicantly
        # but requires more GPU memory
        o_batch_size = 128
        o_max_sequence_length = 150

    # default bert model parameters
    if architecture.find("BERT") != -1:
        o_batch_size = 32
        o_early_stop = True
        o_max_sequence_length = 150
        o_max_epoch = 50
        o_embeddings_name = None

    if dataset_type == 'conll2012':
        o_multiprocessing = False

    if patience > 0:
        o_patience = patience

    if batch_size > 0:
        o_batch_size = batch_size

    if max_sequence_length > 0:
        o_max_sequence_length = max_sequence_length

    if max_epoch > 0:
        o_max_epoch = max_epoch

    if early_stop is not None:
        o_early_stop = early_stop

    return o_batch_size, o_max_sequence_length, o_patience, o_recurrent_dropout, o_early_stop, o_max_epoch, o_embeddings_name, o_word_lstm_units, o_multiprocessing


# train a model with all available for a given dataset 
def train(dataset_type='conll2003', lang='en', embeddings_name=None, architecture='BidLSTM_CRF',

          transformer=None, data_path=None, use_ELMo=False, max_sequence_length=-1,
          batch_size=-1, patience=-1, learning_rate=None, max_epoch=-1, early_stop=None, multi_gpu=False):

    batch_size, max_sequence_length, patience, recurrent_dropout, early_stop, max_epoch, embeddings_name, word_lstm_units, multiprocessing = \
        configure(architecture, dataset_type, lang, embeddings_name, use_ELMo, max_sequence_length, batch_size, patience, max_epoch, early_stop)

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

        model_name = 'ner-en-conll2003-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        model = Sequence(model_name, 
                        max_epoch=max_epoch, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embeddings_name,
                        architecture=architecture,
                        transformer_name=transformer,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        early_stop=early_stop,
                        patience=patience,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        multiprocessing=multiprocessing,
                        learning_rate=learning_rate)

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

        model_name = 'ner-en-conll2012-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        model = Sequence(model_name, 
                        max_epoch=max_epoch, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embeddings_name, 
                        architecture=architecture,
                        transformer_name=transformer,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        early_stop=early_stop,
                        patience=patience,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        multiprocessing=multiprocessing,
                        learning_rate=learning_rate)
    elif (lang == 'fr'):
        print('Loading data...')
        dataset_type = 'lemonde'
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        shuffle_arrays([x_all, y_all])
        x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-fr-lemonde-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        model = Sequence(model_name, 
                        max_epoch=max_epoch, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embeddings_name, 
                        architecture=architecture,
                        transformer_name=transformer,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        early_stop=early_stop,
                        patience=patience,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        multiprocessing=multiprocessing,
                        learning_rate=learning_rate)
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return

    #elif (dataset_type == 'ontonotes') and (lang == 'en'):
    #    model = sequenceLabelling.Sequence('ner-en-ontonotes', max_epoch=60, embeddings_name=embeddings_name)
    #elif (lang == 'fr'):
    #    model = sequenceLabelling.Sequence('ner-fr-lemonde', max_epoch=60, embeddings_name=embeddings_name)

    start_time = time.time()
    model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    model.save()


# train and usual eval on dataset, e.g. eval with CoNLL 2003 eng.testb for CoNLL 2003 
def train_eval(embeddings_name=None, 
                dataset_type='conll2003', 
                lang='en', 
                architecture='BidLSTM_CRF', 
                transformer=None, 
                fold_count=1, 
                train_with_validation_set=False,
                data_path=None, 
                use_ELMo=False,
                patience=-1,
                batch_size=-1,
                max_sequence_length=-1,
                learning_rate=None,
                max_epoch=-1,
                early_stop=None,
                multi_gpu=False):

    batch_size, max_sequence_length, patience, recurrent_dropout, early_stop, max_epoch, embeddings_name, word_lstm_units, multiprocessing = \
        configure(architecture, dataset_type, lang, embeddings_name, use_ELMo,
                  max_sequence_length=max_sequence_length, batch_size=batch_size, patience=patience, max_epoch=max_epoch, early_stop=early_stop)

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL 2003 data...')
        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-conll2003-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        if not train_with_validation_set: 
            # restrict training on train set, use validation set for early stop, as in most papers
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            early_stop=True,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results (~ +0.5 f1 score with CoNLL-2003)
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            early_stop=False, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)

    elif (dataset_type == 'ontonotes-all') and (lang == 'en'):
        print("Loading all Ontonotes 5.0 XML data, evaluation will be on 10\% random partition")
        x_all, y_all = load_data_and_labels_ontonotes(data_path)
        x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-ontonotes-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        model = Sequence(model_name, 
                        max_epoch=max_epoch, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embeddings_name, 
                        fold_number=fold_count,
                        architecture=architecture,
                        transformer_name=transformer,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        early_stop=early_stop,
                        patience=patience,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        multiprocessing=multiprocessing,
                        learning_rate=learning_rate)

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.train')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.dev')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-conll2012-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        if not train_with_validation_set: 
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            early_stop=True,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)
        else:
            # also use validation set to train (no early stop, hyperparameters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results 
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            early_stop=False, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            patience=patience, 
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)

    elif (lang == 'fr') and (dataset_type == 'ftb' or dataset_type is None):
        print('Loading data for ftb...')
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        shuffle_arrays([x_all, y_all])
        x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        model = Sequence(model_name, 
                        max_epoch=max_epoch, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embeddings_name, 
                        fold_number=fold_count,
                        architecture=architecture,
                        transformer_name=transformer,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        early_stop=early_stop,
                        patience=patience,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        multiprocessing=multiprocessing,
                        learning_rate=learning_rate)
    elif (lang == 'fr') and (dataset_type == 'ftb_force_split'):
        print('Loading data for ftb_force_split...')
        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/leMonde/ftb6_train.conll')
        shuffle_arrays([x_train, y_train])
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/leMonde/ftb6_dev.conll')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/leMonde/ftb6_test.conll')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde-force-split-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        if not train_with_validation_set: 
            # restrict training on train set, use validation set for early stop, as in most papers
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            early_stop=True, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results (~ +0.5 f1 score with CoNLL-2003)
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            early_stop=False, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)
    elif (lang == 'fr') and (dataset_type == 'ftb_force_split_xml'):
        print('Loading data for ftb_force_split_xml...')
        x_train, y_train = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.train.xml')
        shuffle_arrays([x_train, y_train])
        x_valid, y_valid = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.dev.xml')
        x_eval, y_eval = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.test.xml')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde-force-split-xml-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'

        if not train_with_validation_set: 
            # restrict training on train set, use validation set for early stop, as in most papers
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            early_stop=True, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results (~ +0.5 f1 score with CoNLL-2003)
            model = Sequence(model_name, 
                            max_epoch=max_epoch, 
                            recurrent_dropout=recurrent_dropout,
                            embeddings_name=embeddings_name, 
                            early_stop=False, 
                            fold_number=fold_count,
                            architecture=architecture,
                            transformer_name=transformer,
                            word_lstm_units=word_lstm_units,
                            batch_size=batch_size,
                            patience=patience,
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            multiprocessing=multiprocessing,
                            learning_rate=learning_rate)
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return        

    start_time = time.time()
    if fold_count == 1:
        model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid, multi_gpu=multi_gpu)
    else:
        model.train_nfold(x_train, y_train, x_valid=x_valid, y_valid=y_valid, multi_gpu=multi_gpu)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    print("\nEvaluation on test set:")
    model.eval(x_eval, y_eval)

    # # saving the model (must be called after eval for multiple fold training)
    model.save()


# usual eval on CoNLL 2003 eng.testb 
def eval(dataset_type='conll2003', 
         lang='en', 
         architecture='BidLSTM_CRF', 
         data_path=None,
         use_ELMo=False): 

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL-2003 NER data...')
        x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2003-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'
        model = Sequence(model_name)
        model.load()

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2012-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'
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
             file_in=None, 
             file_out=None,
             use_ELMo=False,
             multi_gpu=False):
    if file_in is None:
        raise ValueError("an input file to be annotated must be provided")

    if not os.path.isfile(file_in):
        raise ValueError("the provided input file is not valid")
        
    annotations = []

    if (dataset_type == 'conll2003') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2003-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'
        model = Sequence(model_name)
        model.load()

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2012-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'
        model = Sequence(model_name)
        model.load()

    elif (lang == 'fr'):
        model_name = 'ner-fr-lemonde-' + architecture
        if use_ELMo:
            model_name += '-with_ELMo'
        model = Sequence(model_name)
        model.load()
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return 

    start_time = time.time()

    model.tag_file(file_in=file_in, output_format=output_format, file_out=file_out, multi_gpu=multi_gpu)
    runtime = round(time.time() - start_time, 3)

    print("runtime: %s seconds " % (runtime))


if __name__ == "__main__":

    architectures_word_embeddings = [
                     'BidLSTM', 'BidLSTM_CRF', 'BidLSTM_ChainCRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 
                     ]

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']

    architectures_transformers_based = [
                    'BERT', 'BERT_CRF', 'BERT_ChainCRF', 'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES'
                     ]

    pretrained_transformers_examples = [ 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased' ]

    architectures = architectures_word_embeddings + architectures_transformers_based

    parser = argparse.ArgumentParser(description="Neural Named Entity Recognizers based on DeLFT")

    parser.add_argument("action", help="one of [train, train_eval, eval, tag]")
    parser.add_argument("--fold-count", type=int, default=1, help="number of folds or re-runs to be used when training")
    parser.add_argument("--lang", default='en', help="language of the model as ISO 639-1 code (en, fr, de, etc.)")
    parser.add_argument("--dataset-type",default='conll2003', help="dataset to be used for training the model")
    parser.add_argument("--train-with-validation-set", action="store_true", help="Use the validation set for training together with the training set")
    parser.add_argument("--architecture", default='BidLSTM_CRF', help="type of model architecture to be used, one of "+str(architectures))
    parser.add_argument("--data-path", default=None, help="path to the corpus of documents for training (only use currently with Ontonotes corpus in orginal XML format)") 
    parser.add_argument("--file-in", default=None, help="path to a text file to annotate") 
    parser.add_argument("--file-out", default=None, help="path for outputting the resulting JSON NER annotations")
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument(
        "--embedding", 
        default=None,
        help="The desired pre-trained word embeddings using their descriptions in the file. " + \
            "For local loading, use delft/resources-registry.json. " + \
            "Be sure to use here the same name as in the registry, e.g. " + str(word_embeddings_examples) + \
            " and that the path in the registry to the embedding file is correct on your system."
    )
    parser.add_argument(
        "--transformer", 
        default=None,
        help="The desired pre-trained transformer to be used in the selected architecture. " + \
            "For local loading use, delft/resources-registry.json, and be sure to use here the same name as in the registry, e.g. " + \
            str(pretrained_transformers_examples) + \
            " and that the path in the registry to the model path is correct on your system. " + \
            "HuggingFace transformers hub will be used otherwise to fetch the model, see https://huggingface.co/models " + \
            "for model names"
    )

    parser.add_argument("--max-sequence-length", type=int, default=-1, help="max-sequence-length parameter to be used.")
    parser.add_argument("--batch-size", type=int, default=-1, help="batch-size parameter to be used.")
    parser.add_argument("--patience", type=int, default=-1, help="patience, number of extra epochs to perform after "
                                                                 "the best epoch before stopping a training.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial learning rate")

    parser.add_argument("--max-epoch", type=int, default=-1,
                        help="Maximum number of epochs.")
    parser.add_argument("--early-stop", type=t_or_f, default=None,
                        help="Force early training termination when metrics scores are not improving " + 
                             "after a number of epochs equals to the patience parameter.")

    parser.add_argument("--multi-gpu", default=False,
                        help="Enable the support for distributed computing (the batch size needs to be set accordingly using --batch-size)",
                        action="store_true")


    args = parser.parse_args()

    action = args.action    
    if action not in ('train', 'tag', 'eval', 'train_eval'):
        print('action not specified, must be one of [train, train_eval, eval, tag]')
    lang = args.lang
    dataset_type = args.dataset_type
    train_with_validation_set = args.train_with_validation_set
    architecture = args.architecture
    if architecture not in architectures:
        print('unknown model architecture, must be one of', architectures)
    transformer = args.transformer
    data_path = args.data_path
    file_in = args.file_in
    file_out = args.file_out
    use_ELMo = args.use_ELMo
    patience = args.patience
    max_sequence_length = args.max_sequence_length
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_epoch = args.max_epoch
    early_stop = args.early_stop
    multi_gpu = args.multi_gpu

    # name of embeddings refers to the file delft/resources-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    # below we set the default embeddings value
    embeddings_name = args.embedding

    if action == 'train':
        train( 
            dataset_type=dataset_type, 
            lang=lang, 
            embeddings_name=embeddings_name,
            architecture=architecture, 
            transformer=transformer,
            data_path=data_path,
            use_ELMo=use_ELMo,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            patience=patience,
            learning_rate=learning_rate,
            max_epoch=max_epoch,
            early_stop=early_stop,
            multi_gpu=multi_gpu
        )

    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(
            dataset_type=dataset_type, 
            lang=lang, 
            embeddings_name=embeddings_name, 
            architecture=architecture, 
            transformer=transformer,
            fold_count=args.fold_count, 
            train_with_validation_set=train_with_validation_set, 
            data_path=data_path,
            use_ELMo=use_ELMo,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            patience=patience,
            learning_rate=learning_rate,
            max_epoch=max_epoch,
            early_stop=early_stop,
            multi_gpu=multi_gpu
            )

    if action == 'eval':
        eval(
            dataset_type=dataset_type, 
            lang=lang, 
            architecture=architecture, 
            #transformer=transformer,
            use_ELMo=use_ELMo)

    if action == 'tag':
        if lang != 'en' and lang != 'fr':
            print("Language not supported:", lang)
        else: 
            print(file_in)
            annotate("json",
                            dataset_type,
                            lang,
                            architecture=architecture,
                            #transformer=transformer,
                            file_in=file_in,
                            file_out=file_out,
                            use_ELMo=use_ELMo,
                            multi_gpu=multi_gpu)
            """
            if result is not None:
                if file_out is None:
                    print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
            """
