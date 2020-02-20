import os
import numpy as np
from delft.sequenceLabelling import Sequence
from delft.utilities.Tokenizer import tokenizeAndFilter
from delft.utilities.Embeddings import Embeddings,test
from delft.utilities.Utilities import stats
from delft.sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll, load_data_and_labels_lemonde, load_data_and_labels_ontonotes
from sklearn.model_selection import train_test_split
import keras.backend as K
import argparse
import time

# train a model with all available CoNLL 2003 data 
def train(embedding_name, dataset_type='conll2003', lang='en', architecture='BidLSTM_CRF', use_ELMo=False, use_BERT=False, data_path=None): 

    max_sequence_length = 300

    if (architecture == "BidLSTM_CNN_CRF"):
        word_lstm_units = 200
        recurrent_dropout=0.5
    else:
        word_lstm_units = 100
        recurrent_dropout=0.5

    if use_ELMo:
        batch_size = 100
    elif architecture.lower().find("bert") != -1:
        batch_size = 32
        max_sequence_length = 150
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
        elif use_BERT:
            model_name += '-with_BERT'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=60, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embedding_name,
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        use_BERT=use_BERT)
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
        elif use_BERT:
            model_name += '-with_BERT'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=80, 
                        recurrent_dropout=0.20,
                        embeddings_name=embedding_name, 
                        early_stop=True, 
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        use_BERT=use_BERT)
    elif (lang == 'fr'):
        print('Loading data...')
        dataset_type = 'lemonde'
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
        elif use_BERT:
            model_name += '-with_BERT'
        model_name += '-' + architecture

        model = Sequence(model_name, 
                        max_epoch=60, 
                        recurrent_dropout=recurrent_dropout,
                        embeddings_name=embedding_name, 
                        model_type=architecture,
                        word_lstm_units=word_lstm_units,
                        batch_size=batch_size,
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        use_BERT=use_BERT)
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
                use_BERT=False, 
                data_path=None): 

    max_sequence_length = 300
    if (architecture == "BidLSTM_CNN_CRF"):
        word_lstm_units = 200
        max_epoch = 30
        recurrent_dropout=0.5
    else:        
        word_lstm_units = 100
        max_epoch = 25
        recurrent_dropout=0.5

    if use_ELMo or use_BERT:
        batch_size = 120
    elif architecture.lower().find("bert") != -1:
        batch_size = 32
        max_sequence_length = 150
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
        elif use_BERT:
            model_name += '-with_BERT'
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)

    elif (dataset_type == 'ontonotes-all') and (lang == 'en'):
        print('Loading Ontonotes 5.0 XML data...')
        x_all, y_all = load_data_and_labels_ontonotes(data_path)
        x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-ontonotes'
        if use_ELMo:
            model_name += '-with_ELMo'
        elif use_BERT:
            model_name += '-with_BERT'
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
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        use_BERT=use_BERT)

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.train')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.dev')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        elif use_BERT:
            model_name += '-with_BERT'
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)

    elif (lang == 'fr') and (dataset_type == 'ftb' or dataset_type is None):
        print('Loading data...')
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
            # custom batch size for French ELMo
            batch_size = 20
        elif use_BERT:
            # need to find a French BERT :/
            model_name += '-with_BERT'
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
                        max_sequence_length=max_sequence_length,
                        use_ELMo=use_ELMo,
                        use_BERT=use_BERT)
    elif (lang == 'fr') and (dataset_type == 'ftb_force_split'):
        print('Loading data...')
        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/leMonde/ftb6_train.conll')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/leMonde/ftb6_dev.conll')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/leMonde/ftb6_test.conll')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde-force-split'
        if use_ELMo:
            model_name += '-with_ELMo'
            # custom batch size for French ELMo
            batch_size = 20
        elif use_BERT:
            # need to find a French BERT :/
            model_name += '-with_BERT'
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)
    elif (lang == 'fr') and (dataset_type == 'ftb_force_split_xml'):
        print('Loading data...')
        x_train, y_train = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.train.xml')
        x_valid, y_valid = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.dev.xml')
        x_eval, y_eval = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.test.xml')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-fr-lemonde-force-split-xml'
        if use_ELMo:
            model_name += '-with_ELMo'
            # custom batch size for French ELMo
            batch_size = 20
        elif use_BERT:
            # need to find a French BERT :/
            model_name += '-with_BERT'
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)
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
                            max_sequence_length=max_sequence_length,
                            use_ELMo=use_ELMo,
                            use_BERT=use_BERT)
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
         use_BERT=False, 
         data_path=None): 

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL-2003 NER data...')
        x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        elif use_BERT:
            model_name += '-with_BERT'
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
        elif use_BERT:
            model_name += '-with_BERT'
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
             use_BERT=False, 
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
        elif use_BERT:
            model_name += '-with_BERT'
        model_name += '-' + architecture
        model = Sequence(model_name)
        model.load()

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        elif use_BERT:
            model_name += '-with_BERT'
        model_name += '-' + architecture
        model = Sequence(model_name)
        model.load()

    elif (lang == 'fr'):
        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
        elif use_BERT:
            model_name += '-with_BERT'
        model_name += '-' + architecture
        model = Sequence(model_name)
        model.load()
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return 

    start_time = time.time()

    model.tag_file(file_in=file_in, output_format=output_format, file_out=file_out)
    runtime = round(time.time() - start_time, 3)

    print("runtime: %s seconds " % (runtime))


if __name__ == "__main__":

    architectures = ['BidLSTM_CRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 
                     'bert-base-en', 'bert-large-en', 'scibert', 'biobert']

    parser = argparse.ArgumentParser(
        description = "Neural Named Entity Recognizers")

    parser.add_argument("action", help="one of [train, train_eval, eval, tag]")
    parser.add_argument("--fold-count", type=int, default=1, help="number of folds or re-runs to be used when training")
    parser.add_argument("--lang", default='en', help="language of the model as ISO 639-1 code")
    parser.add_argument("--dataset-type",default='conll2003', help="dataset to be used for training the model")
    parser.add_argument("--train-with-validation-set", action="store_true", help="Use the validation set for training together with the training set")
    parser.add_argument("--architecture",default='BidLSTM_CRF', help="type of model architecture to be used, one of "+str(architectures))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--use-BERT", action="store_true", help="Use BERT extracted features (embeddings)") 
    parser.add_argument("--data-path", default=None, help="path to the corpus of documents for training (only use currently with Ontonotes corpus in orginal XML format)") 
    parser.add_argument("--file-in", default=None, help="path to a text file to annotate") 
    parser.add_argument("--file-out", default=None, help="path for outputting the resulting JSON NER anotations") 
    parser.add_argument(
        "--embedding", default=None,
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )

    args = parser.parse_args()

    action = args.action    
    if action not in ('train', 'tag', 'eval', 'train_eval'):
        print('action not specifed, must be one of [train, train_eval, eval, tag]')
    lang = args.lang
    dataset_type = args.dataset_type
    train_with_validation_set = args.train_with_validation_set
    use_ELMo = args.use_ELMo
    use_BERT = args.use_BERT
    architecture = args.architecture
    if architecture not in architectures and architecture.lower().find("bert") == -1:
        print('unknown model architecture, must be one of', architectures)
    data_path = args.data_path
    file_in = args.file_in
    file_out = args.file_out

    # name of embeddings refers to the file embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    # below we set the default embeddings value
    if args.embedding is None:
        embeddings_name = 'glove-840B'
        if lang == 'en':
            if dataset_type == 'conll2012':
                embeddings_name = 'fasttext-crawl'
        elif lang == 'fr':
            embeddings_name = 'wiki.fr'
    else:
        embeddings_name = args.embedding

    if action == 'train':
        train(embeddings_name, 
            dataset_type, 
            lang, 
            architecture=architecture, 
            use_ELMo=use_ELMo,
            use_BERT=use_BERT,
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
            use_BERT=use_BERT,
            data_path=data_path)

    if action == 'eval':
        eval(dataset_type, lang, architecture=architecture, use_ELMo=use_ELMo, use_BERT=use_BERT)

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
                            use_BERT=use_BERT,
                            file_in=file_in, 
                            file_out=file_out)
            """if result is not None:
                if file_out is None:
                    print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
            """

    try:
        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()
    except:
        # TF could complain in some case
        print("\nLeaving TensorFlow...")
