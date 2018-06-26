import os
import json
import numpy as np
import sequenceLabelling
from utilities.Tokenizer import tokenizeAndFilter
from utilities.Embeddings import Embeddings
from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll, load_data_and_labels_lemonde, load_data_and_labels_ontonotes
from sklearn.model_selection import train_test_split
import keras.backend as K
import argparse
import time

# produce some statistics
def stats(x_train=None, y_train=None, x_valid=None, y_valid=None, x_eval=None, y_eval=None):
    if x_train is not None:
        print(len(x_train), 'train sequences')
        nb_tokens = 0
        for sentence in x_train:
            nb_tokens += len(sentence)
        print("\t","nb. tokens", nb_tokens)
    if y_train is not None:
        nb_entities = 0
        for labels in y_train:
            for label in labels:
                if label != 'O':
                    nb_entities += 1
        print("\t","with nb. entities", nb_entities)
    if x_valid is not None:
        print(len(x_valid), 'validation sequences')
        nb_tokens = 0
        for sentence in x_valid:
            nb_tokens += len(sentence)
        print("\t","nb. tokens", nb_tokens)
    if y_valid is not None:
        nb_entities = 0
        for labels in y_valid:
            for label in labels:
                if label != 'O':
                    nb_entities += 1
        print("\t","with nb. entities", nb_entities)
    if x_eval is not None:
        print(len(x_eval), 'evaluation sequences')
        nb_tokens = 0
        for sentence in x_eval:
            nb_tokens += len(sentence)
        print("\t","nb. tokens", nb_tokens)
    if y_eval is not None:
        nb_entities = 0
        for labels in y_eval:
            for label in labels:
                if label != 'O':
                    nb_entities += 1
        print("\t","with nb. entities", nb_entities)


# train a model with all available CoNLL 2003 data 
def train(embedding_name, dataset_type='conll2003', lang='en', architecture='BidLSTM_CRF', use_ELMo=False, data_path=None): 
    
    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading data...')
        x_train1, y_train1 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
        x_train2, y_train2 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')

        # we concatenate train and valid sets
        x_train = np.concatenate((x_train1, x_train2), axis=0)
        y_train = np.concatenate((y_train1, y_train2), axis=0)

        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'

        model = sequenceLabelling.Sequence(model_name, 
                                        max_epoch=60, 
                                        recurrent_dropout=0.5,
                                        embeddings_name=embedding_name,
                                        model_type=architecture,
                                        use_ELMo=use_ELMo)
    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_train1, y_train1 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.train')
        x_train2, y_train2 = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.dev')

        # we concatenate train and valid sets
        x_train = np.concatenate((x_train1, x_train2), axis=0)
        y_train = np.concatenate((y_train1, y_train2), axis=0)

        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_train, y_train, x_valid, y_valid)

        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'

        model = sequenceLabelling.Sequence(model_name, 
                                        max_epoch=60, 
                                        recurrent_dropout=0.20,
                                        embeddings_name=embedding_name, 
                                        early_stop=True, 
                                        fold_number=fold_count,
                                        model_type=architecture,
                                        use_ELMo=use_ELMo)
    elif (lang == 'fr'):
        print('Loading data...')
        dataset_type = 'lemonde'
        x_all, y_all = load_data_and_labels_lemonde('data/sequenceLabelling/leMonde/ftb6_ALL.EN.docs.relinked.xml')
        x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)
        stats(x_train, y_train, x_valid, y_valid)

        model = sequenceLabelling.Sequence('ner-fr-lemonde', 
                                        max_epoch=60, 
                                        embeddings_name=embedding_name, 
                                        model_type=architecture,
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
    root = os.path.join(os.path.dirname(__file__), '../data/sequence/')

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL 2003 data...')
        x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
        x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_train, y_train, x_valid, y_valid, x_eval, y_eval)

        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'

        if not train_with_validation_set: 
            # restrict training on train set, use validation set for early stop, as in most papers
            model = sequenceLabelling.Sequence(model_name, 
                                            max_epoch=60, 
                                            recurrent_dropout=0.5,
                                            embeddings_name=embedding_name, 
                                            early_stop=True, 
                                            fold_number=fold_count,
                                            model_type=architecture,
                                            use_ELMo=use_ELMo)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results (~ +0.5 f1 score with CoNLL-2003)
            model = sequenceLabelling.Sequence(model_name, 
                                            max_epoch=25, 
                                            recurrent_dropout=0.5,
                                            embeddings_name=embedding_name, 
                                            early_stop=False, 
                                            fold_number=fold_count,
                                            model_type=architecture,
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

        model = sequenceLabelling.Sequence(model_name, 
                                        max_epoch=60, 
                                        recurrent_dropout=0.20,
                                        embeddings_name=embedding_name, 
                                        early_stop=True, 
                                        fold_number=fold_count,
                                        model_type=architecture,
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

        if not train_with_validation_set: 
            model = sequenceLabelling.Sequence(model_name, 
                                            max_epoch=60, 
                                            recurrent_dropout=0.20,
                                            embeddings_name=embedding_name, 
                                            early_stop=True, 
                                            fold_number=fold_count,
                                            model_type=architecture,
                                            use_ELMo=use_ELMo)
        else:
            # also use validation set to train (no early stop, hyperparmeters must be set preliminarly), 
            # as (Chui & Nochols, 2016) and (Peters and al., 2017)
            # this leads obviously to much higher results 
            model = sequenceLabelling.Sequence(model_name, 
                                            max_epoch=30, 
                                            recurrent_dropout=0.20,
                                            embeddings_name=embedding_name, 
                                            early_stop=False, 
                                            fold_number=fold_count,
                                            model_type=architecture,
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

        model = sequenceLabelling.Sequence(model_name, 
                                        max_epoch=60, 
                                        recurrent_dropout=0.50,
                                        embeddings_name=embedding_name, 
                                        early_stop=True, 
                                        fold_number=fold_count,
                                        model_type=architecture,
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
def eval(dataset_type='conll2003', lang='en', use_ELMo=False, data_path=None): 
    root = os.path.join(os.path.dirname(__file__), '../data/sequence/')

    if (dataset_type == 'conll2003') and (lang == 'en'):
        print('Loading CoNLL-2003 NER data...')
        x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        model = sequenceLabelling.Sequence(model_name)
        model.load()

    elif (dataset_type == 'conll2012') and (lang == 'en'):
        print('Loading Ontonotes 5.0 CoNLL-2012 NER data...')

        x_eval, y_eval = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2012-NER/eng.test')
        stats(x_eval=x_test, y_eval=y_test)

        # load model
        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        model = sequenceLabelling.Sequence(model_name)
        model.load()

    else:
        print("dataset/language combination is not supported for fixed eval:", dataset_type, lang)
        return

    start_time = time.time()

    print("\nEvaluation on test set:")
    model.eval(x_test, y_test)
    runtime = round(time.time() - start_time, 3)
    
    print("runtime: %s seconds " % (runtime))


# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts, output_format, dataset_type='conll2003', lang='en', use_ELMo=False, data_path=None):
    annotations = []

    if (dataset_type == 'conll2003') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2003'
        if use_ELMo:
            model_name += '-with_ELMo'
        model = sequenceLabelling.Sequence(model_name)
        model.load()
    
    elif (dataset_type == 'conll2012') and (lang == 'en'):
        # load model
        model_name = 'ner-en-conll2012'
        if use_ELMo:
            model_name += '-with_ELMo'
        model = sequenceLabelling.Sequence(model_name)
        model.load()

    elif (lang == 'fr'):
        model_name = 'ner-fr-lemonde'
        if use_ELMo:
            model_name += '-with_ELMo'
        model = sequenceLabelling.Sequence(model_name)
        model.load()
    else:
        print("dataset/language combination is not supported:", dataset_type, lang)
        return 

    start_time = time.time()

    annotations = model.tag(texts, output_format)
    runtime = round(time.time() - start_time, 3)

    if output_format is 'json':
        annotations["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return annotations


if __name__ == "__main__":
    #test()
    parser = argparse.ArgumentParser(
        description = "Named Entity Recognizer")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1, help="number of folds or re-runs to be used when training")
    parser.add_argument("--lang", default='en', help="language of the model as ISO 639-1 code")
    parser.add_argument("--dataset-type",default='conll2003', help="dataset to be used for training the model")
    parser.add_argument("--train-with-validation-set", action="store_true", help="Use the validation set for training together with the training set")
    parser.add_argument("--architecture",default='BidLSTM_CRF', help="type of model architecture to be used (BidLSTM_CRF, BidLSTM_CNN_CRF or BidLSTM_CNN_CRF)")
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--data-path", default=None, help="path to the corpus of documents to process") 

    args = parser.parse_args()
    
    action = args.action    
    if (action != 'train') and (action != 'tag') and (action != 'eval') and (action != 'train_eval'):
        print('action not specifed, must be one of [train,train_eval,eval,tag]')
    lang = args.lang
    dataset_type = args.dataset_type
    train_with_validation_set = args.train_with_validation_set
    use_ELMo = args.use_ELMo
    architecture = args.architecture
    if (architecture != 'BidLSTM_CRF') and (architecture != 'BidLSTM_CNN_CRF') and (architecture != 'BidLSTM_CNN_CRF'):
        print('unknown model architecture, must be one of [BidLSTM_CRF,BidLSTM_CNN_CRF,BidLSTM_CNN_CRF]')
    data_path = args.data_path

    # change bellow for the desired pre-trained word embeddings using their descriptions in the file 
    # embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'), 
    # and that the path in the registry to the embedding file is correct on your system
    if lang == 'en':
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
        eval(dataset_type, lang, use_ELMo=use_ELMo)

    if action == 'tag':
        if (lang == 'en'):
            someTexts = ['The University of California has found that 40 percent of its students suffer food insecurity. At four state universities in Illinois, that number is 35 percent.',
                         'President Obama is not speaking anymore from the White House.']
        elif (lang == 'fr'):
            someTexts = ['Elargie à l’Italie et au Canada, puis à la Russie en 1998, elle traversa une première crise en 2014 après l’annexion de la Crimée par Moscou.',
                         'Or l’Allemagne pourrait préférer la retenue, de peur que Donald Trump ne surtaxe prochainement les automobiles étrangères.']
        else:
            print("Language not supported:", lang)
            someTexts = []

        result = annotate(someTexts, "json", dataset_type, lang, use_ELMo=use_ELMo, data_path=data_path)
        if result is not None:
            print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
