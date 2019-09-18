import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_dataseer_corpus_csv
from delft.textClassification.reader import vectorize as vectorizer
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import keras.backend as K
import time
from delft.textClassification.models import modelTypes
import numpy as np

"""
    Classifier for deciding if a sentence introduce a dataset or not, and prediction of the 
    dataset type. 
"""

def train(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru", cascaded=False): 
    print('loading binary dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    class_weights = None
    batch_size = 256
    maxlen = 300
    model_name = 'dataseer-binary'
    if use_ELMo:
        batch_size = 20
        model_name += '-with_ELMo'
    elif use_BERT:
        batch_size = 50
        model_name += '-with_BERT'

    # default bert model parameters
    if architecture.lower().find("bert") != -1:
        batch_size = 32
        maxlen = 100

    model = Classifier(model_name, model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size,
        class_weights=class_weights)

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()

    print('loading first-level dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-1.csv")

    model_name = 'dataseer-first'
    if use_ELMo:
        model_name += '-with_ELMo'
    elif use_BERT:
        model_name += '-with_BERT'

    model = Classifier(model_name, model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
    use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size,
         class_weights=class_weights)

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def train_and_eval(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru", cascaded=False): 
    if cascaded:
        return train_eval_cascaded(embeddings_name, fold_count, use_ELMo, use_BERT, architecture)

    print('loading dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-1.csv")
    #xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    # distinct values of classes
    print(list_classes)
    print(len(list_classes), "classes")

    print(len(xtr), "texts")
    print(len(y), "classes")

    class_weights = None
    batch_size = 256
    maxlen = 300
    if use_ELMo:
        batch_size = 20
    elif use_BERT:
        batch_size = 50

    # default bert model parameters
    if architecture.find("bert") != -1:
        batch_size = 32
        maxlen = 100

    model = Classifier('dataseer', model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size, maxlen=maxlen,
        class_weights=class_weights)

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    print(len(x_train), "train texts")
    print(len(y_train), "train classes")

    print(len(x_test), "eval texts")
    print(len(y_test), "eval classes")

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


def classify(texts, output_format, architecture="gru", cascaded=False):
    '''
        Classify a list of texts with an existing model
    '''
    # load model
    model = Classifier('dataseer', model_type=architecture)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)
    if output_format is 'json':
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return result

def train_eval_cascaded(embeddings_name, fold_count, use_ELMo=False, use_BERT=False, architecture="gru"):
    # general setting of parameters
    class_weights = None
    batch_size = 256
    maxlen = 300
    if use_ELMo:
        batch_size = 20
    elif use_BERT:
        batch_size = 50

    # default bert model parameters
    if architecture.find("bert") != -1:
        batch_size = 32
        maxlen = 100

    # first binary classifier: dataset or no_dataset 
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    print(list_classes)

    model_binary = Classifier('dataseer-binary', model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size, maxlen=maxlen,
        class_weights=class_weights)

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    if fold_count == 1:
        model_binary.train(x_train, y_train)
    else:
        model_binary.train_nfold(x_train, y_train)
    model_binary.eval(x_test, y_test)
    
    x_test_binary = x_test
    y_test_binary = y_test

    # second, the first level datatype taxonomy for sentences classified as dataset
    xtr, y_classes, y_subclasses, y_leafclasses, list_classes, list_subclasses, list_leaf_classes = load_dataseer_corpus_csv("data/textClassification/dataseer/all-1.csv")
    # ignore the no_dataset, ignore the first eval set, build first level classifier
    
    ind = list_classes.index('no_dataset')
    to_remove = vectorizer(ind, len(list_classes))

    x_train, y_train = filter_exclude_class(xtr, y_classes, to_remove)
    y_train2 = np.zeros(shape=(len(y_train), len(list_classes)-1))
    for i in range(0,len(y_train)):
        y_train2[i] = np.delete(y_train[i], ind)
    y_train = y_train2

    list_classes.remove('no_dataset')

    model_first = Classifier('dataseer-first', model_type=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, patience=10,
        use_roc_auc=True, embeddings_name=embeddings_name, use_ELMo=use_ELMo, use_BERT=use_BERT, batch_size=batch_size, maxlen=maxlen,
        class_weights=class_weights)

    if fold_count == 1:
        model_first.train(x_train, y_train)
    else:
        model_first.train_nfold(x_train, y_train)
    model_first.eval(x_test, y_test)

    # eval by cascading
    result_binary = model_binary.predict(x_test_binary, output_format='default')
    result_first = model_first.predict(x_test, output_format='default')

    # select sequences classified as dataset
    result_intermediate = np.asarray([np.argmax(line) for line in result_binary])    
    def vectorize(index, size):
        result = np.zeros(size)
        if index < size:
            result[index] = 1
        return result
    result_binary = np.array([vectorize(xi, len(list_classes)) for xi in result_intermediate])


    


def filter_exclude_class(xtr, y_classes, the_class):
    # apply the filter
    new_xtr = []
    new_classes = []
    for i in range(0,len(y_classes)):
        if not np.array_equal(y_classes[i], the_class):
            new_xtr.append(xtr[i])
            new_classes.append(y_classes[i])

    return np.array(new_xtr), np.array(new_classes)


def build_prior_class_distribution():
    """
    Inject count from the training data to the classification taxonomy of data types
    """
    _, y_classes, y_subclasses, y_leafclasses, list_classes, list_subclasses, list_leaf_classes = load_dataseer_corpus_csv("data/textClassification/dataseer/all-1.csv")

    with open('data/textClassification/dataseer/DataTypes.json') as json_file:
        distribution = json.load(json_file)

    # init count everywhere
    for key1 in distribution:
        print(key1)
        if type(distribution[key1]) is dict:
            distribution[key1]['count'] = 0
            for key2 in distribution[key1]:
                print(key1, key2)
                if type(distribution[key1][key2]) is dict:
                    distribution[key1][key2]['count'] = 0
                    for key3 in distribution[key1][key2]:
                        print(key1, key2, key3)
                        if type(distribution[key1][key2][key3]) is dict:
                            distribution[key1][key2][key3]['count'] = 0

    # inject counts in the json
    for i in range(0, len(y_classes)):
        pos_class = np.where(y_classes[i] == 1)
        pos_subclass = np.where(y_subclasses[i] == 1)
        pos_leafclass = np.where(y_leafclasses[i] == 1)
        print(list_classes[pos_class[0][0]], list_subclasses[pos_subclass[0][0]], list_leaf_classes[pos_leafclass[0][0]])
        if list_classes[pos_class[0][0]] != "no_dataset":
            the_class = list_classes[pos_class[0][0]]
            if list_subclasses[pos_subclass[0][0]] != "nan":
                the_subclass = list_subclasses[pos_subclass[0][0]]
                if list_leaf_classes[pos_leafclass[0][0]] != "nan":
                    the_leafclass = list_leaf_classes[pos_leafclass[0][0]]
                    print(distribution[the_class][the_subclass][the_leafclass])
                    if 'count' in distribution[the_class][the_subclass][the_leafclass]:
                        distribution[the_class][the_subclass][the_leafclass]['count'] = distribution[the_class][the_subclass][the_leafclass]['count'] + 1
                else:
                    if 'count' in distribution[the_class][the_subclass]:
                        distribution[the_class][the_subclass]['count'] = distribution[the_class][the_subclass]['count'] + 1
            else:
                if 'count' in distribution[the_class]:
                    distribution[the_class]['count'] = distribution[the_class]['count'] + 1

    # save the extended json
    with open('data/textClassification/dataseer/DataTypesWithCounts.json', 'w') as outfile:
        json.dump(distribution, outfile, sort_keys=False, indent=4)

if __name__ == "__main__":
    #build_prior_class_distribution()  

    parser = argparse.ArgumentParser(
        description = "Dataset identification and classification for scientific literature")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture",default='gru', help="type of model architecture to be used, one of "+str(modelTypes))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 
    parser.add_argument("--use-BERT", action="store_true", help="Use BERT contextual embeddings") 
    parser.add_argument("--cascaded", action="store_true", help="Use models in cascade (train, eval, predict)") 
    parser.add_argument(
        "--embedding", default='word2vec',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )

    args = parser.parse_args()

    if args.action not in ('train', 'train_eval', 'classify'):
        print('action not specifed, must be one of [train,train_eval,classify]')

    embeddings_name = args.embedding
    use_ELMo = args.use_ELMo
    use_BERT = args.use_BERT
    cascaded = args.cascaded

    architecture = args.architecture
    if architecture not in modelTypes:
        print('unknown model architecture, must be one of '+str(modelTypes))

    if args.action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(embeddings_name, args.fold_count, use_ELMo=use_ELMo, use_BERT=use_BERT, architecture=architecture, cascaded=cascaded)

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        y_test = train_and_eval(embeddings_name, args.fold_count, use_ELMo=use_ELMo, use_BERT=use_BERT, architecture=architecture, cascaded=cascaded)    

    if args.action == 'classify':
        someTexts = ['Labeling yield and radiochemical purity was analyzed by instant thin layered chromatography (ITLC).', 
            'NOESY and ROESY spectra64,65 were collected with typically 128 scans per t1 increment, with the residual water signal removed by the WATERGATE sequence and 1 s relaxation time.', 
            'The concentrations of Cd and Pb in feathers were measured by furnace atomic absorption spectrometry (VARIAN 240Z).']
        result = classify(someTexts, "json", architecture=architecture, cascaded=cascaded)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))  

    # See https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()