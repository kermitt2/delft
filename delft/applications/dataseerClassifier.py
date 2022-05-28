import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_dataseer_corpus_csv
from delft.textClassification.reader import vectorize as vectorizer
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import time
from delft.textClassification.models import architectures
import numpy as np

"""
    Classifier for deciding if a sentence introduce a dataset or not, and prediction of the 
    dataset type. 
"""

def configure(architecture):
    # default RNN model parameters
    batch_size = 200
    maxlen = 300
    patience = 5
    early_stop = True
    max_epoch = 50

    # default transformer model parameters
    if architecture == "bert":
        batch_size = 16
        early_stop = False
        max_epoch = 5
        maxlen = 300

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(embeddings_name, fold_count, architecture="gru", transformer=None, cascaded=False): 
    print('loading binary dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    model_name = 'dataseer-binary_'+architecture
    class_weights = None

    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)
    
    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()

    print('loading reuse dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-reuse.csv")

    model_name = 'dataseer-reuse_' + architecture
    class_weights = {0: 1.5, 1: 1.}

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count,
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)
    
    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()

    print('loading first-level dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-multilevel.csv")

    model_name = 'dataseer-first_' + architecture

    class_weights = None

    model = Classifier(model_name, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
    use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
         class_weights=class_weights, transformer_name=transformer)

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()
    
    '''
    print('training second-level dataset subtype corpus...')
    xtr, y1, y2, _, list_classes, list_subclasses, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-multilevel.csv")
    # aggregate by class, we will have one training set per class
    datatypes_y = {}
    datatypes_xtr = {}
    datatypes_list_subclasses = {}
    for i in range(0,len(xtr)):
        datatype = y1[i]
        datasubtype = y2[i]
        if datatype in datatypes_y:
            datatypes_y[datatype].append(datasubtype)
            datatypes_xtr[datatype].append(xtr[i])
            if not y2[i] in datatypes_list_subclasses[datatype]:
                datatypes_list_subclasses[datatype].append(y2[i])
        else:
            datatypes_y[datatype] = []
            datatypes_y[datatype].append(datasubtype)
            datatypes_xtr[datatype] = []
            datatypes_xtr[datatype].append(xtr[i])
            datatypes_list_subclasses[datatype] = []
            datatypes_list_subclasses[datatype].append(y2[i])

    for the_class in list_classes:
        print('training', the_class)

        model_name = 'dataseer-' + the_class + "_" + architecture

        model = Classifier(model_name, architecture=architecture, list_classes=datatypes_list_subclasses[the_class], max_epoch=max_epoch, 
            fold_number=fold_count, patience=patience, use_roc_auc=True, embeddings_name=embeddings_name, 
            batch_size=batch_size, class_weights=class_weights, early_stop=early_stop, transformer_name=transformer)

        if fold_count == 1:
            model.train(datatypes_xtr[the_class], datatypes_y[the_class])
        else:
            model.train_nfold(datatypes_xtr[the_class], datatypes_y[the_class])
        # saving the model
        model.save()
    '''

def train_and_eval(embeddings_name=None, fold_count=1, architecture="gru", transformer=None, cascaded=False): 
    if cascaded:
        return train_eval_cascaded(embeddings_name, fold_count, architecture=architecture, transformer=transformer)

    # classifier for deciding if we have a dataset or not in a sentence
    train_and_eval_binary(embeddings_name, fold_count, architecture=architecture, transformer=transformer)

    # classifier for deciding if the introduced dataset is a reuse of an existing one or is a new dataset
    train_and_eval_reuse(embeddings_name, fold_count, architecture=architecture, transformer=transformer)

    # classifier for first level data type hierarchy
    train_and_eval_primary(embeddings_name, fold_count, architecture=architecture, transformer=transformer)

    # classifier for second level data type hierarchy (subtypes)
    #train_and_eval_secondary(embeddings_name, fold_count, architecture=architecture, transformer=transformer)

def train_and_eval_binary(embeddings_name, fold_count, architecture="gru", transformer=None): 
    print('loading dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    # distinct values of classes
    print(list_classes)
    print(len(list_classes), "classes")

    print(len(xtr), "texts")
    print(len(y), "classes")

    class_weights = None

    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier('dataseer-binary_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count,  
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

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

def train_and_eval_reuse(embeddings_name, fold_count, architecture="gru", transformer=None): 
    print('loading dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-reuse.csv")

    # distinct values of classes
    print(list_classes)
    print(len(list_classes), "classes")

    print(len(xtr), "texts")
    print(len(y), "classes")

    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    class_weights = {0: 1.5, 1: 1.}

    model = Classifier('dataseer-reuse_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

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
    
def train_and_eval_primary(embeddings_name, fold_count, architecture="gru", transformer=None): 
    print('loading dataset type corpus...')
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-multilevel.csv")

    # distinct values of classes
    print(list_classes)
    print(len(list_classes), "classes")

    print(len(xtr), "texts")
    print(len(y), "classes")

    class_weights = None
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier('dataseer-first_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

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

def train_and_eval_secondary(embeddings_name, fold_count, architecture="gru", transformer=None): 
    print('training second-level dataset subtype corpus...')
    xtr, y1, y2, _, list_classes, list_subclasses, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-multilevel.csv")
    # aggregate by class, we will have one training set per class

    print(list_classes)
    print(list_subclasses)
    print(len(list_classes), "classes")
    print(len(list_subclasses), "sub-classes")

    class_weights = None
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    datatypes_y = {}
    datatypes_xtr = {}
    datatypes_list_subclasses = {}
    for i in range(0,len(xtr)):
        #print(np.where(y2[i] == 1))
        ind1= np.where(y1[i] == 1)[0][0]
        ind2 = np.where(y2[i] == 1)[0][0]
        #print(ind2)
        datatype = list_classes[ind1]
        datasubtype = list_subclasses[ind2]
        #print(str(xtr[i]), datatype, datasubtype)
        if datatype in datatypes_y:
            datatypes_y[datatype].append(datasubtype)
            datatypes_xtr[datatype].append(xtr[i])
            if not datasubtype in datatypes_list_subclasses[datatype]:
                datatypes_list_subclasses[datatype].append(datasubtype)
        else:
            datatypes_y[datatype] = []
            datatypes_y[datatype].append(datasubtype)
            datatypes_xtr[datatype] = []
            datatypes_xtr[datatype].append(xtr[i])
            datatypes_list_subclasses[datatype] = []
            datatypes_list_subclasses[datatype].append(datasubtype)

    print(datatypes_list_subclasses)

    for the_class in list_classes:
        print('\ntraining', the_class)
        if not the_class in datatypes_list_subclasses:
            print('no subclass for', the_class)
            continue

        if len(datatypes_list_subclasses[the_class]) <= 1:
            print('only one subclass for', the_class)
            continue

        if len(datatypes_list_subclasses[the_class]) == 2 and 'nan' in datatypes_list_subclasses[the_class]:
            continue     

        if the_class == 'Protein Data':
            continue

        print('subtypes to be classified:', datatypes_list_subclasses[the_class])

        model_name = 'dataseer-' + the_class + "_" + architecture

        model = Classifier(model_name, architecture=architecture, list_classes=datatypes_list_subclasses[the_class], max_epoch=max_epoch, 
            fold_number=fold_count, use_roc_auc=True, embeddings_name=embeddings_name, 
            batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop, 
            class_weights=class_weights, transformer_name=transformer)

        # we need to vectorize the y according to the actual list of classes
        local_y = []
        for the_y in datatypes_y[the_class]:
            the_ind = datatypes_list_subclasses[the_class].index(the_y)
            local_y.append(vectorizer(the_ind, len(datatypes_list_subclasses[the_class])))

        # segment train and eval sets
        x_train, y_train, x_test, y_test = split_data_and_labels(np.asarray(datatypes_xtr[the_class]), np.asarray(local_y), 0.9)

        if fold_count == 1:
            model.train(x_train, y_train)
        else:
            model.train_nfold(x_train, y_train)
        model.eval(x_test, y_test)
        # saving the model
        model.save()
    
def classify(texts, output_format, architecture="gru"):
    '''
        Classify a list of texts with an existing model
    '''
    # load model
    model = Classifier('dataseer-binary_'+architecture)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)
    if output_format == 'json':
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return result

def train_eval_cascaded(embeddings_name, fold_count, architecture="gru", transformer=None):
    # general setting of parameters
    class_weights = None
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    # first binary classifier: dataset or no_dataset 
    xtr, y, _, _, list_classes, _, _ = load_dataseer_corpus_csv("data/textClassification/dataseer/all-binary.csv")

    print(list_classes)

    model_binary = Classifier('dataseer-binary_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer=transformer)

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
    xtr, y_classes, y_subclasses, y_leafclasses, list_classes, list_subclasses, list_leaf_classes = load_dataseer_corpus_csv("data/textClassification/dataseer/all-multilevel.csv")
    # ignore the no_dataset, ignore the first eval set, build first level classifier
    
    ind = list_classes.index('no_dataset')
    to_remove = vectorizer(ind, len(list_classes))

    x_train, y_train = filter_exclude_class(xtr, y_classes, to_remove)
    y_train2 = np.zeros(shape=(len(y_train), len(list_classes)-1))
    for i in range(0,len(y_train)):
        y_train2[i] = np.delete(y_train[i], ind)
    y_train = y_train2

    list_classes.remove('no_dataset')

    model_first = Classifier('dataseer-first_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=100, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
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
    _, y_classes, y_subclasses, y_leafclasses, list_classes, list_subclasses, list_leaf_classes = load_dataseer_corpus_csv("data/textClassification/dataseer/all-multilevel.csv")

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
        description = "Dataset passage identification and classification for scientific literature based on DeLFT")

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']
    pretrained_transformers_examples = [ 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased' ]

    parser.add_argument("action", help="one of [train, train_eval, classify]")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture",default='gru', help="type of model architecture to be used, one of "+str(architectures))
    parser.add_argument("--cascaded", action="store_true", help="Use models in cascade (train, eval, predict)") 
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

    args = parser.parse_args()

    if args.action not in ('train', 'train_eval', 'classify'):
        print('action not specifed, must be one of [train,train_eval,classify]')

    embeddings_name = args.embedding
    cascaded = args.cascaded
    transformer = args.transformer

    architecture = args.architecture
    if architecture not in architectures:
        print('unknown model architecture, must be one of '+str(architectures))

    if transformer == None and embeddings_name == None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if args.action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(embeddings_name=embeddings_name, fold_count=args.fold_count, architecture=architecture, transformer=transformer, cascaded=cascaded)

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        y_test = train_and_eval(embeddings_name=embeddings_name, fold_count=args.fold_count, architecture=architecture, transformer=transformer, cascaded=cascaded)    

    if args.action == 'classify':
        someTexts = ['Labeling yield and radiochemical purity was analyzed by instant thin layered chromatography (ITLC).', 
            'NOESY and ROESY spectra64,65 were collected with typically 128 scans per t1 increment, with the residual water signal removed by the WATERGATE sequence and 1 s relaxation time.', 
            'The concentrations of Cd and Pb in feathers were measured by furnace atomic absorption spectrometry (VARIAN 240Z).']
        result = classify(someTexts, "json", architecture=architecture)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))  
