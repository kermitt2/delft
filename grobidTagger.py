# -*- coding: utf-8 -*-

import argparse
import json
import time

from sklearn.model_selection import train_test_split

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.models import *
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file
from delft.sequenceLabelling.reader import load_data_crf_string

import keras.backend as K

from delft.utilities.misc import parse_number_ranges

MODEL_LIST = ['affiliation-address', 'citation', 'date', 'header', 'name-citation', 'name-header', 'software']

# train a GROBID model with all available data
def train(model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False, input_path=None, output_path=None,
          features_indices=None, max_sequence_length=-1, batch_size=-1):

    print('Loading data...')
    if input_path is None:
        x_all, y_all, f_all = load_data_and_labels_crf_file('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')
    else:
        x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)

    print(len(x_all), 'total sequences')

    x_train, x_valid, y_train, y_valid, f_train, f_valid = train_test_split(x_all, y_all, f_all, test_size=0.1, shuffle=True)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    batch_size, max_sequence_length, model_name = configure(model, architecture, output_path, use_ELMo, max_sequence_length, batch_size)

    model = Sequence(model_name,
                     max_epoch=100,
                     recurrent_dropout=0.50,
                     embeddings_name=embeddings_name,
                     model_type=architecture,
                     use_ELMo=use_ELMo,
                     batch_size=batch_size,
                     max_sequence_length=max_sequence_length,
                     features_indices=features_indices)

    start_time = time.time()
    model.train(x_train, y_train, f_train, x_valid, y_valid, f_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    if (output_path):
        model.save(output_path)
    else:
        model.save()


# split data, train a GROBID model and evaluate it
def train_eval(model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False,
               input_path=None, output_path=None, fold_count=1,
               features_indices=None, max_sequence_length=-1, batch_size=-1):
    print('Loading data...')
    if input_path is None:
        x_all, y_all, f_all = load_data_and_labels_crf_file('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')
    else:
        x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)

    x_train_all, x_eval, y_train_all, y_eval, f_train_all, f_eval = train_test_split(x_all, y_all, f_all, test_size=0.1, shuffle=True)
    x_train, x_valid, y_train, y_valid, f_train, f_valid = train_test_split(x_train_all, y_train_all, f_train_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_eval), 'evaluation sequences')

    batch_size, max_sequence_length, model_name = configure(model, architecture, output_path, use_ELMo, max_sequence_length, batch_size)

    model = Sequence(model_name,
                    max_epoch=100,
                    recurrent_dropout=0.50,
                    embeddings_name=embeddings_name,
                    model_type=architecture,
                    use_ELMo=use_ELMo,
                    max_sequence_length=max_sequence_length,
                    batch_size=batch_size,
                    fold_number=fold_count,
                    features_indices=features_indices)

    start_time = time.time()

    if fold_count == 1:
        model.train(x_train, y_train, f_train=f_train, x_valid=x_valid, y_valid=y_valid, f_valid=f_valid)
    else:
        model.train_nfold(x_train, y_train, f_train=f_train, x_valid=x_valid, y_valid=y_valid, f_valid=f_valid, fold_number=fold_count)

    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % runtime)

    # evaluation
    print("\nEvaluation:")
    model.eval(x_eval, y_eval, features=f_eval)

    # saving the model
    if (output_path):
        model.save(output_path)
    else:
        model.save()


def configure(model, architecture, output_path=None, use_ELMo=False, max_sequence_length=-1, batch_size=-1):
    '''
    Set up the default parameters based on the model type.
    '''
    if output_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if max_sequence_length == -1:
        max_sequence_length = 3000

    if architecture.lower().find("bert") != -1:
        if batch_size == -1:
            batch_size = 6
        if max_sequence_length == -1 or max_sequence_length > 512:
            # 512 is the largest sequence for BERT input
            max_sequence_length = 512

    if model == "software":
        if use_ELMo:
            if batch_size == -1:
                batch_size = 7
        elif batch_size == -1:
            # class are more unbalanced, so we need to extend the batch size
            batch_size = 50
        if max_sequence_length == -1:
            max_sequence_length = 1500

    model_name += '-' + architecture;

    if use_ELMo:
        model_name += '-with_ELMo'
    
    '''
        if model_name.startswith('software') or model_name.startswith('grobid-software'):
            if batch_size == -1:
                batch_size = 7
    '''

    if batch_size == -1:
        batch_size = 20

    return batch_size, max_sequence_length, model_name


# split data, train a GROBID model and evaluate it
def eval_(model, use_ELMo=False, input_path=None, architecture='BidLSTM_CRF'):
    print('Loading data...')
    if input_path is None:
        # it should never be the case
        print("A Grobid evaluation data file must be specified for evaluating a grobid model for the eval action, use parameter --input ")
        return
    else:
        x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)

    print(len(x_all), 'evaluation sequences')

    model_name = 'grobid-' + model
    model_name += '-'+architecture

    if use_ELMo and not 'bert' in model.lower():
        model_name += '-with_ELMo'

    start_time = time.time()

    # load the model
    model = Sequence(model_name)
    model.load()

    # evaluation
    print("\nEvaluation:")
    model.eval(x_all, y_all, features=f_all)

    runtime = round(time.time() - start_time, 3)
    print("Evaluation runtime: %s seconds " % (runtime))


# annotate a list of texts, this is relevant only of models taking only text as input 
# (so not text with layout information) 
def annotate_text(texts, model, output_format, use_ELMo=False, architecture='BidLSTM_CRF', features=None):
    annotations = []

    # load model
    model_name = 'grobid-'+model
    model_name += '-'+architecture

    if use_ELMo and not 'bert' in model.lower():
        model_name += '-with_ELMo'

    model = Sequence(model_name)
    model.load()

    start_time = time.time()

    annotations = model.tag(texts, output_format, features=features)
    runtime = round(time.time() - start_time, 3)

    if output_format is 'json':
        annotations["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return annotations

class Tasks:
    TRAIN = 'train'
    TRAIN_EVAL = 'train_eval'
    EVAL = 'eval'
    TAG = 'tag'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Trainer for GROBID models")

    actions = [Tasks.TRAIN, Tasks.TRAIN_EVAL, Tasks.EVAL, Tasks.TAG]
    architectures = [BidLSTM_CRF.name, BidLSTM_CNN.name, BidLSTM_CNN_CRF.name, BidGRU_CRF.name, BidLSTM_CRF_CASING.name,
                     BidLSTM_CRF_FEATURES.name,
                     'bert-base-en', 'bert-large-en', 'scibert', 'biobert']

    parser.add_argument("model", help="Name of the model.")
    parser.add_argument("action", choices=actions)
    parser.add_argument("--fold-count", type=int, default=1, help="Number of fold to use when evaluating with n-fold "
                                                                  "cross validation.")
    parser.add_argument("--architecture", default='BidLSTM_CRF', choices=architectures,
                        help="Type of model architecture to be used, one of "+str(architectures))
    parser.add_argument("--embedding", default='glove-840B',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )
    parser.add_argument("--use-ELMo", action="store_true", default=False, help="Use ELMo contextual embeddings.")
    parser.add_argument("--output", help="Directory where to save a trained model.")
    parser.add_argument("--input", help="Grobid data file to be used for training (train action), for training and "
                                        "evaluation (train_eval action) or just for evaluation (eval action).")
    parser.add_argument(
        "--feature-indices",
        type=parse_number_ranges,
        help="The feature indices to use. e.g. 7:10. If blank, all of the features will be used."
    )

    parser.add_argument("--max-sequence-length", type=int, default=-1, help="max-sequence-length parameter to be used.")
    parser.add_argument("--batch-size", type=int, default=-1, help="batch-size parameter to be used.")

    args = parser.parse_args()

    model = args.model
    action = args.action
    use_ELMo = args.use_ELMo
    architecture = args.architecture
    output = args.output
    input_path = args.input
    embeddings_name = args.embedding
    feature_indices = args.feature_indices
    max_sequence_length = args.max_sequence_length
    batch_size = args.batch_size

    if action == Tasks.TRAIN:
        train(model, 
            embeddings_name, 
            architecture=architecture, 
            use_ELMo=use_ELMo, 
            input_path=input_path, 
            output_path=output,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size)

    if action == Tasks.EVAL:
        if args.fold_count is not None:
            print("The argument fold-count argument will be ignored. For n-fold cross-validation, please use "
                  "it in combination with " + str(Tasks.TRAIN_EVAL))
        if input_path is None:
            raise ValueError("A Grobid evaluation data file must be specified to evaluate a grobid model")
        eval_(model, use_ELMo=use_ELMo, input_path=input_path, architecture=architecture)

    if action == Tasks.TRAIN_EVAL:
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(model, 
                embeddings_name, 
                architecture=architecture, 
                use_ELMo=use_ELMo, 
                input_path=input_path, 
                output_path=output, 
                fold_count=args.fold_count,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size)

    if action == Tasks.TAG:
        someTexts = []

        if model == 'date':
            someTexts.append("January 2006")
            someTexts.append("March the 27th, 2001")
            someTexts.append('2018')
        elif model == 'citation':
            someTexts.append("N. Al-Dhahir and J. Cioffi, \“On the uniform ADC bit precision and clip level computation for a Gaussian signal,\” IEEE Trans. Signal Processing, pp. 434–438, Feb. 1996.")
            someTexts.append("T. Steinherz, E. Rivlin, N. Intrator, Off-line cursive script word recognition—a survey, Int. J. Doc. Anal. Recognition 2(3) (1999) 1–33.")
        elif model == 'name-citation':
            someTexts.append("L. Romary and L. Foppiano")
            someTexts.append("Maniscalco, S., Francica, F., Zaffino, R.L.")
        elif model == 'name-header':
            someTexts.append("He-Jin Wu 1 · Zhao Jin 2 · Ai-Dong Zhu 1")
            someTexts.append("Irène Charon ⋆ and Olivier Hudry")
        elif model == 'software':
            someTexts.append("Wilcoxon signed-ranks tests were performed to calculate statistical significance of comparisons between  alignment programs, which include ProbCons (version 1.10) (23), MAFFT (version 5.667) (11) with several options, MUSCLE (version 3.52) (10) and ClustalW (version 1.83) (7).")
            someTexts.append("The statistical analysis was performed using IBM SPSS Statistics v. 20 (SPSS Inc, 2003, Chicago, USA).")

        if architecture.find("FEATURE") == -1:
            result = annotate_text(someTexts, model, "json", use_ELMo=use_ELMo, architecture=architecture)
            print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
        else:
            print("The model " + architecture + " cannot be used without supplying features as input and it's disabled. "
                                                "Please supply an architecture without features. ")

        # test with the use input file with features
        if model == 'software':
            with open("tests/sequence_labelling/test_data/input-software.crf", 'r') as file:
                input_crf_string = file.read()
            x_all, f_all = load_data_crf_string(input_crf_string)
            result = annotate_text(x_all, model, None, use_ELMo=use_ELMo, architecture=architecture, features=f_all)
            print(result)
        
    try:
        # see https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()
    except:
        # TF could complain in some case
        print("\nLeaving TensorFlow...")

