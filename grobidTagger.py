import json
import numpy as np

from delft.sequenceLabelling.preprocess import prepare_preprocessor
from delft.utilities.Embeddings import Embeddings
import delft.sequenceLabelling
from delft.sequenceLabelling import Sequence
from delft.utilities.Tokenizer import tokenizeAndFilter
from sklearn.model_selection import train_test_split
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file
from delft.sequenceLabelling.reader import load_data_and_labels_crf_string
from delft.sequenceLabelling.reader import load_data_crf_string
import keras.backend as K
import argparse
import time

models = ['affiliation-address', 'citation', 'date', 'header', 'name-citation', 'name-header', 'software']


# train a GROBID model with all available data 
def train(model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False, input_path=None, output_path=None): 
    print('Loading data...')
    if input_path is None:
        x_all, y_all, f_all = load_data_and_labels_crf_file('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')
    else:
        x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)
    x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-'+model

    if use_ELMo:
        model_name += '-with_ELMo'

    model = Sequence(model_name, 
                    max_epoch=100, 
                    recurrent_dropout=0.50,
                    embeddings_name=embeddings_name, 
                    model_type=architecture,
                    use_ELMo=use_ELMo)

    start_time = time.time()
    model.train(x_train, y_train, x_valid, y_valid)
    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # saving the model
    if (output_path):
        model.save(output_path)
    else:
        model.save()

# split data, train a GROBID model and evaluate it 
def train_eval(model, embeddings_name, architecture='BidLSTM_CRF', use_ELMo=False, input_path=None, output_path=None, fold_count=1): 
    print('Loading data...')
    if input_path is None:
        x_all, y_all, f_all = load_data_and_labels_crf_file('data/sequenceLabelling/grobid/'+model+'/'+model+'-060518.train')
    else:
        x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)

    x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_eval), 'evaluation sequences')

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-'+model

    batch_size = 20
    max_sequence_length = 3000

    if model == "software":
        # class are more unbalanced, so we need to extend the batch size  
        batch_size = 50
        max_sequence_length = 1500
    
    if use_ELMo:
        model_name += '-with_ELMo'
        if model_name == 'software-with_ELMo' or model_name == 'grobid-software-with_ELMo':
            batch_size = 3

    model = Sequence(model_name, 
                    max_epoch=100, 
                    recurrent_dropout=0.50,
                    embeddings_name=embeddings_name, 
                    model_type=architecture,
                    use_ELMo=use_ELMo,
                    max_sequence_length=max_sequence_length,
                    batch_size=batch_size,
                    fold_number=fold_count)

    start_time = time.time()

    if fold_count == 1:
        model.train(x_train, y_train, x_valid, y_valid)
    else:
        model.train_nfold(x_train, y_train, x_valid, y_valid, fold_number=fold_count)

    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % (runtime))

    # evaluation
    print("\nEvaluation:")
    model.eval(x_eval, y_eval)

    # saving the model
    if (output_path):
        model.save(output_path)
    else:
        model.save()


# split data, train a GROBID model and evaluate it
def eval_(model, use_ELMo=False, input_path=None, output_path=None):
    print('Loading data...')
    if input_path is None:
        x_all, y_all, f_all = load_data_and_labels_crf_file(
            'data/sequenceLabelling/grobid/' + model + '/' + model + '-060518.train')
    else:
        x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)

    print(len(x_all), 'evaluation sequences')

    batch_size = 20
    max_sequence_length = 3000

    if output_path:
        model_name = model
    else:
        model_name = 'grobid-' + model

    if model == "software":
        # class are more unbalanced, so we need to extend the batch size
        batch_size = 50
        max_sequence_length = 1500

    if use_ELMo:
        model_name += '-with_ELMo'
        if model_name == 'software-with_ELMo' or model_name == 'grobid-software-with_ELMo':
            batch_size = 3

    start_time = time.time()

    # load the model
    model = Sequence(model_name)
    model.load()
    # model.p.fit(x_all, y_all)
    # model.model_config.char_vocab_size = len(model.p.vocab_char)
    # model.model_config.case_vocab_size = len(model.p.vocab_case)

    # evaluation
    print("\nEvaluation:")
    model.eval(x_all, y_all)

    runtime = round(time.time() - start_time, 3)
    print("evaluation runtime: %s seconds " % (runtime))


# annotate a list of texts, this is relevant only of models taking only text as input 
# (so not text with layout information) 
def annotate_text(texts, model, output_format, use_ELMo=False):
    annotations = []

    # load model
    model_name = 'grobid-'+model
    if use_ELMo:
        model_name += '-with_ELMo'
    model = Sequence(model_name)
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
        description = "Trainer for GROBID models")

    actions = ['train', 'tag', 'train_eval', 'eval']
    architectures = ['BidLSTM_CRF', 'BidLSTM_CNN', 'BidLSTM_CNN_CRF', 'BidGRU-CRF']

    parser.add_argument("model")
    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture", default='BidLSTM_CRF',
                        help="type of model architecture to be used, one of " + str(architectures))
    parser.add_argument(
        "--embedding", default='glove-840B',
        help=(
            "The desired pre-trained word embeddings using their descriptions in the file"
            " embedding-registry.json."
            " Be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),"
            " and that the path in the registry to the embedding file is correct on your system."
        )
    )
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")
    parser.add_argument("--output", help="directory where to save a trained model")
    parser.add_argument("--input", help="provided training file")
    
    args = parser.parse_args()

    model = args.model    
    #if not model in models:
    #    print('invalid model, should be one of', models)

    action = args.action
    if action not in actions:
        print('action not specified, must be one of ' + str(actions))

    use_ELMo = args.use_ELMo
    architecture = args.architecture
    if architecture not in architectures:
        print('unknown model architecture, must be one of ' + str(architectures))

    output = args.output
    input_path = args.input
    embeddings_name = args.embedding

    if action == 'train':
        train(model, embeddings_name, architecture=architecture, use_ELMo=use_ELMo, input_path=input_path, output_path=output)

    if action == 'eval':
        eval_(model, use_ELMo=use_ELMo, input_path=input_path, output_path=output)

    if action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(model, embeddings_name, architecture=architecture, use_ELMo=use_ELMo, input_path=input_path, output_path=output, fold_count=args.fold_count)

    if action == 'tag':
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

        result = annotate_text(someTexts, model, "json", use_ELMo=use_ELMo)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
