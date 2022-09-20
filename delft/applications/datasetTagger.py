import os
import argparse
import json
import time
import numpy as np

from sklearn.model_selection import train_test_split

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_json_offsets
from delft.utilities.misc import parse_number_ranges

def configure(architecture, output_path=None, max_sequence_length=-1, batch_size=-1, embeddings_name=None, max_epoch=-1, use_ELMo=False):
    """
    Set up the default parameters based on the model type.
    """
    model_name = 'datasets'

    multiprocessing = True
    max_epoch = 60
    early_stop = True

    if "BERT" in architecture:
        # architectures with some transformer layer/embeddings inside
        if batch_size == -1:
            #default
            batch_size = 20
        if max_sequence_length == -1:
            #default 
            max_sequence_length = 200

        if max_sequence_length > 512:
            # 512 is the largest sequence for BERT input
            max_sequence_length = 512

        embeddings_name = None

    else:
        # RNN-only architectures
        if batch_size == -1:
            batch_size = 20
        if max_sequence_length == -1:
            max_sequence_length = 1500
        multiprocessing = False

    model_name += '-' + architecture

    if use_ELMo:
        model_name += '-with_ELMo'

    return batch_size, max_sequence_length, model_name, embeddings_name, max_epoch, multiprocessing, early_stop


# train a model with all available data
def train(embeddings_name=None, architecture='BidLSTM_CRF', transformer=None,
               input_path=None, output_path=None, fold_count=1,
               features_indices=None, max_sequence_length=-1, batch_size=-1, max_epoch=-1, use_ELMo=False):
    print('Loading data...')
    if input_path is None:
        x_all1 = y_all1 = x_all2 = y_all2 = x_all3 = y_all3 = []
        dataseer_sentences_path = "data/sequenceLabelling/datasets/dataseer_sentences.json"
        if os.path.exists(dataseer_sentences_path):
            x_all1, y_all1 = load_data_and_labels_json_offsets(dataseer_sentences_path)
        ner_dataset_recognition_sentences_path = "data/sequenceLabelling/datasets/ner_dataset_recognition_sentences.json"
        if os.path.exists(ner_dataset_recognition_sentences_path):
            x_all2, y_all2 = load_data_and_labels_json_offsets(ner_dataset_recognition_sentences_path)
        coleridge_sentences_path = "data/sequenceLabelling/datasets/coleridge_sentences.json.gz"
        if os.path.exists(coleridge_sentences_path):    
            x_all3, y_all3 = load_data_and_labels_json_offsets(coleridge_sentences_path)
        x_all = np.concatenate((x_all1, x_all2, x_all3[:1000]), axis=0)
        y_all = np.concatenate((y_all1, y_all2, y_all3[:1000]), axis=0)
    else:
        x_all, y_all = load_data_and_labels_json_offsets(input_path)

    x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1, shuffle=True)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    batch_size, max_sequence_length, model_name, embeddings_name, max_epoch, multiprocessing, early_stop = configure(architecture, 
                                                                            output_path, 
                                                                            max_sequence_length, 
                                                                            batch_size, 
                                                                            embeddings_name,
                                                                            max_epoch,
                                                                            use_ELMo)
    model = Sequence(model_name,
                    recurrent_dropout=0.50,
                    embeddings_name=embeddings_name,
                    architecture=architecture,
                    transformer_name=transformer,
                    max_sequence_length=max_sequence_length,
                    batch_size=batch_size,
                    fold_number=fold_count,
                    features_indices=features_indices,
                    max_epoch=max_epoch, 
                    use_ELMo=use_ELMo,
                    multiprocessing=multiprocessing,
                    early_stop=early_stop)

    start_time = time.time()
    model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
    runtime = round(time.time() - start_time, 3)

    print("training runtime: %s seconds " % runtime)

    # saving the model
    if output_path:
        model.save(output_path)
    else:
        model.save()


# split data, train a model and evaluate it
def train_eval(embeddings_name=None, architecture='BidLSTM_CRF', transformer=None,
               input_path=None, output_path=None, fold_count=1,
               features_indices=None, max_sequence_length=-1, batch_size=-1, max_epoch=-1, use_ELMo=False):
    print('Loading data...')
    if input_path is None:
        x_all1 = y_all1 = x_all2 = y_all2 = x_all3 = y_all3 = []
        dataseer_sentences_path = "data/sequenceLabelling/datasets/dataseer_sentences.json"
        if os.path.exists(dataseer_sentences_path):
            x_all1, y_all1 = load_data_and_labels_json_offsets(dataseer_sentences_path)
        ner_dataset_recognition_sentences_path = "data/sequenceLabelling/datasets/ner_dataset_recognition_sentences.json"
        if os.path.exists(ner_dataset_recognition_sentences_path):
            x_all2, y_all2 = load_data_and_labels_json_offsets(ner_dataset_recognition_sentences_path)
        coleridge_sentences_path = "data/sequenceLabelling/datasets/coleridge_sentences.json.gz"
        if os.path.exists(coleridge_sentences_path):    
            x_all3, y_all3 = load_data_and_labels_json_offsets(coleridge_sentences_path)
        x_all = np.concatenate((x_all1, x_all2, x_all3[:1000]), axis=0)
        y_all = np.concatenate((y_all1, y_all2, y_all3[:1000]), axis=0)
    else:
        x_all, y_all = load_data_and_labels_json_offsets(input_path)

    x_train_all, x_eval, y_train_all, y_eval = train_test_split(x_all, y_all, test_size=0.1, shuffle=True)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1)

    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_eval), 'evaluation sequences')

    batch_size, max_sequence_length, model_name, embeddings_name, max_epoch, multiprocessing, early_stop = configure(architecture, 
                                                                            output_path, 
                                                                            max_sequence_length, 
                                                                            batch_size, 
                                                                            embeddings_name,
                                                                            max_epoch,
                                                                            use_ELMo)
    model = Sequence(model_name,
                    recurrent_dropout=0.50,
                    embeddings_name=embeddings_name,
                    architecture=architecture,
                    transformer_name=transformer,
                    max_sequence_length=max_sequence_length,
                    batch_size=batch_size,
                    fold_number=fold_count,
                    features_indices=features_indices,
                    max_epoch=max_epoch, 
                    use_ELMo=use_ELMo,
                    multiprocessing=multiprocessing,
                    early_stop=early_stop)

    start_time = time.time()

    if fold_count == 1:
        model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid,)
    else:
        model.train_nfold(x_train, y_train, x_valid=x_valid, y_valid=y_valid)

    runtime = round(time.time() - start_time, 3)
    print("training runtime: %s seconds " % runtime)

    # evaluation
    print("\nEvaluation:")
    model.eval(x_eval, y_eval)

    # saving the model (must be called after eval for multiple fold training)
    if output_path:
        model.save(output_path)
    else:
        model.save()


def eval_(input_path=None, architecture=None):
    return


# annotate a list of texts
def annotate_text(texts, output_format, architecture='BidLSTM_CRF', features=None, use_ELMo=False):
    annotations = []

    # load model
    model_name = 'datasets'
    model_name += '-'+architecture
    if use_ELMo:
        model_name += '-with_ELMo'

    model = Sequence(model_name)
    model.load()

    start_time = time.time()

    annotations = model.tag(texts, output_format, features=features)
    runtime = round(time.time() - start_time, 3)

    if output_format == 'json':
        annotations["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))

    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Trainer for dataset recognition models using the DeLFT library")

    actions = ["train", "train_eval", "eval", "tag"]

    architectures_word_embeddings = [
                     'BidLSTM', 'BidLSTM_CRF', 'BidLSTM_ChainCRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 
                     'BidLSTM_CRF_FEATURES', 'BidLSTM_ChainCRF_FEATURES', 
                     ]

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']

    architectures_transformers_based = [
                    'BERT', 'BERT_CRF', 'BERT_ChainCRF', 'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES'
                     ]

    architectures = architectures_word_embeddings + architectures_transformers_based

    pretrained_transformers_examples = ['bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased']

    parser.add_argument("action", choices=actions)
    parser.add_argument("--fold-count", type=int, default=1, help="Number of fold to use when evaluating with n-fold "
                                                                  "cross validation.")
    parser.add_argument("--architecture", help="Type of model architecture to be used, one of "+str(architectures))
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings") 

    # group_embeddings = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument("--output", help="Directory where to save a trained model.")
    parser.add_argument("--input", help="Grobid data file to be used for training (train action), for training and " +
                                        "evaluation (train_eval action) or just for evaluation (eval action).")
    parser.add_argument("--max-sequence-length", type=int, default=-1, help="max-sequence-length parameter to be used.")
    parser.add_argument("--batch-size", type=int, default=-1, help="batch-size parameter to be used.")

    args = parser.parse_args()

    action = args.action
    architecture = args.architecture
    output = args.output
    input_path = args.input
    embeddings_name = args.embedding
    max_sequence_length = args.max_sequence_length
    batch_size = args.batch_size
    transformer = args.transformer
    use_ELMo = args.use_ELMo

    if transformer is None and embeddings_name is None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if action == "train":
            train(embeddings_name=embeddings_name, 
            architecture=architecture, 
            transformer=transformer,
            input_path=input_path, 
            output_path=output,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            use_ELMo=use_ELMo)

    if action == "eval":
        if args.fold_count is not None and args.fold_count > 1:
            print("The argument fold-count argument will be ignored. For n-fold cross-validation, please use "
                  "it in combination with train_eval")
        if input_path is None:
            raise ValueError("A Grobid evaluation data file must be specified to evaluate a grobid model with the parameter --input")
        eval_(input_path=input_path, architecture=architecture)

    if action == "train_eval":
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(embeddings_name=embeddings_name, 
                architecture=architecture, 
                transformer=transformer,
                input_path=input_path, 
                output_path=output, 
                fold_count=args.fold_count,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size,
                use_ELMo=use_ELMo)

    if action == "tag":
        someTexts = []
        someTexts.append("The DEGs were annotated using the following databases: the NR protein database (NCBI), Swiss Prot, Gene Ontology (GO), the Kyoto Encyclopedia of Genes and Genomes (KEGG) database, and the Clusters of Orthologous Groups database (COG) according to the methods of described by Zhou et al")
        someTexts.append("The electrochemiluminescence immunoassay was used to measure serum concentration of 25-hydroxyvitamin D using Roche Modular E170 Analyzer (Roche Diagnostics, Basel, Switzerland).")
        someTexts.append("We found that this technique works very well in practice, for the MNIST and NORB datasets (see below).")
        someTexts.append("We also compare ShanghaiTechRGBD with other RGB-D crowd counting datasets in , and we can see that ShanghaiTechRGBD is the most challenging RGB-D crowd counting dataset in terms of the number of images and heads.")

        result = annotate_text(someTexts, "json", architecture=architecture, use_ELMo=use_ELMo)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
        

