import os
import json
from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_xml_file
import argparse
import time

from delft.utilities.Utilities import t_or_f


def configure(architecture, embeddings_name, batch_size=-1, max_epoch=-1, early_stop=None):

    maxlen = 300
    patience = 5
    o_early_stop = True
    if max_epoch == -1:
        max_epoch = 50

    if batch_size == -1:
        batch_size = 20

    # default bert model parameters
    if architecture.find("BERT") != -1:
        if batch_size == -1:
            batch_size = 10
        o_early_stop = False
        if max_epoch == -1:
            max_epoch = 3

        embeddings_name = None

    if early_stop is not None:
        o_early_stop = early_stop

    return batch_size, maxlen, patience, o_early_stop, max_epoch, embeddings_name

def train(embeddings_name=None, architecture='BidLSTM_CRF', transformer=None,
          use_ELMo=False, learning_rate=None,
          batch_size=-1, max_epoch=-1, early_stop=None, multi_gpu=False):
    batch_size, maxlen, patience, early_stop, max_epoch, embeddings_name = configure(architecture, 
                                                                                    embeddings_name, 
                                                                                    batch_size, 
                                                                                    max_epoch, 
                                                                                    early_stop)
    root = 'data/sequenceLabelling/toxic/'

    train_path = os.path.join(root, 'corrected.xml')
    valid_path = os.path.join(root, 'valid.xml')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_xml_file(train_path)
    x_valid, y_valid = load_data_and_labels_xml_file(valid_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    model_name = 'insult-' + architecture
    if use_ELMo:
        model_name += '-with_ELMo'

    model = Sequence(model_name, max_epoch=max_epoch, batch_size=batch_size, max_sequence_length=maxlen, 
        embeddings_name=embeddings_name, architecture=architecture, patience=patience, early_stop=early_stop,
        transformer_name=transformer, use_ELMo=use_ELMo, learning_rate=learning_rate)
    model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid, multi_gpu=multi_gpu)
    print('training done')

    # saving the model (must be called after eval for multiple fold training)
    model.save()


# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts, output_format, architecture='BidLSTM_CRF', transformer=None, use_ELMo=False, multi_gpu=False):
    annotations = []

    model_name = 'insult-' + architecture
    if use_ELMo:
        model_name += '-with_ELMo'

    # load model
    model = Sequence(model_name, architecture=architecture, transformer_name=transformer, use_ELMo=use_ELMo)
    model.load()

    start_time = time.time()

    annotations = model.tag(texts, output_format, multi_gpu=multi_gpu)
    runtime = round(time.time() - start_time, 3)

    if output_format == 'json':
        annotations["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return annotations


if __name__ == "__main__":

    architectures_word_embeddings = [
                     'BidLSTM', 'BidLSTM_CRF', 'BidLSTM_ChainCRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 
                     ]

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']

    architectures_transformers_based = [
                    'BERT', 'BERT_CRF', 'BERT_ChainCRF', 'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES'
                     ]

    architectures = architectures_word_embeddings + architectures_transformers_based

    pretrained_transformers_examples = [ 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased' ]
    parser = argparse.ArgumentParser(
        description = "Experimental insult recognizer for the Wikipedia toxic comments dataset")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture", default='BidLSTM_CRF', 
                        help="Type of model architecture to be used, one of "+str(architectures))
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
    parser.add_argument("--use-ELMo", action="store_true", help="Use ELMo contextual embeddings")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--max-epoch", type=int, default=-1,
                        help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=-1, help="batch-size parameter to be used.")
    parser.add_argument("--early-stop", type=t_or_f, default=None,
                        help="Force early training termination when metrics scores are not improving " + 
                             "after a number of epochs equals to the patience parameter.")

    parser.add_argument("--multi-gpu", default=False,
                        help="Enable the support for distributed computing (the batch size needs to be set accordingly using --batch-size)",
                        action="store_true")


    args = parser.parse_args()

    if args.action not in ('train', 'tag'):
        print('action not specified, must be one of [train,tag]')

    embeddings_name = args.embedding
    architecture = args.architecture
    transformer = args.transformer
    use_ELMo = args.use_ELMo
    learning_rate = args.learning_rate

    batch_size = args.batch_size
    max_epoch = args.max_epoch
    early_stop = args.early_stop
    multi_gpu = args.multi_gpu

    if transformer == None and embeddings_name == None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if args.action == 'train':
        train(embeddings_name=embeddings_name,
              architecture=architecture,
              transformer=transformer,
              use_ELMo=use_ELMo,
              learning_rate=learning_rate,
              batch_size=batch_size,
              max_epoch=max_epoch,
              early_stop=early_stop,
              multi_gpu=multi_gpu)

    if args.action == 'tag':
        someTexts = ['This is a gentle test.', 
                     'you\'re a moronic wimp who is too lazy to do research! die in hell !!', 
                     'This is a fucking test.']
        result = annotate(someTexts, "json", architecture=architecture, transformer=transformer, use_ELMo=use_ELMo)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

