import json
from delft.utilities.Utilities import split_data_and_labels
from delft.textClassification.reader import load_citation_sentiment_corpus
from delft.textClassification import Classifier
import argparse
import time
from delft.textClassification.models import architectures

list_classes = [
                    "negative", 
                    "neutral", 
                    "positive"
                ]

class_weights = {
                    0: 25.,
                    1: 1.,
                    2: 9.
                }

def configure(architecture):
    batch_size = 256
    maxlen = 150
    patience = 5
    early_stop = True
    max_epoch = 60

    # default bert model parameters
    if architecture == "bert":
        batch_size = 32
        early_stop = False
        max_epoch = 3

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(embeddings_name, fold_count, architecture="gru", transformer=None):
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier('citations_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

    print('loading citation sentiment corpus...')
    xtr, y = load_citation_sentiment_corpus("data/textClassification/citations/citation_sentiment_corpus.txt")

    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()


def train_and_eval(embeddings_name, fold_count, architecture="gru", transformer=None): 
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier('citations_'+architecture, architecture=architecture, list_classes=list_classes, max_epoch=max_epoch, fold_number=fold_count, 
        use_roc_auc=True, embeddings_name=embeddings_name, batch_size=batch_size, maxlen=maxlen, patience=patience, early_stop=early_stop,
        class_weights=class_weights, transformer_name=transformer)

    print('loading citation sentiment corpus...')
    xtr, y = load_citation_sentiment_corpus("data/textClassification/citations/citation_sentiment_corpus.txt")

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    
    # saving the model
    model.save()

    model.eval(x_test, y_test)

    
# classify a list of texts
def classify(texts, output_format, architecture="gru", embeddings_name=None, transformer=None):
    # load model
    model = Classifier('citations_'+architecture, architecture=architecture, list_classes=list_classes, embeddings_name=embeddings_name, transformer_name=transformer)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)
    if output_format == 'json':
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Sentiment classification of citation contexts based on DeLFT")

    word_embeddings_examples = ['glove-840B', 'fasttext-crawl', 'word2vec']
    pretrained_transformers_examples = [ 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased' ]

    parser.add_argument("action", help="one of [train, train_eval, classify]")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--architecture",default='gru', help="type of model architecture to be used, one of "+str(architectures))
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

        train(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(embeddings_name, args.fold_count, architecture=architecture, transformer=transformer)    

    if args.action == 'classify':
        someTexts = ['One successful strategy [15] computes the set-similarity involving (multi-word) keyphrases about the mentions and the entities, collected from the KG.', 
            'Unfortunately, fewer than half of the OCs in the DAML02 OC catalog (Dias et al. 2002) are suitable for use with the isochrone-fitting method because of the lack of a prominent main sequence, in addition to an absence of radial velocity and proper-motion data.', 
            'However, we found that the pairwise approach LambdaMART [41] achieved the best performance on our datasets among most learning to rank algorithms.']
        result = classify(someTexts, "json", architecture=architecture, embeddings_name=embeddings_name, transformer=transformer)
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
