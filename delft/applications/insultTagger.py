import argparse
import json
import os
import time

from sklearn.model_selection import train_test_split

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_xml_file
from delft.utilities.Utilities import set_random_seed, t_or_f


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


def train(
    embeddings_name=None,
    architecture="BidLSTM_CRF",
    transformer=None,
    learning_rate=None,
    batch_size=-1,
    max_epoch=-1,
    early_stop=None,
    multi_gpu=False,
    report_to_wandb=False,
    wandb_project=None,
    num_workers=None,
):
    batch_size, maxlen, patience, early_stop, max_epoch, embeddings_name = configure(
        architecture, embeddings_name, batch_size, max_epoch, early_stop
    )
    root = "data/sequenceLabelling/toxic/"

    train_path = os.path.join(root, "corrected.xml")
    valid_path = os.path.join(root, "valid.xml")

    print("Loading data...")
    x_train, y_train = load_data_and_labels_xml_file(train_path)
    x_valid, y_valid = load_data_and_labels_xml_file(valid_path)
    print(len(x_train), "train sequences")
    print(len(x_valid), "validation sequences")

    model_name = "insult-" + architecture

    model = Sequence(
        model_name,
        max_epoch=max_epoch,
        batch_size=batch_size,
        max_sequence_length=maxlen,
        embeddings_name=embeddings_name,
        architecture=architecture,
        patience=patience,
        early_stop=early_stop,
        transformer_name=transformer,
        learning_rate=learning_rate,
        report_to_wandb=report_to_wandb,
        wandb_project=wandb_project,
        nb_workers=num_workers,
        short_model_name="insult",
    )
    model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid, multi_gpu=multi_gpu)
    print("training done")

    # saving the model (must be called after eval for multiple fold training)
    model.save()


# train on a part of the training set and evaluate on the validation set
def train_eval(
    embeddings_name=None,
    architecture="BidLSTM_CRF",
    transformer=None,
    fold_count=1,
    learning_rate=None,
    batch_size=-1,
    max_epoch=-1,
    early_stop=None,
    multi_gpu=False,
    report_to_wandb=False,
    wandb_project=None,
    num_workers=None,
):
    batch_size, maxlen, patience, early_stop, max_epoch, embeddings_name = configure(
        architecture, embeddings_name, batch_size, max_epoch, early_stop
    )
    root = "data/sequenceLabelling/toxic/"

    print("Loading data...")
    x_all, y_all = load_data_and_labels_xml_file(os.path.join(root, "corrected.xml"))
    x_eval, y_eval = load_data_and_labels_xml_file(os.path.join(root, "valid.xml"))
    # the validation set of the corpus is the evaluation set: the one of the training is held out of the train set
    x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1, shuffle=True)
    print(len(x_train), "train sequences")
    print(len(x_valid), "validation sequences")
    print(len(x_eval), "evaluation sequences")

    model = Sequence(
        "insult-" + architecture,
        max_epoch=max_epoch,
        batch_size=batch_size,
        max_sequence_length=maxlen,
        embeddings_name=embeddings_name,
        architecture=architecture,
        fold_number=fold_count,
        patience=patience,
        early_stop=early_stop,
        transformer_name=transformer,
        learning_rate=learning_rate,
        report_to_wandb=report_to_wandb,
        wandb_project=wandb_project,
        nb_workers=num_workers,
        short_model_name="insult",
    )
    if fold_count == 1:
        model.train(x_train, y_train, x_valid=x_valid, y_valid=y_valid, multi_gpu=multi_gpu)
    else:
        model.train_nfold(x_train, y_train, x_valid=x_valid, y_valid=y_valid, multi_gpu=multi_gpu)
    print("training done")

    print("\nEvaluation:")
    model.eval(x_eval, y_eval)

    # saving the model (must be called after eval for multiple fold training)
    model.save()


# annotate a list of texts, provides results in a list of offset mentions
def annotate(
    texts,
    output_format,
    architecture="BidLSTM_CRF",
    transformer=None,
    multi_gpu=False,
):
    annotations = []

    model_name = "insult-" + architecture

    # load model
    model = Sequence(
        model_name,
        architecture=architecture,
        transformer_name=transformer,
    )
    model.load()

    start_time = time.time()

    annotations = model.tag(texts, output_format, multi_gpu=multi_gpu)
    runtime = round(time.time() - start_time, 3)

    if output_format == "json":
        annotations["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))
    return annotations


if __name__ == "__main__":
    architectures_word_embeddings = [
        "BidLSTM",
        "BidLSTM_CRF",
        "BidLSTM_ChainCRF",
        "BidLSTM_CNN_CRF",
        "BidGRU_CRF",
        "BidLSTM_CNN",
        "BidLSTM_CRF_CASING",
    ]

    word_embeddings_examples = ["glove-840B", "fasttext-crawl", "word2vec", "potion-base-8M"]

    architectures_transformers_based = [
        "BERT",
        "BERT_CRF",
        "BERT_ChainCRF",
        "BERT_CRF_FEATURES",
    ]

    architectures = architectures_word_embeddings + architectures_transformers_based

    pretrained_transformers_examples = [
        "bert-base-cased",
        "bert-large-cased",
        "allenai/scibert_scivocab_cased",
    ]
    parser = argparse.ArgumentParser(
        description="Experimental insult recognizer for the Wikipedia toxic comments dataset"
    )

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed of the random number generators, to run a training again with the same split of the data, the "
        + "same initial weights and the same order of the batches. Default: not seeded, every run differs.",
    )
    parser.add_argument(
        "--architecture",
        default="BidLSTM_CRF",
        help="Type of model architecture to be used, one of " + str(architectures),
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help="The desired pre-trained word embeddings using their descriptions in the file. "
        + "For local loading, use delft/resources-registry.json. "
        + "Be sure to use here the same name as in the registry, e.g. "
        + str(word_embeddings_examples)
        + ". Contextual embeddings include scibert-contextual and bert-base-cased-contextual, and can also be "
        + "given as contextual:<HuggingFace model or local path>. Paths in the registry must be correct on your system.",
    )
    parser.add_argument(
        "--transformer",
        default=None,
        help="The desired pre-trained transformer to be used in the selected architecture. "
        + "For local loading use, delft/resources-registry.json, and be sure to use here the same name as in the registry, e.g. "
        + str(pretrained_transformers_examples)
        + " and that the path in the registry to the model path is correct on your system. "
        + "HuggingFace transformers hub will be used otherwise to fetch the model, see https://huggingface.co/models "
        + "for model names",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--max-epoch", type=int, default=-1, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=-1, help="batch-size parameter to be used.")
    parser.add_argument(
        "--early-stop",
        type=t_or_f,
        default=None,
        help="Force early training termination when metrics scores are not improving "
        + "after a number of epochs equals to the patience parameter.",
    )

    parser.add_argument(
        "--multi-gpu",
        default=False,
        help="Enable the support for distributed computing (the batch size needs to be set accordingly using --batch-size)",
        action="store_true",
    )

    parser.add_argument(
        "--wandb",
        default=False,
        help="Enable logging to Weights and Biases",
        action="store_true",
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Wandb project name. If unset, falls back to the WANDB_PROJECT env var, then to wandb's default.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for data loading. Default: cpu_count - 1 for train.",
    )

    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.action not in ("train", "train_eval", "tag"):
        print("action not specified, must be one of [train,train_eval,tag]")

    embeddings_name = args.embedding
    architecture = args.architecture
    transformer = args.transformer
    learning_rate = args.learning_rate

    batch_size = args.batch_size
    max_epoch = args.max_epoch
    early_stop = args.early_stop
    multi_gpu = args.multi_gpu
    wandb = args.wandb
    wandb_project = args.wandb_project
    num_workers = args.num_workers

    if args.action in ("train", "train_eval"):
        if embeddings_name is None and not (architecture and "BERT" in architecture):
            # No word embeddings, and no transformer inside the architecture: train character-only (issue #216).
            print("No --embedding given: training without word embeddings (char-only).")
    elif transformer is not None or embeddings_name is not None:
        print(
            f"Warning: --transformer and --embedding are ignored by {args.action}, which uses what the "
            "configuration of the model says: the ones it was trained with."
        )

    if args.action == "train":
        train(
            embeddings_name=embeddings_name,
            architecture=architecture,
            transformer=transformer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            early_stop=early_stop,
            multi_gpu=multi_gpu,
            report_to_wandb=wandb,
            wandb_project=wandb_project,
            num_workers=num_workers,
        )

    if args.action == "train_eval":
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train_eval(
            embeddings_name=embeddings_name,
            architecture=architecture,
            transformer=transformer,
            fold_count=args.fold_count,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            early_stop=early_stop,
            multi_gpu=multi_gpu,
            report_to_wandb=wandb,
            wandb_project=wandb_project,
            num_workers=num_workers,
        )

    if args.action == "tag":
        someTexts = [
            "This is a gentle test.",
            "you're a moronic wimp who is too lazy to do research! die in hell !!",
            "This is a fucking test.",
        ]
        result = annotate(
            someTexts,
            "json",
            architecture=architecture,
            transformer=transformer,
        )
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))
