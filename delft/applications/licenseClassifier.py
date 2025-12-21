import json
from delft.utilities.Embeddings import Embeddings
from delft.utilities.Utilities import split_data_and_labels
from delft.utilities.numpy import concatenate_or_none, shuffle_triple_with_view
from delft.textClassification.reader import vectorize as vectorizer
from delft.textClassification.reader import (
    normalize_classes,
    load_texts_and_classes_pandas_no_id,
)
import delft.textClassification
from delft.textClassification import Classifier
import argparse
import time
from delft.textClassification.models import architectures
import numpy as np

"""
    Two multiclass classifiers to be used in combination with Grobid to classify a license/copyrights section
    extracted from a scientific article into two dimensions:  
    - copyright owner: publisher, authors or undecidable (changed from NA to avoid issues with pandas)
    - license associated to the article file: explicit copyrights (no restriction), creative commons licenses 
      (CC-0, CC-BY, CC-BY-NC, etc.), other, or undecidable (changed from NA to avoid issues with pandas)
    
    Note: when the license is undecidable, this means normal copyrights when a copyright owner exists.
"""

list_classes_copyright = ["publisher", "authors", "undecided"]
class_weights_copyright = {0: 1.0, 1: 1.0, 2: 1.0}
list_classes_licenses = [
    "CC-0",
    "CC-BY",
    "CC-BY-NC",
    "CC-BY-NC-ND",
    "CC-BY-SA",
    "CC-BY-NC-SA",
    "CC-BY-ND",
    "copyright",
    "other",
    "undecided",
]
class_weights_licenses = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}


def configure(architecture):
    batch_size = 256
    maxlen = 300
    patience = 5
    early_stop = True
    max_epoch = 60

    # default bert model parameters
    if architecture == "bert":
        batch_size = 16
        early_stop = False
        max_epoch = 6
        maxlen = 200

    return batch_size, maxlen, patience, early_stop, max_epoch


def train(embeddings_name, fold_count, architecture="gru", transformer=None):
    print("loading multiclass copyright/license dataset...")
    xtr, y_copyrights = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="copyrights",
    )

    report_training_copyrights(y_copyrights)

    # copyright ownwer classifier
    model_name = "copyright_" + architecture
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier(
        model_name,
        architecture=architecture,
        list_classes=list_classes_copyright,
        max_epoch=max_epoch,
        fold_number=fold_count,
        patience=patience,
        use_roc_auc=True,
        embeddings_name=embeddings_name,
        batch_size=batch_size,
        maxlen=maxlen,
        early_stop=early_stop,
        class_weights=class_weights_copyright,
        transformer_name=transformer,
    )

    if fold_count == 1:
        model.train(xtr, y_copyrights)
    else:
        model.train_nfold(xtr, y_copyrights)
    # saving the model
    model.save()

    # license classifier
    xtr, y_licenses = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="licenses",
    )
    model_name = "license_" + architecture
    class_weights_licenses = None
    # batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    model = Classifier(
        model_name,
        architecture=architecture,
        list_classes=list_classes_licenses,
        max_epoch=max_epoch,
        fold_number=fold_count,
        patience=patience,
        use_roc_auc=True,
        embeddings_name=embeddings_name,
        batch_size=batch_size,
        maxlen=maxlen,
        early_stop=early_stop,
        class_weights=class_weights_licenses,
        transformer_name=transformer,
    )

    if fold_count == 1:
        model.train(xtr, y_licenses)
    else:
        model.train_nfold(xtr, y_licenses)
    # saving the model
    model.save()


def train_and_eval(embeddings_name, fold_count, architecture="gru", transformer=None):
    print("loading multiclass copyright/license dataset...")
    xtr, y_copyrights = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="copyrights",
    )

    report_training_copyrights(y_copyrights)

    # copyright ownwer classifier
    model_name = "copyright_" + architecture

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y_copyrights, 0.9)
    batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

    print(list_classes_copyright)

    model = Classifier(
        model_name,
        architecture=architecture,
        list_classes=list_classes_copyright,
        max_epoch=max_epoch,
        fold_number=fold_count,
        patience=patience,
        use_roc_auc=True,
        embeddings_name=embeddings_name,
        batch_size=batch_size,
        maxlen=maxlen,
        early_stop=early_stop,
        class_weights=class_weights_copyright,
        transformer_name=transformer,
    )

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()

    # license classifier
    xtr, y_licenses = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="licenses",
    )

    model_name = "license_" + architecture

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y_licenses, 0.9)
    # batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)
    class_weights_licenses = None

    model = Classifier(
        model_name,
        architecture=architecture,
        list_classes=list_classes_licenses,
        max_epoch=max_epoch,
        fold_number=fold_count,
        patience=patience,
        use_roc_auc=True,
        embeddings_name=embeddings_name,
        batch_size=batch_size,
        maxlen=maxlen,
        early_stop=early_stop,
        class_weights=class_weights_licenses,
        transformer_name=transformer,
    )

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model (not so useful...)
    model.save()


def train_binary(embeddings_name, fold_count, architecture="gru", transformer=None):
    print("loading multiclass copyright/license dataset...")
    xtr, y_copyrights = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="copyrights",
    )

    report_training_copyrights(y_copyrights)

    for class_rank in range(len(list_classes_copyright)):
        model_name = (
            "copyright_" + list_classes_copyright[class_rank] + "_" + architecture
        )

        # we could experiment with binary weighting the class maybe
        class_weights = None

        batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

        y_train_class_rank = [
            [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_copyrights
        ]
        y_train_class_rank = np.array(y_train_class_rank)

        list_classes_rank = [
            list_classes_copyright[class_rank],
            "not_" + list_classes_copyright[class_rank],
        ]

        model = Classifier(
            model_name,
            architecture=architecture,
            list_classes=list_classes_rank,
            max_epoch=max_epoch,
            fold_number=fold_count,
            patience=patience,
            use_roc_auc=True,
            embeddings_name=embeddings_name,
            batch_size=batch_size,
            maxlen=maxlen,
            early_stop=early_stop,
            class_weights=class_weights,
            transformer_name=transformer,
        )

        if fold_count == 1:
            model.train(xtr, y_train_class_rank)
        else:
            model.train_nfold(xtr, y_train_class_rank)
        # saving the model
        model.save()

    xtr, y_licenses = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="licenses",
    )
    for class_rank in range(len(list_classes_licenses)):
        model_name = (
            "licenses_" + list_classes_licenses[class_rank] + "_" + architecture
        )

        # we could experiment with binary weighting the class maybe
        class_weights = None

        batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

        y_train_class_rank = [
            [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_licenses
        ]
        y_train_class_rank = np.array(y_train_class_rank)

        list_classes_rank = [
            list_classes_licenses[class_rank],
            "not_" + list_classes_licenses[class_rank],
        ]

        model = Classifier(
            model_name,
            architecture=architecture,
            list_classes=list_classes_rank,
            max_epoch=max_epoch,
            fold_number=fold_count,
            patience=patience,
            use_roc_auc=True,
            embeddings_name=embeddings_name,
            batch_size=batch_size,
            maxlen=maxlen,
            early_stop=early_stop,
            class_weights=class_weights,
            transformer_name=transformer,
        )

        if fold_count == 1:
            model.train(xtr, y_train_class_rank)
        else:
            model.train_nfold(xtr, y_train_class_rank)
        # saving the model
        model.save()


def train_and_eval_binary(
    embeddings_name, fold_count, architecture="gru", transformer=None
):
    print("loading multiclass copyright/license dataset...")
    xtr, y_copyrights = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="copyrights",
    )

    report_training_copyrights(y_copyrights)

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y_copyrights, 0.9)

    for class_rank in range(len(list_classes_copyright)):
        model_name = (
            "copyright_" + list_classes_copyright[class_rank] + "_" + architecture
        )
        class_weights = None

        batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

        y_train_class_rank = [
            [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_train
        ]
        y_test_class_rank = [[1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_test]

        y_train_class_rank = np.array(y_train_class_rank)
        y_test_class_rank = np.array(y_test_class_rank)

        list_classes_rank = [
            list_classes_copyright[class_rank],
            "not_" + list_classes_copyright[class_rank],
        ]

        model = Classifier(
            model_name,
            architecture=architecture,
            list_classes=list_classes_rank,
            max_epoch=max_epoch,
            fold_number=fold_count,
            patience=patience,
            use_roc_auc=True,
            embeddings_name=embeddings_name,
            batch_size=batch_size,
            maxlen=maxlen,
            early_stop=early_stop,
            class_weights=class_weights,
            transformer_name=transformer,
        )

        if fold_count == 1:
            model.train(x_train, y_train_class_rank)
        else:
            model.train_nfold(x_train, y_train_class_rank)
        model.eval(x_test, y_test_class_rank)

        # saving the model
        # model.save()

    xtr, y_licenses = _read_data(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv",
        data_type="licenses",
    )

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y_licenses, 0.9)

    for class_rank in range(len(list_classes_licenses)):
        model_name = (
            "licenses_" + list_classes_licenses[class_rank] + "_" + architecture
        )
        class_weights = None

        batch_size, maxlen, patience, early_stop, max_epoch = configure(architecture)

        y_train_class_rank = [
            [1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_train
        ]
        y_test_class_rank = [[1, 0] if y[class_rank] == 1.0 else [0, 1] for y in y_test]

        y_train_class_rank = np.array(y_train_class_rank)
        y_test_class_rank = np.array(y_test_class_rank)

        list_classes_rank = [
            list_classes_licenses[class_rank],
            "not_" + list_classes_licenses[class_rank],
        ]

        model = Classifier(
            model_name,
            architecture=architecture,
            list_classes=list_classes_rank,
            max_epoch=max_epoch,
            fold_number=fold_count,
            patience=patience,
            use_roc_auc=True,
            embeddings_name=embeddings_name,
            batch_size=batch_size,
            maxlen=maxlen,
            early_stop=early_stop,
            class_weights=class_weights,
            transformer_name=transformer,
        )

        if fold_count == 1:
            model.train(x_train, y_train_class_rank)
        else:
            model.train_nfold(x_train, y_train_class_rank)
        model.eval(x_test, y_test_class_rank)

        # saving the model
        # model.save()


# classify a list of texts
def classify(
    texts, output_format, embeddings_name=None, architecture="gru", transformer=None
):
    # load model
    model = Classifier("copyright_" + architecture)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    runtime = round(time.time() - start_time, 3)

    model = Classifier("license_" + architecture)
    model.load()
    result2 = model.predict(texts, output_format)

    runtime = round(time.time() - start_time, 3)
    if output_format == "json":
        result["runtime"] = runtime
    else:
        print("runtime: %s seconds " % (runtime))

    return result, result2


def _read_data(file_path, data_type="copyrights"):
    """
    Classification training data is expected in the following input format (csv or tsv):
    text_0    copyright_owner_0   license_0
    text_1    copyright_owner_1   license_1
    ...
    text_n    copyright_owner_n   license_n

    One of the value copyright_owner_i or license_i can be empty, in this case the training
    case is only relevant to one classifier.

    data_type indicates the data to retun according to the classifier, it is either
    "copyrights" or "licences".

    Note: final classification data include vectorized class values and the whole set is shuffled.
    """
    x, y = load_texts_and_classes_pandas_no_id(
        "data/textClassification/licenses/copyrights-licenses-data-validated.csv"
    )
    y_target = None
    list_classes = None
    if data_type == "copyrights":
        y_target = y[:, 0]
        list_classes = list_classes_copyright
    elif data_type == "licenses":
        y_target = y[:, 1]
        list_classes = list_classes_licenses
    else:
        print("invalid data type:", data_type)
        return None, None

    # remove rows when class valus is MISSINGVALUE
    def delete_rows(vecA, vecB, value):
        new_vecA = []
        new_vecB = []
        for i in range(len(vecB)):
            if vecB[i] != value and vecB[i] != "nan":
                new_vecA.append(vecA[i])
                new_vecB.append(vecB[i])
        return np.asarray(new_vecA), np.asarray(new_vecB)

    x, y_target = delete_rows(x, y_target, "MISSINGVALUE")

    xtr, y_target, _ = shuffle_triple_with_view(x, y_target)

    y_target_final = []
    for i in range(len(y_target)):
        index = list_classes.index(y_target[i])
        y_target_final.append(vectorizer(index, len(list_classes)))

    return xtr, np.array(y_target_final)


def report_training_copyrights(y):
    """
    Quick and dirty stats on the classes for copyrights owner
    """
    nb_publisher = 0
    nb_authors = 0
    nb_undecidable = 0
    for the_class in y:
        if the_class[0] == 1:
            nb_publisher += 1
        elif the_class[1] == 1:
            nb_authors += 1
        elif the_class[2] == 1:
            nb_undecidable += 1

    print("\ntotal context training cases:", len(y))

    print("\t- publisher owner:", nb_publisher)
    print("\t  not publisher owner:", str(len(y) - nb_publisher))

    print("\t- authors owner:", nb_authors)
    print("\t  not authors owner:", str(len(y) - nb_authors))

    print("\t- undecidable:", nb_undecidable)
    print("\t  not undecidable:", str(len(y) - nb_undecidable))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a copyright/license section using the DeLFT library"
    )

    word_embeddings_examples = ["glove-840B", "fasttext-crawl", "word2vec"]
    pretrained_transformers_examples = [
        "bert-base-cased",
        "bert-large-cased",
        "allenai/scibert_scivocab_cased",
    ]

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument(
        "--architecture",
        default="gru",
        help="type of model architecture to be used, one of " + str(architectures),
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help="The desired pre-trained word embeddings using their descriptions in the file. "
        + "For local loading, use delft/resources-registry.json. "
        + "Be sure to use here the same name as in the registry, e.g. "
        + str(word_embeddings_examples)
        + " and that the path in the registry to the embedding file is correct on your system.",
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

    args = parser.parse_args()

    if args.action not in (
        "train",
        "train_eval",
        "classify",
        "train_binary",
        "train_eval_binary",
    ):
        print(
            "action not specified, must be one of [train,train_binary,train_eval,train_eval_binary,classify]"
        )

    embeddings_name = args.embedding
    transformer = args.transformer

    architecture = args.architecture
    if architecture not in architectures:
        print("unknown model architecture, must be one of " + str(architectures))

    if transformer == None and embeddings_name == None:
        # default word embeddings
        embeddings_name = "glove-840B"

    if args.action == "train":
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(
            embeddings_name,
            args.fold_count,
            architecture=architecture,
            transformer=transformer,
        )

    if args.action == "train_binary":
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train_binary(
            embeddings_name,
            args.fold_count,
            architecture=architecture,
            transformer=transformer,
        )

    if args.action == "train_eval":
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(
            embeddings_name,
            args.fold_count,
            architecture=architecture,
            transformer=transformer,
        )

    if args.action == "train_eval_binary":
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval_binary(
            embeddings_name,
            args.fold_count,
            architecture=architecture,
            transformer=transformer,
        )

    if args.action == "classify":
        someTexts = [
            "© 2005 Elsevier Inc. All rights reserved.",
            "This is an open-access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by-nc/4.0), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited. Copyright © 2022 The Korean Association for Radiation Protection.",
            "© The Author(s) 2023.",
        ]
        result1, result2 = classify(
            someTexts,
            "json",
            architecture=architecture,
            embeddings_name=embeddings_name,
            transformer=transformer,
        )
        print(json.dumps(result1, sort_keys=False, indent=4, ensure_ascii=False))
        print(json.dumps(result2, sort_keys=False, indent=4, ensure_ascii=False))
