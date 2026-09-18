"""The texts of a data set too big for memory are read from disk (issue #27)."""

import multiprocessing
import pickle

import numpy as np
import pytest

from delft.textClassification.reader import TextsOnDisk, load_texts_and_classes
from delft.textClassification.wrapper import split_train_validation

LINES = [
    ("0", "Jim Henson was a puppeteer", ["1", "0"]),
    ("1", "", ["0", "1"]),
    ("2", "in Mississippi, today", ["0", "1"]),
    ("3", "Ωmega and Ünicode are fine", ["1", "1"]),
    ("4", "the last line has no newline", ["0", "0"]),
]


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "texts.tsv"
    content = "\n\n".join("\t".join([identifier, text] + classes) for identifier, text, classes in LINES)
    path.write_text(content, encoding="utf-8")  # blank lines between, none at the end
    return str(path)


def test_the_same_texts_and_classes_as_when_loaded_in_memory(data_file):
    texts, classes = load_texts_and_classes(data_file)
    texts_on_disk, classes_on_disk = load_texts_and_classes(data_file, in_memory=False)
    assert isinstance(texts_on_disk, TextsOnDisk)
    assert list(texts_on_disk) == list(texts)
    assert [texts_on_disk[i] for i in range(len(texts_on_disk))] == list(texts)
    assert np.array_equal(classes_on_disk, classes)


def test_a_slice_or_indices_give_the_texts_on_disk_of_those_lines(data_file):
    texts, _ = load_texts_and_classes(data_file, in_memory=False)
    assert list(texts[1:3]) == [LINES[1][1], LINES[2][1]]
    assert list(texts[[4, 0]]) == [LINES[4][1], LINES[0][1]]
    assert list(texts[np.array([3, 1])]) == [LINES[3][1], LINES[1][1]]
    assert list(texts[np.array([True, False, False, False, True])]) == [LINES[0][1], LINES[4][1]]
    assert texts[np.int64(2)] == LINES[2][1]
    assert texts[[4, 0]].shape == (2,)


def test_the_split_of_the_training_data_keeps_the_texts_on_disk_and_paired(data_file):
    texts, classes = load_texts_and_classes(data_file, in_memory=False)
    x_train, y_train, x_valid, y_valid = split_train_validation(texts, classes, split_ratio=0.6)
    assert isinstance(x_train, TextsOnDisk) and isinstance(x_valid, TextsOnDisk)
    assert len(x_train) == len(y_train) == 3
    assert len(x_valid) == len(y_valid) == 2
    expected = {text: classes for _, text, classes in LINES}
    for x, y in ((x_train, y_train), (x_valid, y_valid)):
        for text, label in zip(x, y):
            assert list(label) == expected[text]


def _read(texts, index):
    return texts[index]


def test_read_from_another_process_as_the_workers_of_a_dataloader_do(data_file):
    texts, _ = load_texts_and_classes(data_file, in_memory=False)
    assert texts[0] == LINES[0][1]  # the file is open in this process
    restored = pickle.loads(pickle.dumps(texts))
    assert list(restored) == list(texts)
    with multiprocessing.get_context("fork").Pool(2) as pool:
        assert pool.starmap(_read, [(texts, i) for i in range(len(texts))]) == [text for _, text, _ in LINES]
