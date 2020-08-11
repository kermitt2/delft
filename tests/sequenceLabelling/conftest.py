import os

import pytest

from .test_data import TEST_DATA_PATH

# derived from https://github.com/elifesciences/sciencebeam-trainer-delft/tree/develop/tests

@pytest.fixture
def sample_preprocessor1():
    return os.path.join(
        TEST_DATA_PATH,
        'preprocessor.json'
    )

@pytest.fixture
def sample_preprocessor2():
    return os.path.join(
        TEST_DATA_PATH,
        'preprocessor2.json'
    )