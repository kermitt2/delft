import os

import pytest

from .test_data import TEST_DATA_PATH

# derived from https://github.com/elifesciences/sciencebeam-trainer-delft/tree/develop/tests

@pytest.fixture
def sample_train_file():
    return os.path.join(
        TEST_DATA_PATH,
        'test-header.train'
    )
