import os

import pytest

from .test_data import TEST_DATA_PATH


@pytest.fixture
def sample_train_file():
    return os.path.join(
        TEST_DATA_PATH,
        'test-header.train'
    )
