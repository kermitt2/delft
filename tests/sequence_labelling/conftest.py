import os

import pytest

from .test_data import TEST_DATA_PATH


@pytest.fixture
def preprocessor1():
    return os.path.join(
        TEST_DATA_PATH,
        'preprocessor.json'
    )


@pytest.fixture
def preprocessor2():
    return os.path.join(
        TEST_DATA_PATH,
        'preprocessor2.json'
    )

