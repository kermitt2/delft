import os
from unittest.mock import MagicMock, patch

from delft.utilities.Utilities import download_file

URL = "https://example.org/files/vectors.bin"


def _response(chunks, status_code=200, content_length=None):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {} if content_length is None else {"content-length": str(content_length)}
    response.iter_content.side_effect = lambda chunk_size: iter(chunks)
    return response


def _download(tmp_path, response):
    with patch("delft.utilities.Utilities.requests.get", return_value=response):
        return download_file(URL, str(tmp_path))


def test_with_a_content_length(tmp_path):
    path = _download(tmp_path, _response([b"abc", b"def"], content_length=6))
    assert path == str(tmp_path / "vectors.bin")
    assert (tmp_path / "vectors.bin").read_bytes() == b"abcdef"


def test_without_a_content_length(tmp_path):
    """The file was written, and the download reported as failed."""
    path = _download(tmp_path, _response([b"abc", b"def"]))
    assert path == str(tmp_path / "vectors.bin")
    assert (tmp_path / "vectors.bin").read_bytes() == b"abcdef"


def test_an_error_status_downloads_nothing(tmp_path):
    assert _download(tmp_path, _response([b"Not Found"], status_code=404)) is None
    assert os.listdir(tmp_path) == []


def test_a_download_that_fails_half_way_leaves_nothing(tmp_path):
    def chunks(chunk_size):
        yield b"abc"
        raise ConnectionError("connection lost")

    response = _response([])
    response.iter_content.side_effect = chunks
    assert _download(tmp_path, response) is None
    assert os.listdir(tmp_path) == []


def test_a_destination_that_is_not_a_directory(tmp_path):
    with patch("delft.utilities.Utilities.requests.get") as get:
        assert download_file(URL, str(tmp_path / "missing")) is None
    get.assert_not_called()
