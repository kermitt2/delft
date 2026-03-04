import pickle

import numpy as np
import pytest

from delft.utilities.Embeddings import _check_lmdb_format


class TestCheckLmdbFormat:
    def test_accepts_raw_float32(self):
        """Raw float32 bytes should pass validation without error."""
        vector = np.random.rand(300).astype(np.float32)
        raw_bytes = vector.tobytes()
        _check_lmdb_format(raw_bytes)  # should not raise

    def test_rejects_pickle(self):
        """Pickled numpy arrays should raise ValueError."""
        vector = np.zeros(300, dtype=np.float32)
        pickled = pickle.dumps(vector)
        with pytest.raises(ValueError, match="legacy pickle format"):
            _check_lmdb_format(pickled)

    def test_accepts_short_values(self):
        """Values shorter than 2 bytes should pass (edge case)."""
        _check_lmdb_format(b"")
        _check_lmdb_format(b"\x00")

    def test_no_false_positive_on_0x80(self):
        """A float32 vector that happens to start with 0x80 should not be rejected."""
        # Craft a float32 array whose first byte is 0x80 (e.g., -0.0 in IEEE 754 is 0x80000000)
        vector = np.array([-0.0] + [1.0] * 299, dtype=np.float32)
        raw_bytes = vector.tobytes()
        assert raw_bytes[0] == 0x00  # -0.0 little-endian is 00 00 00 80

        # Directly craft bytes starting with 0x80 to be sure
        crafted = bytes([0x80, 0x03]) + b"\x00" * 1198  # 0x80 + protocol 3, but no 'numpy'
        _check_lmdb_format(crafted)  # should not raise (no b'numpy' in first 50 bytes)

    def test_rejects_all_pickle_protocols(self):
        """Pickle protocols 2-5 should all be detected."""
        vector = np.zeros(300, dtype=np.float32)
        for protocol in (2, 3, 4, 5):
            pickled = pickle.dumps(vector, protocol=protocol)
            with pytest.raises(ValueError, match="legacy pickle format"):
                _check_lmdb_format(pickled)
