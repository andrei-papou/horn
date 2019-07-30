import json
import sys
from typing import Dict, Optional, Tuple, List

import numpy as np


class DataEncoder:
    """
    Data format spec:

    | {BYTES_PER_ENTRY_SIZE} bytes | JSON spec of the model encoded as UTF-8 string |
    --- 0 or more ---
    | {BYTES_PER_TENSOR_KEY} bytes | {BYTES_PER_ENTRY_SIZE} bytes | Flattened weight tensor (as f64 data type) |
    -----------------
    """

    BYTES_PER_TENSOR_KEY = 2
    BYTES_PER_ENTRY_SIZE = 4
    TENSOR_DATA_TYPE = 'float64'

    class EncodeError(Exception):
        pass

    def __init__(self):
        self._header_entry: Optional[Dict] = None
        self._tensor_entries: List[Tuple[int, np.ndarray]] = []

    def _prepend_entry_size(self, b: bytes) -> bytes:
        return len(b).to_bytes(self.BYTES_PER_ENTRY_SIZE, sys.byteorder) + b

    def add_header_entry(self, header: Dict):
        self._header_entry = header

    def add_tensor_entry(self, te: Tuple[int, np.ndarray]):
        self._tensor_entries.append(te)

    def encode(self) -> bytes:
        if self._header_entry is None:
            raise self.EncodeError('Header entry should be provided before encoding')

        result = self._prepend_entry_size(bytes(json.dumps(self._header_entry).encode('utf8')))
        for key, ten in self._tensor_entries:
            result += key.to_bytes(self.BYTES_PER_TENSOR_KEY, sys.byteorder)
            result += self._prepend_entry_size(ten.astype(self.TENSOR_DATA_TYPE).flatten().tobytes())
        return result
