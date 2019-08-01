import json
import struct
import sys

from typing import Dict
import numpy as np


SWAP_NEEDED = sys.byteorder != 'big'
TENSOR_DATA_TYPE = 'float64'


def _prepend_bytes_size(b):
    # type: (bytes) -> bytes
    return struct.pack('>L', len(b)) + b


def encode_json(jsn):
    # type: (Dict) -> bytes
    jsn_repr = json.dumps(jsn).encode('utf8')
    return _prepend_bytes_size(jsn_repr)


def encode_tensor_id(tid):
    # type: (int) -> bytes
    return struct.pack('>H', tid)


def encode_tensor_flat(tensor):
    # type: (np.ndarray) -> bytes
    tensor = tensor.astype(TENSOR_DATA_TYPE)
    tensor = tensor.byteswap().newbyteorder('>') if SWAP_NEEDED else tensor
    tensor_repr = tensor.flatten().tobytes()
    return _prepend_bytes_size(tensor_repr)


def encode_tensor(tensor):
    # type: (np.ndarray) -> bytes
    shape_repr = bytes()
    for d in tensor.shape:
        shape_repr += struct.pack('>L', d)
    shape_repr = _prepend_bytes_size(shape_repr)
    return shape_repr + encode_tensor_flat(tensor)


def encode_model(spec, weights):
    # type: (Dict, Dict[int, np.ndarray]) -> bytes
    """
    Model binary format spec:

    | {BYTES_PER_ENTRY_SIZE} bytes | JSON spec of the model encoded as UTF-8 string |
    --- 0 or more ---
    | {BYTES_PER_TENSOR_KEY} bytes | {BYTES_PER_ENTRY_SIZE} bytes | Flattened weight tensor (as f64 data type) |
    -----------------
    """
    model_repr = encode_json(spec)
    for k, w in weights.items():
        model_repr += encode_tensor_id(k)
        model_repr += encode_tensor_flat(w)
    return model_repr
