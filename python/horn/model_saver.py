from enum import Enum
from typing import List, Dict, Callable, Optional

import numpy as np
from keras import Model, backend, activations
from keras.layers import Layer, Dense, Activation
from keras.layers.advanced_activations import Softmax

from data_encoder import DataEncoder
from exceptions import NotSupportedException


def get_layer_type(layer: Layer) -> str:
    return layer.__class__.__name__


class LayerType(Enum):
    DENSE = 'Dense'
    SIGMOID = 'Sigmoid'
    SOFTMAX = 'Softmax'
    TANH = 'Tanh'
    RELU = 'Relu'

    @classmethod
    def get_for_layer(cls, layer: Layer) -> 'Optional[LayerType]':
        for option in cls:
            if option.value == get_layer_type(layer):
                return option

    @classmethod
    def get_for_activation_func_name(cls, func_name: str) -> 'Optional[LayerType]':
        if func_name == activations.softmax.__name__:
            return cls.SOFTMAX
        elif func_name == activations.sigmoid.__name__:
            return cls.SIGMOID
        elif func_name == activations.tanh.__name__:
            return cls.TANH
        elif func_name == activations.relu.__name__:
            return cls.RELU

    @classmethod
    def get_for_activation_func(cls, func: Callable) -> 'Optional[LayerType]':
        return cls.get_for_activation_func_name(func.__name__)


class ModelSaver:

    def __init__(self):
        self._weight_id = 0
        self._data_encoder = DataEncoder()

    def _get_weight_id(self) -> int:
        self._weight_id += 1
        return self._weight_id

    def _save_weight(self, w: np.ndarray) -> int:
        wid = self._get_weight_id()
        self._data_encoder.add_tensor_entry((wid, w))
        return wid

    def _save_activation_func(self, name: str, func: Callable) -> Dict:
        layer_type = LayerType.get_for_activation_func(func)
        if layer_type is None:
            raise NotSupportedException(f'Activation function "{func.__name__}" is not supported.')
        return {
            'name': name,
            'type': layer_type.value,
        }

    def _save_activation_layer(self, layer: Activation) -> Dict:
        layer_type = LayerType.get_for_activation_func_name(layer.activation)
        if layer_type is None:
            raise NotSupportedException(f'Activation function "{layer.activation}" is not supported.')
        return {
            'name': layer.name,
            'type': layer_type.value,
        }

    def _save_softmax_layer(self, layer: Softmax) -> Dict:
        return {
            'name': layer.name,
            'type': LayerType.SOFTMAX.value,
            'axis': layer.axis if layer.axis != -1 else None
        }

    def _save_dense(self, layer: Dense) -> List[Dict]:
        weights: List[np.ndarray] = backend.batch_get_value(layer.weights)
        layer_dicts = [
            {
                'name': layer.name,
                'type': LayerType.DENSE.value,
                'w_shape': weights[0].shape,
                'w_id': self._save_weight(weights[0]),
                'b_shape': weights[1].shape if layer.use_bias else None,
                'b_id': self._save_weight(weights[1]) if layer.use_bias else None,
            }
        ]
        if layer.activation is not None and layer.activation is not activations.linear:
            layer_dicts.append(self._save_activation_func(name=f'{layer.name}__activation', func=layer.activation))
        return layer_dicts

    def save_model(self, model: Model, file_path: 'str'):
        layer_dicts = []
        layers: List[Layer] = model.layers

        for l in layers:
            lt = get_layer_type(l)

            if lt == Dense.__name__:
                layer_dicts.extend(self._save_dense(l))
            elif lt == Activation.__name__:
                layer_dicts.append(self._save_activation_layer(l))
            elif lt == Softmax.__name__:
                layer_dicts.append(self._save_softmax_layer(l))

        self._data_encoder.add_header_entry({
            'name': model.name,
            'layers': layer_dicts
        })

        with open(file_path, 'wb') as f:
            f.write(self._data_encoder.encode())
