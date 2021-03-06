from enum import Enum
from typing import List, Dict, Callable, Optional, AnyStr, Union

import numpy as np
from keras import Model, backend, activations
from keras.layers import Layer, Dense, Activation, Conv2D, Convolution2D, MaxPool2D, MaxPooling2D, AvgPool2D, \
    AveragePooling2D, Flatten
from keras.layers.advanced_activations import Softmax
from keras.models import load_model

from horn.binary_format import encode_model, encode_tensor
from horn.exceptions import NotSupportedException
from horn.validation import validate_padding, validate_data_format


def get_layer_type(layer):
    # type: (Layer) -> str
    return layer.__class__.__name__


class LayerType(Enum):
    DENSE = u'Dense'
    SIGMOID = u'Sigmoid'
    SOFTMAX = u'Softmax'
    TANH = u'Tanh'
    RELU = u'Relu'
    CONV2D = u'Conv2D'
    MAX_POOL2D = u'MaxPool2D'
    AVG_POOL2D = u'AvgPool2D'
    FLATTEN = u'Flatten'

    @classmethod
    def get_for_layer(cls, layer):
        # type: (Layer) -> Optional[LayerType]
        for option in cls:
            if option.value == get_layer_type(layer):
                return option

    @classmethod
    def get_for_activation_func_name(cls, func_name):
        # type: (str) -> Optional[LayerType]
        if func_name == activations.softmax.__name__:
            return cls.SOFTMAX
        elif func_name == activations.sigmoid.__name__:
            return cls.SIGMOID
        elif func_name == activations.tanh.__name__:
            return cls.TANH
        elif func_name == activations.relu.__name__:
            return cls.RELU

    @classmethod
    def get_for_activation_func(cls, func):
        # type: (Callable) -> Optional[LayerType]
        return cls.get_for_activation_func_name(func.__name__)


class ModelSaver(object):

    def __init__(self):
        self._weight_id = 0
        self._weights = {}  # type: Dict[int, np.ndarray]

    def _get_weight_id(self):
        # type: () -> int
        self._weight_id += 1
        return self._weight_id

    def _save_weight(self, w):
        # type: (np.ndarray) -> int
        wid = self._get_weight_id()
        self._weights[wid] = w
        return wid

    def _save_activation_func(self, name, func):
        # type: (AnyStr, Callable) -> Dict
        layer_type = LayerType.get_for_activation_func(func)
        if layer_type is None:
            raise NotSupportedException('Activation function "{}" is not supported.'.format(func.__name__))
        return {
            u'name': name,
            u'type': layer_type.value,
        }

    def _save_activation_layer(self, layer):
        # type: (Activation) -> Dict
        layer_type = LayerType.get_for_activation_func_name(layer.activation)
        if layer_type is None:
            raise NotSupportedException('Activation function "{}" is not supported.'.format(layer.activation))
        return {
            u'name': layer.name,
            u'type': layer_type.value,
        }

    def _save_softmax_layer(self, layer):
        # type: (Softmax) -> Dict
        return {
            u'name': layer.name,
            u'type': LayerType.SOFTMAX.value,
            u'axis': layer.axis if layer.axis != -1 else None
        }

    def _save_dense(self, layer):
        # type: (Dense) -> List[Dict]
        weights = backend.batch_get_value(layer.weights)  # type: List[np.ndarray]
        layer_dicts = [
            {
                u'name': layer.name,
                u'type': LayerType.DENSE.value,
                u'w_shape': weights[0].shape,
                u'w_id': self._save_weight(weights[0]),
                u'b_shape': weights[1].shape if layer.use_bias else None,
                u'b_id': self._save_weight(weights[1]) if layer.use_bias else None,
            }
        ]
        if layer.activation is not None and layer.activation is not activations.linear:
            layer_dicts.append(self._save_activation_func(
                name=u'{}__activation'.format(layer.name),
                func=layer.activation
            ))
        return layer_dicts

    def _save_conv2d(self, layer):
        # type: (Union[Conv2D, Convolution2D]) -> List[Dict]
        validate_padding(layer.padding)
        validate_data_format(layer.data_format)

        weights = backend.batch_get_value(layer.weights)  # type: List[np.ndarray]
        layer_dicts = [
            {
                u'name': layer.name,
                u'type': LayerType.CONV2D.value,
                u'f_shape': weights[0].shape,
                u'f_id': self._save_weight(weights[0]),
                u'b_shape': weights[1].shape if layer.use_bias else None,
                u'b_id': self._save_weight(weights[1]) if layer.use_bias else None,
                u'strides': layer.strides,
                u'padding': layer.padding,
                u'data_format': layer.data_format,
            }
        ]
        if layer.activation is not None and layer.activation is not activations.linear:
            layer_dicts.append(self._save_activation_func(
                name=u'{}__activaion'.format(layer.name),
                func=layer.activation
            ))
        return layer_dicts

    def _save_pool2d(self, layer, layer_type):
        # type: (Union[MaxPool2D, MaxPooling2D, AvgPool2D, AveragePooling2D], str) -> Dict
        validate_padding(layer.padding)
        validate_data_format(layer.data_format)

        return {
            u'name': layer.name,
            u'type': layer_type,
            u'pool_window': layer.pool_size,
            u'strides': layer.pool_size,
            u'padding': layer.padding,
            u'data_format': layer.data_format,
        }

    def _save_flatten(self, layer):
        # type: (Flatten) -> Dict
        return {
            u'name': layer.name,
            u'type': LayerType.FLATTEN.value,
        }

    def save_model(self, model, file_path):
        # type: (Model, str) -> None
        layer_dicts = []
        layers = model.layers  # type: List[Layer]

        for l in layers:
            lt = get_layer_type(l)

            if lt == Dense.__name__:
                layer_dicts.extend(self._save_dense(l))
            elif lt == Activation.__name__:
                layer_dicts.append(self._save_activation_layer(l))
            elif lt == Softmax.__name__:
                layer_dicts.append(self._save_softmax_layer(l))
            elif lt == Conv2D.__name__ or lt == Convolution2D.__name__:
                layer_dicts.extend(self._save_conv2d(l))
            elif lt == MaxPool2D.__name__ or lt == MaxPooling2D.__name__:
                layer_dicts.append(self._save_pool2d(l, LayerType.MAX_POOL2D.value))
            elif lt == AvgPool2D.__name__ or lt == AveragePooling2D.__name__:
                layer_dicts.append(self._save_pool2d(l, LayerType.AVG_POOL2D.value))
            elif lt == Flatten.__name__:
                layer_dicts.append(self._save_flatten(l))

        spec = {
            u'name': model.name,
            u'layers': layer_dicts
        }
        model_repr = encode_model(spec, self._weights)

        with open(file_path, 'wb') as f:
            f.write(model_repr)


def save_model(model, file_path):
    # type: (Model, str) -> None
    ModelSaver().save_model(model, file_path=file_path)


def convert_model(keras_model_file_path, new_model_file_path):
    # type: (str, str) -> None
    model = load_model(keras_model_file_path)
    ModelSaver().save_model(model=model, file_path=new_model_file_path)


def save_tensor(tensor, file_path):
    # type: (np.ndarray, str) -> None
    with open(file_path, 'wb') as f:
        f.write(encode_tensor(tensor))
