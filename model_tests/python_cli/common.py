import os
import warnings
from argparse import _SubParsersAction, Namespace
from typing import Callable

import tensorflow as tf
from numpy import ndarray
from keras import Model
from keras.models import load_model


SubParsers = _SubParsersAction
Args = Namespace


def shut_the_logging_up():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelSpec:
    name: str
    xs: ndarray
    ys: ndarray

    @classmethod
    def get_model(cls) -> Model:
        raise NotImplementedError()


def get_artifacts_file_path(file_name: str) -> str:
    return os.path.join('../artifacts', file_name)


def get_model_file_path(model_name: str) -> str:
    return get_artifacts_file_path('{}.model'.format(model_name))


def get_keras_model_file_path(model_name: str) -> str:
    return get_artifacts_file_path('{}.h5'.format(model_name))


def get_data_file_path(model_name: str, data_name: str) -> str:
    return get_artifacts_file_path('{}.{}.data'.format(model_name, data_name))


def get_model(model_name: str, model_getter: Callable[[], Model]) -> Model:
    keras_model_path = get_keras_model_file_path(model_name)
    if os.path.exists(keras_model_path) and os.path.isfile(keras_model_path):
        model = load_model(keras_model_path)
        print('Loaded model from {}'.format(keras_model_path))
    else:
        print('Model not found in {}. Building and training...'.format(keras_model_path))
        model = model_getter()
        model.save(keras_model_path)
    return model
