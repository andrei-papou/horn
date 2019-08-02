import os
import sys
import warnings
from time import time
from typing import Callable

import numpy as np
import tensorflow as tf
from keras import Model
from keras.models import load_model

from horn import save_model, save_tensor


COMMAND_CREATE_MODEL = 'create_model'
COMMAND_CREATE_TEST_DATA = 'create_test_data'
COMMAND_TEST_PERFORMANCE = 'test_performance'
COMMANDS = {COMMAND_CREATE_MODEL, COMMAND_CREATE_TEST_DATA, COMMAND_TEST_PERFORMANCE}


def _artifacts_file_path(file_name):
    # type: (str) -> str
    return os.path.join('../artifacts', file_name)


def _model_file_path(model_name):
    # type: (str) -> str
    return _artifacts_file_path('{}.model'.format(model_name))


def _keras_model_file_path(model_name):
    # type: (str) -> str
    return _artifacts_file_path('{}.h5'.format(model_name))


def _data_file_path(data_name):
    # type: (str) -> str
    return _artifacts_file_path('{}.data'.format(data_name))


def _get_model(model_name, model_getter):
    # type: (str, Callable[[], Model]) -> Model
    keras_model_path = _keras_model_file_path(model_name)
    if os.path.exists(keras_model_path) and os.path.isfile(keras_model_path):
        model = load_model(keras_model_path)
        print('Loaded model from {}'.format(keras_model_path))
    else:
        print('Model not found in {}. Building and training...'.format(keras_model_path))
        model = model_getter()
        model.save(keras_model_path)
    return model


def _create_model(model_name, model_getter):
    # type: (str, Callable[[], Model]) -> None
    model_path = _model_file_path(model_name)
    model = _get_model(model_name, model_getter)
    save_model(model, model_path)


def _create_test_data(xs, ys):
    xs_file_path = _data_file_path('x')
    ys_file_path = _data_file_path('y')
    if not os.path.exists(xs_file_path) or not os.path.isfile(xs_file_path):
        save_tensor(xs, xs_file_path)
    if not os.path.exists(ys_file_path) or not os.path.isfile(ys_file_path):
        save_tensor(ys, ys_file_path)


def _test_performance(model_name, model_getter, xs, ys):
    # type: (str, Callable[[], Model], np.ndarray, np.ndarray) -> None
    model = _get_model(model_name, model_getter)
    cumulative_time = 0.0
    for i in range(1000):
        start = time()
        model.evaluate(xs, ys)
        cumulative_time += time() - start

    print('Keras (Python) performance: {}'.format(cumulative_time * 1e6))


def handle_cli_command(model_name, model_getter, xs, ys):
    # type: (str, Callable[[], Model], np.ndarray, np.ndarray) -> None
    if len(sys.argv) < 2:
        raise ValueError('Please specify the command.')
    command = sys.argv[1]
    if command not in COMMANDS:
        raise ValueError('Unknown command.')
    if command == COMMAND_CREATE_MODEL:
        _create_model(model_name, model_getter)
    elif command == COMMAND_CREATE_TEST_DATA:
        _create_test_data(xs, ys)
    elif command == COMMAND_TEST_PERFORMANCE:
        _test_performance(model_name, model_getter, xs, ys)


def shut_the_logging_up():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
