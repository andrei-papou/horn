import os
import sys
from time import time

from horn import save_model, save_tensor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.activations import softmax
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras.layers.advanced_activations import Softmax
from keras.optimizers import Adam


KERAS_MODEL_FILE_PATH = '../artifacts/iris.h5'
HORN_MODEL_FILE_PATH = '../artifacts/iris.model'
TEST_DATA_X_FILE_PATH = '../artifacts/x.data'
TEST_DATA_Y_FILE_PATH = '../artifacts/y.data'


def custom_axis_softmax(axis):
    def soft(arg):
        return softmax(arg, axis=axis)
    return soft


iris_data = load_iris()  # load the iris dataset

print('Example data: ')
print(iris_data.data[:5])
print('Example labels: ')
print(iris_data.target[:5])

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)  # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


def _get_model() -> Model:
    if os.path.exists(KERAS_MODEL_FILE_PATH) and os.path.isfile(KERAS_MODEL_FILE_PATH):
        model = load_model(KERAS_MODEL_FILE_PATH)
        print('Loaded model from {}'.format(model))
    else:
        print('Model not found in {}. Building and training...'.format(KERAS_MODEL_FILE_PATH))
        model = Sequential()

        model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
        model.add(Dense(10, activation='relu', name='fc2'))
        model.add(Dense(3, activation='linear', name='fc4'))
        model.add(Softmax(name='output', axis=1))

        # Adam optimizer with learning rate of 0.001
        optimizer = Adam(lr=0.001)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print('Neural Network Model Summary: ')
        print(model.summary())

        # Train the model
        model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

        # Test on unseen data

        results = model.evaluate(test_x, test_y)

        print('Final test set loss: {:4f}'.format(results[0]))
        print('Final test set accuracy: {:4f}'.format(results[1]))

        model.save(KERAS_MODEL_FILE_PATH)
    return model


def create_model():
    model = _get_model()
    save_model(model, HORN_MODEL_FILE_PATH)


def create_testing_data():
    if not os.path.exists(TEST_DATA_X_FILE_PATH) or not os.path.isfile(TEST_DATA_X_FILE_PATH):
        save_tensor(test_x, TEST_DATA_X_FILE_PATH)
    if not os.path.exists(TEST_DATA_Y_FILE_PATH) or not os.path.isfile(TEST_DATA_Y_FILE_PATH):
        save_tensor(test_y, TEST_DATA_Y_FILE_PATH)


def measure_performance():
    model = _get_model()
    start = time()
    for _ in range(1000):
        model.evaluate(test_x, test_y)
    print('Keras (Python) performance: {}'.format((time() - start) / 1000))


COMMAND_CREATE_MODEL = 'create_model'
COMMAND_CREATE_TESTING_DATA = 'create_testing_data'
COMMAND_MEASURE_PERFORMANCE = 'measure_performance'
COMMANDS_MAP = {
    COMMAND_CREATE_MODEL: create_model,
    COMMAND_CREATE_TESTING_DATA: create_testing_data,
    COMMAND_MEASURE_PERFORMANCE: measure_performance,
}


def main():
    if len(sys.argv) < 2:
        raise ValueError('Please specify the command')
    command = sys.argv[1]
    if command not in COMMANDS_MAP:
        raise ValueError('Unknown command')
    COMMANDS_MAP[command]()


if __name__ == '__main__':
    main()
