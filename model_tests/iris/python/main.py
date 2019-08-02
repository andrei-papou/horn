from horn.test_model_utils import handle_cli_command, shut_the_logging_up
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers.advanced_activations import Softmax
from keras.optimizers import Adam


shut_the_logging_up()


iris_data = load_iris()  # load the iris dataset

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)  # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


def get_model() -> Model:
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

    return model


if __name__ == '__main__':
    handle_cli_command(
        model_name='iris',
        model_getter=get_model,
        xs=test_x,
        ys=test_y,
    )
