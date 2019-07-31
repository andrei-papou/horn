from keras.models import load_model, Model
from model_saver import ModelSaver


def save_model(model, file_path):
    # type: (Model, str) -> None
    ModelSaver().save_model(model, file_path=file_path)


def convert_model(keras_model_file_path, new_model_file_path):
    # type: (str, str) -> None
    model = load_model(keras_model_file_path)
    ModelSaver().save_model(model=model, file_path=new_model_file_path)
