from keras.models import load_model
from model_saver import ModelSaver


def main():
    ms = ModelSaver()
    model = load_model('../artifacts/model.h5')
    ms.save_model(model, '../artifacts/model.horn')


if __name__ == '__main__':
    main()
