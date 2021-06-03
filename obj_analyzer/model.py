import tensorflow as tf
import os

from obj_analyzer.segmentation import Unet

MODELS_PATH = './models'


def add_model(model_name, data_path, model_path=None):
    sgm = Unet(model_name)
    sgm.fit(data_path)


def get_model_by_name(name):
    sgm = Unet(name)
    sgm.get_model()
    return sgm


def load_model(name, model_path=MODELS_PATH):
    model = tf.keras.models.load_model(os.path.join(model_path, f"{name}.hdf5"), compile=False)
    return model


def save_model(model, name, model_path=MODELS_PATH):
    model.save_weights(os.path.join(model_path, f"{name}.hdf5"))


def get_all_models():
    return {'models': [name[:-5] for name in os.listdir(MODELS_PATH)]}
