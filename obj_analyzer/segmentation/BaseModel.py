import tensorflow as tf


class BaseModel:
    def __init__(self):
        self.model = None
        self.model_path = None

    def create_model(self):
        pass

    def get_model(self):
        if self.model is None:
            return self.load_model()
        else:
            return self.model

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model = model

        return model

    def save_model(self):
        model_json = self.model.to_json()
        with open(self.model_path, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(self.model_path)

    def predict(self, img):
        pass
