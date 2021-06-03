import os
import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, \
    UpSampling2D, Add, Concatenate, MaxPool2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2 as cv
import numpy as np

from obj_analyzer.segmentation.DataGenerator import DataGenerator
from obj_analyzer.segmentation.BaseModel import BaseModel

epsilon = 1e-5
smooth = 1


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def resblock(X, f):
    X_copy = X  # copy of input

    X = Conv2D(f, kernel_size=(1, 1), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)

    X_copy = Conv2D(f, kernel_size=(1, 1), kernel_initializer='he_normal')(X_copy)
    X_copy = BatchNormalization()(X_copy)

    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X


def upsample_concat(x, skip):
    X = UpSampling2D((2, 2))(x)
    merge = Concatenate()([X, skip])

    return merge


class Unet(BaseModel):
    def __init__(self, name):
        super().__init__()
        self.model_name = name
        self.model_path = f'models/{name}.hdf5'
        print(os.listdir('models'))

    def fit(self, data_path):
        model = self.create_model()

        imgs = np.array(glob.glob(os.path.join(data_path, "images", "*")))
        masks = np.array(glob.glob(os.path.join(data_path, "masks", "*")))

        data = [{'image_path': img, 'mask_path': mask} for img, mask in zip(imgs, masks)]
        X_train, X_val = train_test_split(data, test_size=0.15)
        X_test, X_val = train_test_split(X_val, test_size=0.5)

        train_ids = [item['image_path'] for item in X_train]
        train_mask = [item['mask_path'] for item in X_train]

        val_ids = [item['image_path'] for item in X_val]
        val_mask = [item['mask_path'] for item in X_val]

        earlystopping = EarlyStopping(monitor='val_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=20
                                      )

        checkpointer = ModelCheckpoint(filepath=self.model_path,
                                       verbose=1,
                                       save_best_only=True
                                       )

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=10,
                                      min_delta=0.0001,
                                      factor=0.2
                                      )

        train_data = DataGenerator(train_ids, train_mask)
        val_data = DataGenerator(val_ids, val_mask)

        h = model.fit(
            train_data,
            epochs=2,
            validation_data=val_data,
            callbacks=[checkpointer, earlystopping, reduce_lr]
        )

        model.save(self.model_path)

        return h

    def create_model(self):
        input_shape = (256, 256, 3)
        X_input = Input(input_shape)  # iniating tensor of input shape

        # Stage 1
        conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
        conv_1 = BatchNormalization()(conv_1)
        pool_1 = MaxPool2D((2, 2))(conv_1)

        # stage 2
        conv_2 = resblock(pool_1, 32)
        pool_2 = MaxPool2D((2, 2))(conv_2)

        # Stage 3
        conv_3 = resblock(pool_2, 64)
        pool_3 = MaxPool2D((2, 2))(conv_3)

        # Stage 4
        conv_4 = resblock(pool_3, 128)
        pool_4 = MaxPool2D((2, 2))(conv_4)

        # Stage 5 (bottle neck)
        conv_5 = resblock(pool_4, 256)

        # Upsample Stage 1
        up_1 = upsample_concat(conv_5, conv_4)
        up_1 = resblock(up_1, 128)

        # Upsample Stage 2
        up_2 = upsample_concat(up_1, conv_3)
        up_2 = resblock(up_2, 64)

        # Upsample Stage 3
        up_3 = upsample_concat(up_2, conv_2)
        up_3 = resblock(up_3, 32)

        # Upsample Stage 4
        up_4 = upsample_concat(up_3, conv_1)
        up_4 = resblock(up_4, 16)

        # final output
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same', activation='sigmoid')(up_4)

        seg_model = Model(X_input, out)

        self.model = seg_model
        self.compile()
        return seg_model

    def compile(self):
        adam = tf.keras.optimizers.Adam(lr=0.05, epsilon=0.1)
        self.model.compile(optimizer=adam,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=[]
                           )

    def predict(self, img):
        X = np.empty((1, 256, 256, 3))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)
        img -= img.mean()
        img /= img.std()
        X[0,] = img
        prd = self.model.predict(X)
        mask = np.array(prd).squeeze().round()
        return mask
