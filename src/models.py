from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, Reshape)
from tensorflow.keras.models import Model, Sequential


def get_AlexNet():
    return Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
               padding="same", activation="relu", input_shape=(227, 227, 3)),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
               padding="same", activation="relu"),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
               padding="same", activation="relu"),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
               padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
               padding="same", activation="relu"),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.4),
        Dense(4096, activation="relu"),
        Dropout(0.4),
        Dense(2, activation="softmax")
    ])


def get_fMnist():
    return Sequential([
        Conv2D(64, (5, 5), padding="same",
               activation="relu", input_shape=(56, 56, 3)),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2), padding="valid"),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(2, activation="softmax"),
    ])


def get_fcnfMnist():
    entrada = Input(shape=(None, None, 3))
    x = Conv2D(64, (5, 5), padding="same", activation="relu")(entrada)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="valid")(x)
    x = Conv2D(32, (13, 13), activation="relu", padding="valid")(x)
    #x = Dropout(0.4)(x)
    res = Conv2D(1, (1, 1), activation="sigmoid", padding="valid")(x)
    training_res = Reshape((-1, 1))(res)

    # Return two models since training metrics require flattening
    return (Model(entrada, [training_res]), Model(entrada, [res]))


# Implemented batch normalization but gives poor results
def test():
    entrada = Input(shape=(None, None, 3))
    x = Conv2D(64, (5, 5), padding="same", activation="relu")(entrada)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="valid")(x)
    x = Conv2D(256, (13, 13), activation="relu", padding="valid")(x)
    x = Dropout(0.4)(x)
    res = Conv2D(1, (1, 1), activation="sigmoid", padding="valid")(x)
    training_res = Reshape((1,))(res)

    return (Model(entrada, [training_res]), Model(entrada, [res]))
