import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, Input

def get_AlexNet():
    return Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same",
            activation="relu", input_shape=(227, 227, 3)),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", 
            activation="relu"),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
            activation="relu"),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
            activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
            activation="relu"),
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
        Conv2D(256,(3,3),activation="relu", padding="same"),
        Conv2D(256,(3,3),activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2,2), padding="valid"),
        # Conv2D(512,(3,3),activation="relu", padding="same"),
        # Conv2D(512,(3,3),activation="relu", padding="same"),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(10, activation="softmax"),
    ])

def get_fcnfMnist():
    return Sequential([
        Conv2D(64, (5, 5), padding="same",
            activation="relu", input_shape=(None, None, 3)),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),
        Conv2D(256,(3,3),activation="relu", padding="same"),
        Conv2D(256,(3,3),activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2,2), padding="valid"),
        Conv2D(256,(13,13),activation="relu", padding="valid"),
        Dropout(0.4),
        Conv2D(1,(1,1),activation="sigmoid", padding="valid"),
        Reshape((1,))
    ])

def test():
    entrada=Input(shape=(None, None, 3))
    x=Conv2D(64, (5, 5), padding="same", activation="relu")(entrada)
    x=MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    x=Conv2D(256,(3,3),activation="relu", padding="same")(x)
    x=Conv2D(256,(3,3),activation="relu", padding="same")(x)
    x=MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x=Conv2D(256,(13,13),activation="relu", padding="valid")(x)
    x=Dropout(0.4)(x)
    res=Conv2D(1,(1,1),activation="sigmoid", padding="valid")(x)
    training_res = Reshape((1,))(res)
    
    return (Model(entrada, [training_res]), Model(entrada, [res]))
    