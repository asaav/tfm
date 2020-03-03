from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import random
from sklearn.metrics import classification_report, confusion_matrix
from models import AlexNet, fMnist, fcn_fMnist
import sys

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )


def train_test_model(hparams, dataset_train, dataset_test, dataset_val, architecture="A"):
    model = None
    if architecture == "A":
        model = AlexNet
    elif architecture == "MNIST":
        model = fMnist
    elif architecture == "FCN":
        model = fcn_fMnist

    model.summary()

    model.compile(
        optimizer="adam",
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(dataset_train, epochs=60, validation_data=dataset_val)#, verbose=0)
    Y_pred = model.predict(dataset_test)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(dataset_test.classes, y_pred))
    accuracy = model.evaluate(dataset_test, verbose=0)[1]
    print("Accuracy: ", accuracy)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams, dataset_train, dataset_test, dataset_val, sys.argv[1])
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0
seed = 42
random.seed(seed)
tf.random.set_seed(seed)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

input_size=(56,56)
batch_size=64
# load and iterate training dataset
dataset_train = datagen.flow_from_directory('dataset/train', class_mode='binary',
                                            batch_size=batch_size, target_size=input_size, seed=seed)
dataset_test = datagen.flow_from_directory('dataset/test', class_mode='binary',
                                           batch_size=batch_size, target_size=input_size, seed=seed, shuffle=False)
dataset_val = datagen.flow_from_directory('dataset/val', class_mode='binary',
                                           batch_size=batch_size, target_size=input_size, seed=seed)

for optimizer in HP_OPTIMIZER.domain.values:
     hparams = {
          HP_OPTIMIZER: optimizer
     }
     run_name = "run-%d" % session_num
     print('--- Starting trial: %s' % run_name)
     print({h.name: hparams[h] for h in hparams})
     run('logs/hparam_tuning/' + run_name, hparams)
     session_num += 1
