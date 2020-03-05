import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import random
from sklearn.metrics import classification_report, confusion_matrix
import models
import sys
from numpy.random import seed

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

def train_test_model(hparams, dataset_train, dataset_test, dataset_val, architecture="A"):
    model = None
    if architecture == "A":
        model = models.get_AlexNet()
    elif architecture == "MNIST":
        model = models.get_fMnist()
    elif architecture == "FCN":
        model = models.get_fcnfMnist()

    model.summary()

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(dataset_train, epochs=60, validation_data=dataset_val)#, verbose=0)
    Y_pred = model.predict(dataset_test)
    y_pred = [np.argmax(p) for p in Y_pred]
    print('Confusion Matrix')
    print(confusion_matrix(dataset_test.classes, y_pred))
    accuracy = model.evaluate(dataset_test, verbose=0)[1]
    print("Accuracy: ", accuracy)
    return accuracy

def run(hparams):
    train_test_model(hparams, dataset_train, dataset_test, dataset_val, sys.argv[1])


session_num = 0
seed = 42
np.random.seed(seed)
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
     run(hparams)
     session_num += 1
