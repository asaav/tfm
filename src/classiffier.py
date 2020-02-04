from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import random

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100, 150, 200, 250, 300, 350]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
     hp.hparams_config(
     hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
     metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
     )


def train_test_model(hparams, dataset_train, dataset_test):
     model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(56, 56, 3)),
          tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
          tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
          tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
          tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
          #tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
          tf.keras.layers.Dense(2, activation=tf.nn.softmax),
     ])
     model.compile(
          optimizer=hparams[HP_OPTIMIZER],
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'],
     )

     model.fit(dataset_train, epochs=60, verbose=0)
     _, accuracy = model.evaluate(dataset_test, verbose=1)
     return accuracy

def run(run_dir, hparams):
     with tf.summary.create_file_writer(run_dir).as_default():
          hp.hparams(hparams)  # record the values used in this trial
          accuracy = train_test_model(hparams, dataset_train, dataset_test)
          tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0
seed = 42
random.seed(seed)
tf.random.set_seed(seed)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# load and iterate training dataset
dataset_train = datagen.flow_from_directory('dataset/train', class_mode='binary',
     batch_size=25, target_size=(56,56), seed=seed)
dataset_test = datagen.flow_from_directory('dataset/test', class_mode='binary',
     batch_size=25, target_size=(56,56), seed=seed)

# image_batch, label_batch = next(dataset_train)
# show_batch(image_batch, label_batch)

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(56, 56, 3)),
#   tf.keras.layers.Dense(300, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(300, activation='relu'),
#   tf.keras.layers.Dense(200, activation='relu'),
#   tf.keras.layers.Dense(2, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(dataset_train, epochs=25)
# model.evaluate(dataset_test)


for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1


