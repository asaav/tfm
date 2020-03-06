import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import comargs
from tensorboard.plugins.hparams import api as hp
import random
from sklearn.metrics import classification_report, confusion_matrix
import models
import sys
from numpy.random import seed

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
results = []
session_num = 0
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

args = comargs.classifier_train_args()
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

batch_size = args.batch
input_size = (args.input, args.input)
# load datasets
dataset_train = datagen.flow_from_directory(args.dataset + 'train', class_mode='binary',
                                            batch_size=batch_size, target_size=input_size, seed=seed)
dataset_val = datagen.flow_from_directory(args.dataset + 'val', class_mode='binary',
                                        batch_size=batch_size, target_size=input_size, seed=seed)
#test dataset is unshuffled to keep order for confusion matrix calculations
dataset_test = datagen.flow_from_directory(args.dataset + 'test', class_mode='binary',
                                        batch_size=batch_size, target_size=input_size, seed=seed, shuffle=False)

def train_test_model(hparams):
    
    architecture = args.architecture
    
    # Two different models in the case of FCN because we need a reshape layer for training
    train_model = None
    ev_model = None
    if architecture == "AlexNet":
        train_model = models.get_AlexNet()
        ev_model = train_model
    elif architecture == "MNIST":
        train_model = models.get_fMnist()
        ev_model = train_model
    elif architecture == "FCN":
        train_model, ev_model = models.test()

    if args.summarize:
        train_model.summary()

    train_model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    train_model.fit(dataset_train, epochs=60, validation_data=dataset_val)#, verbose=0)
    Y_pred = train_model.predict(dataset_test)
    y_pred = [np.argmax(p) for p in Y_pred]
    print('Confusion Matrix')
    print(confusion_matrix(dataset_test.classes, y_pred))
    accuracy = train_model.evaluate(dataset_test, verbose=0)[1]

    # zero = np.zeros((1, 58,58, 3))
    # print(ev_model.predict(zero))
    print("Accuracy: ", accuracy)
    return ev_model, accuracy

def run(hparams):
    model, acc = train_test_model(hparams)
    results.append({"model": model, "accuracy": acc})


def main():
    session_num = 0
    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_OPTIMIZER: optimizer
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(hparams)
        session_num += 1
    print("-------RESULTS-------")
    for i, elem in enumerate(results):
        print("Model ", i+1, " accuracy: ", elem['accuracy'])
    print("Best model is ", 
            np.argmax([elem['accuracy'] for elem in results])+1)
    if comargs.query_yes_no("Save any model?"):
        while True:
            choice = int(input("Which model to save? "))
            if (choice > 0 and choice < len(results)):
                results[choice-1]['model'].save("models/" + args.architecture + ".h5")
                break

if __name__ == "__main__":
    main()
