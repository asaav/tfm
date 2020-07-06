import os

import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import comargs
import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

args = comargs.classifier_train_args()

train_datagen = ImageDataGenerator(rescale=1./255, vertical_flip=True,
                                   rotation_range=60, width_shift_range=.2,
                                   height_shift_range=.2, zoom_range=0.5)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = args.batch

# Set up input size depending on chosen architecture
if(args.architecture == "AlexNet"):
    input_size = (227, 227)
else:
    input_size = (56, 56)

# load datasets
dataset_train = train_datagen.flow_from_directory(
    args.dataset + 'train',
    class_mode='binary', batch_size=batch_size,
    target_size=input_size, seed=seed
)
dataset_val = test_datagen.flow_from_directory(
    args.dataset + 'val',
    class_mode='binary', batch_size=batch_size,
    target_size=input_size, seed=seed
)
# test dataset is unshuffled to keep order for confusion matrix calculations
dataset_test = test_datagen.flow_from_directory(
    args.dataset + 'test',
    class_mode='binary', batch_size=batch_size,
    target_size=input_size, seed=seed, shuffle=False
)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    log_dir = os.path.join(".", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, run_id)


def train_test_model():

    architecture = args.architecture

    # Two different models in the case of FCN
    # because we need a reshape layer for training
    train_model = None
    ev_model = None
    if architecture == "AlexNet":
        train_model = models.get_AlexNet()
        ev_model = train_model
    elif architecture == "MNIST":
        train_model = models.get_fMnist()
        ev_model = train_model
    elif architecture == "FCN":
        train_model, ev_model = models.get_fcnfMnist()

    if args.summarize:
        # Print summary if arg if needed
        train_model.summary()

    # Set up tensorboard callbacks
    run_logdir = get_run_logdir()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

    # Compile and train model with EarlyStopping
    train_model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50, min_delta=.001,
        restore_best_weights=True
    )

    train_model.fit(dataset_train, epochs=1000, validation_data=dataset_val,
                    callbacks=[early_stopping, tensorboard_cb])

    # Print confusion matrix after training
    Y_pred = train_model.predict(dataset_test)
    Y_pred = Y_pred.reshape(Y_pred.shape[0])
    y_pred = [round(p) for p in Y_pred]
    print('Confusion Matrix')
    print(confusion_matrix(dataset_test.classes, y_pred))
    accuracy = train_model.evaluate(dataset_test, verbose=0)[1]

    print("Accuracy: ", accuracy)
    return ev_model, accuracy


def main():
    model, acc = train_test_model()
    if comargs.query_yes_no("Save model?"):
        model.save("models/" + args.architecture + ".h5")


if __name__ == "__main__":
    main()
