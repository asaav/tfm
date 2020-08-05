import os
import re
import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import get_fcnfMnist
from comargs import hnmin_args

# Print only tensorflow error and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

input_size = (56, 56)
batch_size = 64
seed = 42


def train_test_model(model, dataset):

    train_datagen = ImageDataGenerator(rescale=1./255, vertical_flip=True,
                                       rotation_range=60, width_shift_range=.2,
                                       height_shift_range=.2, zoom_range=0.5)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # load datasets
    dataset_train = train_datagen.flow_from_directory(
        os.path.join(dataset, 'train'),
        class_mode='binary', batch_size=batch_size,
        target_size=input_size, seed=seed
    )
    dataset_val = test_datagen.flow_from_directory(
        os.path.join(dataset, 'val'),
        class_mode='binary', batch_size=batch_size,
        target_size=input_size, seed=seed
    )
    # Test dataset is unshuffled to keep order
    # for confusion matrix calculations
    dataset_test = test_datagen.flow_from_directory(
        os.path.join(dataset, 'test'),
        class_mode='binary', batch_size=batch_size,
        target_size=input_size, seed=seed, shuffle=False
    )
    train_model, ev_model = model

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
                    callbacks=[early_stopping])

    # Print confusion matrix after training
    Y_pred = train_model.predict(dataset_test)
    Y_pred = Y_pred.reshape(Y_pred.shape[0])
    y_pred = [round(p) for p in Y_pred]
    print('Confusion Matrix')
    print(confusion_matrix(dataset_test.classes, y_pred))
    accuracy = train_model.evaluate(dataset_test, verbose=0)[1]

    print("Accuracy: ", accuracy)
    return ev_model, accuracy


def continue_training(model_path, data):
    train_model, ev_model = get_fcnfMnist()
    train_model.load_weights(model_path)
    print("Model " + model_path + " successfully loaded")
    it_regex = r'(it(\d+))(.+?)\.'
    match = re.match(it_regex, os.path.basename(model_path))
    iteration = match.group(1)
    it_number = int(match.group(2))
    name = match.group(3)
    datasetPath = os.path.join(data, name, iteration)
    if not os.path.exists(datasetPath):
        sys.exit(datasetPath + " does not exist")

    model, _ = train_test_model((train_model, ev_model), datasetPath)
    model_path = os.path.join("models", f'it{it_number+1}{name}.h5')
    print("Saving model in " + model_path)
    model.save(model_path)


def main():
    args = hnmin_args()
    if not os.path.exists(args.model):
        sys.exit("Model " + args.model + " does not exist")

    if not os.path.exists(args.data):
        sys.exit(args.data + " does not exist")
    continue_training(args.model, args.data)


if __name__ == '__main__':
    main()
