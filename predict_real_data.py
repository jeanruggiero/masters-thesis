import os
import logging
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
import boto3

import numpy as np

from real_data_wrangling.pipelines import preprocess_real_data
from modeling.metrics import f1_score_post_epoch, precision_post_epoch, recall_post_epoch
from modeling.modeling import expand_dim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure logging
fh = logging.FileHandler("prediction_remove.log")
fh.setLevel(logging.INFO)

fhv = logging.FileHandler("prediction_remove_verbose.log")
fhv.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[fh, fhv, sh])


def predict_real(experiment_name, X_test, y_test, model=None):

    logging.info('.........................................................................................')
    logging.info("Generating predictions for experiment: " + experiment_name)

    if model is None:
        alpha = 0.05
        model = keras.models.Sequential([
            # keras.layers.InputLayer(shape=[144, 480, 1]),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(0.2),
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1)),
            keras.layers.Conv2D(filters=20, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(0.2),
                                activation='relu', padding='same'),
            keras.layers.Conv2D(filters=20, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(0.2),
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            keras.layers.Conv2D(filters=25, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(0.2),
                                activation='relu', padding='same'),
            keras.layers.Conv2D(filters=25, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(0.2),
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile()

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(name="jean-masters-thesis")

        # Get all objects pertaining to specified model
        objs = bucket.objects.filter(Prefix='models/' + experiment_name + '.')

        for obj in objs:
            logging.info(f"Downloading {obj.key}")
            bucket.download_file(obj.key, obj.key)

        # Restore the weights
        model.load_weights('models/' + experiment_name)

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # logging.debug("y_test, y_pred, y_pred_proba")
    # for yt, yp, ypp in zip(y_test, y_pred, y_pred_proba):
    #     logging.debug(f"{yt:5.3f}  {yp:5.3f}  {ypp:5.3f}")

    logging.info(f"f1-score = {f1_score_post_epoch(y_test, y_pred_proba):.2f}")
    logging.info(f"precision = {precision_post_epoch(y_test, y_pred_proba):.2f}")
    logging.info(f"recall = {recall_post_epoch(y_test, y_pred_proba):.2f}")


def load_real_data(balance='bootstrap', cached=True):
    # balance can either be bootstrap, remove, or None

    if cached:
        # Read cached data
        logging.info("Reading cached y_test, X_test")
        y_test = np.loadtxt('realdata/y_test', dtype=np.int32)
        X_test = expand_dim(np.array([np.loadtxt(fname) for fname in glob.glob('realdata/x_*')]))

    else:
        # Load & cache data
        logging.info("Loading y_test, X_test from s3")
        X_test, y_test = preprocess_real_data('thesis_real_data_labels.csv', 'real_data_metadata.csv')
        X_test = np.transpose(expand_dim(X_test), axes=(0, 2, 1, 3))

        np.savetxt('realdata/y_test', y_test)

        for i, x in enumerate(X_test):
            np.savetxt(f'realdata/x_{i}', x[:,:,0])

    logging.info("Preprocessing complete")
    logging.info(f"X.shape = {X_test.shape}")
    logging.info(f"y.shape = {y_test.shape}")

    n_samples = y_test.shape[0]
    n_positive = np.sum(y_test)
    n_negative = n_samples - np.sum(y_test)

    logging.info(f"\nTotal samples: {n_samples}")
    logging.info(f"Total positive: {n_positive}")
    logging.info(f"Total negative: {n_negative}")

    logging.info('Balancing dataset.')

    if balance:
        # Get indices for negative & positive samples
        negatives = np.flatnonzero(y_test == 0)
        positives = np.flatnonzero(y_test)

        if n_positive > n_negative:
            # More positive than negative scans.
            diff = n_positive - n_negative


            if balance == 'bootstrap':
                # Randomly sample (with replacement) from negative indices and add scans to the dataset
                new_negatives = np.random.choice(negatives, diff)
                logging.debug(f"new_negatives = {new_negatives}")

                logging.debug(f"X_test[new_negatives].shape = {X_test[new_negatives].shape}")

                # Select new negatives and concatenate with existing X
                X_test = np.concatenate([X_test, X_test[new_negatives]])
                y_test = np.concatenate([y_test, y_test[new_negatives]])

            elif balance == 'remove':
                # Randomly remove positives
                old_positives = np.random.choice(positives, diff, replace=False)

                logging.debug(f"old_positives = {old_positives}")

                X_test = np.delete(X_test, old_positives, axis=0)
                y_test = np.delete(y_test, old_positives)

        elif n_negative > n_positive:
            # More negative than positive scans.
            diff = n_negative - n_positive

            if balance == 'bootstrap':
                # Randomly sample (with replacement) from positive indices
                new_positives = np.random.choice(positives, diff)
                logging.debug(f"new_positives = {new_positives}")

                # Select new positives and concatenate with existing X
                X_test = np.concatenate([X_test, X_test[new_positives]])
                y_test = np.concatenate([y_test, y_test[new_positives]])

            elif balance == 'remove':
                # Randomly remove negatives
                old_negatives = np.random.choice(negatives, diff, replace=False)

                X_test = np.delete(X_test, old_negatives, axis=0)
                y_test = np.delete(y_test, old_negatives)

    n_samples = y_test.shape[0]
    n_positive = np.sum(y_test)
    n_negative = n_samples - np.sum(y_test)

    logging.info(f"\nTotal samples: {n_samples}")
    logging.info(f"Total positive: {n_positive}")
    logging.info(f"Total negative: {n_negative}")
    logging.info(f"X_test.shape = {X_test.shape}")

    return X_test, y_test

if __name__ == '__main__':

    X_test, y_test = load_real_data(cached=True, balance='remove')

    experiment_names = [
        'experiment1_balanced', 'experiment2_balanced_n_10', 'experiment4_balanced', 'experiment5_balanced_5_99',
        'experiment7_balanced_n_10', 'experiment8_balanced_n_10',
        'experiment9_balanced', 'experiment10_balanced_n_10'
    ]

    for experiment_name in experiment_names:
        predict_real(experiment_name, X_test, y_test)

    # Upload training logs to s3
    s3_client = boto3.client('s3')
    s3_client.upload_file("prediction_remove.log", 'jean-masters-thesis', f'models/prediction_remove.log')
    s3_client.upload_file("prediction_remove_verbose.log", 'jean-masters-thesis',
                          f'models/prediction_remove_verbose.log')
