import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import boto3

import matplotlib.pyplot as plt

from labeling import DataSetGenerator, S3DataLoader
from preprocessing import preprocess
from modeling import train_model
from modeling.metrics import mean_jaccard_index, f1_score, mean_overlap, object_detection_f1_score, object_size_rmse, \
    object_center_rmse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

import logging
logging.basicConfig(filename='training.log', level=logging.INFO)


def scheduler(epoch, lr):
    print(f"Learning rate in previous epoch: {lr}")
    if epoch <= 3:
        return 0.001
    if epoch <= 15:
        return 0.00001
    else:
        return 0.000001


def scheduler_after_first_batch(epoch, lr):
    print(f"Learning rate in previous epoch: {lr}")
    return 0.000001


def run_model(model, name):

    # Load geometry files
    s3_client = boto3.client('s3')
    filenames = ['geometry_spec.csv', 'geometry_spec2.csv', 'geometry_spec3.csv']

    # Download geometry files from s3 if not present
    if not os.path.isfile('geometry_spec.csv'):
        for filename in filenames:
            s3_client.download_file('jean-masters-thesis', 'geometry/' + filename, filename)

    geometry_spec = pd.concat(
        [pd.read_csv(filename, index_col=0) for filename in [
            'geometry_spec.csv', 'geometry_spec2.csv', 'geometry_spec3.csv'
        ]]
    )

    # Load raw data
    loader = S3DataLoader('jean-masters-thesis', 'simulations/merged/')

    # Generate bootstrapped training set
    data_generator = DataSetGenerator(loader, geometry_spec, 20, n=50, scan_max_col=100, random_seed=42)

    # Reshaping parameters
    output_time_range = 120
    sample_rate = 10  # samples per ns

    # Training callbacks
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    lr_scheduler_after_first_batch = tf.keras.callbacks.LearningRateScheduler(scheduler_after_first_batch)
    es = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, restore_best_weights=True)

    callbacks = {
        es: None,
        # lr_scheduler: [0],
        # lr_scheduler_after_first_batch: list(range(1, 11))
    }

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        metrics=['accuracy', mean_jaccard_index, f1_score, Precision(), Recall(), mean_overlap,
                 object_detection_f1_score, object_size_rmse]
    )

    # Train model
    history, model, X_val, y_val = train_model(model, data_generator, output_time_range, sample_rate,
                                               callbacks=callbacks, epochs=5, plots=False)

    s3_client.upload_file("training.log", 'jean-masters-thesis', f'models/{name}_training.log')
    np.save_txt(f"{name}_X_val.csv", X_val, delimiter=",")
    np.save_txt(f"{name}_y_val.csv", y_val, delimiter=",")
    s3_client.upload_file(name, 'jean-masters-thesis', f'models/{name}_X_val.csv')
    s3_client.upload_file(name, 'jean-masters-thesis', f'models/{name}_y_val.csv')

    # Save model to disk & s3
    model.save(name)
    s3_client.upload_file(name, 'jean-masters-thesis', f'models/{name}.model')

    # Save history to disk
    with open(f"{name}_history.txt", 'w') as f:
        f.write(json.dumps(history))
    s3_client.upload_file(f'{name}_history.txt', 'jean-masters-thesis', f'models/{name}_history.txt')


if __name__ == '__main__':
    logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    model = keras.models.Sequential([
        keras.layers.Masking(mask_value=0, input_shape=[None, 1200]),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(100, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(200, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        # keras.layers.BatchNormalization(),
        # keras.layers.LSTM(300, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        # keras.layers.BatchNormalization(),
        # keras.layers.LSTM(500, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        # keras.layers.BatchNormalization(),
        # keras.layers.LSTM(500, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        # keras.layers.BatchNormalization(),
        # keras.layers.LSTM(300, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(200, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(100, return_sequences=True, kernel_regularizer=l2(0.2), dropout=0.5),
        keras.layers.Dense(2, activation='softmax')
    ])

    run_model(model, 'lstm1')
