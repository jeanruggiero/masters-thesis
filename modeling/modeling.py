import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import logging
import numpy as np

from tensorflow.keras.optimizers import Adam

from preprocessing import preprocess
from modeling.metrics import mean_jaccard_index_post_epoch, f1_score_post_epoch, precision_post_epoch, \
    recall_post_epoch


def plot_history(history):
    fig, axes = plt.subplots(1, 5)

    metrics = ['loss', 'mean_overlap', 'object_detection_f1_score', 'object_size_rmse']

    for ax, metric in zip(axes, metrics):
        try:
            ax.plot(history.history[metric])
            ax.plot(history.history['val_' + metric])
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.set_xlabel('epoch')
        except KeyError:
            continue

    axes[0].legend(['train', 'test'], loc='upper left')


def expand_dim(array):
    return array[...,np.newaxis]


def apply_window(X, y, n):
    X_w = expand_dim(np.array([np.array([X[j, i:i+n, :] for i in range(X.shape[1] - n)]).T for j in range(X.shape[0])]))
    y_w = y[:, 0:-n]

    return X_w, y_w


def train_model(model, data_generator, output_time_range, sample_rate, callbacks={}, plots=True, resample=False,
                epochs=30, sliding_window_size=None):
    # Callbacks argument should be a dict of callback_fn: list of batches or None pairs. If list of batches is None
    # the callback will be applied to all batches

    # Use the first batch for validation.
    logging.info("Loading validation set.")
    X_val, y_val = preprocess(data_generator.generate_batch(0), output_time_range, sample_rate, resample=resample)

    if sliding_window_size is not None:
        X_val, y_val = apply_window(X_val, y_val, sliding_window_size)
    else:
        X_val = expand_dim(X_val)

    logging.info(f'X_val.shape = {X_val.shape}')
    logging.info(f'y_val.shape = {y_val.shape}')

    histories = []
    for i in range(1, data_generator.num_batches):
        logging.info(f"Loading batch {i}")

        X_train, y_train = preprocess(data_generator.generate_batch(i), output_time_range, sample_rate)

        if sliding_window_size is not None:
            X_train, y_train = apply_window(X_train, y_train, sliding_window_size)
        else:
            X_train = expand_dim(X_train)

        logging.info(f"X_train.shape = {X_train.shape}")
        logging.info(f"y_train.shape = {y_train.shape}")

        # Select callbacks to apply to this batch
        batch_callbacks = [key for key, batches in callbacks.items() if not batches or i in batches]

        logging.info(f"Training model on batch {i}")
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=batch_callbacks)
        histories.append(history)

        y_pred = model.predict(X_val)
        logging.info(f"Mean Jaccard Index = {mean_jaccard_index_post_epoch(y_val, y_pred):.2f}")
        logging.info(f"f1-score = {f1_score_post_epoch(y_val, y_pred):.2f}")
        logging.info(f"Precision = {precision_post_epoch(y_val, y_pred):.2f}")
        logging.info(f"Recall = {recall_post_epoch(y_val, y_pred):.2f}")

        if plots:
            plot_history(history)

    return histories, model, X_val, y_val
