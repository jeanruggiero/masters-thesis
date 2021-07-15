import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import logging
import numpy as np

from tensorflow.keras.optimizers import Adam

from preprocessing import preprocess, Noiser
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
    X_w = np.transpose(
        expand_dim(np.array([[X[j, i:i+n, :] for i in range(X.shape[1] - n)] for j in range(X.shape[0])])),
        axes=[0, 1, 3, 2, 4]
    )
    y_w = y[:, 0:-n]

    return X_w, y_w


def load_batch(batch, data_generator, gulkana_data_generator, noiser, output_time_range, sample_rate, resample,
               sliding_window_size):
    X, y = preprocess(data_generator.generate_batch(batch), output_time_range, sample_rate, resample=resample)

    logging.debug(f'X.shape = {X.shape}')
    logging.debug(f'y.shape = {y.shape}')

    if noiser:
        X_val = np.array([noiser.noise(X[i, :, :]) for i in range(X.shape[0])])

    logging.debug(f'X.shape = {X.shape}')
    logging.debug(f'y.shape = {y.shape}')

    n_positive = np.sum(y)
    n_negative = y.shape[0] - n_positive

    if gulkana_data_generator:
        if gulkana_data_generator.balance and n_positive > n_negative:
            size = n_positive - n_negative
        else:
            size = None

        X_gulkana, y_gulkana = gulkana_data_generator.generate_batch(batch, size=size)

        logging.info(f'X_gulkana.shape = {X_gulkana.shape}')
        logging.info(f'y_gulkana.shape = {y_gulkana.shape}')

        X = np.concatenate([X, X_gulkana])
        y = np.concatenate([y, y_gulkana])

    if sliding_window_size is not None:
        X, y = apply_window(X, y, sliding_window_size)
    else:
        X = expand_dim(X)

    X = np.array([Noiser.normalize(X[i, :, :]) for i in range(X.shape[0])])

    logging.info(f'X.shape = {X.shape}')
    logging.info(f'y.shape = {y.shape}')

    return X, y


def train_model(model, data_generator, output_time_range, sample_rate, callbacks={}, plots=True, resample=False,
                epochs=50, sliding_window_size=None, gulkana_data_generator=None, noiser=None, X_test=None,
                y_test=None):
    # Callbacks argument should be a dict of callback_fn: list of batches or None pairs. If list of batches is None
    # the callback will be applied to all batches

    # Use the first batch for validation.
    logging.info("Loading validation set.")
    X_val, y_val = load_batch(
        0, data_generator, gulkana_data_generator, noiser, output_time_range, sample_rate, resample, sliding_window_size
    )

    histories = []
    total_training_set_size = 0
    total_training_positives = 0
    total_training_negatives = 0
    for i in range(1, data_generator.num_batches):
        logging.info(f"Loading batch {i}")

        X_train, y_train = load_batch(
            i, data_generator, gulkana_data_generator, noiser, output_time_range, sample_rate, resample,
            sliding_window_size
        )

        total_training_set_size += X_train.shape[0]
        total_training_positives += np.sum(y_train)
        total_training_negatives += y_train.shape[0] - np.sum(y_train)

        # Select callbacks to apply to this batch
        batch_callbacks = [key for key, batches in callbacks.items() if not batches or i in batches]

        logging.info(f"Training model on batch {i}")
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=batch_callbacks)
        histories.append(history)

        y_pred = model.predict(X_val)

        logging.debug("\ny_pred validation set")
        logging.debug(f"{np.argmax(y_pred, axis=1)}")

        logging.debug("\ny_pred_proba validation set")
        for y_pp, y_v in zip(y_pred, y_val):
            logging.debug(f"{y_pp}, {y_v}")

        logging.info(f"\nTotal samples in training set: {total_training_set_size}")
        logging.info(f"Total positive (training): {total_training_positives}")
        logging.info(f"Total negative (training): {total_training_negatives}")

        y_pred_train = model.predict(X_train)
        logging.info(f"\nResults on last training batch:")
        logging.info(f"Training f1-score = {f1_score_post_epoch(y_train, y_pred_train):.2f}")
        logging.info(f"Training Precision = {precision_post_epoch(y_train, y_pred_train):.2f}")
        logging.info(f"Training Recall = {recall_post_epoch(y_train, y_pred_train):.2f}")

        logging.info(f"\nTotal samples in validation set: {y_val.shape[0]}")
        logging.info(f"Total positive: {np.sum(y_val)}")
        logging.info(f"Total negative: {y_val.shape[0] - np.sum(y_val)}")

        # logging.info(f"Mean Jaccard Index = {mean_jaccard_index_post_epoch(y_val, y_pred):.2f}")
        logging.info(f"\nf1-score = {f1_score_post_epoch(y_val, y_pred):.2f}")
        logging.info(f"Precision = {precision_post_epoch(y_val, y_pred):.2f}")
        logging.info(f"Recall = {recall_post_epoch(y_val, y_pred):.2f}")


        if plots:
            plot_history(history)

    logging.info("Results on test set:")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    logging.info(f"f1-score = {f1_score_post_epoch(y_test, y_pred_proba):.2f}")
    logging.info(f"precision = {precision_post_epoch(y_test, y_pred_proba):.2f}")
    logging.info(f"recall = {recall_post_epoch(y_test, y_pred_proba):.2f}")

    return histories, model, X_val, y_val
