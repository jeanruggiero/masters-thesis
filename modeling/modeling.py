import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from preprocessing import preprocess


def plot_history(history):
    fig, axes = plt.subplots(1, 5)

    metrics = ['loss', 'mean_overlap', 'object_detection_f1_score', 'object_size_rmse']

    for ax, metric in zip(axes, metrics):
        try:
            ax.plot(history.history[metric])
            ax.plot(history.history['val_' + metric])
            ax.title(metric)
            ax.set_ylabel(metric)
            ax.set_xlabel('epoch')
        except KeyError:
            continue

    axes[0].legend(['train', 'test'], loc='upper left')
    fig.show()


def train_model(model, data_generator, output_time_range, sample_rate, callbacks={}, plots=True, resample=False):
    # Callbacks argument should be a dict of callback_fn: list of batches or None pairs. If list of batches is None
    # the callback will be applied to all batches

    batches = data_generator.generate_batches(10)

    # Use the first batch for validation.
    X_val, y_val = preprocess(next(batches), output_time_range, sample_rate, resample=resample)
    print(X_val.shape)
    print(y_val.shape)

    histories = []
    for i, batch in enumerate(batches):
        print(f"Training model on batch {i}")
        X_train, y_train = preprocess(batch, output_time_range, sample_rate)

        # Select callbacks to apply to this batch
        batch_callbacks = [key for key, batches in callbacks.items() if not batches or i in batches]

        history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), callbacks=batch_callbacks)
        histories.append(history)

        if plots:
            plot_history(history)

    return histories, model, X_val, y_val
