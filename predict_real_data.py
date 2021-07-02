import os
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler("predictions.log"),
    logging.StreamHandler()
])

import numpy as np

from real_data_wrangling.pipelines import preprocess_real_data
from modeling.metrics import f1_score_post_epoch, precision_post_epoch, recall_post_epoch
from modeling.modeling import expand_dim

alpha = 0.05

model = keras.models.Sequential([
    #keras.layers.InputLayer(shape=[144, 480, 1]),
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

# Restore the weights
model.load_weights('conv1')

X_test, y_test = preprocess_real_data('thesis_real_data_labels.csv', 'real_data_metadata.csv')
X_test = np.transpose(expand_dim(X_test), axes=(0, 2, 1, 3))

print("Preprocessing complete")
print(f"type(X) = {type(X_test)}")
print(f"X.shape = {X_test.shape}")
print(f"y.shape = {y_test.shape}")


## Balance dataset by omitting positive or negative samples as needed
n_positive = np.sum(y_test)
n_negative = y_test.shape[0] - n_positive

print(f"Dataset contains {n_positive} positive samples and {n_negative} negative samples.")

# y_test_positive, X_test_positive = np.where(y_test, y_test, X_test)
# y_test_negative, X_test_negative = np.where(~y_test, y_test, X_test)
#
# if n_positive > n_negative:
#     # More positive than negative samples
#     positive_indices = np.random.choice(len(n_positive), size=n_negative, replace=False)
#     y_test = np.concatenate([y_test_negative, y_test_positive[positive_indices]])
#     X_test = np.concatenate([X_test_negative, X_test_positive[positive_indices, :, :]])
# else:
#     # More negative than positive samples
#     negative_indices = np.random.choice(len(n_negative), size=n_positive, replace=False)
#     y_test = np.concatenate([y_test_positive, y_test_negative[negative_indices]])
#     X_test = np.concatenate([X_test_positive, X_test_negative[negative_indices, :, :]])

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print(y_test)
print(y_pred)

print(y_pred_proba)

print(f"\nTotal samples: {y_pred.shape[0]}")
print(f"Total positive: {np.sum(y_test)}")
print(f"Total negative: {y_test.shape[0] - np.sum(y_test)}")

print(f"f1-score = {f1_score_post_epoch(y_test, y_pred_proba):.2f}")
print(f"precision = {precision_post_epoch(y_test, y_pred_proba):.2f}")
print(f"recall = {recall_post_epoch(y_test, y_pred_proba):.2f}")
