import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

import numpy as np

from real_data_wrangling.pipelines import preprocess_real_data

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

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
