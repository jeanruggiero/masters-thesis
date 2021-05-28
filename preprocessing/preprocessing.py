import numpy as np
import random
import math
import logging

from real_data_wrangling.reshape import resample_y


def resample_scan(scan, sample_rate, output_time_range):
    input_time_range = 60 if scan.shape[0] == 10057 else 120
    return resample_y(scan, input_time_range, sample_rate, output_time_range)


def pad_scan(scan, n_cols):
    if scan.shape[1] == n_cols:
        return scan
    return np.concatenate((scan, np.zeros((scan.shape[0], n_cols - scan.shape[1]))), axis=1)


def pad_label(label, n_cols):
    if type(label) == int:
        return label
    if len(label) == n_cols:
        return label
    return label + [0] * (n_cols - len(label))


def pad(scans, labels, n_cols):
    padded_scans = [pad_scan(scan, n_cols) for scan in scans]
    padded_labels = [pad_label(label, n_cols) for label in labels]
    return padded_scans, padded_labels


def preprocess(batch, output_time_range, sample_rate, n_cols=144, resample=False):
    data, labels = batch

    if resample:
        data = [resample_scan(d, sample_rate, output_time_range) for d in data]

    logging.debug(f"Padding scans to size: {max((scan.shape[1] for scan in data))}")

    padded_scans, padded_labels = pad(data, labels, max((scan.shape[1] for scan in data)))

    X = np.array([d.T for d in padded_scans])
    y = np.array(padded_labels)
    return X, y


def train_test_split(X, y, test_size=0.2):
    test_indices = random.sample(range(y.shape[0]), math.floor(test_size * y.shape[0]))
    train_indices = [i for i in range(y.shape[0]) if i not in test_indices]

    return X[train_indices, :, :], X[test_indices, :, :], y[train_indices, :], y[test_indices, :]