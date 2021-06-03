import pandas as pd
import numpy as np
from .reshape import preprocess_scan
import io
import h5py
import boto3
import math


def read_scan(id):
    s3 = boto3.resource('s3')

    with io.BytesIO() as b:
        s3.Object("jean-masters-thesis", f"gold_data/ifsttar/data/{id}.hdf5").download_fileobj(b)

        with h5py.File(b, 'r') as f:
            d = np.zeros(f['data'].shape, dtype=float)
            f['data'].read_direct(d)
    return d


def preprocess_real_data(label_filename, metadata_filename):

    labels = pd.read_csv(label_filename).set_index('id')
    labels = labels.join(pd.read_csv(metadata_filename).set_index('id'))

    X = []
    y = []

    for id, label in labels.iterrows():

        print(f"Loading scan {id}")
        # Load data from s3
        data = read_scan(id)
        print(f"data.shape = {data.shape}")

        # Select rows
        data = data[:, label['start_col']:label['end_col'] + 1]

        input_time_range = label['range']
        output_sample_rate = 4
        output_time_range = 120
        x_range = data.shape[1] / label['scans_per_meter']
        output_size = math.floor(50 * x_range)
        window_size = 144
        overlap = 50

        bootstrapped_scans = preprocess_scan(data, input_time_range, output_sample_rate, output_time_range, x_range,
                                             output_size, window_size, overlap=overlap)

        bootstrapped_labels = [label['label']] * len(bootstrapped_scans)

        X.extend(bootstrapped_scans)
        y.extend(bootstrapped_labels)

    return X, y
