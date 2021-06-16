import pandas as pd
import numpy as np
from .reshape import preprocess_scan
import io
import h5py
import boto3
import math
import glob
import os

from utils.dt1io import readdt1, readdt1Header


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
        window_size = 100
        overlap = 50

        bootstrapped_scans = preprocess_scan(data, input_time_range, output_sample_rate, output_time_range, x_range,
                                             output_size, window_size, overlap=overlap)

        print(f"len(bootstrapped_scans = {len(bootstrapped_scans)}")
        bootstrapped_labels = [label['label']] * len(bootstrapped_scans)

        X.extend(bootstrapped_scans)
        y.extend(bootstrapped_labels)

    return np.array(X), np.array(y)


def preprocess_gulkana_real_data():

    base_dir = "/Users/jeanruggiero/Projects/masters-thesis/gulkana"
    data_filenames = glob.glob(os.path.join(base_dir, '*.DT1'))
    X = []

    for data_filename in data_filenames:
        print(f"Processing file {data_filename}")

        header_filename = base_dir + '/' + data_filename.split('/')[-1][:-4] + '.HD'

        data = np.asarray(readdt1(data_filename))
        header = readdt1Header(header_filename)

        print(header)
        print(f"data.shape = {data.shape}")

        input_time_range = header['Total_time_window']
        output_sample_rate = 4
        output_time_range = 120
        x_range = header['Final_pos'] - header['Start_pos']
        output_size = math.floor(50 * x_range)
        window_size = 100
        overlap = 50

        bootstrapped_scans = preprocess_scan(data, input_time_range, output_sample_rate, output_time_range, x_range,
                                             output_size, window_size, overlap=overlap)

        print(f"len(bootstrapped_scans) = {len(bootstrapped_scans)}")
        X.extend(bootstrapped_scans)

    return np.array(X), np.array([0] * len(X))
