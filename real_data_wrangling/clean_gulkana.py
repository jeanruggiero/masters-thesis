"""This file contains code to read the IFSTTAR scans and convert them into a standard hdf5 format."""

import os
import glob
import uuid
import h5py
import boto3
import math

from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.dt1io import readdt1, readdt1Header
from .reshape import preprocess_scan


def load_gulkana(key, bucket):
    header_key = key[:-4] + '.HD'

    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, 'tmp.DT1')
    s3_client.download_file(bucket, header_key, 'tmp.HD')

    data = np.asarray(readdt1('tmp.DT1'))
    header = readdt1Header('tmp.HD')

    os.remove('tmp.DT1')
    os.remove('tmp.HD')

    return data, header


def preprocess_gulkana_real_data():

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name="jean-masters-thesis")
    scan_path = "raw_data/gulkanaGlacier_rawGPR_2017/GPR_data/DATA01/LINE00"
    keys = [obj.key for obj in bucket.objects.filter(Prefix=scan_path) if obj.key[-4:] == '.DT1']
    X = []

    for key in keys:
        print(f"Processing {key}")

        data, header = load_gulkana(key, "jean-masters-thesis")

        input_time_range = header['Total_time_window']
        output_sample_rate = 4
        output_time_range = 120
        x_range = header['Final_pos'] - header['Start_pos']
        output_size = math.floor(50 * x_range)
        window_size = 100
        overlap = 10

        bootstrapped_scans = preprocess_scan(data, input_time_range, output_sample_rate, output_time_range, x_range,
                                             output_size, window_size, overlap=overlap, method_y='last',
                                             method_x='last')

        print(f"len(bootstrapped_scans) = {len(bootstrapped_scans)}")
        X.extend(bootstrapped_scans)

    return np.array(X), np.array([0] * len(X))


if __name__ == '__main__':
    X, y = preprocess_gulkana_real_data()

    print(X.shape, y.shape)


# s3 = boto3.resource('s3')
# bucket = s3.Bucket(name="jean-masters-thesis")
# scan_path = "raw_data/gulkanaGlacier_rawGPR_2017/GPR_data/DATA01/"
#
#
# for obj in bucket.objects.filter(Prefix=scan_path):
#     if obj.key[-4:] == ".DT1":
#         print(obj.key)

# data_filenames = glob.glob(os.path.join(base_dir, '*.DT1'))
#
# print(readdt1Header(base_dir + '/LINE00.HD'))
#
# for data_filename in data_filenames:
#
#     header_filename = data_filename.split('/')[-1][:-4] + '.HD'
#
#     data = readdt1(data_filename)
#     header = readdt1Header(header_filename)
#
#     print(header)



# for dir, soil_type in zip(dirs, soil_types):
#     paths = glob.glob(os.path.join(base_dir, dir, '*'))
#     for path in paths:
#
#
#         file_extension = path.split('.')[-1].lower()
#         filename = path.split('/')[-1]
#
#         if file_extension == 'dzt':
#             # DZT file
#             data, info = readdzt(path)
#             data = np.asarray(data)
#
#         elif file_extension == 'rd3':
#             # MALA geoscience GPR - two files per scan - .rad and .rd3
#             # Remove file extension as required by read function
#             path = path.split('.')[0]
#             data, info = readMALA(path)
#             data = np.asarray(data)
#
#         elif file_extension == 'csv':
#             # IDS GPR
#             # https://github.com/emanuelhuber/RGPR/blob/master/R/readIDS.R
#             # Data loaded into R and converted to csv
#             data = pd.read_csv(path)
#             data = data.drop(columns=['Unnamed: 0'])
#             data = data.to_numpy()
#
#         else:
#             if file_extension not in ['dt', 'rad']:
#                 # print(f"Could not process file: {filename}")
#                 pass
#             continue
#
#         # Center frequency
#         frequency = filename[:3]
#
#         # Whether or not the profile was generated in the reverse direction
#         reverse = "rev" in filename
#
#         # Read file metadata if available
#         file_metadata = metadata[metadata['File Name'].str.lower() == filename.lower()]
#         if not file_metadata.empty:
#             scans_per_meter = file_metadata['Scans/m'].squeeze()
#             range = file_metadata['Range (ns)'].squeeze()
#         else:
#             print(f"No metadata for {filename}")
#             scans_per_meter = np.nan
#             range = np.nan
#
#         # File contains two half-profiles in the same file, so split it into two different files
#         if filename == "200MHz_silt_h2h1.dzt":
#             to_hdf5(data[:, :596], frequency, scans_per_meter, range, soil_type, profile_encodings[dir]['h2'], reverse,
#                     output_dir)
#             to_hdf5(data[:, 597:], frequency, scans_per_meter, range, soil_type, profile_encodings[dir]['h1'], reverse,
#                     output_dir)
#         else:
#             # Determine the profile
#             if dir == "MULTI-LAYER":
#                 profile = profile_encodings["MULTI-LAYER"]
#             else:
#                 profile_description = filename.split('.')[0].split('_')[2]
#                 try:
#                     profile = profile_encodings[dir][profile_description]
#                 except:
#                     print(filename)
#
#             to_hdf5(data, frequency, scans_per_meter, range, soil_type, profile, reverse, output_dir)
