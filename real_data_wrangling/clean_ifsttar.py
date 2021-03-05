"""This file contains code to read the IFSTTAR scans and convert them into a standard hdf5 format."""

import os
import glob
import uuid
import h5py

from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.read_dzt import readdzt
from utils.read_rd3 import readMALA

from profiles import profile_encodings


def to_hdf5(data: np.ndarray, center_frequency: Number, scans_per_meter: int, range: int, soil_type: str, profile: int,
            reverse: bool, output_dir: str = "."):
    id = uuid.uuid4()

    scan_width = np.nan if not scans_per_meter else data.shape[1] / scans_per_meter

    with h5py.File(os.path.join(output_dir, f"{id}.hdf5"), 'w') as f:
        dataset = f.create_dataset("data", data.shape, dtype="f", data=data)
        dataset.attrs['center_frequency'] = frequency
        dataset.attrs['scan_width'] = scan_width
        dataset.attrs['scans_per_meter'] = scans_per_meter
        dataset.attrs['scan_count'] = data.shape[1]
        dataset.attrs['range'] = range
        dataset.attrs['samples_per_ascan'] = data.shape[0]
        dataset.attrs['soil_type'] = soil_type
        dataset.attrs['id'] = str(id)
        dataset.attrs['profile'] = profile
        dataset.attrs['reverse'] = reverse

metadata = pd.read_csv('/Users/jeanruggiero/Projects/masters-thesis/IFSTTAR_scans/metadata.csv', na_values=" NaN")

output_dir = "/Users/jeanruggiero/Projects/masters-thesis/IFSTTAR_scans/cleaned"

base_dir = "/Users/jeanruggiero/Projects/masters-thesis/IFSTTAR_scans/Database_2018"
dirs = ['SILT', 'GNEISS0-20', 'GNEISS14-20', 'LIMESTONE', 'MULTI-LAYER']
soil_types = ['silt', 'gneiss0-20', 'gneiss14-20', 'limestone', 'NA']

for dir, soil_type in zip(dirs, soil_types):
    paths = glob.glob(os.path.join(base_dir, dir, '*'))
    for path in paths:


        file_extension = path.split('.')[-1].lower()
        filename = path.split('/')[-1]

        if file_extension == 'dzt':
            # DZT file
            data, info = readdzt(path)
            data = np.asarray(data)

        elif file_extension == 'rd3':
            # MALA geoscience GPR - two files per scan - .rad and .rd3
            # Remove file extension as required by read function
            path = path.split('.')[0]
            data, info = readMALA(path)
            data = np.asarray(data)

        elif file_extension == 'csv':
            # IDS GPR
            # https://github.com/emanuelhuber/RGPR/blob/master/R/readIDS.R
            # Data loaded into R and converted to csv
            data = pd.read_csv(path)
            data = data.drop(columns=['Unnamed: 0'])
            data = data.to_numpy()

        else:
            if file_extension not in ['dt', 'rad']:
                # print(f"Could not process file: {filename}")
                pass
            continue

        # Center frequency
        frequency = filename[:3]

        # Whether or not the profile was generated in the reverse direction
        reverse = "rev" in filename

        # Read file metadata if available
        file_metadata = metadata[metadata['File Name'].str.lower() == filename.lower()]
        if not file_metadata.empty:
            scans_per_meter = file_metadata['Scans/m'].squeeze()
            range = file_metadata['Range (ns)'].squeeze()
        else:
            print(f"No metadata for {filename}")
            scans_per_meter = np.nan
            range = np.nan

        # File contains two half-profiles in the same file, so split it into two different files
        if filename == "200MHz_silt_h2h1.dzt":
            to_hdf5(data[:, :596], frequency, scans_per_meter, range, soil_type, profile_encodings[dir]['h2'], reverse,
                    output_dir)
            to_hdf5(data[:, 597:], frequency, scans_per_meter, range, soil_type, profile_encodings[dir]['h1'], reverse,
                    output_dir)
        else:
            # Determine the profile
            if dir == "MULTI-LAYER":
                profile = profile_encodings["MULTI-LAYER"]
            else:
                profile_description = filename.split('.')[0].split('_')[2]
                try:
                    profile = profile_encodings[dir][profile_description]
                except:
                    print(filename)

            to_hdf5(data, frequency, scans_per_meter, range, soil_type, profile, reverse, output_dir)
