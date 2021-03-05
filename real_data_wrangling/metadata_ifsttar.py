import os
import glob
import uuid
import h5py
import pandas as pd


def get_metadata():
    dir = "/Users/jeanruggiero/Projects/masters-thesis/IFSTTAR_scans/cleaned/*"
    paths = glob.glob(dir)
    metadata = {}

    for path in paths:
        with h5py.File(path, 'r') as f:
            attrs = f['data']
            for key, val in f['data'].attrs.items():
                metadata[key] = metadata.get(key, []) + [val]

    df = pd.DataFrame(metadata)
    df['center_frequency'] = df['center_frequency'].astype(int)
    df['sample_rate'] = df['samples_per_ascan'] / df['range'] / 1e-9
    return df

if __name__ == "__main__":
    print(get_metadata())
