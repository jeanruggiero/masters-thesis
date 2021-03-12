import os
import glob
import uuid
import h5py
import boto3
import io
import pandas as pd


def get_metadata():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name="jean-masters-thesis")

    try:
        with io.BytesIO() as b:
            s3.Object(bucket.name, "gold_data/ifsttar/data/scan_metadata.csv").download_fileobj(b)
            return pd.read_csv(b)
    except:
        metadata = {}

        for obj in bucket.objects.filter(Prefix="gold_data/ifsttar/data/"):
            with io.BytesIO() as b:
                s3.Object(bucket.name, obj.key).download_fileobj(b)

                with h5py.File(b, 'r') as f:
                    attrs = f['data']
                    for key, val in f['data'].attrs.items():
                        metadata[key] = metadata.get(key, []) + [val]

        df = pd.DataFrame(metadata)
        df['center_frequency'] = df['center_frequency'].astype(int)
        df['sample_rate'] = df['samples_per_ascan'] / df['range'] / 1e-9
        return df

if __name__ == "__main__":
    print(get_metadata())
