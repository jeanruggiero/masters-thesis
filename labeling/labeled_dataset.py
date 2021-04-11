import boto3
import re
import io
import h5py
import numpy as np
import random
import pickle
from .labelers import S3ScanLabeler
from preprocessing import resample_scan


class BScanMergeCrawler:
    """
    Crawls an S3 bucket looking for simulated scans and merges ascans into bscans.
    """

    def __init__(self, bucket_name, scan_path, resample=False, overwrite=False):
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(name=bucket_name)
        self.scan_path = scan_path
        self.resample = resample
        self.overwrite = overwrite

    def scans(self):
        """Returns an iterable of scan numbers found in the S3 bucket."""
        keys = [obj.key[len(self.scan_path):] for obj in self.bucket.objects.filter(Prefix=self.scan_path)]
        return set([int(key.split('/')[0]) for key in keys if key.split('/')[0].isnumeric()])

    def merge_scan(self, scan_number):
        ascan_objs = sorted(list(self.bucket.objects.filter(Prefix=f"{self.scan_path}{scan_number}/")),
                            key=(lambda obj: int(re.findall(r'\d+', obj.key.split('/')[-1])[0])))

        bscan = []
        for i, obj in enumerate(ascan_objs):
            with io.BytesIO() as b:
                self.s3.Object(self.bucket.name, obj.key).download_fileobj(b)

                with h5py.File(b, 'r') as f:
                    group = f['/rxs/rx1/']
                    bscan.append(group['Ez'][()])

        output_time_range = 120
        sample_rate = 10

        return resample_scan(np.array(bscan, dtype=float), sample_rate, output_time_range)

    def write_bscan(self, bscan, scan_number):

        with io.BytesIO() as b:
            np.savetxt(b, bscan, delimiter=",")
            b.seek(0)
            self.bucket.put_object(Key=f"{self.scan_path}merged/{scan_number}_merged.csv", Body=b)

    def merge_and_write(self, scan_number):
        # Only write the scan if we'd like to force overwriting or if the scan doesn't exist
        if self.overwrite or not self.merged_scan_exists(scan_number):
            self.write_bscan(self.merge_scan(scan_number), scan_number)

    def merge_all(self):
        for scan_number in self.scans():
            print(f"Merging scan {scan_number}")
            try:
                self.merge_and_write(scan_number)
            except ValueError:
                print(f"Scan {scan_number} resulted in an error...skipping.")
                continue

    def merged_scan_exists(self, scan_number):
        key = f"{self.scan_path}merged/{scan_number}_merged.csv"
        for obj in self.bucket.objects.filter(Prefix=key):
            if obj.key == key:
                return True

        return False


class S3DataLoader:
    def __init__(self, bucket_name, prefix):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(name=bucket_name)

    def scan_numbers(self):
        return [
            int(re.findall(r'\d+', obj.key[len(self.prefix):])[0]) for obj in
            self.bucket.objects.filter(Prefix=self.prefix)
        ]

    def load(self, filename, resample=False):
        output_time_range = 120
        sample_rate = 10  # samples per ns

        x = []

        for scan_number in self.scan_numbers():
            with io.BytesIO() as b:
                self.s3.Object(self.bucket_name, f"{self.prefix}{scan_number}_merged.csv").download_fileobj(b)
                b.seek(0)
                if resample:
                    x.append(resample_scan(np.loadtxt(b, delimiter=","), output_time_range, sample_rate))
                else:
                    x.append(np.loadtxt(b, delimiter=","))

        with open(filename, 'wb') as f:
            pickle.dump(x, f)


class DataSetGenerator:

    def __init__(self, pickle_filename, scan_numbers, bucket_name, geometry_spec, scan_min_col=50, scan_max_col=None,
                 n=1000):
        # Iterate through all available scans, using a sliding window to create multiple inputs from each scan - with
        # labels
        with open(pickle_filename, 'rb') as f:
            self.data = pickle.load(f)

        # print(type(self.data))
        # print(self.data[0])
        # print(type(self.data[0]))

        self.scan_numbers = scan_numbers

        self.labeler = S3ScanLabeler(bucket_name, '', geometry_spec)

        self.scan_min_col = scan_min_col
        self.scan_max_col = scan_max_col
        self.n = n

    def bootstrap_scan(self, scan, label):
        # Generate a number of input matrices from the base scan

        scan_max_col = self.scan_max_col if self.scan_max_col and self.scan_max_col <= scan.shape[1] else scan.shape[1]
        scan_min_col = min(self.scan_min_col, scan.shape[1])

        # Generate n scans from each b-scan
        scans = []
        labels = []
        for i in range(self.n):
            # Randomly pick a scan length between scan_min_col and scan_max_col
            scan_length = random.randint(scan_min_col, scan_max_col)

            # Randomly pick a scan starting point from the range of possible values
            scan_start = random.randint(0, scan.shape[1] - scan_length)

            # Select the columns from the input scan
            scans.append(scan[:, scan_start:scan_start + scan_length])
            labels.append(label[scan_start:scan_start + scan_length])

        return scans, labels

    def generate(self, indices=None):

        x = []
        y = []

        for i, (scan_number, d) in enumerate(zip(self.scan_numbers, self.data)):
            if not indices or i in indices:
                data, labels = self.bootstrap_scan(d.T, self.labeler.label_scan_inside_outside(scan_number))
                x.extend(data)
                y.extend(labels)

        return x, y

    def generate_batches(self, n):
        batched_indices = self.partition(list(range(len(self.scan_numbers))), n)
        return (self.generate(indices) for indices in batched_indices)

    @staticmethod
    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]

