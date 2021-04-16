import boto3
import re
import io
import h5py
import numpy as np
import random
import pickle
import multiprocessing
import itertools
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

        if self.resample:
            return resample_scan(np.array(bscan, dtype=float), sample_rate, output_time_range)
        else:
            return np.array(bscan, dtype=float)

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

    @staticmethod
    def load_scan(bucket_name, key, output_time_range=None, sample_rate=None, resample=False):

        s3 = boto3.resource('s3')

        with io.BytesIO() as b:
            s3.Object(bucket_name, key).download_fileobj(b)
            b.seek(0)
            scan = np.loadtxt(b, delimiter=",")
            if scan.shape[1] != 10057:
                print(f"Skipping {key} - wrong size.")
            elif resample:
                return resample_scan(scan, output_time_range, sample_rate)
            else:
                return scan

    def load(self, scan_numbers=None, filename=None, resample=False):
        output_time_range = 120
        sample_rate = 10  # samples per ns

        scan_numbers = scan_numbers if scan_numbers else self.scan_numbers()
        keys = [f"{self.prefix}{scan_number}_merged.csv" for scan_number in scan_numbers]

        with multiprocessing.Pool(8) as p:
            x = p.starmap(self.load_scan, itertools.product([self.bucket_name], keys))

        return x
        # if filename:
        #     with open(filename, 'wb') as f:
        #         pickle.dump(x, f)


class DataSetGenerator:

    def __init__(self, data_loader, geometry_spec, scan_min_col=100, scan_max_col=None,
                 n=1000, random_seed=None):

        self.scan_numbers = data_loader.scan_numbers()

        self.data_loader = data_loader
        self.labeler = S3ScanLabeler(data_loader.bucket_name, '', geometry_spec)

        self.scan_min_col = scan_min_col
        self.scan_max_col = scan_max_col
        self.n = n

        if random_seed:
            random.seed(random_seed)

    @staticmethod
    def bootstrap(scan, label, max_col, min_col, n):

        if scan is None:
            return None

        # Generate a number of input matrices from the base scan
        max_col = max_col if max_col and max_col <= scan.shape[1] else scan.shape[1]
        min_col = min(min_col, scan.shape[1])

        lengths = np.random.randint(min_col, high=max_col + 1, size=n)
        starts = [random.randint(0, scan.shape[1] - length) for length in lengths]

        # Generate n scans from each b-scan
        return DataSetGenerator.bootstrap_scan(scan.T, starts, lengths), \
               DataSetGenerator.bootstrap_label(label, starts, lengths)

    @staticmethod
    def bootstrap_label(label, starts, lengths):
        return [label[start:start + length] for start, length in zip(starts, lengths)]

    @staticmethod
    def bootstrap_scan(scan, starts, lengths):
        return [scan[:, start:start + length] for start, length in zip(starts, lengths)]

    def generate(self, indices=None):

        scan_numbers = [sn for i, sn in enumerate(self.scan_numbers) if i in indices]

        scans = self.data_loader.load(scan_numbers=scan_numbers)
        labels = (self.labeler.label_scan_inside_outside(scan_number) for scan_number in scan_numbers)

        with multiprocessing.Pool(8) as p:
            scan_labels = p.starmap(self.bootstrap, zip(
                scans, labels, itertools.repeat(self.scan_max_col), itertools.repeat(self.scan_min_col),
                itertools.repeat(self.n)
            ))

        x = list(itertools.chain.from_iterable((scan_label[0] for scan_label in scan_labels if scan_label)))
        y = list(itertools.chain.from_iterable((scan_label[1] for scan_label in scan_labels if scan_label)))

        return x, y

    def generate_batches(self, n):
        batched_indices = self.partition(list(range(len(self.scan_numbers))), n)
        return (self.generate(indices) for indices in batched_indices)

    @staticmethod
    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]

