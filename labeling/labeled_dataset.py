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
from real_data_wrangling.clean_gulkana import preprocess_gulkana_real_data
import logging


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
            logging.info(f"Merging scan {scan_number}")
            try:
                self.merge_and_write(scan_number)
            except ValueError:
                logging.warning(f"Scan {scan_number} resulted in an error...skipping.")
                continue
            except Exception as e:
                logging.warning(f"Scan {scan_number} resulted in an error...skipping...error: {e}")
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
                logging.warning(f"Skipping {key} - wrong size.")
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

    def __init__(self, data_loader, geometry_spec, num_batches, scan_min_col=100, scan_max_col=None,
                 n=1000, random_seed=None, num_threads=8):

        self.scan_numbers = data_loader.scan_numbers()
        self.num_batches = num_batches
        self.num_threads = num_threads

        self.data_loader = data_loader

        self.labeler = S3ScanLabeler(data_loader.bucket_name, '', geometry_spec)

        self.scan_min_col = scan_min_col
        self.scan_max_col = scan_max_col
        self.n = n

        if random_seed:
            random.seed(random_seed)

        self.batched_indices = self.partition(list(range(len(self.scan_numbers))), num_batches)

    @staticmethod
    def bootstrap(scan, label, max_col, min_col, n):

        if scan is None:
            logging.debug("[DataSetGenerator.bootstrap] scan = None")
            return None

        logging.debug(f"[DataSetGenerator.bootstrap] scan.shape = {scan.shape}")
        logging.debug(f"[DataSetGenerator.bootstrap] len(label) = {len(label)}")

        # Generate a number of input matrices from the base scan
        max_col = max_col if max_col and max_col <= scan.shape[0] else scan.shape[0]
        min_col = min(min_col, scan.shape[0])

        logging.debug(f"[DataSetGenerator.bootstrap] max_col: {max_col}")
        logging.debug(f"[DataSetGenerator.bootstrap] min_col: {min_col}")

        lengths = np.random.randint(min_col, high=max_col + 1, size=n)
        starts = [random.randint(0, scan.shape[0] - length) for length in lengths]

        logging.debug(f"lengths = {lengths}")
        logging.debug(f"starts = {starts}")

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

        logging.debug(f"Generating batch with indices {indices}")
        scan_numbers = [sn for i, sn in enumerate(self.scan_numbers) if i in indices]
        logging.debug(f"Batch includes {len(scan_numbers)} scans: {scan_numbers}")

        for i in range(5):
            try:
                scans = list(self.data_loader.load(scan_numbers=scan_numbers))
                logging.info("Finished loading scans")
                break
            except Exception as e:
                logging.warning("Error loading scans...retrying...")
                logging.warning(f"{e}")
                continue

        labels = [self.labeler.label_scan_inside_outside(scan_number) for scan_number in scan_numbers]
        logging.info("Finished labeling")

        # logging.info(f"scan1.number = {scan_numbers[0]}")
        # logging.info(f"scan1.label = {labels[0]}")
        # logging.info(f"scan1.shape = {scans[0].shape if scans[0] is not None else None}")

        with multiprocessing.Pool(self.num_threads) as p:
            scan_labels = p.starmap(self.bootstrap, zip(
                scans, labels, itertools.repeat(self.scan_max_col), itertools.repeat(self.scan_min_col),
                itertools.repeat(self.n)
            ))

        # scan_labels = []
        #
        # for scan, label, max_col, min_col, n in zip(scans, labels, itertools.repeat(self.scan_max_col),
        #                                             itertools.repeat(self.scan_min_col), itertools.repeat(self.n)):
        #     scan_labels.append(self.bootstrap(scan, label, max_col, min_col, n))

        x = list(itertools.chain.from_iterable((scan_label[0] for scan_label in scan_labels if scan_label)))
        y = list(itertools.chain.from_iterable((scan_label[1] for scan_label in scan_labels if scan_label)))

        if self.scan_max_col != max((scan.shape[1] for scan in x)):
            logging.warning(f"Max shape of x ({max((scan.shape[1] for scan in x))}) not equal to scan_max_col"
                            f"({self.scan_max_col}).")

        if self.scan_min_col != min((scan.shape[1] for scan in x)):
            logging.warning(f"Min shape of x ({min((scan.shape[1] for scan in x))}) not equal to scan_min_col"
                            f"({self.scan_min_col}).")

        if self.scan_max_col != max((len(label) for label in y)):
            logging.warning(f"Max shape of y ({max((len(label) for label in y))}) not equal to scan_max_col"
                            f"({self.scan_max_col}).")

        if self.scan_min_col != min((len(label) for label in y)):
            logging.warning(f"Min shape of y ({min((len(label) for label in y))}) not equal to scan_min_col"
                            f"({self.scan_min_col}).")

        return x, y

    def generate_batch(self, batch_number):
        return self.generate(indices=self.batched_indices[batch_number])

    def generate_batches(self):
        # print("batched indices = ", batched_indices)
        return (self.generate(indices=indices) for indices in self.batched_indices)

    @staticmethod
    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]


class BScanDataSetGenerator:

    def __init__(self, data_loader, num_batches, scan_min_col=100, scan_max_col=None, n=1, random_seed=None,
                 num_threads=8, random_cropping=True):

        self.scan_numbers = data_loader.scan_numbers()
        self.num_batches = num_batches
        self.num_threads = num_threads

        self.data_loader = data_loader

        self.scan_min_col = scan_min_col
        self.scan_max_col = scan_max_col
        self.n = n
        self.random_cropping = random_cropping

        if random_seed:
            random.seed(random_seed)

        self.batched_indices = self.partition(list(range(len(self.scan_numbers))), num_batches)

    @staticmethod
    def bootstrap(scan, label, max_col, min_col, n, random_cropping):
        # This must be a static method in order for multithreading to work properly

        if scan is None:
            logging.debug("[DataSetGenerator.bootstrap] scan = None")
            return None

        logging.debug(f"[DataSetGenerator.bootstrap] scan.shape = {scan.shape}")
        logging.debug(f"[DataSetGenerator.bootstrap] label = {label}")

        max_col = max_col if max_col and max_col <= scan.shape[0] else scan.shape[0]
        min_col = min(min_col, scan.shape[0])

        logging.debug(f"[DataSetGenerator.bootstrap] max_col: {max_col}")
        logging.debug(f"[DataSetGenerator.bootstrap] min_col: {min_col}")
        logging.debug(f"Random cropping is {'ON' if random_cropping else 'OFF'}")

        if random_cropping:
            # Generate a number of input matrices from the base scan
            lengths = np.random.randint(min_col, high=max_col + 1, size=n)
            starts = [random.randint(0, scan.shape[0] - length) for length in lengths]

        else:
            starts = [int((scan.shape[0] - max_col) / 2)]
            lengths = [max_col]

        logging.debug(f"lengths = {lengths}")
        logging.debug(f"starts = {starts}")

        # Generate n scans from each b-scan
        return BScanDataSetGenerator.bootstrap_scan(scan.T, starts, lengths), \
               BScanDataSetGenerator.bootstrap_label(label, starts, lengths)

    @staticmethod
    def bootstrap_label(label, starts, lengths):
        l = [label for start in starts]
        logging.debug(f"[DataSetGenerator.bootstrap_label] label = {label} -> {l}")
        return l

    @staticmethod
    def bootstrap_scan(scan, starts, lengths):
        return [scan[:, start:start + length] for start, length in zip(starts, lengths)]

    def generate(self, indices=None):

        logging.debug(f"Generating batch with indices {indices}")
        scan_numbers = [sn for i, sn in enumerate(self.scan_numbers) if i in indices]
        logging.info(f"Batch includes {len(scan_numbers)} scans: {scan_numbers}")

        for i in range(5):
            try:
                scans = list(self.data_loader.load(scan_numbers=scan_numbers))
                logging.info("Finished loading scans")
                break
            except Exception as e:
                logging.warning("Error loading scans...retrying...")
                logging.warning(f"{e}")
                continue

        labels = [1 if scan_number < 20000 else 0 for scan_number in scan_numbers]
        print("Unbootstrapped labels")
        print(labels)
        print("\n")

        logging.info("Finished labeling")

        # logging.info(f"scan1.number = {scan_numbers[0]}")
        # logging.info(f"scan1.label = {labels[0]}")
        # logging.info(f"scan1.shape = {scans[0].shape if scans[0] is not None else None}")

        with multiprocessing.Pool(self.num_threads) as p:
            scan_labels = p.starmap(self.bootstrap, zip(
                scans, labels, itertools.repeat(self.scan_max_col), itertools.repeat(self.scan_min_col),
                itertools.repeat(self.n), itertools.repeat(self.random_cropping)
            ))

        # print(scan_labels)
        # scan_labels = []
        #
        # for scan, label, max_col, min_col, n in zip(scans, labels, itertools.repeat(self.scan_max_col),
        #                                             itertools.repeat(self.scan_min_col), itertools.repeat(self.n)):
        #     scan_labels.append(self.bootstrap(scan, label, max_col, min_col, n))

        x = list(itertools.chain.from_iterable((scan_label[0] for scan_label in scan_labels if scan_label)))
        y = list(itertools.chain.from_iterable((scan_label[1] for scan_label in scan_labels if scan_label)))

        print(f"len(X) = {len(x)}")
        print(f"\nTotal samples: {len(y)}")
        print(f"Total positive: {np.sum(y)}")
        print(f"Total negative: {len(y) - np.sum(y)}")

        if self.scan_max_col != max((scan.shape[1] for scan in x)):
            logging.warning(f"Max shape of x ({max((scan.shape[1] for scan in x))}) not equal to scan_max_col"
                            f"({self.scan_max_col}).")

        if self.scan_min_col != min((scan.shape[1] for scan in x)):
            logging.warning(f"Min shape of x ({min((scan.shape[1] for scan in x))}) not equal to scan_min_col"
                            f"({self.scan_min_col}).")

        # if self.scan_max_col != max((len(label) for label in y)):
        #     logging.warning(f"Max shape of y ({max((len(label) for label in y))}) not equal to scan_max_col"
        #                     f"({self.scan_max_col}).")
        #
        # if self.scan_min_col != min((len(label) for label in y)):
        #     logging.warning(f"Min shape of y ({min((len(label) for label in y))}) not equal to scan_min_col"
        #                     f"({self.scan_min_col}).")

        return x, y

    def generate_batch(self, batch_number):
        return self.generate(indices=self.batched_indices[batch_number])

    def generate_batches(self):
        # print("batched indices = ", batched_indices)
        return (self.generate(indices=indices) for indices in self.batched_indices)

    @staticmethod
    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]


class HybridBScanDataSetGenerator:

    def __init__(self, data_loader, num_batches, scan_min_col=100, scan_max_col=None, n=1, random_seed=None,
                 num_threads=8):

        self.scan_numbers = data_loader.scan_numbers()
        self.num_batches = num_batches
        self.num_threads = num_threads

        self.data_loader = data_loader

        self.scan_min_col = scan_min_col
        self.scan_max_col = scan_max_col
        self.n = n

        if random_seed:
            random.seed(random_seed)

        self.batched_indices = self.partition(list(range(len(self.scan_numbers))), num_batches)

        self.gulkana_X, self.gulkana_y = preprocess_gulkana_real_data()

        self.batched_indices_gulkana = self.partition(list(range(len(self.gulkana_y))), num_batches)

    @staticmethod
    def bootstrap(scan, label, max_col, min_col, n):

        if scan is None:
            logging.debug("[DataSetGenerator.bootstrap] scan = None")
            return None

        logging.debug(f"[DataSetGenerator.bootstrap] scan.shape = {scan.shape}")
        logging.debug(f"[DataSetGenerator.bootstrap] label = {label}")

        # Generate a number of input matrices from the base scan
        max_col = max_col if max_col and max_col <= scan.shape[0] else scan.shape[0]
        min_col = min(min_col, scan.shape[0])

        logging.debug(f"[DataSetGenerator.bootstrap] max_col: {max_col}")
        logging.debug(f"[DataSetGenerator.bootstrap] min_col: {min_col}")

        lengths = np.random.randint(min_col, high=max_col + 1, size=n)
        starts = [random.randint(0, scan.shape[0] - length) for length in lengths]

        logging.debug(f"lengths = {lengths}")
        logging.debug(f"starts = {starts}")

        # Generate n scans from each b-scan
        return HybridBScanDataSetGenerator.bootstrap_scan(scan.T, starts, lengths), \
               HybridBScanDataSetGenerator.bootstrap_label(label, starts, lengths)

    @staticmethod
    def bootstrap_label(label, starts, lengths):
        l = [label for start in starts]
        logging.debug(f"[DataSetGenerator.bootstrap_label] label = {label} -> {l}")
        return l

    @staticmethod
    def bootstrap_scan(scan, starts, lengths):
        return [scan[:, start:start + length] for start, length in zip(starts, lengths)]

    def generate(self, indices=None, gulkana_indices=None):

        logging.debug(f"Generating batch with indices {indices}")
        scan_numbers = [sn for i, sn in enumerate(self.scan_numbers) if i in indices]
        logging.info(f"Batch includes {len(scan_numbers)} scans: {scan_numbers}")

        for i in range(5):
            try:
                scans = list(self.data_loader.load(scan_numbers=scan_numbers))
                logging.info("Finished loading scans")
                break
            except Exception as e:
                logging.warning("Error loading scans...retrying...")
                logging.warning(f"{e}")
                continue

        labels = [1 if scan_number < 20000 else 0 for scan_number in scan_numbers]
        print("Unbootstrapped labels")
        print(labels)
        print("\n")

        logging.info("Finished labeling")

        # logging.info(f"scan1.number = {scan_numbers[0]}")
        # logging.info(f"scan1.label = {labels[0]}")
        # logging.info(f"scan1.shape = {scans[0].shape if scans[0] is not None else None}")

        with multiprocessing.Pool(self.num_threads) as p:
            scan_labels = p.starmap(self.bootstrap, zip(
                scans, labels, itertools.repeat(self.scan_max_col), itertools.repeat(self.scan_min_col),
                itertools.repeat(self.n)
            ))

        x = list(itertools.chain.from_iterable((scan_label[0] for scan_label in scan_labels if scan_label)))
        y = list(itertools.chain.from_iterable((scan_label[1] for scan_label in scan_labels if scan_label)))

        print(f"len(X) = {len(x)}")
        print(f"\nTotal samples: {len(y)}")
        print(f"Total positive: {np.sum(y)}")
        print(f"Total negative: {len(y) - np.sum(y)}")

        if self.scan_max_col != max((scan.shape[1] for scan in x)):
            logging.warning(f"Max shape of x ({max((scan.shape[1] for scan in x))}) not equal to scan_max_col"
                            f"({self.scan_max_col}).")

        if self.scan_min_col != min((scan.shape[1] for scan in x)):
            logging.warning(f"Min shape of x ({min((scan.shape[1] for scan in x))}) not equal to scan_min_col"
                            f"({self.scan_min_col}).")


        gx = self.gulkana_X[gulkana_indices]

        print(f"gx.shape = {gx.shape}")
        print(f"x.shape = {x.shape}")

        return np.concatenate([x, self.gulkana_X[gulkana_indices]]), np.concatenate([y, self.gulkana_y[
            gulkana_indices]])

    def generate_batch(self, batch_number):
        return self.generate(
            indices=self.batched_indices[batch_number],
            gulkana_indices = self.batched_indices_gulkana[batch_number]
        )

    def generate_batches(self):
        return (self.generate(indices=indices, gulkana_indices = gulkana_indices) for indices, gulkana_indices in
                zip(self.batched_indices, self.batched_indices_gulkana))

    @staticmethod
    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]


class GulkanaBScanDataSetGenerator:

    def __init__(self, num_batches, random_seed=None, prefix=None, keys=None):

        self.num_batches = num_batches

        if random_seed:
            random.seed(random_seed)

        self.X, self.y = preprocess_gulkana_real_data(prefix=prefix, keys=keys)
        self.batched_indices = self.partition(list(range(len(self.y))), num_batches)

    def generate(self, indices=None):
        logging.debug(f"Generating batch with indices {indices}")
        return self.X[indices], self.y[indices]

    def generate_batch(self, batch_number):
        return self.generate(indices=self.batched_indices[batch_number])

    def generate_batches(self):
        return (self.generate(indices=indices) for indices in self.batched_indices)

    @staticmethod
    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]