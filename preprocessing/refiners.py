import numpy as np
from scipy.signal import butter, lfilter
import logging
from real_data_wrangling.clean_gulkana import preprocess_gulkana_real_data

class Noiser:

    def __init__(self, cutoff, percentile):
        self.cutoff = cutoff
        self.percentile = percentile
        self.order = 6
        self.fs = 30.0

        # load noise scans & resample/slice to get list of (480, 100) size scans
        self.real_scans, _ = preprocess_gulkana_real_data(prefix='DATA02/', keys=['LINE00'])

        # log warning if not enough noise scans
        logging.info(f"Number of noise scans available: {self.real_scans.shape}")

    @staticmethod
    def normalize(scan):
        return (scan - np.min(scan)) / (np.max(scan) - np.min(scan))

    def filter(self, data):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    def extract_noise(self, ascan):
        y = self.filter(ascan)
        c = np.percentile(np.abs(y), self.percentile)
        return np.clip(y, -c, c)

    def extract_noise_bscan(self, scan):
        # Each column of the input matrix should represent an ascan
        return np.apply_along_axis(self.extract_noise, 0, self.normalize(scan))

    def noise(self, simulated_scan):
        print(f"Simulated scan shape: {simulated_scan.shape}")
        print(f"self.real_scans[np.random.choice(self.real_scans.shape[0]).shape = "
              f"{self.real_scans[np.random.choice(self.real_scans.shape[0])].shape}")

        noise = self.extract_noise_bscan(
            self.real_scans[np.random.choice(self.real_scans.shape[0])]
        )

        print(f"noise.shape = {noise.shape}")

        return self.normalize(simulated_scan) + self.extract_noise_bscan(
            self.real_scans[np.random.choice(self.real_scans.shape[0])]
        )
