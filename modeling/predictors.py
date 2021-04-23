import matplotlib.pyplot as plt
import numpy as np
from modeling.metrics import mean_jaccard_index_post_epoch
import logging


class ObjectDetector:

    def __init__(self, model=None, y_true=None, y_pred=None):
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred

    def label_scan(self, scan, y_true=None, start_true=None, end_true=None):

        if y_true is None and start_true and end_true:
            y_true = [1 if start_true <= i <= end_true else 0 for i in range(scan.shape[0])]
        elif y_true is not None:
            start_true = np.min(np.where(y_true))
            end_true = np.max(np.where(y_true))

        if y_true is not None:
            print(y_true)

        y_pred = np.argmax(self.model.predict(scan), axis=2)
        print(y_pred[0])

        fig, ax = plt.subplots()
        ax.imshow(scan.T, cmap='gray', interpolation='nearest', aspect='auto')

        try:
            start_pred = np.min(np.where(y_pred[0]))
            end_pred = np.max(np.where(y_pred[0]))
            # Plot predicted start and end
            ax.axvline(x=start_pred, color="r")
            ax.axvline(x=end_pred, color="r")
        except ValueError:
            logging.error("Could not label prediction - object not detected.")

        if y_true is not None:
            # Plot true start and end
            ax.axvline(x=start_true, color="chartreuse")
            ax.axvline(x=end_true, color="chartreuse")
            ax.set_title(f"J = {mean_jaccard_index_post_epoch(y_true, y_pred)}")
