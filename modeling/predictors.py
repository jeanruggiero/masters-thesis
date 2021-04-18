import matplotlib.pyplot as plt
import numpy as np
from modeling.metrics import mean_jaccard_index_post_epoch


class ObjectDetector:

    def __init__(self, model):
        self.model = model

    def label_scan(self, scan, y_true=None, start_true=None, end_true=None):

        if not y_true and start_true and end_true:
            y_true = [1 if start_true <= i <= end_true else 0 for i in range(scan.shape[0])]
        elif y_true:
            start_true = np.min(np.where(y_true))
            end_true = np.max(np.where(y_true))

        if y_true:
            print(y_true)

        y_pred = np.argmax(self.model.predict(scan), axis=1)
        print(y_pred)

        start_pred = np.min(np.where(y_pred))
        end_pred = np.max(np.where(y_pred))

        fig, ax = plt.subplots()
        ax.imshow(scan.T, cmap='gray')

        # Plot predicted start and end
        ax.axvline(x=start_pred, color="r")
        ax.axvline(x=end_pred, color="r")

        if y_true:
            # Plot true start and end
            ax.axvline(x=start_true, color="g")
            ax.axvline(x=end_true, color="g")

            ax.set_title(f"J = {mean_jaccard_index_post_epoch(y_true, y_pred)}")


