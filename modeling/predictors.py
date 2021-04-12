import matplotlib.pyplot as plt
import numpy as np


class ObjectDetector:

    def __init__(self, model):
        self.model = model

    def label_scan(self, scan, y_true=None, start_true=None, end_true=None):
        if start_true and end_true:
            y_true = [1 if start_true <= i <= end_true else 0 for i in range(scan.shape[0])]

        if y_true:
            print(y_true)

        y_pred = np.round(self.model.predict(scan))
        print(y_pred)

        start_pred = np.argmin(np.where(y_pred))
        end_pred = np.argmax(np.where(y_pred))

        fig, ax = plt.subplots()
        ax.imshow(scan.T, cmap='gray')

        # Plot predicted start and end
        ax.axvline(x=start_pred, color="r")
        ax.axvline(x=end_pred, color="r")

        if y_true:
            # Plot true start and end
            ax.axvline(x=start_true, color="g")
            ax.axvline(x=end_true, color="g")

            # Compute Jaccard similarity
            m11 = np.sum(np.logical_and(y_true, y_pred))
            m01 = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
            m10 = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))

            jaccard_index = m11 / (m01 + m10 + m11)

            ax.set_title(f"J = {jaccard_index}")


