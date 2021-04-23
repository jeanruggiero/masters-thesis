from tensorflow.keras.metrics import Metric
import tensorflow as tf
import numpy as np
import logging


def boolean_f1_score(y_true, y_pred):

    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.logical_not(y_true), y_pred), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.math.logical_not(y_pred)), tf.float32))

    return tf.math.divide(
        true_positives, tf.math.add(true_positives, tf.math.multiply(
            0.5, tf.math.add(false_positives, false_negatives)
        ))
    )


def mean_jaccard_index_post_epoch(y_true, y_pred):
    # If needed, convert from predict probabilities to class labels
    y_pred = y_pred if len(y_pred.shape) == 2 else np.argmax(y_pred, 2)

    m11 = np.sum(y_true & y_pred, axis=1)
    m01 = np.sum(np.logical_not(y_true) & y_pred, axis=1)
    m10 = np.sum(y_true & np.logical_not(y_pred), axis=1)

    with np.errstate(divide='ignore'):
        j = m11 / (m01 + m10 + m11)

    return np.nanmean(j)


def f1_score_post_epoch(y_true, y_pred):
    # If needed, convert from predict probabilities to class labels
    y_pred = y_pred if len(y_pred.shape) == 2 else np.argmax(y_pred, 2)

    true_positives = np.sum(y_true & y_pred)
    false_positives = np.sum(np.logical_not(y_true) & y_pred)
    false_negatives = np.sum(y_true & np.logical_not(y_pred))

    logging.info(f"TP = {true_positives}")
    logging.info(f"FP = {false_positives}")
    logging.info(f"FN = {false_negatives}")

    return true_positives / (0.5 * (false_positives + false_negatives))


def precision_post_epoch(y_true, y_pred):
    # If needed, convert from predict probabilities to class labels
    y_pred = y_pred if len(y_pred.shape) == 2 else np.argmax(y_pred, 2)

    true_positives = np.sum(y_true & y_pred)
    false_positives = np.sum(np.logical_not(y_true) & y_pred)

    return true_positives / (true_positives + false_positives)


def recall_post_epoch(y_true, y_pred):
    # If needed, convert from predict probabilities to class labels
    y_pred = y_pred if len(y_pred.shape) == 2 else np.argmax(y_pred, 2)

    true_positives = np.sum(y_true & y_pred)
    false_negatives = np.sum(y_true & np.logical_not(y_pred))

    return true_positives / (true_positives + false_negatives)



def mean_jaccard_index(y_true, y_pred):

    tf.print(tf.shape(y_true))
    tf.print(tf.shape(y_pred))

    y_pred = tf.math.argmax(y_pred, 2)

    tf.print(tf.shape(y_pred))

    # Convert probability to boolean
    y_pred = tf.cast(y_pred, tf.bool)
    y_true = tf.cast(y_true, tf.bool)

    m11 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true, y_pred), tf.int8), 1)
    m01 = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true), y_pred), tf.int8), 1)
    m10 = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true, tf.math.logical_not(y_pred)), tf.int8), 1)

    tf.print(tf.shape(m11))
    tf.print(tf.shape(m01))
    tf.print(tf.shape(m10))

    j = tf.math.divide(m11, tf.math.add(tf.math.add(m01, m10), m11))
    tf.print(tf.shape(j))
    return tf.math.reduce_mean(tf.boolean_mask(j, tf.math.is_finite(j)))


def f1_score(y_true, y_pred):

    y_pred = tf.math.argmax(y_pred, 2)

    # Convert probability to 1 or 0
    y_pred = tf.cast(y_pred, tf.bool)
    y_true = tf.cast(y_true, tf.bool)

    return boolean_f1_score(y_true, y_pred)


def mean_overlap(y_true, y_pred):

    # Convert probability to 1 or 0
    y_pred = tf.math.argmax(y_pred, 2)

    # Compute true width of object and overlap of true & predicted objects
    width = tf.math.count_nonzero(y_true, 1)
    overlap = tf.math.count_nonzero(tf.logical_and(tf.cast(y_pred, tf.bool), tf.cast(y_true, tf.bool)), 1)

    # Compute the percent overlap between true and predicted objects
    percent_overlap = tf.math.divide(overlap, width)

    # Compute mean percent overlap, ignoring NaNs
    return tf.math.reduce_mean(tf.boolean_mask(percent_overlap, tf.math.is_finite(percent_overlap)))


def object_detection_f1_score(y_true, y_pred):

    # Convert probability to 1 or 0
    y_pred = tf.math.argmax(y_pred, 2)

    # Convert y_true and y_pred into boolean true/false for each sample: object detected or not
    y_true = tf.cast(tf.math.count_nonzero(y_true, 1), tf.bool)
    y_pred = tf.cast(tf.math.count_nonzero(y_pred, 1), tf.bool)

    return boolean_f1_score(y_true, y_pred)


def object_size_rmse(y_true, y_pred):
    # Convert probability to 1 or 0
    y_pred = tf.math.argmax(y_pred, 2)

    # Compute width of true and predicted objects
    width_true = tf.math.count_nonzero(y_true, 1)
    width_pred = tf.math.count_nonzero(y_pred, 1)

    squared_error = tf.math.pow(tf.math.subtract(width_true, width_pred), 2)

    return tf.math.sqrt(tf.reduce_mean(tf.cast(squared_error, tf.float64)))


def object_center(y):
    # Compute width of objects
    width = tf.cast(tf.math.count_nonzero(y, 1), tf.float32)

    # Compute start index of objects
    start = tf.cast(tf.math.argmin(tf.where(tf.cast(y, tf.bool))), tf.float32)

    # Compute center location of true and predicted objects
    return tf.math.add(start, tf.math.multiply(0.5, width))


def object_center_rmse(y_true, y_pred):
    # Convert probability to 1 or 0
    y_pred = tf.math.round(y_pred)

    # Compute squared error of true & predicted center locations
    squared_error = tf.math.pow(tf.math.subtract(object_center(y_true), object_center(y_pred)), 2.0)

    return tf.math.sqrt(tf.reduce_mean(squared_error))


class MeanObjectOverlap(Metric):
    def __init__(self, name='object_overlap', **kwargs):
        super(MeanObjectOverlap, self).__init__(name=name, **kwargs)
        self.overlap_sum = self.add_weight(name='num', initializer='zeros')
        self.count = self.add_weight(name='den', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        print(y_true)
        print(y_pred)

        y_true = np.nditer(y_true.numpy())
        y_pred = np.nditer(y_pred.numpy())

        width = sum(1 for y in y_pred if y)
        overlap = sum(1 for y_t, y_p in zip(y_true, y_pred) if y_t and y_p)

        self.overlap_sum.assign_add(overlap / width)
        self.count.assign_add(1)

    def result(self):
        return self.overlap_sum / self.count


class ObjectDetectionF1Score(Metric):
    def __init__(self, name='object_detection_f1_score', **kwargs):
        super(ObjectDetectionF1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight:
            raise NotImplementedError('Sample weighting not implemented.')

        y_true = bool(sum(1 for y in y_true if y))
        y_pred = bool(sum(1 for y in y_pred if y))

        if y_true and y_pred:
            self.true_positives.assign_add(1)
        elif y_true and not y_pred:
            self.false_negatives.assign_add(1)
        elif not y_true and y_pred:
            self.false_positives.assign_add(1)

    def result(self):
        return self.true_positives / (self.true_positives + 1 / 2 * (self.false_positives + self.false_negatives))


class ObjectSizeRMSE(Metric):
    def __init__(self, name='object_size_rmse', **kwargs):
        super(ObjectSizeRMSE, self).__init__(name=name, **kwargs)
        self.sum_squared_errors= self.add_weight(name='sse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight:
            raise NotImplementedError('Sample weighting not implemented.')

        y_true = sum(1 for y in y_true if y)
        y_pred = sum(1 for y in y_pred if y)

        self.count.assign_add(1)
        self.sum_squared_errors.assign_add((y_true - y_pred)**2)

    def result(self):
        return np.sqrt(self.sum_squared_errors / self.count)


class ObjectCenterRMSE(Metric):
    def __init__(self, name='object_size_rmse', **kwargs):
        super(ObjectSizeRMSE, self).__init__(name=name, **kwargs)
        self.sum_squared_errors= self.add_weight(name='sse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight:
            raise NotImplementedError('Sample weighting not implemented.')



        self.count.assign_add(1)
        self.sum_squared_errors.assign_add((y_true - y_pred)**2)

    def result(self):
        return np.sqrt(self.sum_squared_errors / self.count)
