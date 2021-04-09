from tensorflow.keras.metrics import Metric
import tensorflow as tf
import numpy as np


def mean_overlap(y_true, y_pred):

    # Apply argmax to convert probability distribution to most likely label
    y_pred = tf.math.argmax(y_pred, 2)

    # Compute true width of object and overlap of true & predicted objects
    width = tf.math.count_nonzero(y_true, 1)
    overlap = tf.math.count_nonzero(tf.logical_and(tf.cast(y_pred, tf.bool), tf.cast(y_true, tf.bool)), 1)

    # Compute the percent overlap between true and predicted objects
    percent_overlap = tf.math.divide(overlap, width)

    # Compute mean percent overlap, ignoring NaNs
    return tf.math.reduce_mean(tf.boolean_mask(percent_overlap, tf.math.is_finite(percent_overlap)))


def object_detection_f1_score(y_true, y_pred):

    # Apply argmax to convert probability distribution to most likely label
    y_pred = tf.math.argmax(y_pred, 2)

    # Convert y_true and y_pred into boolean true/false for each sample: object detected or not
    y_true = tf.cast(tf.math.count_nonzero(y_true, 1), tf.bool)
    y_pred = tf.cast(tf.math.count_nonzero(y_pred, 1), tf.bool)

    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.logical_not(y_true), y_pred), tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.math.logical_not(y_pred)), tf.float32))

    return tf.math.divide(
        true_positives, tf.math.add(true_positives, tf.math.multiply(
            0.5, tf.math.add(false_positives, false_negatives)
        ))
    )


def object_size_rmse(y_true, y_pred):
    # Apply argmax to convert probability distribution to most likely label
    y_pred = tf.math.argmax(y_pred, 2)

    # Compute width of true and predicted objects
    width_true = tf.math.count_nonzero(y_true, 1)
    width_pred = tf.math.count_nonzero(y_pred, 1)

    squared_error = tf.math.pow(tf.math.subtract(width_true, width_pred), 2)

    return tf.math.sqrt(tf.reduce_mean(tf.cast(squared_error, tf.float64)))


def get_first_occurrence_indices(sequence, eos_idx):
    '''
    args:
        sequence: [batch, length]
        eos_idx: scalar
    '''
    batch_size, maxlen = sequence.get_shape().as_list()
    eos_idx = tf.convert_to_tensor(eos_idx)
    tensor = tf.concat(
        [sequence, tf.tile(eos_idx[None, None], [batch_size, 1])], axis=-1)
    index_all_occurrences = tf.where(tf.equal(tensor, eos_idx))
    index_all_occurrences = tf.cast(index_all_occurrences, tf.int32)
    index_first_occurrences = tf.segment_min(index_all_occurrences[:, 1],
                                             index_all_occurrences[:, 0])
    index_first_occurrences.set_shape([batch_size])
    index_first_occurrences = tf.minimum(index_first_occurrences + 1, maxlen)

    return index_first_occurrences


def object_center(y):
    # Compute width of objects
    width = tf.cast(tf.math.count_nonzero(y, 1), tf.float64)

    # Compute start index of objects
    start = tf.cast(get_first_occurrence_indices(tf.cast(y, tf.bool), True), tf.float64)

    # Compute center location of true and predicted objects
    return tf.math.add(start, tf.math.multiply(0.5, width))


def object_center_rmse(y_true, y_pred):
    # Apply argmax to convert probability distribution to most likely label
    y_pred = tf.math.argmax(y_pred, 2)

    # Compute squared error of true & predicted center locations
    squared_error = tf.math.pow(tf.math.subtract(object_center(y_true), object_center(y_pred)), 2)

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
