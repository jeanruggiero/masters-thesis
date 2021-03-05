import math
import numpy as np


def resample_y(data, input_time_range, output_sample_rate, output_time_range):
    """
    Resamples the provided data in the y direction to the specified output time range and sample rate.

    Probably not this - Takes a 2D matrix as input and returns a 3D matrix with third dimension containing 3
    elements: (0) min, (1), max, (2) mean.

    The y-axis represents time. It is assumed that the acquisition rate of the data is constant over the entire time
    range and that the first sample is acquired at time t=0. An approximation to the input data is constructed in the
    form of a sum of sines. The approximated waveform is then sampled at the requested rate.

    :param data: 2D matrix of data to be resampled
    :param time_range: time range over which data was collected
    :param sample_rate: output sample rate in samples per second
    :return: the resampled data
    """

    resampled_data = np.zeros((output_sample_rate * output_time_range, data.shape[1]))

    # Iterate over columns of the input data, resampling each one
    for j in range(data.shape[1]):
        resampled_data[:, j] = dft_resample(data[:, j], input_time_range, output_sample_rate, output_time_range)

    return resampled_data


def timeseries_pad(y, input_time_range, output_time_range):
    """
    Truncate or pad the provides time series as needed to achieve a series with the provided output time range.

    :param y: input time series
    :param input_time_range: time range of the input series
    :param output_time_range: desired time range of the output series
    :return: the input time series adjusted to the output_time_range
    """

    input_sample_rate = len(y) / input_time_range

    if input_time_range > output_time_range:
        # Truncate series to output_time_range ns
        return y[:math.floor(input_time_range * input_sample_rate)]
    elif input_time_range < output_time_range:
        # Pad the end of the series with the last measured value
        return np.concatenate((y,
            np.full(math.floor((output_time_range - input_time_range) * input_sample_rate), y[-1])))
    else:
        # The input series has the appropriate time range
        return y


def dft_resample(y, input_time_range, output_sample_rate, output_time_range):
    """
    Uses a discrete Fourier transform to resample the provided series y over the given time range to a series over
    the provided output time range and with the specified output sample rate. Applies padding or truncation to the
    series y as needed to obtain a series over the output time range.

    It is assumed that the acquisition rate of the data is constant over the entire time
    range and that the first sample is acquired at time t=0. An approximation to the input data is constructed in the
    form of a sum of sines. The approximated waveform is then sampled at the requested rate.

    :param y: the time series to resample
    :param input_time_range: the time range corresponding to the input time series
    :param output_sample_rate: the sample rate of the output series
    :param output_time_range: the time range of the output series
    :return: the resampled series
    """

    # Adjust time range as needed
    y = timeseries_pad(y, input_time_range, output_time_range)

    # Use the FFT to convert to the frequency domain
    f = np.fft.fft(y)
    # Shift zero frequency to center of spectrum
    f = np.fft.fftshift(f)
    if output_sample_rate > len(y) / input_time_range:
        # Output sample rate is greater than input sample rate, so we need to pad

        # Number of samples in the resampled series
        n_samples = output_time_range * output_sample_rate
        # Number of additional samples required in the output series
        padding = n_samples - len(y)

        # Pad both ends evenly with zeros
        f = np.concatenate((np.zeros(math.ceil(padding / 2)), f, np.zeros(math.floor(padding / 2))))
    else:
        raise NotImplementedError("Downsampling not implemented.")

    # Inverse shift
    f = np.fft.ifftshift(f)
    # Inverse fft & rescale
    return np.real(np.fft.ifft(f) * len(f) / len(y))


def resample_x(data, ):
    """
    Resamples
    :return:
    """
    pass

def normalize():
    pass
