import math
import numpy as np


def resample_y(data, input_time_range, output_sample_rate, output_time_range):
    """
    Resamples the provided data in the y direction to the specified output time range and sample rate.

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


def polynomial_resample(x, x_range, output_size):
    """
    Assume samples are taken at fixed distance intervals
    :param x: the series to resample
    :param x_range: the distance range covered by the input points
    :param output_size: the number of columns in the output
    :return:
    """

    # Distance coordinates for each column in the input data
    d = np.linspace(0, x_range, len(x))

    # Distance coordinates for each column in the output data
    d_output = np.linspace(0, x_range,  output_size)

    return np.vectorize(lambda d_val: quadratic_interpolate(d, x, d_val))(d_output)


def linear_resample(x, x_range, output_size):
    """
    Resamples the provided input series x to the specified output size using linear interpolation. Assume samples are
    taken at fixed distance intervals.

    :param x: the series to resample
    :param x_range: the distance range covered by the input points in meters
    :param output_size: the number of columns in the output
    :return: the input series resampled to the specified size
    """

    # Distance coordinates for each column in the input data
    d = np.linspace(0, x_range, len(x))

    # Distance coordinates for each column in the output data
    d_output = np.linspace(0, x_range,  output_size)

    return np.vectorize(lambda d_val: linear_interpolate(d, x, d_val))(d_output)


def find_between_index(array, element):
    for i, a in enumerate(array):
        if a > element:
            return i - 1

    return len(array)


def quadratic_interpolate(x, y, x_val):
    if x_val in x:
        # The x-value is in the input series, so return its corresponding y value
        return y[np.where(x == x_val)[0]]

    j = find_between_index(x, x_val)
    if j < 0:
        # Right-sided interpolation
        x_fit = x[:2]
        y_fit = x[:2]
    elif j < 1:
        # Asymmetrical interpolation - one point on left side, two on right
        x_fit = np.concatenate((x[:1], x[1:3]))
        y_fit = np.concatenate((y[:1], y[1:3]))
    elif j > len(x) - 1:
        # Left-sided interpolation
        x_fit = x[-2:]
        y_fit = y[-2:]
    elif j > len(x) - 2:
        # Asymmetrical interpolation - two points on left side, one on right
        x_fit = np.concatenate((x[-3:-1], x[-1:]))
        y_fit = np.concatenate((y[-3:-1], y[-1:]))
    else:
        # Normal case
        x_fit = np.concatenate((x[j - 1:j + 1], x[j + 1:j + 3]))
        y_fit = np.concatenate((y[j - 1:j + 1], y[j + 1:j + 3]))

    # except:
    #     print(j)
    #     print(x[:1], x[2:4])
    #     print(y[:1], y[2:4])

    # Fit 2nd degree polynomial and perform interpolation
    p = np.poly1d(np.polyfit(x_fit, y_fit, 2))
    return p(x_val)


def linear_interpolate(x, y, x_val):
    if x_val in x:
        # The x-value is in the input series, so return its corresponding y value
        return y[np.where(x == x_val)[0]]


    j = find_between_index(x, x_val)
    if j < 0:
        # Right-sided interpolation
        return y[0]
    elif j > len(x) - 1:
        # Left-sided interpolation
        return y[-1]
    else:
        # Normal case
        x_fit = np.concatenate((x[j:j + 1], x[j + 1:j + 2]))
        y_fit = np.concatenate((y[j:j + 1], y[j + 1:j + 2]))

    # Perform linear interpolation
    return np.interp(x_val, x_fit, y_fit)


def resample_x(data, x_range, output_size):
    """
    Resamples each row of the provided input data to the specified output size using linear interpolation.
    Assume samples are taken at fixed distance intervals.

    :param x: the series to resample
    :param x_range: the distance range covered by the input points in meters
    :param output_size: the number of columns in the output
    :return: the input series resampled to the specified size
    """

    resampled_data = np.zeros((data.shape[0], output_size))

    # Iterate over rows, resampling each one
    for i in range(data.shape[0]):
        resampled_data[i, :] = linear_resample(data[i, :], x_range, output_size)

    return resampled_data


def resample_xy(data, input_time_range, output_sample_rate, output_time_range, x_range, output_size):
    """
    Resample the provided matrix of data.

    :param data: 2D matrix of data to be resampled
    :param input_time_range: time range over which data was collected
    :param output_sample_rate: y-direction output sample rate in samples per second
    :param x_range: the distance range covered by the input points in meters
    :param output_size: the number of columns in the output
    :return: the input matrix resampled according to the parameters provided
    """
    return resample_x(resample_y(data, input_time_range, output_sample_rate, output_time_range), x_range, output_size)


def slice_scan(data, window_size):
    """
    Slices the provided b-scan data into fixed size chunks, each containing window_size columns.

    :param data: the b_scan to slice
    :param window_size: the number of columns in each slice
    :return: a sequence of slices of the input data
    """

    n_steps = data.shape[1] - window_size + 1
    return [data[:, n:n + window_size] for n in range(n_steps)]


def preprocess_scan(data, input_time_range, output_sample_rate, output_time_range, x_range, output_size, window_size):
    """
    Pre-processes the b-scan represented by the provided data to prepare it for use as input to a neural network. The
    provided data is resampled in both the x and y directions according to the parameters provided. Resampling is
    performed first in the y-direction using a DFT resampling method along with padding or truncation is needed.
    Next, the data is resampled in the x direction using linear interpolatio so that all scans are transformed to
    include the same number of samples per meter. The resulting scan is then sliced using a sliding window to
    generate a large number of shorter scans according to the provided window_size. The shape of each slice in the
    output data is (output_sample_rate * output_time_range, window_size).

    :param data: 2D matrix of data to be resampled
    :param input_time_range: time range over which data was collected
    :param output_sample_rate: y-direction output sample rate in samples per second
    :param x_range: the distance range covered by the input points in meters
    :param output_size: the number of columns in the output
    :param window_size: the number of columns in each slice
    :return: a sequence of slices of the resampled input data
    """

    return slice_scan(
        resample_xy(data, input_time_range, output_sample_rate, output_time_range, x_range, output_size), window_size
    )


def normalize():
    pass
