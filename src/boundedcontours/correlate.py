from numba import jit
import numpy as np


# @jit
def correlate1d(input_array, weights, mode="symmetric", padded_input=None):
    # Ensure input and weights are numpy arrays
    input_array = np.asarray(input_array)
    weights = np.asarray(weights)

    # Calculate the size of the weights and the padding on each side
    filter_size = len(weights)

    # Initialize the output array
    output = np.zeros_like(input_array)

    # Perform the correlation operation
    for i in range(len(input_array)):
        window = padded_input[i : i + filter_size]
        output[i] = np.sum(window * weights)

    return output


def _gaussian_kernel1d(sigma, radius):
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    return phi_x


def _gaussian_kernel2d(sigma, radius):
    weights = _gaussian_kernel1d(sigma, radius)
    weights = np.outer(weights, weights)
    return weights


def _gaussian_kernel2d_2(sigma, radius):
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(x, y)
    x_mean = 0
    y_mean = 0
    weights = np.exp(-0.5 * ((x - x_mean) ** 2 + (y - y_mean) ** 2) / sigma**2)
    # weights = weights / weights.sum()
    return weights


def gaussian_filter1d(input, sigma, mode="symmetric", truncate=4.0, *, radius=None):
    """FIXME: consider removing the mode argument and just using symmetric padding (since we hard code symmetric padding for reflection at boundaries other than the edge)."""
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, lw)[::-1]

    padded_input = np.pad(
        input,
        pad_width=((lw, lw),),
        mode=mode,
    )
    return correlate1d(input, weights, mode)


def pad(input_array, cond, pad_width, axis, mode="symmetric"):
    """Where there is a boundary s.t. there is a transition from True to False values in cond,
    pad the False (0) values with the values from the other side of the boundary. There will only be a maximum of two boundaries per row.
    e.g. for the 2-boundary cond = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0] and with pad_width=1, the symmetric mode, and the input
    [0,0,1,2,3,0,0,0,0,0], the output is [0,1,1,2,3,3,0,0,0,0].

    Parameters
    ----------
    input_array : _type_
        _description_
    cond : _type_
        _description_
    pad_width : _type_
        _description_
    axis : _type_
        _description_
    mode : str, optional
        _description_, by default "symmetric"

    Returns
    -------
    _type_
        _description_
    """
    padded_input = np.pad(input_array, pad_width, mode=mode)

    return padded_input


def gaussian_filter2d(
    input, sigma, output=None, mode="symmetric", truncate=4.0, cond=None
):
    output = np.zeros_like(input)
    # pad_width = int(truncate * sigma + 0.5)
    for axis in [0, 1]:
        # if axis == 0:
        # input = pad(input, cond, pad_width, axis, mode=mode)
        for i in range(input.shape[axis]):
            output[i] = gaussian_filter1d(
                input[i],
                sigma,
                mode=mode,
                truncate=truncate,
            )
            input = output
        # else:
        # if cond is not None:
        # remove the padding from the previous axis
        # this time we use the previous output as input
        # output[~cond] = 0
        # breakpoint()
        # # output = pad(output, cond, pad_width, axis, mode=mode)
        # for i in range(input.shape[axis]):
        #     output[:, i] = gaussian_filter1d(
        #         output[:, i],
        #         sigma,
        #         mode=mode,
        #         truncate=truncate,
        #     )

    if cond is not None:
        # remove the padding around the boundary.
        output[~cond] = 0

    return output


def convolve2d_with_symmetric_padding(input_array, kernel):
    """
    Perform a 2D convolution with symmetric padding.

    :param input_array: 2D array representing the input.
    :param kernel: 2D array representing the kernel/filter.
    :return: 2D array representing the convolved output.
    """

    # Dimensions of the input and kernel
    input_rows, input_cols = input_array.shape
    kernel_rows, kernel_cols = kernel.shape

    # Padding width
    pad_height = kernel_rows // 2
    pad_width = kernel_cols // 2

    # Pad the input array symmetrically
    padded_input = np.pad(input_array, (pad_height, pad_width), mode="symmetric")

    # Dimensions of the padded input
    padded_rows, padded_cols = padded_input.shape

    # Output dimensions
    output_rows = padded_rows - kernel_rows + 1
    output_cols = padded_cols - kernel_cols + 1

    # Initialize the output array
    output = np.zeros((output_rows, output_cols))

    # Perform the convolution
    for i in range(output_rows):
        for j in range(output_cols):
            output[i, j] = np.sum(
                padded_input[i : i + kernel_rows, j : j + kernel_cols] * kernel
            )
        # breakpoint()

    return output


def gaussian_filter_2d_convolve_2d(
    input, sigma, mode="symmetric", truncate=4.0, radius=None, cond=None
):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    kernel = _gaussian_kernel2d(sigma, lw)
    breakpoint()

    return convolve2d_with_symmetric_padding(input, kernel)
