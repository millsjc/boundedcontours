from numba import jit
import numpy as np


@jit
def correlate1d(padded_input, weights):
    # Ensure input and weights are numpy arrays
    padded_input = np.asarray(padded_input)
    weights = np.asarray(weights)

    # Calculate the size of the weights and the padding on each side
    filter_size = len(weights)

    # Initialize the output arra
    n = len(padded_input) - filter_size + 1
    output = np.zeros(n)

    # Perform the correlation operation
    for i in range(n):
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


def pad_boundary(input_array, cond, pad_width):
    """Where there is a boundary s.t. there is a transition from True to False values in cond,
    pad the False (0) values with the symmetric reflected values from the other side of the boundary.
    There will only be a maximum of two boundaries per row.
    e.g. for the 2-boundary cond = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0].astype(bool) and with pad_width=1, the symmetric mode, and the input
    [0,0,1,2,3,0,0,0,0,0], the output is [0,1,1,2,3,3,0,0,0,0].
    """
    padded_input = np.copy(input_array)
    # Convert cond to int for diff calculation
    cond_int = cond.astype(int)
    # Calculate the diff to find transitions
    diff_cond = np.diff(cond_int)

    n_transitions = np.sum(np.abs(diff_cond))
    assert n_transitions <= 2, "There should be at most two boundaries per row."
    if n_transitions == 2:
        # check the boundaries are at least pad_width apart
        change_indices = np.where(diff_cond != 0)[0]
        assert change_indices[1] - change_indices[0] >= pad_width, (
            "The boundaries are too close together, choose a different kernel"
            " radius/truncation."
        )

    for i in range(len(diff_cond)):
        if diff_cond[i] == -1:  # True to False transition
            # Reflect values to the right
            for j in range(1, pad_width + 1):
                if i + j < len(input_array):
                    padded_input[i + j] = input_array[i - j + 1]
        elif diff_cond[i] == 1:  # False to True transition
            # Reflect values to the left
            for j in range(1, pad_width + 1):
                if i + 1 - j >= 0:
                    padded_input[i + 1 - j] = input_array[i + j]

    return padded_input


def gaussian_filter1d(input, sigma, truncate=4.0, *, radius=None, cond=None):
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sigma + 0.5)
    if radius is not None:
        lw = radius
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, lw)[::-1]

    padded_input = np.pad(
        input,
        pad_width=((lw, lw),),
        mode="symmetric",
    )
    if cond is not None:
        # pad symmetrically around the boundary
        padded_input[lw:-lw] = pad_boundary(input, cond, lw)

    out = correlate1d(padded_input, weights)
    if cond is not None:
        # breakpoint()
        # remove the padding around the boundary.
        out[~(cond.astype(bool))] = 0
    return out


def gaussian_filter2d(input, sigma, output=None, truncate=4.0, cond=None):
    output = np.zeros_like(input)
    if cond is not None:
        # check that input is zero where cond is False
        assert np.all(
            input[~(cond.astype(bool))] == 0
        ), "The input should be zero where cond is False."
    for axis, slice_func in enumerate(
        (
            lambda i: (i),  # [i, :]
            lambda i: (slice(None), i),  # [:, i]
        )
    ):
        for i in range(input.shape[axis]):
            output[slice_func(i)] = gaussian_filter1d(
                input[slice_func(i)],
                sigma,
                truncate=truncate,
                cond=cond[slice_func(i)] if cond is not None else None,
            )

        input = output

    # if cond is not None:
    #     # remove the padding around the boundary.
    #     output[~(cond).astype(bool)] = 0

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

    return convolve2d_with_symmetric_padding(input, kernel)
