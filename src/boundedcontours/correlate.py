import numbers
from numba import jit
import numpy as np
import scipy.ndimage


# @jit
def _correlate1d(padded_input, weights, output_dtype=np.float64):
    # FIXME: change dtype back to floats for zero array (here and below), hacked to allow sympy floats.
    # Ensure input and weights are numpy arrays
    padded_input = np.asarray(padded_input)
    weights = np.asarray(weights)

    # Calculate the size of the weights and the padding on each side
    filter_size = len(weights)

    # Initialize the output array
    n = len(padded_input) - filter_size + 1
    output = np.zeros(n, dtype=output_dtype)

    # Perform the correlation operation
    for i in range(n):
        window = padded_input[i : i + filter_size]
        output[i] = np.sum(window * weights)

    return output


def correlate1d(
    input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    try:
        return scipy.ndimage.correlate1d(
            input, weights, axis, output, mode, cval, origin=origin
        )
    except (ValueError, RuntimeError):
        print("Error in scipy.ndimage.correlate1d, using custom correlate1d.")
        pw = len(weights) // 2
        padded_input = np.pad(input, (pw, pw), mode="symmetric")
        output_dtype = input.dtype
        return _correlate1d(padded_input, weights, output_dtype=output_dtype)


def _gaussian_kernel1d(sigma, radius):
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    return phi_x


def find_non_zero_islands(arr):
    arr = np.array(arr)
    # Identify all non-zero elements
    non_zero_indices = np.nonzero(arr)[0]

    if non_zero_indices.size == 0:
        # No non-zero islands
        slices = []
    else:
        # Find differences between consecutive non-zero indices
        diffs = np.diff(non_zero_indices)
        # Identify where the difference is greater than 1, indicating a new island
        island_starts = non_zero_indices[np.where(diffs > 1)[0] + 1]
        island_ends = non_zero_indices[np.where(diffs > 1)[0]]

        # Add the start of the first island and the end of the last island
        island_starts = [non_zero_indices[0]] + island_starts.tolist()
        island_ends = island_ends.tolist() + [non_zero_indices[-1]]

        # Create slice objects for each island
        slices = [
            slice(start, end + 1, None)
            for start, end in zip(island_starts, island_ends)
        ]
    return slices


def gaussian_filter1d(
    input,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    radius=None,
    cond=None,
):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")
    # Since we are calling correlate, not convolve, revert the kernel
    # weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = _gaussian_kernel1d(sigma, lw)[::-1]

    if cond is None:
        output = correlate1d(input, weights, axis, output, mode, cval, 0)
    else:
        if output is None:
            output = np.zeros_like(input)
        slices = find_non_zero_islands(cond)
        for s in slices:
            output[s] = correlate1d(input[s], weights, axis, output[s], mode, cval, 0)
    return output


def gaussian_filter2d(input, sigma, truncate=4.0, cond=None, output_dtype=np.float64):
    output = np.zeros_like(input, dtype=output_dtype)
    if cond is not None:
        # below is zero when there is no input outside the condition
        # FIXME: change this to be an option.
        P_outside_cond = np.sum(input[~(cond.astype(bool))]) / np.sum(input)
        if P_outside_cond > 0:
            input = input.copy()
            input /= 1 - P_outside_cond
    #     # check that input is zero where cond is False
    #     try:
    #         assert np.all(
    #             input[~(cond.astype(bool))] == 0
    #         ), "The input should be zero where cond is False."
    #     except AssertionError:
    #         breakpoint()
    for axis, slice_func in enumerate((
        lambda i: (i, slice(None)),  # [i, :]
        lambda i: (slice(None), i),  # [:, i]
    )):
        for i in range(input.shape[axis]):
            output[slice_func(i)] = gaussian_filter1d(
                input[slice_func(i)],
                sigma,
                axis=-1,
                order=0,
                output=output[slice_func(i)],
                mode="reflect",
                cval=0.0,
                truncate=truncate,
                cond=cond[slice_func(i)] if cond is not None else None,
            )

        input = output

    # if cond is not None:
    #     # remove the padding around the boundary.
    #     output[~(cond).astype(bool)] = 0

    return output
