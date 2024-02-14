import numpy as np
import scipy.ndimage

def find_non_zero_islands(arr):
    arr = np.array(arr)
    non_zero_indices = np.nonzero(arr)[0]

    if non_zero_indices.size == 0:
        slices = []
    else:
        diffs = np.diff(non_zero_indices)
        # Identify where the difference is greater than 1, indicating the start of a new island and the end of the previous one
        idx = np.where(diffs > 1)[0]
        island_starts = non_zero_indices[idx + 1]
        island_ends = non_zero_indices[idx]

        # Add the first and last island
        island_starts = [non_zero_indices[0]] + island_starts.tolist()
        island_ends = island_ends.tolist() + [non_zero_indices[-1]]

        # Create slice objects to select each island
        slices = [
            slice(start, end + 1, None)
            for start, end in zip(island_starts, island_ends)
        ]
    return slices


def gaussian_filter2d(
    input,
    sigma,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    # radius=None,
    cond=None,
):
    output = input.copy()

    for axis, slice_func in enumerate((
        lambda i: (i, ...),  # [i, :]
        lambda i: (..., i),  # [:, i]
    )):
        for i in range(input.shape[axis]):
            if cond is None:
                island_slices = [...]
            else:
                island_slices = find_non_zero_islands(cond[slice_func(i)])
            for s in island_slices:
                output[slice_func(i)][s] = scipy.ndimage.gaussian_filter1d(
                    input[slice_func(i)][s],
                    sigma,
                    axis=-1,
                    order=order,
                    output=output[slice_func(i)][s],
                    mode=mode,
                    cval=cval,
                    truncate=truncate,
                )

        input = output

    return output
