from typing import List
import numpy as np
import scipy.ndimage


def find_non_zero_islands(arr: np.ndarray) -> List[slice]:
    """
    Find contiguous non-zero islands in a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input array in which to find non-zero islands.

    Returns
    -------
    List[slice]
        A list of slice objects, each corresponding to a contiguous non-zero island in the input array.

    Notes
    -----
    An island is defined as a sequence of one or more consecutive non-zero elements in the array.

    Examples
    --------
    >>> import numpy as np
    >>> from boundedcontours.filter import find_non_zero_islands
    >>> find_non_zero_islands([0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    [slice(4, 7, None), slice(9, 12, None)]
    """
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
    input: np.ndarray,
    sigma: float,
    order: int = 0,
    output: np.ndarray = None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    cond: np.ndarray = None,
) -> np.ndarray:
    """
    Apply a Gaussian filter to a 2D array, optionally within regions defined by a condition array.

    Parameters
    ----------
    input : np.ndarray
        The input array to filter.
    sigma : float
        The standard deviation for Gaussian kernel.
    order : int, optional
        The order of the filter along each axis. Default is 0, which means a Gaussian blur.
    output : np.ndarray, optional
        The array to store the output of the filter. If None, a new array will be created.
    mode : str, optional
        The mode parameter determines how the input array is extended beyond its boundaries. Default is 'reflect'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    cond : np.ndarray, optional
        An optional condition array. The filter is applied only to the regions where cond is non-zero.

    Returns
    -------
    np.ndarray
        The filtered array.

    Notes
    -----
    This function applies a Gaussian filter along each axis of a 2D array, with the ability to apply the filter only within regions specified by a condition array.

    Examples
    --------
    >>> import numpy as np
    >>> from boundedcontours.filter import gaussian_filter2d
    >>> h = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> x, y = np.indices(h.shape)
    >>> condition_func = lambda x, y: x >= y
    >>> x, y = np.indices(h.shape)
    >>> cond = condition_func(x, y)
    >>> h[~cond] = 0
    >>> gaussian_filter2d(h, sigma=1, truncate=1, cond=cond)
    np.array([[1, 0, 0], [4, 5, 0], [6, 6, 8]])
    """
    output = input.copy()

    for axis, slice_func in enumerate((
        lambda i: (i, ...),  # [i, :]
        lambda i: (..., i),  # [:, i]
    )):
        if cond is None:
            output = scipy.ndimage.gaussian_filter1d(
                input,
                sigma,
                axis=axis,
                order=order,
                output=output,
                mode=mode,
                cval=cval,
                truncate=truncate,
            )

        else:  # Apply line by line, island by island
            for i in range(input.shape[axis]):
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
