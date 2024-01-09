from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter


def level_from_credible_interval(H: np.ndarray, p_level: float = 0.9) -> float:
    """
    Find the level of a credible interval from a histogram.

    Parameters
    ----------
    H : 2D array
        2D array of the histogram
    p_level : float, optional
        The desired credible interval. Default is 0.9.
    Returns
    -------
    level : float
        The value of H at the desired credible interval.
    """
    # Flatten the histogram into a 1D array and sort it in descending order
    H_flat = sorted(H.flatten(), reverse=True)
    # Calculate the cumulative probability of each bin
    cumulative_prob = np.cumsum(H_flat)
    # Find the bin with a cumulative probability greater than the desired
    # level and return the value of the bin
    i_level = np.where(cumulative_prob / cumulative_prob[-1] >= p_level)[0][0]
    level = H_flat[i_level]
    return level


def symmetric_2d_gaussian(
    x: np.ndarray, y: np.ndarray, x_mean: float, y_mean: float, sigma: float = 1
) -> np.ndarray:
    return np.exp(-0.5 * ((x - x_mean) ** 2 + (y - y_mean) ** 2) / sigma**2)


def smooth_with_condition(
    H: np.ndarray, cond: np.ndarray, sigma: float = 1.0, truncate: int = 4
) -> np.ndarray:
    """Smooth using a Gaussian filter, but only over pixels where the condition is met.

    Parameters
    ----------
    H : 2d array
        the input array in [x, y] order (see Notes)
    cond : 2d array of bools
        must be the same shape as H
    sigma : float, optional
        the std of the smoothing gaussian, by default 1
    truncate : int, optional
        the number of stds to truncate the gaussian at, by default 4

    Returns
    -------
    2d array
        the smoothed array

    Notes
    -----
    Input array must be in [x,y] order as required by contour and
    gaussian_filter, but not returned by histogram2d!

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from contour import smooth_with_condition
    >>> x = np.random.normal(0, 1, 1000)
    >>> y = np.random.normal(0, 1, 1000)
    >>> H, x_bins, y_bins = np.histogram2d(x, y, bins=100)
    >>> X, Y = np.meshgrid((x_bins[1:] + x_bins[:-1])/2, (y_bins[1:] + y_bins[:-1])/2)
    >>> cond = H > 0
    >>> H_smooth = smooth_with_condition(H, cond, sigma=1, truncate=4)
    >>> plt.contour(X, Y, H_smooth.T, levels=2, cmap='plasma', label='smoothed')
    >>> plt.contout(X, Y, H.T, levels=2, cmap='reds', label='original')
    >>> plt.legend()
    >>> plt.show()

    Dev Notes
    ---------
    Code outline: For each pixel, make a 2d gaussian with that pixel at the centre
    with a window of size "truncate". It should also be zeroed where
    the condition "cond" is not met. Normalize it. This will be the
    weight array. The new value of the pixel is the sum of the weights
    times the values of the pixels in the window.
    A good test at the end is that the sum of the final 2d array should
    be the same as the sum of the original array.
    """
    assert H.shape == cond.shape, "H and cond must have the same shape"
    assert H.ndim == 2, "H must be a 2D array"
    assert cond.ndim == 2, "cond must be a 2D array"

    # Determine the size of the Gaussian window to use based on truncate and sigma
    window_radius = int(truncate * sigma + 0.5)

    # Create an array to hold the smoothed image
    H_smooth = np.zeros_like(H)

    # For each pixel in the image...
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if not cond[i, j]:
                continue
            # Define the bounds of the window
            x_start = max(0, i - window_radius)
            x_end = min(H.shape[0], i + window_radius + 1)
            y_start = max(0, j - window_radius)
            y_end = min(H.shape[1], j + window_radius + 1)

            # Create a Gaussian kernel centered on the pixel
            x = np.arange(x_start, x_end, dtype=H.dtype)[:, np.newaxis]
            y = np.arange(y_start, y_end, dtype=H.dtype)

            kernel = symmetric_2d_gaussian(x, y, i, j, sigma)

            assert kernel.shape == (
                x_end - x_start,
                y_end - y_start,
            ), "kernel has the wrong shape"

            # Zero out the kernel where the condition is not met
            kernel[~cond[x_start:x_end, y_start:y_end]] = 0

            # Normalize the kernel so it sums to 1
            kernel = kernel / np.sum(kernel)

            # Compute the new value for this pixel
            H_smooth[i, j] = np.sum(kernel * H[x_start:x_end, y_start:y_end])

    return H_smooth


def smooth_2d_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, np.ndarray, Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = 100,
    sigma_smooth: float = 1.0,
    gaussian_filter_kwargs: dict = {},
    condition: Union[str, None] = None,
    truncate: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed 2D histogram.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        x samples
    y : array_like, shape (n_samples,)
        y samples
    bins : int or array_like or [int, int] or [array, array], optional
        number of bins on each axis. Default is 100.
    sigma_smooth : float, optional
        sigma of gaussian filter. Default is 1.
    gaussian_filter_kwargs : dict, optional
        kwargs to pass to `scipy.ndimage.gaussian_filter`. Default is {}.
    condition : str, optional
        condition to apply to the histogram. Default is None. Must be one of: "X>Y".
    truncate : int, optional
        number of stds to truncate the gaussian at. Default is 4.

    Returns
    -------
    X : array_like, shape (bins, bins)
        X meshgrid
    Y : array_like, shape (bins, bins)
        Y meshgrid
    H : array_like, shape (bins, bins)
        histogram in [x, y] order (not [y, x] as returned by `np.histogram2d`)
    """

    H, bins_x, bins_y = np.histogram2d(x, y, bins=bins, density=True)
    X, Y = np.meshgrid((bins_x[1:] + bins_x[:-1]) / 2, (bins_y[1:] + bins_y[:-1]) / 2)
    H = H.T  # output of histogram2d is [y, x] but we want [x, y]
    # do some smoothing
    if condition is None:
        H = gaussian_filter(H, sigma=sigma_smooth, **gaussian_filter_kwargs)
    elif condition.lower() == "x>y":
        H = smooth_with_condition(H, cond=X > Y, sigma=sigma_smooth, truncate=truncate)
    else:
        raise NotImplementedError(f"condition {condition} not implemented")

    return X, Y, H


def contour_at_level(
    x: np.ndarray,
    y: np.ndarray,
    p_levels: Union[float, List[float]] = 0.9,
    bins: Union[int, np.ndarray, Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = 100,
    sigma_smooth: float = 1,
    ax: Optional[mpl.axes.Axes] = None,
    plot_pcolormesh: bool = False,
    gaussian_filter_kwargs: dict = {},
    condition: Union[str, None] = None,
    mpl_contour_kwargs=dict(colors="k", linewidths=2, linestyles="--"),
    truncate: int = 4,
) -> mpl.axes.Axes:
    """Plot the contour at the level corresponding to the p_levels credible interval.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        x samples
    y : array_like, shape (n_samples,)
        y samples
    p_levels : float or list of floats, optional
        The credible interval(s) to plot. Default is 0.9.
    bins : int or array_like or [int, int] or [array, array], optional
        Number of bins to use in the 2d histogram. Default is 100.
    sigma_smooth : float, optional
        Sigma for the Gaussian filter applied to the 2d histogram. Default is 1.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, a new figure is created.
    plot_pcolormesh : bool, optional
        If True, plot a pcolormesh of the 2d histogram. Default is False.
    gaussian_filter_kwargs : dict, optional
        Additional keyword arguments to pass to `scipy.ndimage.gaussian_filter`.
        Default is {}.
    condition : str, optional
        Condition to apply to the histogram. Default is None. Must be one of: "X>Y".
    mpl_contour_kwargs : dict, optional
        Additional keyword arguments to pass to `matplotlib.axes.Axes.contour`.
        Default is dict(colors="k", linewidths=2, linestyles="--").
    truncate : int, optional
        Number of stds to truncate the gaussian at. Default is 4.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the contour.
    """
    X, Y, H = smooth_2d_histogram(
        x,
        y,
        bins=bins,
        sigma_smooth=sigma_smooth,
        gaussian_filter_kwargs=gaussian_filter_kwargs,
        condition=condition,
        truncate=truncate,
    )

    p_levels = sorted(np.ravel(p_levels), reverse=True)  # ravel handles list or float

    levels = [level_from_credible_interval(H, p) for p in p_levels]

    if ax is None:
        fig, ax = plt.subplots()
    if plot_pcolormesh:
        ax.pcolormesh(X, Y, H)
    ax.contour(X, Y, H, levels=levels, **mpl_contour_kwargs)

    return ax


def get_min_max_sample(
    samples_list: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[float, float]:
    """
    Get the min and max values of x and y from a list of samples.

    Parameters
    ----------
    samples_list : list of tuples of np.ndarray
        A list of sets of x and y samples.

    Returns
    -------
    min_sample : float
        The minimum x value found in samples_list.
    max_sample : float
        The maximum x value found in samples_list.
    """
    ax_min, ax_max = [], []
    for x_samp, y_samp in samples_list:
        ax_min.append(min([min(x_samp), min(y_samp)]))
        ax_max.append(max([max(x_samp), max(y_samp)]))
    return min(ax_min), max(ax_max)


def get_2d_bins(
    x: List[float],
    y: List[float],
    target_nbins: int = 100,
    lower_bound_safety_factor: float = 0.7,
    min_bin_width: Union[int, float] = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attempt to make target_nbins bins, but increase the number of bins
    until the computed bin width is less than min_bin_width (up to rounding/truncation
    errors).

    Parameters
    ----------
    x : list of float
        The data for the x-axis.
    y : list of float
        The data for the y-axis.
    target_nbins : int or float, optional
        The desired number of bins. Default is 100.
    lower_bound_safety_factor : float, optional
        A factor used to shrink the minimum x and y values to ensure contours drawn nicely. Default is 0.7.
    min_bin_width : int, optional
        The minimum bin spacing. If the computed bin width based on the range of the data and the target number of bins is less than this, this value will be used instead. Default is 4.

    Returns
    -------
    tuple of np.ndarray
        A tuple of two 1D numpy arrays containing the bin boundaries for the x and y axes.
    """

    def _get_lower_bound(min_value, safety_factor):
        safety_factor = 1 / safety_factor if min_value < 0 else safety_factor
        return safety_factor * min_value

    xmin, xmax = _get_lower_bound(min(x), lower_bound_safety_factor), max(x) + 0.5
    ymin, ymax = _get_lower_bound(min(y), lower_bound_safety_factor), max(y) + 0.5
    x_width = xmax - xmin
    y_width = ymax - ymin
    bin_width = min(min_bin_width, x_width / target_nbins, y_width / target_nbins)
    x_bins = np.linspace(xmin, xmax, int(x_width / bin_width + 0.5))
    y_bins = np.linspace(ymin, ymax, int(y_width / bin_width + 0.5))
    return x_bins, y_bins


def contour_plots(
    samples_list: List[Tuple[np.ndarray, np.ndarray]],
    labels: List[str],
    axes_labels: List[str],
    condition: Optional[str] = None,
    ax: Optional[mpl.axes.Axes] = None,
    legend_ax: Optional[mpl.axes.Axes] = None,
    colors: Optional[List[str]] = None,
    target_nbins: int = 100,
    min_bin_width: float = 4,
    linewidth: float = 1.5,
    sigma_smooth: float = 2,
    truncate: int = 4,
    p_levels: List[float] = [0.9],
    axes_lims: Optional[List[float]] = None,
    bins_2d_kwargs: dict = {},
    plot_pcolormesh: bool = False,
) -> Tuple[mpl.axes.Axes, mpl.axes.Axes]:
    """Creates contour plots of given list of samples and annotates them with labels.

    Parameters
    ----------
    samples_list : list of tuple of np.ndarray
        A list of sets of x and y samples.
    labels : list of str
        Labels for each set of samples.
    axes_labels : list of str
        Labels for the x and y axes.
    condition : str, optional
        Condition to apply to the histogram during smoothing. Default is None. Must be one of: ["X>Y"].
    ax : mpl.axes.Axes, optional
        The axes on which to plot the contour. If None, a new axes object is created. Default is None.
    legend_ax : mpl.axes.Axes, optional
        The axes on which to place the legend. If None, the legend is placed on the plot axes. Default is None.
    colors : list of str, optional
        The colors to use for each set of samples. If None, the default matplotlib color cycle is used. Default is None.
    target_nbins : int, optional
        The target number of bins for the histogram. Default is 100.
    min_bin_width : float, optional
        The minimum bin width for the histogram, nbins will increase until this is met. Default is 4.
    linewidth : float, optional
        The line width of the contour lines. Default is 1.5.
    sigma_smooth : float, optional
        The standard deviation of the Gaussian kernel used for smoothing. Default is 2.
    truncate : int, optional
        The truncation radius for Gaussian kernel. Default is 4.
    p_levels : list of float, optional
        The percentile levels for the contour plot. Default is [0.9].
    axes_lims : list of float, optional
        The limits for the x and y axes. Default is None.
    bins_2d_kwargs : dict, optional
        Additional keyword arguments to pass to `get_2d_bins`. Default is {}.
    plot_pcolormesh : bool, optional
        If True, plot a pcolormesh of the 2d histogram. Default is False.

    Returns
    -------
    tuple of mpl.axes.Axes, mpl.axes.Axes
        A tuple containing the contour and the legend axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    legend_handles = []
    for i, ss in enumerate(samples_list):
        bins_x, bins_y = get_2d_bins(
            ss[0],
            ss[1],
            target_nbins=target_nbins,
            min_bin_width=min_bin_width,
            **bins_2d_kwargs,
        )
        linestyle = "-"
        if axes_lims is not None:
            mmax = axes_lims[1]
            out_of_bounds = sum((ss[0] > mmax) | (ss[1] > mmax)) / len(ss[1])
            if out_of_bounds > 0.1:
                print(
                    f"Warning: {int(out_of_bounds * 100 + 0.5)}% (>10%) of samples are"
                    f" outside of axes limits for {labels[i]}"
                )
                print("Plotting dashed contour")
                linestyle = "--"

        color = colors[i] if colors is not None else None
        ax = contour_at_level(
            ss[0],
            ss[1],
            condition=condition,
            bins=(bins_x, bins_y),
            sigma_smooth=sigma_smooth,
            truncate=truncate,
            ax=ax,
            p_levels=p_levels,
            plot_pcolormesh=plot_pcolormesh,
            mpl_contour_kwargs=dict(
                colors=color, linestyles=linestyle, linewidths=linewidth
            ),
        )
        legend_handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth * 2,
                label=labels[i],
            )
        )

    ax.set_xlabel(axes_labels[0], fontsize=16)
    ax.set_ylabel(axes_labels[1], fontsize=16)

    # Set the tick label font size
    ax.tick_params(axis="both", which="major", labelsize=14)

    if legend_ax is None:
        legend_ax = ax

    if not all([l is None for l in labels]):
        legend = legend_ax.legend(
            handles=legend_handles,
            handlelength=1,
            ncol=1,
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
            fontsize=14,
        )

    return ax, legend_ax


def shade_mass_gap_region(
    ax: mpl.axes.Axes,
    mmax: float = 10000,
    mmin: float = 0,
    mass_gap_strict: Tuple[float, float] = (90, 120),
    mass_gap_loose: Tuple[float, float] = (50, 175),
) -> mpl.axes.Axes:
    """Shade the pair instability mass gap region.

    Parameters
    ----------
    ax : mpl.axes.Axes
        The axis to shade
    mmax : float, optional
        The maximum mass (M_sun) to shade. Default is 10000.
    mmin : float, optional
        The minimum mass (M_sun) to shade. Default is 0.
    mass_gap_strict : tuple of float, optional
        The inner mass gap region to shade in darker colour in M_sun. Default is (60, 120).
    mass_gap_loose : tuple of float, optional
        The outer mass gap region to shade in lighter colour in M_sun. Default is (50, 175).

    Returns
    -------
    mpl.axes.Axes
        The axis with the shaded region.
    """
    for mass_gap, color, alpha in zip(
        [mass_gap_loose, mass_gap_strict], ["gray", "gray"], [0.1, 0.3]
    ):
        reg = Polygon(
            [
                [mass_gap[0], mmin],
                [mass_gap[0], mass_gap[0]],
                [mass_gap[1], mass_gap[1]],
                [mmax, mass_gap[1]],
                [mmax, mass_gap[0]],
                [mass_gap[1], mass_gap[0]],
                [mass_gap[1], mmin],
            ],
            color=color,
            alpha=alpha,
            hatch="//",
        )
        ax.add_patch(reg)
    return ax


def shade_undefined_mass_region(
    ax: mpl.axes.Axes, mmax: float = 10000, mmin: float = 0
) -> mpl.axes.Axes:
    """
    Shades the undefined mass region where m2 > m1 in the plot.

    Parameters
    ----------
    ax : mpl.axes.Axes
        The axes to add the patch to.
    mmax : float, optional
        The maximum mass to be plotted. Default is 10000.
    mmin : float, optional
        The minimum mass to be plotted. Default is 0.

    Returns
    -------
    mpl.axes.Axes
        The axes with the shaded region added.
    """
    reg = Polygon(
        [[mmin, mmin], [mmin, mmax], [mmax, mmax]],
        hatch="\\",
        fill=False,
        color="gray",
    )
    ax.add_patch(reg)
    return ax


def add_injected_value_2d(
    ax: mpl.axes.Axes, injected_value: List[float]
) -> mpl.axes.Axes:
    """
    Adds a scatter point representing an injected value to the plot.

    Parameters
    ----------
    ax : mpl.axes.Axes
        The axes to add the scatter point to.
    injected_value : list of float
        The coordinates of the injected value.

    Returns
    -------
    mpl.axes.Axes
        The axes with the scatter point added.
    """
    ax.scatter(
        injected_value[0],
        injected_value[1],
        marker="x",
        color="k",
        linewidth=2.0,
        s=50,
        zorder=3,
    )
    return ax
