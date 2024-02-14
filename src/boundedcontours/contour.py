from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter

from . import correlate

gaussian_filter2d = correlate.gaussian_filter2d


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
    # Sort it in descending order and calculate the cumulative probability
    H_flat = sorted(H.flatten(), reverse=True)
    cumulative_prob = np.cumsum(H_flat)
    # Find the bin with a cumulative probability greater than the desired
    # level and return the value of the bin
    i_level = np.where(cumulative_prob / cumulative_prob[-1] >= p_level)[0][0]
    level = H_flat[i_level]
    return level


def smooth_2d_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, np.ndarray, Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = 100,
    sigma_smooth: float = 1.0,
    gaussian_filter2d_kwargs: dict = {},
    condition_function=None,
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
    condition_function : callable, optional
        condition to apply to the histogram. Default is None.
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

    H, bins_x, bins_y = np.histogram2d(x, y, bins=bins, density=True)  # type: ignore
    X, Y = np.meshgrid((bins_x[1:] + bins_x[:-1]) / 2, (bins_y[1:] + bins_y[:-1]) / 2)
    H = H.T  # output of histogram2d is [y, x] but we want [x, y]

    cond = condition_function(X, Y) if condition_function is not None else None
    H = gaussian_filter2d(
        H, sigma=sigma_smooth, truncate=truncate, cond=cond, **gaussian_filter2d_kwargs
    )

    return X, Y, H


def contour_at_level(
    x: np.ndarray,
    y: np.ndarray,
    p_levels: Union[float, List[float]] = 0.9,
    bins: Union[int, np.ndarray, Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = 100,
    sigma_smooth: float = 1,
    ax: Optional[mpl.axes.Axes] = None,
    plot_pcolormesh: bool = False,
    gaussian_filter2d_kwargs: dict = {},
    condition_function=None,
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
    condition_function : callable, optional
        A function that takes x and y and returns a boolean array of the same shape.
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
        gaussian_filter2d_kwargs=gaussian_filter2d_kwargs,
        condition_function=condition_function,
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
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
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
    condition_function=None,
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
    condition_function : callable, optional
        Condition to apply to the histogram during smoothing. Default is None. The function should take x and y as arguments and return a boolean array of the same shape.
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
            condition_function=condition_function,
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
