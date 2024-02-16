"""Utility functions for testing and demoing the boundedcontours package."""

from typing import Tuple
import numpy as np
import scipy
import scipy.stats
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from boundedcontours.filter import gaussian_filter2d
from boundedcontours.contour import level_from_credible_interval

get_levels = lambda x, p_levels: [level_from_credible_interval(x, p) for p in p_levels]


def make_bins(nbins, bin_range):
    """Generate bins for a 2D histogram.

    Parameters
    ----------
    nbins : int
        The number of bins in each dimension.
    bin_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the bins.

    Returns
    -------
    Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the bin widths in each dimension, the bin edges in each dimension,
        and the meshgrid of bin centers in each dimension.
    """
    ((x_min, x_max), (y_min, y_max)) = bin_range
    dx = (x_max - x_min) / nbins
    dy = (y_max - y_min) / nbins
    x_edges = np.linspace(x_min, x_max, nbins + 1)
    y_edges = np.linspace(y_min, y_max, nbins + 1)
    X, Y = np.meshgrid(x_edges[:-1] + dx / 2, y_edges[:-1] + dy / 2)
    return dx, dy, x_edges, y_edges, X, Y


def hist(x_data, y_data, x_edges, y_edges):
    h, x_edges, y_edges = np.histogram2d(
        x_data,
        y_data,
        bins=[x_edges, y_edges],  # type: ignore
        density=True,
    )
    # Transpose the histogram to match the orientation of the PDF
    h = h.T
    return h


def generate_data_and_pdf(
    distribution, sample_size, X, Y, bin_range, condition_func=lambda x, y: x >= y
):
    """Use the random samples to normalize the pdf to the region of interest"""
    data = distribution.rvs(sample_size)
    x_data, y_data = data[:, 0], data[:, 1]

    mask = (
        condition_func(x_data, y_data)
        if condition_func is not None
        else np.ones_like(x_data, dtype=bool)
    )
    # Calculate the total probability within the bin range and mask
    (x_min, x_max), (y_min, y_max) = bin_range
    within_region = (
        (x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max)
    )
    P_region = np.sum(mask & within_region) / sample_size

    # Truncate the data at the boundary
    x_data = x_data[mask]
    y_data = y_data[mask]

    # get the pdf values
    pdf_values = distribution.pdf(np.dstack([X, Y]))
    # Truncate and renormalize
    cond = (
        condition_func(X, Y)
        if condition_func is not None
        else np.ones_like(X, dtype=bool)
    )
    pdf_values_truncated = np.where(cond, pdf_values, 0)

    pdf_values_truncated = pdf_values_truncated / P_region

    return x_data, y_data, pdf_values_truncated


def get_pdf_and_estimates(
    sample_size: int,
    distribution: scipy.stats.rv_continuous,  # type: ignore
    bin_range: Tuple[Tuple[float, float], Tuple[float, float]],
    nbins: int,
    sigma_smooth: float,
    truncate_smooth: float,
    condition_func=lambda x, y: x >= y,
) -> np.ndarray:
    """
    Generate a 2d PDF and estimates of the PDF from samples. The estimates are:
    - histogram
    - histogram smoothed with scipy.ndimage.gaussian_filter
    - histogram smoothed with scipy.ndimage.gaussian_filter and zeroed outside the boundary
    - histogram smoothed with boundedcontours.gaussian_filter2d with symmetric padding around all boundaries

    Parameters
    ----------
    sample_size : int
        The size of the sample.
    distribution : scipy.stats.rv_continuous
        The 2d distribution of interest. Should have pdf and rvs methods.
    bin_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the bins.
    sigma_smooth : float
        The standard deviation for smoothing the histograms.
    truncate_smooth : float
        The truncation value for smoothing the histograms.

    Returns
    -------
    np.ndarray
        An array of shape (5, nbins, nbins) containing the analytical PDF at the bin centres, histogram of the data, histogram smoothed with scipy,
        the scipy smoothed histogram zeroed outside of the boundary and histogram smoothed with boundedcontours.
    """
    _, _, x_edges, y_edges, X, Y = make_bins(nbins, bin_range)
    # Generate some data from a truncated 2D Gaussian distribution
    x_data, y_data, p = generate_data_and_pdf(
        distribution, sample_size, X, Y, bin_range, condition_func=condition_func
    )

    # Create a 2D normalized histogram of this data
    h = hist(x_data, y_data, x_edges, y_edges)

    # Smooth the histogram with scipy and boundedcontours
    h_smooth_scipy = gaussian_filter(h, sigma=sigma_smooth, truncate=truncate_smooth)
    h_smooth_scipy_zeroed = h_smooth_scipy.copy()
    cond = condition_func(X, Y) if condition_func is not None else None
    if cond is not None:
        h_smooth_scipy_zeroed[~cond] = 0
    h_smooth_bounded = gaussian_filter2d(
        h,
        sigma=sigma_smooth,
        truncate=truncate_smooth,
        cond=cond,
    )

    return np.array([p, h, h_smooth_scipy, h_smooth_scipy_zeroed, h_smooth_bounded])


def plot_pdf_and_estimates(
    X,
    Y,
    p,
    h,
    h_smooth_scipy,
    h_smooth_scipy_zeroed,
    h_smooth_bounded,
    p_levels,
    plotly=False,
    title="",
):
    estimates = [h, p, h_smooth_scipy, h_smooth_scipy_zeroed, h_smooth_bounded]
    titles = ("hist", "pdf", "smooth", "zeroed smooth", "bounded")
    if plotly:
        ncols = len(estimates)
        fig = make_subplots(
            rows=1,
            cols=ncols,
            subplot_titles=titles,
            specs=[[{"type": "surface"}] * ncols],
        )
        # Define common properties for surface plots
        surface_props = dict(x=X, y=Y, cmax=np.max(estimates), cmin=0, showscale=False)
        for col, h_data in enumerate(estimates, start=1):
            fig.add_trace(
                go.Surface(z=h_data, **surface_props),
                row=1,
                col=col,
            )

        layout = dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Density",
            camera=dict(eye=dict(x=0, y=0, z=3.5)),
        )
        fig_layout = {f"scene{col}": layout for col in range(1, ncols + 1)}
        fig_layout["title_text"] = title
        fig.update_layout(**fig_layout)

        fig.show()

    else:
        fig = plt.figure(figsize=(14, 6))

        for i, h_data in enumerate(estimates):
            ax = fig.add_subplot(151 + i, projection="3d")
            subplot_title = titles[i]
            cmap = "coolwarm" if subplot_title == "pdf" else "viridis"
            _plot_surface_and_contour(
                X, Y, h_data, p_levels, ax, title=subplot_title, cmap=cmap
            )

        plt.show()

        # also make a plot with only contours
        colors = ["k", "y", "g", "r", "b"]
        linestyles = ["-", "--", ":", "-.", ":"]
        fig = plt.figure(figsize=(14, 6))
        for i, h_data in enumerate(estimates):
            ax = fig.add_subplot(151 + i)
            _plot_contour(
                ax, X, Y, h_data, p_levels, color=colors[i], linestyle=linestyles[i]
            )
            ax.set_title(titles[i])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)

        lines = []
        for color, ls in zip(colors, linestyles):
            l = plt.Line2D([0], [0], color=color, linestyle=ls, lw=2)
            lines.append(l)
        ax.legend(lines, titles)

        plt.show()


def _plot_contour(ax, X, Y, h, p_levels, color="k", linestyle="-"):
    ax.contour(
        X,
        Y,
        h,
        levels=get_levels(h, p_levels),
        colors=color,
        linewidths=4,
        linestyles=linestyle,
    )


def _plot_surface_and_contour(X, Y, h, p_levels, ax, cmap="viridis", title=""):
    ax.plot_surface(X, Y, h, cmap=cmap)
    _plot_contour(ax, X, Y, h, p_levels)
    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Density")
    # viewed from above
    ax.view_init(90, 0)
