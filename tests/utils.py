from typing import Union, List, Tuple
import numpy as np
import scipy
from scipy import stats
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from boundedcontours.correlate import gaussian_filter2d
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

    mask = condition_func(x_data, y_data)
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
    cond = condition_func(X, Y)
    pdf_values_truncated = np.where(cond, pdf_values, 0)

    pdf_values_truncated = pdf_values_truncated / P_region

    return x_data, y_data, pdf_values_truncated


def get_pdf_and_histograms(
    sample_size: int,
    distribution: scipy.stats.rv_continuous,  # type: ignore
    bin_range: Tuple[Tuple[float, float], Tuple[float, float]],
    nbins: int,
    sigma_smooth: float,
    truncate_smooth: float,
    condition_func=lambda x, y: x >= y,
) -> np.ndarray:
    """
    Generate a PDF and histograms.

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
        the scipy smoothed histogram zerod outside of the boundary and histogram smoothed with boundedcontours.
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
    cond = condition_func(X, Y)
    h_smooth_scipy_zeroed[~cond] = 0
    h_smooth_bounded = gaussian_filter2d(
        h,
        sigma=sigma_smooth,
        truncate=truncate_smooth,
        cond=cond,
    )

    return np.array([p, h, h_smooth_scipy, h_smooth_scipy_zeroed, h_smooth_bounded])


def format_subplot(ax):
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Density")
    # viewed from above
    ax.view_init(90, 0)


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
):
    if plotly:
        ncols = 5
        fig = make_subplots(
            rows=1,
            cols=ncols,
            subplot_titles=(
                "hist",
                "pdf",
                "smooth",
                "zeroed smoothbounded",
            ),
            specs=[[{"type": "surface"}] * ncols],
        )

        max_value = max(h.max(), p.max())

        fig.add_trace(
            go.Surface(
                z=h,
                x=X,
                y=Y,
                cmax=max_value,
                cmin=0,
                showscale=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Surface(z=p, x=X, y=Y, cmax=max_value, cmin=0),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Surface(z=h_smooth_scipy, x=X, y=Y, cmax=max_value, cmin=0),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Surface(z=h_smooth_scipy_zeroed, x=X, y=Y, cmax=max_value, cmin=0),
            row=1,
            col=4,
        )

        fig.add_trace(
            go.Surface(z=h_smooth_bounded, x=X, y=Y, cmax=max_value, cmin=0),
            row=1,
            col=4,
        )
        # Update layout
        fig.update_layout(
            title_text="Comparing estimated with true distributions",
            scene=dict(
                xaxis_title="X axis", yaxis_title="Y axis", zaxis_title="Density"
            ),
            scene2=dict(
                xaxis_title="X axis", yaxis_title="Y axis", zaxis_title="Density"
            ),
            scene3=dict(
                xaxis_title="X axis", yaxis_title="Y axis", zaxis_title="Density"
            ),
            scene4=dict(
                xaxis_title="X axis", yaxis_title="Y axis", zaxis_title="Density"
            ),
            scene5=dict(
                xaxis_title="X axis", yaxis_title="Y axis", zaxis_title="Density"
            ),
        )

        fig.show()

    else:
        fig = plt.figure(figsize=(14, 6))

        # Truncated and normalized empirical distribution
        ax1 = fig.add_subplot(151, projection="3d")
        ax1.plot_surface(X, Y, h, cmap="viridis")
        hist_levels = get_levels(h, p_levels)
        ax1.contour(
            X,
            Y,
            h,
            levels=hist_levels,
            colors="k",
            linewidths=4,
        )
        ax1.set_title("hist")
        format_subplot(ax1)

        # Truncated and normalized theoretical distribution
        ax2 = fig.add_subplot(152, projection="3d")
        ax2.plot_surface(X, Y, p, cmap="coolwarm")
        pdf_levels = get_levels(p, p_levels)
        ax2.contour(X, Y, p, levels=pdf_levels, colors="k", linewidths=4)
        ax2.set_title("pdf")
        format_subplot(ax2)

        ax3 = fig.add_subplot(153, projection="3d")
        ax3.plot_surface(X, Y, h_smooth_scipy, cmap="viridis")
        smooth_levels = get_levels(h_smooth_scipy, p_levels)
        ax3.contour(
            X,
            Y,
            h_smooth_scipy,
            levels=smooth_levels,
            colors="k",
            linewidths=4,
        )
        ax3.set_title("smooth")
        format_subplot(ax3)

        ax4 = fig.add_subplot(154, projection="3d")
        ax4.plot_surface(X, Y, h_smooth_scipy_zeroed, cmap="viridis")
        smooth_levels = get_levels(h_smooth_scipy_zeroed, p_levels)
        ax4.contour(
            X,
            Y,
            h_smooth_scipy_zeroed,
            levels=smooth_levels,
            colors="k",
            linewidths=4,
        )
        ax4.set_title("zeroed smooth")
        format_subplot(ax4)

        ax5 = fig.add_subplot(155, projection="3d")
        ax5.plot_surface(X, Y, h_smooth_bounded, cmap="viridis")
        bounded_levels = get_levels(h_smooth_bounded, p_levels)
        ax5.contour(
            X,
            Y,
            h_smooth_bounded,
            levels=bounded_levels,
            colors="k",
            linewidths=4,
        )
        ax5.set_title("bounded")
        format_subplot(ax5)

        plt.show()

        # also make a plot with both contours
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(111)
        ax1.contour(
            X,
            Y,
            h,
            levels=hist_levels,
            colors="k",
        )
        ax1.contour(
            X,
            Y,
            p,
            levels=pdf_levels,
            colors="y",
            linestyles="--",
        )
        ax1.contour(
            X,
            Y,
            h_smooth_bounded,
            levels=bounded_levels,
            colors="r",
            linestyles="-.",
        )
        ax1.contour(
            X,
            Y,
            h_smooth_scipy,
            levels=smooth_levels,
            colors="g",
            linestyles=":",
        )
        ax1.contour(
            X,
            Y,
            h_smooth_scipy_zeroed,
            levels=smooth_levels,
            colors="b",
            linestyles=":",
        )

        ax1.set_xlabel("X axis")
        ax1.set_ylabel("Y axis")
        ax1.set_title("Truncated Empirical and Theoretical Distributions")

        # Creating custom legends for contours
        black_line = plt.Line2D([0], [0], color="k", lw=2)
        yellow_line = plt.Line2D([0], [0], color="y", lw=2, linestyle="--")
        green_line = plt.Line2D([0], [0], color="g", lw=2, linestyle=":")
        red_line = plt.Line2D([0], [0], color="r", lw=2, linestyle="-.")
        blue_line = plt.Line2D([0], [0], color="b", lw=2, linestyle=":")
        ax1.legend(
            [black_line, yellow_line, green_line, red_line, blue_line],
            ["Empirical", "PDF", "Smoothed", "Bounded", "Zeroed"],
        )

        plt.show()
