from typing import Union, List
import numpy as np
from scipy.stats import multivariate_normal, norm
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from boundedcontours.correlate import gaussian_filter2d
from boundedcontours.contour import level_from_credible_interval

get_levels = lambda x, p_levels: [level_from_credible_interval(x, p) for p in p_levels]


def generate_data(sample_size, mean, cov):
    data = np.random.multivariate_normal(mean, cov, sample_size)
    x_data, y_data = data[:, 0], data[:, 1]
    # Truncate the data to the region x >= y
    mask = x_data >= y_data
    x_data = x_data[mask]
    y_data = y_data[mask]
    return x_data, y_data


def hist(x_data, y_data, x_edges, y_edges, dx, dy, X, Y):
    h, x_edges, y_edges = np.histogram2d(
        x_data,
        y_data,
        bins=[x_edges, y_edges],  # type: ignore
        density=True,
    )
    # Transpose the histogram to match the orientation of the PDF
    h = h.T

    try:
        region_sum = h[X >= Y].sum()
        assert np.isclose(
            region_sum, h.sum()
        ), "the histogram is not truncated correctly"
        assert region_sum > 0, "the histogram is zero everywhere"
    except AssertionError as e:
        print(e)

    assert np.isclose(
        np.sum(h) * dx * dy, 1
    ), "the histogram is not normalized correctly"
    return h


def pdf(mean, cov, X, Y, bin_range):
    pdf_values = multivariate_normal(mean, cov).pdf(np.dstack([X, Y]))
    # Truncate and renormalize
    pdf_values_truncated = np.where(X < Y, 0, pdf_values) * 2

    # Use the CDF to determine the total probability within the bin range
    # assuming that cov is diagonal
    assert cov[0][1] == 0 and cov[1][0] == 0, "covariance matrix cov is not diagonal"
    (x_min, x_max), (y_min, y_max) = bin_range
    cdf_x_min = norm.cdf(x_min, mean[0], np.sqrt(cov[0][0]))
    cdf_x_max = norm.cdf(x_max, mean[0], np.sqrt(cov[0][0]))
    cdf_y_min = norm.cdf(y_min, mean[1], np.sqrt(cov[1][1]))
    cdf_y_max = norm.cdf(y_max, mean[1], np.sqrt(cov[1][1]))
    P_region = (cdf_x_max - cdf_x_min) * (cdf_y_max - cdf_y_min)

    return pdf_values_truncated / P_region


import numpy as np


def get_pdf_and_histograms(
    sample_size: int,
    mean: Union[List[float], List[int], np.ndarray],
    cov: Union[List[List[float]], List[List[int]], np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_smooth: float,
    truncate_smooth: float,
) -> np.ndarray:
    """
    Generate a PDF and histograms.

    Parameters
    ----------
    sample_size : int
        The size of the sample.
    mean : np.ndarray
        The mean of the 2d distribution.
    cov : np.ndarray
        The covariance matrix of the 2d distribution.
    x_edges : np.ndarray
        The edges of the x-axis bins.
    y_edges : np.ndarray
        The edges of the y-axis bins.
    sigma_smooth : float
        The standard deviation for smoothing the histograms.
    truncate_smooth : float
        The truncation value for smoothing the histograms.

    Returns
    -------
    np.ndarray
        An array of shape (4, nbins, nbins) containing the analytical PDF at the bin centres, histogram of the data, histogram smoothed with scipy,
        and histogram smoothed with boundedcontours.
    """
    bin_range = ((x_edges[0], x_edges[-1]), (y_edges[0], y_edges[-1]))
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    X, Y = np.meshgrid(x_edges[:-1] + dx / 2, y_edges[:-1] + dy / 2)
    # Step 1: Generate some data from a truncated 2D Gaussian distribution
    x_data, y_data = generate_data(sample_size, mean, cov)

    # Step 2: Create a 2D normalized histogram of this data
    h = hist(x_data, y_data, x_edges, y_edges, dx, dy, X, Y)

    # Step 4: Create the theoretical PDF and apply the same truncation and normalization
    p = pdf(mean, cov, X, Y, bin_range)

    # Step 5: Smooth the histogram with scipy and boundedcontours
    h_smooth_scipy = gaussian_filter(h, sigma=sigma_smooth, truncate=truncate_smooth)
    h_smooth_bounded = gaussian_filter2d(
        h,
        sigma=sigma_smooth,
        truncate=truncate_smooth,
        cond=X >= Y,
    )

    return np.array([p, h, h_smooth_scipy, h_smooth_bounded])


def format_subplot(ax):
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Density")
    ax.view_init(
        30, 225
    )  # Adjust the viewing angle for better visibility of truncation


def plot_pdf_and_3_hists(
    X, Y, p, h, h_smooth_scipy, h_smooth_bounded, p_levels, plotly=False
):
    breakpoint()
    if plotly:
        # Create a subplot figure with 1 row and 2 columns
        ncols = 4
        fig = make_subplots(
            rows=1,
            cols=ncols,
            subplot_titles=(
                "Truncated Empirical Distribution",
                "Truncated Theoretical Distribution",
                "Smoothed Empirical Distribution",
                "Bounded Smoothed Empirical Distribution",
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
            go.Surface(z=h_smooth_bounded, x=X, y=Y, cmax=max_value, cmin=0),
            row=1,
            col=4,
        )
        # Update layout
        fig.update_layout(
            title_text="Comparing Truncated Empirical and Theoretical Distributions",
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
        )

        fig.show()

    else:
        fig = plt.figure(figsize=(14, 6))

        # Truncated and normalized empirical distribution
        ax1 = fig.add_subplot(141, projection="3d")
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
        ax1.set_title("Truncated Empirical Distribution (Precise Normalization)")
        format_subplot(ax1)

        # Truncated and normalized theoretical distribution
        ax2 = fig.add_subplot(142, projection="3d")
        ax2.plot_surface(X, Y, p, cmap="coolwarm")
        pdf_levels = get_levels(p, p_levels)
        ax2.contour(X, Y, p, levels=pdf_levels, colors="k", linewidths=4)
        ax2.set_title("Truncated Theoretical Distribution")
        format_subplot(ax2)

        ax3 = fig.add_subplot(143, projection="3d")
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
        ax3.set_title("Smoothed Empirical Distribution")
        format_subplot(ax3)

        ax4 = fig.add_subplot(144, projection="3d")
        ax4.plot_surface(X, Y, h_smooth_bounded, cmap="viridis")
        bounded_levels = get_levels(h_smooth_bounded, p_levels)
        ax4.contour(
            X,
            Y,
            h_smooth_bounded,
            levels=bounded_levels,
            colors="k",
            linewidths=4,
        )
        ax4.set_title("Bounded Smoothed Empirical Distribution")
        format_subplot(ax4)

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

        ax1.set_xlabel("X axis")
        ax1.set_ylabel("Y axis")
        ax1.set_title("Truncated Empirical and Theoretical Distributions")

        # Creating custom legends for contours
        black_line = plt.Line2D([0], [0], color="k", lw=2)
        yellow_line = plt.Line2D([0], [0], color="y", lw=2, linestyle="--")
        green_line = plt.Line2D([0], [0], color="g", lw=2, linestyle=":")
        red_line = plt.Line2D([0], [0], color="r", lw=2, linestyle="-.")
        ax1.legend(
            [black_line, yellow_line, green_line, red_line],
            ["Empirical", "PDF", "Smoothed", "Bounded"],
        )

        plt.show()
