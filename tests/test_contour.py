import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from boundedcontours.contour import contour_at_level, contour_plots
from boundedcontours.correlate import gaussian_filter2d


def _get_test_data(N=2000):
    # multimodal gaussian mixture
    N_1 = int(N / 2)
    N_2 = N - N_1
    np.random.seed(0)  # For reproducible results
    x = np.random.normal(0, 1, N_1) + np.random.normal(0.5, 1, N_2)
    y = np.random.normal(0, 1, N_1) + np.random.normal(0.5, 1, N_2)
    return x, y


def _set_axis(ax, x=(-4, 4), y=(-4, 4)):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_aspect("equal")


def test_contour_at_level(
    N=10000,
    bins=100,
    sigma_smooth=1.0,
    gaussian_filter_kwargs=dict(mode="constant", cval=0.0),
):
    """N is the number of samples to generate. FIXME: This is not suitable for automatic testing, it makes plots."""
    kwargs = dict(
        bins=bins,
        sigma_smooth=sigma_smooth,
        plot_pcolormesh=True,
        p_levels=[0.5, 0.9],
        truncate=4.0,
        gaussian_filter_kwargs=gaussian_filter_kwargs,
    )

    # Generate some random data
    np.random.seed(42)
    x = np.random.normal(size=N)
    y = np.random.normal(size=N)

    # Test with multiple credible intervals
    ax = contour_at_level(x, y, **kwargs)
    err_msg = "Test with multiple credible intervals failed to return an Axes object."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("Multiple credible intervals")
    _set_axis(ax)
    plt.show()

    kwargs["p_levels"] = 0.9

    # Test multimodal data
    x_multi = np.concatenate([x, np.random.normal(loc=5, size=N)])
    y_multi = np.concatenate([y, np.random.normal(loc=5, size=N)])

    ax = contour_at_level(x_multi, y_multi, **kwargs)
    err_msg = "Test with multimodal data failed."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("Multimodal data")
    _set_axis(ax, x=(-4, 10), y=(-4, 10))
    plt.show()

    # Test data with single boundary
    cond = x > y
    x_b = x[cond]
    y_b = y[cond]

    # Test data with single boundary, with condition enabled
    # compare with previous plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1 = contour_at_level(x_b, y_b, ax=ax1, **kwargs)
    ax2 = contour_at_level(x_b, y_b, ax=ax2, condition="X>Y", **kwargs)
    err_msg = "Test with data with single boundary and condition option enabled failed."
    assert isinstance(ax2, mpl.axes.Axes), err_msg
    x_array = np.linspace(-4, 4, 100)
    for ax, opt in zip((ax1, ax2), ("disabled", "enabled")):
        ax.plot(x_array, x_array, color="r", linestyle="--")
        ax.set_title(f"Boundary condition option {opt}")
        _set_axis(ax)
    plt.show()

    # Test data with multiple boundaries
    cond = (x < 1) & (x > y)
    x_b = x[cond]
    y_b = y[cond]

    ax = contour_at_level(x_b, y_b, **kwargs)
    err_msg = "Test with data with multiple boundaries failed."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.plot(x_array, x_array, color="r", linestyle="--")
    ax.set_title("Data with multiple boundaries")
    _set_axis(ax)
    plt.show()

    # Test with array bins
    bins = np.linspace(-3, 3, 10)
    kwargs_no_bins = {k: v for k, v in kwargs.items() if k != "bins"}
    ax = contour_at_level(x, y, bins=bins, **kwargs_no_bins)
    err_msg = "Test with array bins failed to return an Axes object."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("Array bins")
    _set_axis(ax)
    plt.show()

    # Test with double the smoothing
    s = 2 * kwargs.pop("sigma_smooth")
    ax = contour_at_level(x, y, sigma_smooth=s, **kwargs)
    err_msg = f"Test with double smoothing (sigma_smooth={s}) failed."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("More smoothing")
    _set_axis(ax)
    plt.show()


def test_contour_plots():
    """Test the contour_plots function. FIXME: This is not suitable for automatic testing, it makes plots."""
    x, y = _get_test_data()
    samples_list = [[x, y]]
    labels = [None]
    axes_labels = ["x", "y"]
    axes_lims = [-4, 4]
    color = "white"
    ax3, ax2 = contour_plots(
        samples_list,
        labels,
        axes_labels=axes_labels,
        condition_function=lambda x, y: x > y,  # enforce x>y when smoothing histogram
        axes_lims=axes_lims,
        linewidth=3,
        target_nbins=100,
        min_bin_width=4,
        sigma_smooth=3,
        truncate=6,
        plot_pcolormesh=True,
        bins_2d_kwargs=dict(
            lower_bound_safety_factor=0.5,
        ),
        colors=[color],
        p_levels=[0.5, 0.9],
    )
    ax3.set_title("Multimodal gaussian mixture at 50% and 90% credible intervals")
    plt.show()


def test_gaussian_2d():
    for a in [
        np.array(
            [
                [0, 2, 4, 6, 8],
                [10, 12, 14, 16, 18],
                [20, 22, 24, 26, 28],
                [30, 32, 34, 36, 38],
                [40, 42, 44, 46, 48],
            ],
            dtype=float,
        ),
        np.random.random((10, 10)),
    ]:
        out = gaussian_filter2d(a, 1)
        out2 = gaussian_filter(a, 1)
        print("gaussian_filter2d:", out)
        print("gaussian_filter:", out2)
        assert np.allclose(out, out2), "gaussian_filter2d and gaussian_filter disagree"


def test_gaussian_2d_compare_with_scipy():
    truncate = 1.0
    sigma = np.sqrt(-1 / (2 * np.log(0.5)))

    a = np.array(
        [
            [0, 2, 4, 6, 8],
            [10, 12, 14, 16, 18],
            [20, 22, 24, 26, 28],
            [30, 32, 34, 36, 38],
            [40, 42, 44, 46, 48],
        ],
        dtype=float,
    )
    expected_output = np.array(
        [
            [4, 6, 8, 9, 11],
            [10, 12, 14, 15, 17],
            [20, 22, 24, 25, 27],
            [29, 31, 33, 34, 36],
            [35, 37, 39, 40, 42],
        ]
    )
    # time the two methods
    import time
    for cond in [None, a > 20]:
        print(f"{cond=}")
        start = time.time()
        out = np.array(
            gaussian_filter2d(a, sigma, truncate=truncate, cond=cond), dtype=float
        )
        end = time.time()
        print("gaussian_filter2d took", (end - start) * 1000, "milliseconds")
        print(out)
        print(f"{expected_output=}")
        print("Sum of out:", np.sum(out))
        print("Sum of input:", np.sum(a))

        print("\n")
        print("Again with ndimage.gaussian_filter:")
        start = time.time()
        out = gaussian_filter(a, sigma, truncate=truncate)
        end = time.time()
        print("gaussian_filter took", (end - start) * 1000, "milliseconds")
        print(out)
        print("Sum of out:", np.sum(out))


# parser = argparse.ArgumentParser()
# parser.add_argument("--make_plots", action="store_true")
# args = parser.parse_args()
# make_plots = args.make_plots
make_plots = False

test_gaussian_2d()
test_gaussian_2d_compare_with_scipy()

if make_plots:
    test_contour_at_level()
    test_contour_plots()
