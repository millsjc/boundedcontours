import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from contour import smooth_2d_histogram, contour_at_level, contour_plots


def get_test_data(N=2000):
    # multimodal gaussian mixture
    N_1 = int(N / 2)
    N_2 = N - N_1
    np.random.seed(0)  # For reproducible results
    x = np.random.normal(0, 1, N_1) + np.random.normal(0.5, 1, N_2)
    y = np.random.normal(0, 1, N_1) + np.random.normal(0.5, 1, N_2)
    return x, y


def set_axis(ax, x=(-4, 4), y=(-4, 4)):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_aspect("equal")


def test_smooth_2d_histogram(make_plots=True):
    x, y = get_test_data()

    # Test cases
    test_cases = [
        {"bins": 100, "sigma_smooth": 2.0, "condition": "X>Y"},
        {"bins": 30, "sigma_smooth": 0.5, "condition": "X>Y"},
    ]

    # we will only check the largest, middle and smallest non-zero bins
    expected_output = [
        np.array([6.27057731e-10, 5.76411762e-03, 1.04055075e-01]),
        np.array([8.77506173e-10, 5.39031282e-03, 1.17246766e-01]),
    ]
    for i, params in enumerate(test_cases):
        X, Y, H = smooth_2d_histogram(x, y, **params)
        assert H.shape == (params["bins"], params["bins"]), "Histogram shape mismatch"

        # Check that the largest, middle and smallest non-zero bins are as expected
        H_flat = H.flatten()[np.flatnonzero(H)]
        N = len(H_flat)
        vals_to_check = np.array([0, int(N / 2), N - 1])
        out = np.partition(H_flat, vals_to_check)[vals_to_check]

        assert np.allclose(out, expected_output[i]), (
            "Largest, middle and smallest non-zero bins don't match expected values."
            " Expected: {}, got: {}".format(expected_output[i], out)
        )

        if make_plots:
            plt.figure(figsize=(6, 6))
            plt.pcolormesh(X, Y, H)
            plt.title(
                f"Test Case {i+1}: bins={params['bins']},"
                f" sigma={params['sigma_smooth']}"
            )
            plt.colorbar()
            plt.show()

        print(f"Test Case {i+1} passed")

    print("All tests passed (smooth_2d_histogram)")


def test_contour_at_level(
    N=10000,
    bins=100,
    sigma_smooth=1.0,
    gaussian_filter_kwargs=dict(mode="constant", cval=0.0, truncate=4.0),
):
    """N is the number of samples to generate. FIXME: This is not suitable for automatic testing, it makes plots."""
    kwargs = dict(
        bins=bins,
        sigma_smooth=sigma_smooth,
        plot_pcolormesh=True,
        p_levels=[0.5, 0.9],
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
    set_axis(ax)
    plt.show()

    kwargs["p_levels"] = 0.9

    # Test multimodal data
    x_multi = np.concatenate([x, np.random.normal(loc=5, size=N)])
    y_multi = np.concatenate([y, np.random.normal(loc=5, size=N)])

    ax = contour_at_level(x_multi, y_multi, **kwargs)
    err_msg = "Test with multimodal data failed."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("Multimodal data")
    set_axis(ax, x=(-4, 10), y=(-4, 10))
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
        set_axis(ax)
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
    set_axis(ax)
    plt.show()

    # Test with array bins
    bins = np.linspace(-3, 3, 10)
    kwargs_no_bins = {k: v for k, v in kwargs.items() if k != "bins"}
    ax = contour_at_level(x, y, bins=bins, **kwargs_no_bins)
    err_msg = "Test with array bins failed to return an Axes object."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("Array bins")
    set_axis(ax)
    plt.show()

    # Test with double the smoothing
    s = 2 * kwargs.pop("sigma_smooth")
    ax = contour_at_level(x, y, sigma_smooth=s, **kwargs)
    err_msg = f"Test with double smoothing (sigma_smooth={s}) failed."
    assert isinstance(ax, mpl.axes.Axes), err_msg
    ax.set_title("More smoothing")
    set_axis(ax)
    plt.show()


def test_contour_plots():
    """Test the contour_plots function. FIXME: This is not suitable for automatic testing, it makes plots."""
    x, y = get_test_data()
    samples_list = [[x, y]]
    labels = [None]
    axes_labels = ["x", "y"]
    axes_lims = [-4, 4]
    color = "white"
    ax3, ax2 = contour_plots(
        samples_list,
        labels,
        axes_labels=axes_labels,
        condition="X>Y",  # enforce m1 > m2 when smoothing histogram
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


parser = argparse.ArgumentParser()
parser.add_argument("--make_plots", action="store_true")
args = parser.parse_args()
make_plots = args.make_plots

test_smooth_2d_histogram(make_plots)

if make_plots:
    test_contour_at_level()
    test_contour_plots()
