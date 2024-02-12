import numpy as np
from distributions import IndependentBeta2d, IndependentExponential2d, Uniform2d
from utils import (
    get_pdf_and_histograms,
    plot_pdf_and_estimates,
    make_bins,
)
from scipy import stats

plotly = False
sigma_smooth = 1
truncate_smooth = 1
sample_size = int(2e5)
p_levels = [0.9]
# Bins for the histogram
nbins = 30
n_simulations = 10
plot_example = True


def do_nothing(x, y):
    return np.ones_like(x, dtype=bool)


def x_ge_y(x, y):
    return x >= y


def in_circle(x, y, radius=1):
    return x**2 + y**2 <= radius**2


def in_first_quadrant(x, y):
    return (x >= 0) & (y >= 0)


def in_triangle(x, y):
    return x + y <= 1


def sum_constraint(x, y, C=1):
    return x + y <= C


# Print the improvement in MISE.
tests = {
    "unconditioned_normal": {
        "dist": stats.multivariate_normal(np.ones(2), np.eye(2)),
        "condition": do_nothing,
        "bin_range": ((-3, 3), (-3, 3)),
    },
    "uniform_triangle": {
        "dist": Uniform2d(),
        "condition": in_triangle,
        "bin_range": ((0, 1), (0, 1)),
    },
    "normal_circle": {
        "dist": stats.multivariate_normal(np.ones(2), np.eye(2)),
        "condition": in_circle,
        "bin_range": ((-1, 1), (-1, 1)),
    },
    "normal_first_quadrant": {
        "dist": stats.multivariate_normal(np.ones(2), np.eye(2)),
        "condition": in_first_quadrant,
        "bin_range": ((-2, 2), (-5, 5)),
    },
    "normal_x_ge_y": {
        "dist": stats.multivariate_normal(np.ones(2), np.eye(2)),
        "condition": x_ge_y,
        "bin_range": ((-3.05, 3.05), (-4, 4)),
    },
    "normal_x_le_y_covarying": {
        "dist": stats.multivariate_normal(np.ones(2), [[1, 0.5], [0.5, 1]]),
        "condition": x_ge_y,
        "bin_range": ((-3, 3), (-3, 3)),
    },
    "exponential_with_sum_constraint": {
        "dist": IndependentExponential2d(1, 1),
        "condition": sum_constraint,
        "bin_range": ((0, 1), (0, 1)),
    },
    "beta_x_ge_y": {
        "dist": IndependentBeta2d(2, 5, 2, 5),
        "condition": x_ge_y,
        "bin_range": ((0, 1), (0, 1)),
    },
}
for sigma_smooth, truncate_smooth in [(1, 1), (2, 1), (5, 1), (1, 2), (1, 5), (10, 10)][
    1:
]:
    for test_label, test in tests.items():
        if truncate_smooth < 9 or test_label != "normal_x_le_y_covarying":
            continue
        # normal_x_le_y_covarying and sigma_smooth=1 truncate_smooth=5
        print(f"Running test {test_label}")
        dist = test["dist"]
        condition_func = test["condition"]
        bin_range = test["bin_range"]
        dx, dy, _, _, X, Y = make_bins(nbins, bin_range)
        out = np.array([
            get_pdf_and_histograms(
                sample_size,
                dist,
                bin_range,
                nbins,
                sigma_smooth,
                truncate_smooth,
                condition_func=condition_func,
            )
            for i in range(n_simulations)
        ])
        # swap axes to get (n_simulations, nbins, nbins)
        p, h, h_smooth_scipy, h_smooth_scipy_zeroed, h_smooth_bounded = np.einsum(
            "ijkl->jikl", out
        )
        print(np.sum(h), np.sum(h_smooth_bounded))

        mise = {}
        mean_bias = {}
        for label, estimate in zip(
            ["histogram", "smooth", "zeroed", "bounded"],
            [h, h_smooth_scipy, h_smooth_scipy_zeroed, h_smooth_bounded],
        ):
            mise[label] = (
                np.einsum("ijk->", (estimate - p) ** 2) * dx * dy / n_simulations
            )
            mean_bias[label] = np.einsum("ijk->", estimate - p) / (
                n_simulations * nbins**2
            )
            # print(f"MISE {label}: {mise[label]}")
            # print(f"Mean bias {label}: {mean_bias[label]}")

        if (
            mise["smooth"] < mise["bounded"]
            or mise["zeroed"] < mise["bounded"]
            or mean_bias["histogram"] > mean_bias["bounded"]
        ):
            print(
                f"Smoothed estimator is better than bounded estimator for {test_label}"
                f" and {sigma_smooth=} {truncate_smooth=}"
            )
            breakpoint()

        print(
            f"Bounded estimator is {mise['smooth'] / mise['bounded']} times better than"
            f" smoothed estimator for {test_label} and"
            f" {sigma_smooth=} {truncate_smooth=}"
        )
        print(
            f"Bounded estimator is {mise['zeroed'] / mise['bounded']} times better than"
            f" zeroed estimator for {test_label} and {sigma_smooth=} {truncate_smooth=}"
        )
        print(
            f"Bounded estimator is {mean_bias['histogram'] / mean_bias['bounded']} times"
            f" better than histogram estimator for {test_label} and"
            f" {sigma_smooth=} {truncate_smooth=}"
        )

        if plot_example:
            i = 0
            plot_pdf_and_estimates(
                X,
                Y,
                p[i],
                h[i],
                h_smooth_scipy[i],
                h_smooth_scipy_zeroed[i],
                h_smooth_bounded[i],
                p_levels,
                plotly,
            )
