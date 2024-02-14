import argparse
import numpy as np
from boundedcontours.demo.distributions import tests
from boundedcontours.demo.utils import (
    get_pdf_and_estimates,
    plot_pdf_and_estimates,
    make_bins,
)

tests.pop("unconditioned_normal")


def check_bounded_is_better(
    plot_example=False,
    plotly=False,
    sample_size=int(2e5),
    nbins=30,
    n_simulations=10,
):
    """Calculate the mean integrated square error for different smoothing
    methods and for a bunch of distributions and amounts of
    smoothing. Raise an error if the bounded smooth estimator is worse
    than the other smoothed estimators.
    The smoothing estimators considered are:
    - smooth (scipy.ndimage.gaussian_filter smoothing)
    - zeroed ("smooth" with values outside boundary set to zero)
    - bounded (filtering with symmetric padding around all boundaries)

    Parameters
    ----------
    plot_example : bool, optional
        If True, for each test plot an example of the PDF and the different estimators, by default False
    plotly : bool, optional
        If True, use plotly for plotting, by default False
    sample_size : _type_, optional
        Number of samples to draw from the distribution, by default int(2e5)
    nbins : int, optional
        Number of bins for the histogram, by default 30
    n_simulations : int, optional
        Number of simulations to run for each test, by default 10

    Raises
    ------
    ValueError
        If the bounded estimator's error is worse than "zeroed" or "smooth" estimators
    """
    p_levels = [0.9]  # credible interval for contour

    for sigma_smooth, truncate_smooth in [
        (1, 1),
        (2, 1),
        (5, 1),
        (1, 2),
        (1, 5),
        (10, 10),
    ]:
        print(f"Testing smoothing with {sigma_smooth=} {truncate_smooth=}")
        for test_label, test in tests.items():
            print(f"Running test {test_label}")
            dist = test["dist"]
            condition_func = test["condition"]
            bin_range = test["bin_range"]
            dx, dy, _, _, X, Y = make_bins(nbins, bin_range)
            out = np.array([
                get_pdf_and_estimates(
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
            _hsum = np.sum(h)
            _hsmoothsum = np.sum(h_smooth_scipy)
            if not np.isclose(_hsum, _hsmoothsum):
                print(
                    f"Sum of histogram and smooth histogram differ: {_hsum=:.2f} {_hsmoothsum=:.2f}"
                )

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
            value_judgement = "small!" if mise["bounded"] < 0.01 else "seems a bit big!"
            print(
                f"Bounded MISE (raw value): {mise['bounded']:.4f} - {value_judgement}"
            )
            print(
                f"MISE ratios: \n"
                f"smooth: {mise['smooth'] / mise['bounded']:.2f}\n"
                f"zeroed: {mise['zeroed'] / mise['bounded']:.2f}\n"
                f"histogram: {mise['histogram'] / mise['bounded']:.2f}\n"
            )

            def _plot_example():
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
                    title=f"{test_label} - {sigma_smooth=} {truncate_smooth=}",
                )

            if (
                mise["smooth"] < mise["bounded"]
                or mise["zeroed"] < mise["bounded"]
                # or mise["histogram"] < mise["bounded"]
            ):
                print(
                    f"Bounded estimator worse for {test_label}"
                    f" and {sigma_smooth=} {truncate_smooth=}\n"
                )
                _plot_example()
                raise ValueError(
                    f"Bounded estimator worse than other estimators. MISE bounded: {mise['bounded']}, MISE smooth: {mise['smooth']}, MISE zeroed: {mise['zeroed']}"
                )

            if plot_example:
                _plot_example()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_example", action="store_true")
    parser.add_argument("--plotly", action="store_true")
    parser.add_argument("--sample_size", type=int, default=int(2e5))
    parser.add_argument("--nbins", type=int, default=30)
    parser.add_argument("--n_simulations", type=int, default=10)
    args = parser.parse_args()
    check_bounded_is_better(
        args.plot_example,
        args.plotly,
        args.sample_size,
        args.nbins,
        args.n_simulations,
    )
