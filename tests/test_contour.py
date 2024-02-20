import math
import numpy as np
from boundedcontours.contour import (
    level_from_credible_interval,
    smooth_2d_histogram,
    contour_at_level,
    get_2d_bins,
    contour_plots,
)
import matplotlib as mpl
import pytest


class TestLevelFromCredibleInterval:

    # returns the correct level for a given histogram and credible interval
    def test_returns_correct_level_for_given_histogram_and_cri(self):
        H = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        p_level = 0.5
        expected_level = 7
        assert level_from_credible_interval(H, p_level) == expected_level

    # works with a histogram of all zeros
    def test_works_with_histogram_of_all_zeros(self):
        H = np.zeros((3, 3))
        p_level = 0.9
        expected_level = 0
        assert level_from_credible_interval(H, p_level) == expected_level

    # works with a histogram of all ones
    def test_works_with_histogram_of_all_ones(self):
        H = np.ones((3, 3))
        p_level = 0.5
        expected_level = 1
        assert level_from_credible_interval(H, p_level) == expected_level

    # returns the correct level for a histogram with only one bin
    def test_returns_correct_level_for_histogram_with_one_bin(self):
        H = np.array([[5]])
        p_level = 0.9
        expected_level = 5
        assert level_from_credible_interval(H, p_level) == expected_level

    # returns the correct level for a histogram with all values equal to zero except one
    def test_returns_correct_level_for_histogram_with_all_zeros_except_one(self):
        H = np.zeros((3, 3))
        H[1, 1] = 5
        p_level = 0.9
        expected_level = 5
        assert level_from_credible_interval(H, p_level) == expected_level


class TestSmooth2dHistogram:

    # Smooths a 2D histogram with empty x and y arrays
    def test_smooth_2d_histogram_empty_arrays(self):
        x = np.array([])
        y = np.array([])
        X, Y, H = smooth_2d_histogram(x, y)
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(H, np.ndarray)
        assert X.shape == (100, 100)
        assert Y.shape == (100, 100)
        assert H.shape == (100, 100)

    # Smooths a 2D histogram with one element in x and y arrays
    def test_smooth_2d_histogram_one_element(self):
        x = np.array([1])
        y = np.array([1])
        X, Y, H = smooth_2d_histogram(x, y)
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(H, np.ndarray)
        assert X.shape == (100, 100)
        assert Y.shape == (100, 100)
        assert H.shape == (100, 100)

    # Smooths a 2D histogram with bins=1
    def test_smooth_2d_histogram_bins_1(self):
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        bins = 1
        X, Y, H = smooth_2d_histogram(x, y, bins=bins)
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(H, np.ndarray)
        assert X.shape == (bins, bins)
        assert Y.shape == (bins, bins)
        assert H.shape == (bins, bins)

    # Test that with input condition_function=lambda x,y: x>0.5 only the points where X>0.5 are smoothed and the rest are the same as the input.
    def test_smooth_2d_histogram_condition_function(self):
        x = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        y = np.array([0.2, 0.4, 0.7, 0.9, 1.0])
        bins = 5

        H_unsmoothed, bins_x, bins_y = np.histogram2d(x, y, bins=bins, density=True)
        H_unsmoothed = H_unsmoothed.T

        def x_gt_0p5(x, y):
            return x > 0.5

        X, Y, H = smooth_2d_histogram(
            x,
            y,
            sigma_smooth=np.sqrt(-1 / (2 * np.log(1 / 2))),
            truncate=1,
            condition_function=x_gt_0p5,
            bins=bins,
        )

        assert np.allclose(H_unsmoothed[~x_gt_0p5(X, Y)], H[~x_gt_0p5(X, Y)])
        assert not np.allclose(H_unsmoothed[x_gt_0p5(X, Y)], H[x_gt_0p5(X, Y)])

        expected_H = np.array([
            [7.8125, 0.0, 0.0, 0.0, 0.0],
            [0.0, 7.8125, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.46484375, 0.48828125],
            [0.0, 0.0, 0.0, 3.90625, 3.90625],
            [0.0, 0.0, 0.0, 4.39453125, 9.27734375],
        ])
        assert np.allclose(H, expected_H)


class TestContourAtLevel:

    # Given valid input values for x and y, the function should plot a contour at the level corresponding to the p_levels credible interval.
    def test_valid_input_values(self):
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        ax = contour_at_level(x, y, p_levels=0.9, bins=50, sigma_smooth=2.0)
        assert isinstance(ax, mpl.axes.Axes)
        assert len(ax.collections) == 1

    # When p_levels is a float, the function should plot a single contour at the level corresponding to the credible interval.
    def test_p_levels_float(self):
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        ax = contour_at_level(x, y, p_levels=0.9, bins=50, sigma_smooth=2.0)
        assert isinstance(ax, mpl.axes.Axes)
        assert len(ax.collections) == 1

    def test_p_levels_list(self):
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        ax = contour_at_level(x, y, p_levels=[0.9, 0.95], bins=50, sigma_smooth=2.0)
        assert isinstance(ax, mpl.axes.Axes)

    # When x and y are empty arrays, the function should raise an exception.
    def test_empty_arrays(self):
        x = np.array([])
        y = np.array([])
        with pytest.raises(Exception):
            contour_at_level(x, y, p_levels=0.9, bins=50, sigma_smooth=2.0)

    # When condition_function is provided, the function should plot a contour of the 2d histogram that satisfies the condition.
    def test_condition_function(self):
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        condition_function = lambda x, y: (x > 0) & (y > 0)
        ax = contour_at_level(
            x,
            y,
            p_levels=0.9,
            bins=50,
            sigma_smooth=2.0,
            condition_function=condition_function,
        )
        assert isinstance(ax, mpl.axes.Axes)


class TestGet2dBins:

    def test_basic_input(self):
        x, y = np.random.rand(100), np.random.rand(100)
        x_bins, y_bins = get_2d_bins(x, y)
        assert len(x_bins) > 0 and len(y_bins) > 0, "Bins should be non-empty"

    def test_max_bin_width_respected(self):
        max_bin_width = 0.1
        x, y = np.random.rand(100), np.random.rand(100)
        x_bins, y_bins = get_2d_bins(x, y, max_bin_width=max_bin_width)
        assert np.all(
            np.diff(x_bins) <= max_bin_width
        ), "Max bin width for x should be respected"
        assert np.all(
            np.diff(y_bins) <= max_bin_width
        ), "Max bin width for y should be respected"

    def test_negative_values_in_data(self):
        x, y = np.random.randn(100), np.random.randn(100)  # Includes negative values
        x_bins, y_bins = get_2d_bins(x, y)
        assert (
            len(x_bins) > 0 and len(y_bins) > 0
        ), "Bins should be non-empty even with negative values"

    def test_single_value_data(self):
        x, y = [0.5], [0.5]
        x_bins, y_bins = get_2d_bins(x, y)
        assert (
            len(x_bins) > 0 and len(y_bins) > 0
        ), "Bins should be non-empty even with single value data"

    def test_safety_factor(self):
        x, y = np.random.rand(100), np.random.rand(100)
        safety_factor = 2.0
        x_bins, y_bins = get_2d_bins(x, y, safety_factor=safety_factor)
        expected_x_range = (max(x) - min(x)) * safety_factor
        expected_y_range = (max(y) - min(y)) * safety_factor
        rx = x_bins[-1] - x_bins[0]
        ry = y_bins[-1] - y_bins[0]
        assert rx >= expected_x_range or math.isclose(
            rx - expected_x_range, 0
        ), "Safety factor for x is not enforced correctly"
        assert ry >= expected_y_range or math.isclose(
            ry - expected_y_range, 0
        ), "Safety factor for y is not enforced correctly"

class TestContourPlots:

    # Plots the contour for a given list of samples and labels.
    def test_plot_contour(self):
        # create sample data
        x1 = np.random.normal(0, 1, 1000)
        y1 = np.random.normal(0, 1, 1000)
        x2 = np.random.normal(2, 1, 1000)
        y2 = np.random.normal(2, 1, 1000)

        samples_list = [(x1, y1), (x2, y2)]

        labels = ["Sample 1", "Sample 2"]
        axes_labels = ["X", "Y"]

        # call with condition_function=lambda x, y: x<y
        ax, legend_ax = contour_plots(
            samples_list,
            labels,
            axes_labels=axes_labels,
            condition_function=lambda x, y: x < y,
            ax=None,
            legend_ax=None,
            colors=None,
            target_nbins=100,
            max_bin_width=4,
            linewidth=1.5,
            sigma_smooth=2,
            truncate=4,
            p_levels=[0.9],
            axes_lims=None,
            bins_2d_kwargs={},
            plot_pcolormesh=False,
        )
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

if __name__ == "__main__":
    pytest.main()
