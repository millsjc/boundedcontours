import numpy as np
from scipy.ndimage import gaussian_filter
from boundedcontours.filter import find_non_zero_islands, gaussian_filter2d

def test_find_non_zero_islands():
    # Define test cases with expected outputs
    test_cases = [
        (
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [slice(4, 7, None), slice(9, 12, None)],
        ),
        (
            [1, 1, 0, 0, 0, 1, 0, 1, 1],
            [slice(0, 2, None), slice(5, 6, None), slice(7, 9, None)],
        ),
        ([0, 0, 0, 1], [slice(3, 4, None)]),
        ([1, 0, 1, 0, 1], [slice(0, 1, None), slice(2, 3, None), slice(4, 5, None)]),
        ([0, 0, 0, 0], []),
        ([1, 1, 1, 1], [slice(0, 4, None)]),
        # Alternating zeros and ones
        (
            [0, 1, 0, 1, 0, 1, 0],
            [slice(1, 2, None), slice(3, 4, None), slice(5, 6, None)],
        ),
        # Single island with a zero at each end
        ([0, 1, 1, 1, 1, 1, 0], [slice(1, 6, None)]),
    ]

    for input_array, expected_output in test_cases:
        # Call the function with the test case input
        result = find_non_zero_islands(input_array)
        # Assert that the result matches the expected output
        assert (
            result == expected_output
        ), f"Failed on input {input_array}. Expected {expected_output}, got {result}"

    print("Test passed: test_find_non_zero_islands")


def test_gaussian_filter_2d():
    condition_func = lambda x, y: x >= y
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x, y = np.indices(a.shape)
    cond = condition_func(x, y)
    a[~cond] = 0
    # Test with above input with different sigma and truncate values
    for sigma, truncate, expected_output in [
        (1, 1, np.array([[1, 0, 0], [4, 5, 0], [6, 6, 8]])),
        (3, 4, np.array([[3, 0, 0], [4, 5, 0], [4, 6, 8]])),
    ]:
        h_bounded = gaussian_filter2d(a, sigma=sigma, truncate=truncate, cond=cond)
        assert np.allclose(
            h_bounded, expected_output
        ), f"Expected {expected_output}, got {h_bounded}"

    # Test with no condition
    a = np.ones((3, 3))
    a[1, 1] = 3
    sigma = np.sqrt(-1 / (2 * np.log(1 / 2)))
    h_bounded = gaussian_filter2d(a, sigma=sigma, truncate=1)
    expected_output = np.array(
        [[1.125, 1.25, 1.125], [1.25, 1.5, 1.25], [1.125, 1.25, 1.125]]
    )
    assert np.allclose(
        h_bounded, expected_output
    ), f"Expected {expected_output}, got {h_bounded}"
    print("Test passed: test_gaussian_filter_2d_without_cond")

    # Check output is the same as scipy's gaussian_filter
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
        assert np.allclose(out, out2), "gaussian_filter2d and gaussian_filter disagree"

    print("Test passed: test_gaussian_filter_2d")


test_find_non_zero_islands()
test_gaussian_filter_2d()
