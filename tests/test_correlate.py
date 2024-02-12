from boundedcontours.correlate import find_non_zero_islands


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
        ([0, 0, 0, 0], []),  # Test with all zeros
        ([1, 1, 1, 1], [slice(0, 4, None)]),  # Test with all ones
        # Alternating zeros and ones
        (
            [0, 1, 0, 1, 0, 1, 0],
            [slice(1, 2, None), slice(3, 4, None), slice(5, 6, None)],
        ),
        # Single island of length 5 surrounded by a single zero on each side
        ([0, 1, 1, 1, 1, 1, 0], [slice(1, 6, None)]),
    ]

    for input_array, expected_output in test_cases:
        # Call the function with the test case input
        result = find_non_zero_islands(input_array)
        # Assert that the result matches the expected output
        assert (
            result == expected_output
        ), f"Failed on input {input_array}. Expected {expected_output}, got {result}"

    print("All tests passed!")


def test_gaussian_filter_2d():
    import numpy as np
    from sympy import (
        Matrix,
        nsimplify,
    )
    from sympy.abc import a, b, c, d, e, f
    import boundedcontours.correlate

    # Define the matrix A
    A = Matrix([[0, 0, a], [0, b, c], [d, e, f]])
    A_numpy = np.array(A)
    cond = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    sigma = np.sqrt(-1 / (2 * np.log(1 / 2)))
    truncate = 5
    out = boundedcontours.correlate.gaussian_filter2d(
        A_numpy, sigma=sigma, truncate=truncate, cond=cond, output_dtype=object
    )
    out = nsimplify(Matrix(out))
    assert out.sum() == A.sum(), f"Expected sum of {A.sum()}, got {out.sum()}"


test_find_non_zero_islands()
