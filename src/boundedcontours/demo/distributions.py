import numpy as np
from scipy import stats


class IndependentExponential2d:
    def __init__(self, scale1, scale2):
        self.expon_x = stats.expon(scale=scale1)
        self.expon_y = stats.expon(scale=scale2)

    def rvs(self, size):
        x = self.expon_x.rvs(size)
        y = self.expon_y.rvs(size)
        return np.column_stack((x, y))

    def pdf(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        return self.expon_x.pdf(x) * self.expon_y.pdf(y)


class Uniform2d:
    def __init__(self):
        self.uniform_x = stats.uniform(0, 1)
        self.uniform_y = stats.uniform(0, 1)

    def rvs(self, size):
        x = self.uniform_x.rvs(size)
        y = self.uniform_y.rvs(size)
        return np.column_stack((x, y))

    def pdf(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        # Check if points are within the unit square
        return np.where((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1), 1, 0)


class IndependentBeta2d:
    def __init__(self, a1, b1, a2, b2):
        self.beta_x = stats.beta(a1, b1)
        self.beta_y = stats.beta(a2, b2)

    def rvs(self, size):
        x = self.beta_x.rvs(size)
        y = self.beta_y.rvs(size)
        return np.column_stack((x, y))

    def pdf(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        # Compute the product of PDFs for x and y
        return self.beta_x.pdf(x) * self.beta_y.pdf(y)


def no_condition(x, y):
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


tests = {
    "unconditioned_normal": {
        "dist": stats.multivariate_normal(np.zeros(2), np.eye(2)),
        "condition": no_condition,
        "bin_range": ((-3, 3), (-3, 3)),
    },
    "uniform_triangle": {
        "dist": Uniform2d(),
        "condition": in_triangle,
        "bin_range": ((0, 1), (0, 1)),
    },
    "normal_circle": {
        "dist": stats.multivariate_normal(np.zeros(2), np.eye(2)),
        "condition": in_circle,
        "bin_range": ((-1, 1), (-1, 1)),
    },
    "normal_first_quadrant": {
        "dist": stats.multivariate_normal(np.zeros(2), np.eye(2)),
        "condition": in_first_quadrant,
        "bin_range": ((-2, 2), (-5, 5)),
    },
    "normal_x_ge_y": {
        "dist": stats.multivariate_normal(np.zeros(2), np.eye(2)),
        "condition": x_ge_y,
        "bin_range": ((-3.05, 3.05), (-4, 4)),
    },
    "normal_x_le_y_covarying": {
        "dist": stats.multivariate_normal(np.zeros(2), [[1, 0.5], [0.5, 1]]),
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
