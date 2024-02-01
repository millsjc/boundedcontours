# Description: Contains classes for generating random samples from 2d distributions
# and calculating their PDFs.

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
        # Extract x and y components
        x = xy[..., 0]
        y = xy[..., 1]
        # Compute the product of PDFs for x and y
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
