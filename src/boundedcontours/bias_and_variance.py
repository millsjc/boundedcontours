import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from correlate import gaussian_filter2d


def generate_bounded_gaussian_data(samples=1000, mean=[0, 0], cov=[[1, 0], [0, 1]]):
    x, y = np.random.multivariate_normal(mean, cov, samples).T
    mask = x > y
    return np.vstack((x[mask], y[mask])).T


def create_histogram(data, bins=30):
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    return hist, xedges, yedges


def actual_density(x, y, mean, cov):
    rv = multivariate_normal(mean, cov)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    density = rv.pdf(np.array([x, y]).T).T
    density[x <= y] = 0  # Truncation where x > y
    # Normalization
    normalization = 0.5  # half of the area of the unit circle
    return density / normalization


def estimate_density(hist, function, **kwargs):
    dens = function(hist, sigma=1, **kwargs)
    return dens


def calculate_bias_and_variance(estimated_density, actual_density):
    bias = np.mean(estimated_density - actual_density)
    variance = np.var(estimated_density)
    return bias, variance


bins = 30
mean = [0, 0]
cov = [[1, 0], [0, 1]]
data = generate_bounded_gaussian_data()
hist, xedges, yedges = create_histogram(data, bins)

x = xedges[:-1] + (xedges[1] - xedges[0]) / 2
y = yedges[:-1] + (yedges[1] - yedges[0]) / 2
actual_dens = actual_density(x, y, mean, cov)

X, Y = np.meshgrid((xedges[1:] + xedges[:-1]) / 2, (yedges[1:] + yedges[:-1]) / 2)
cond = X < Y
estimated_density_scipy = estimate_density(hist, gaussian_filter)
estimated_density_bounded = estimate_density(hist, gaussian_filter2d, cond=cond)

bias, variance = calculate_bias_and_variance(
    estimated_density_scipy.flatten(), actual_dens
)
print(f"scipy gaussian_filter bias: {bias}")
print(f"scipy gaussian_filter variance: {variance}")

bias, variance = calculate_bias_and_variance(
    estimated_density_bounded.flatten(), actual_dens
)
print(f"bounded gaussian_filter2d bias: {bias}")
print(f"bounded gaussian_filter2d variance: {variance}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(
    hist.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
)
plt.title("Histogram")

plt.subplot(1, 4, 2)
plt.imshow(
    estimated_density_scipy.T,
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
)
plt.title(f"scipy gaussian_filter")

plt.subplot(1, 4, 3)
plt.imshow(
    actual_dens.reshape(bins, bins),
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
)
plt.title("Actual Truncated Gaussian Density")

plt.subplot(1, 4, 4)
plt.imshow(
    estimated_density_bounded.T,
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
)
plt.title(f"bounded gaussian_filter2d")
# plt.colorbar()
plt.show()
