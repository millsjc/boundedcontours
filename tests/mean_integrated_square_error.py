import numpy as np
from utils import plot_pdf_and_3_hists, get_pdf_and_histograms


plotly = True
sigma_smooth = 1
truncate_smooth = 1
sample_size = int(2e4)
p_levels = [0.9]
mean = [0, 0]
cov = [[1, 0], [0, 1]]
# Bins for the histogram
nbins = 30
x_min, x_max, y_min, y_max = -3, 3, -3, 3
bin_range = ((x_min, x_max), (y_min, y_max))
dx = (x_max - x_min) / nbins
dy = (y_max - y_min) / nbins
x_edges = np.linspace(x_min, x_max, nbins + 1)
y_edges = np.linspace(y_min, y_max, nbins + 1)
X, Y = np.meshgrid(x_edges[:-1] + dx / 2, y_edges[:-1] + dy / 2)

n_simulations = 10

out = np.array(
    [
        get_pdf_and_histograms(
            sample_size, mean, cov, x_edges, y_edges, sigma_smooth, truncate_smooth
        )
        for i in range(n_simulations)
    ]
)

p, h, h_smooth_scipy, h_smooth_bounded = np.einsum("ijkl->jikl", out)

mise = {}
for label, estimate in zip(
    ["histogram", "smooth", "bounded"], [h, h_smooth_scipy, h_smooth_bounded]
):
    mise[label] = np.einsum("ijk->", (estimate - p) ** 2) * dx * dy / n_simulations
    print(f"MISE {label}: {mise[label]}")

# Step 6: Plot an example
i = 0
plot_pdf_and_3_hists(
    X, Y, p[i], h[i], h_smooth_scipy[i], h_smooth_bounded[i], p_levels, plotly
)
