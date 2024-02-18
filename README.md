Gaussian smoothing for 2D arrays while obeying boundaries. This is useful for smoothing a noisy 2d histogram but where some condition must be met e.g. x>y. In the default setting this is achieved by reflecting the array about the boundary before convolution with the kernel. See the [notebook](examples/bounded_smoothing_and_contours.ipynb) for a demonstration.

To install:
```bash
pip install boundedcontours
```