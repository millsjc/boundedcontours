Gaussian smoothing for 2D arrays where some condition must be maintained e.g. x>y. In the default setting this is achieved by a reflection about the boundary before the kernel convolution. See example [notebook](https://github.com/millsjc/boundedcontours/blob/main/examples/bounded_smoothing_and_contours.ipynb).

To install:
```bash
pip install boundedcontours
```

Or from source:
```bash
git clone https://github.com/millsjc/boundedcontours
cd boundedcontours
pip install .
```