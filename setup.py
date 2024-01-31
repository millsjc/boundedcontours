from setuptools import setup, find_packages

setup(
    name='boundedcontours',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    url='',
    license='BSD 2-Clause License',
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your project',
    install_requires=[
        'numpy',
        'pytest',
        'plotly',
        'scipy',
        'matplotlib',
        'numba'
    ],
    python_requires='==3.8.*',
)
